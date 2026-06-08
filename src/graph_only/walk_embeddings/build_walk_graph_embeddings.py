#!/usr/bin/env python3
"""
Build lightweight node2vec and metapath2vec-style award embeddings for MethodKG.

This script is intentionally CPU-friendly and dependency-light. It builds embeddings
from the full MethodKG corpus using random walks + gensim Word2Vec.

Important interpretation:
  - node2vec/metapath2vec are transductive graph embedding baselines.
  - They should be reported as graph-embedding baselines, not as strictly inductive GNNs.
  - For temporal-leakage-critical claims, pair these with your leakage-safe historical
    feature baseline and later GraphSAGE/HGT experiments.

Inputs:
  cleaned_nsf_awards_2000_2025.csv
  methodkg_labeled_benchmark_v2_modeling.csv (optional, used for coverage reports)

Outputs:
  node2vec_award_embeddings.csv
  metapath2vec_award_embeddings.csv
  embedding_build_report.csv
  walk_build_summary.json
"""

import argparse
import ast
import csv
import hashlib
import json
import math
import random
import re
import time
import zipfile
from collections import defaultdict, Counter
from pathlib import Path
import sys
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from path_utils import discover_awards, discover_benchmark, find_repo_root, read_csv_or_zip, reset_dir_if_overwrite, resolve_existing_path, resolve_output_path, write_resolved_paths
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def normalize_for_id(x) -> str:
    s = normalize_text(x).lower()
    s = NON_ALNUM_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def stable_id(prefix: str, value: str, n: int = 12) -> str:
    value = normalize_for_id(value)
    if not value:
        value = "unknown"
    h = hashlib.sha1(value.encode("utf-8")).hexdigest()[:n]
    return f"{prefix}_{h}"


def clean_award_id(x) -> str:
    s = normalize_text(x)
    if s.startswith('="') and s.endswith('"'):
        s = s[2:-1]
    s = re.sub(r"\D", "", s)
    return s.strip()


def split_codes(value) -> List[str]:
    s = normalize_text(value)
    if not s:
        return []
    parts = re.split(r"[,;|/\s]+", s)
    out, seen = [], set()
    for p in parts:
        p = re.sub(r"[^A-Za-z0-9]", "", p).upper().strip()
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def load_csv_maybe_zip(path: str | Path, preferred_name: str = "methodkg_labeled_benchmark_v3_modeling.csv") -> pd.DataFrame:
    return read_csv_or_zip(path, preferred_name=preferred_name)


def parse_co_pi_records(value) -> List[Dict[str, str]]:
    """Parse cleaned co_pi_records if present; otherwise return empty."""
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    s = str(value).strip()
    if not s or s in {"[]", "nan", "None"}:
        return []
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            out = []
            for item in obj:
                if isinstance(item, dict):
                    out.append({
                        "person_id": str(item.get("person_id", "")),
                        "name": str(item.get("name", "")),
                        "email": str(item.get("email", "")),
                    })
                elif isinstance(item, str):
                    out.append({"person_id": stable_id("person", item), "name": item, "email": ""})
            return out
    except Exception:
        return []
    return []


def extract_people_from_awards(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Return award_id -> list of person node IDs using lead PI and Co-PI records."""
    people_by_award: Dict[str, List[str]] = defaultdict(list)
    for _, r in df.iterrows():
        aid = clean_award_id(r.get("award_id", r.get("AwardNumber", "")))
        if not aid:
            continue
        seen = set()
        lead_pid = normalize_text(r.get("person_id", r.get("pi_id", "")))
        if not lead_pid:
            name = normalize_text(r.get("pi_clean", r.get("PrincipalInvestigator", "")))
            email = normalize_text(r.get("pi_email_extracted", r.get("PIEmailAddress", ""))).lower()
            key = f"email:{email}" if email else f"name:{name}"
            lead_pid = stable_id("person", key)
        if lead_pid and lead_pid.lower() not in {"nan", "none"}:
            node = f"p:{lead_pid}"
            seen.add(node)
            people_by_award[aid].append(node)

        if "co_pi_records" in df.columns:
            for rec in parse_co_pi_records(r.get("co_pi_records", "")):
                pid = normalize_text(rec.get("person_id", ""))
                if not pid:
                    email = normalize_text(rec.get("email", "")).lower()
                    name = normalize_text(rec.get("name", ""))
                    key = f"email:{email}" if email else f"name:{name}"
                    pid = stable_id("person", key)
                node = f"p:{pid}"
                if pid and node not in seen:
                    seen.add(node)
                    people_by_award[aid].append(node)
    return people_by_award


def extract_award_entities(df: pd.DataFrame) -> Tuple[List[str], Dict[str, Dict[str, List[str]]]]:
    """Extract typed entity nodes for each award."""
    awards = []
    ent: Dict[str, Dict[str, List[str]]] = {}
    people_by_award = extract_people_from_awards(df)

    for _, r in df.iterrows():
        aid = clean_award_id(r.get("award_id", r.get("AwardNumber", "")))
        if not aid:
            continue
        awards.append(aid)
        e = defaultdict(list)

        # Person nodes.
        e["person"].extend(people_by_award.get(aid, []))

        # Institution.
        inst_id = normalize_text(r.get("institution_id", ""))
        if not inst_id:
            org = normalize_text(r.get("organization_clean", r.get("Organization", "")))
            inst_id = stable_id("inst", org) if org else ""
        if inst_id:
            e["institution"].append(f"i:{inst_id}")

        # Program code(s), fallback to Program(s).
        codes = split_codes(r.get("ProgramElementCode(s)", ""))
        if codes:
            for c in codes:
                e["program"].append(f"g:element_{c}")
        else:
            prog = normalize_text(r.get("Program(s)", ""))
            if prog:
                e["program"].append(f"g:{stable_id('program', prog)}")

        org_unit = normalize_text(r.get("NSFOrganization", ""))
        if org_unit:
            e["nsforg"].append(f"o:{org_unit.upper()}")

        directorate = normalize_text(r.get("NSFDirectorate", ""))
        if directorate:
            e["directorate"].append(f"d:{directorate.upper()}")

        state = normalize_text(r.get("State", r.get("OrganizationState", "")))
        if state:
            e["state"].append(f"s:{state.upper()}")

        year = r.get("start_year", r.get("StartDate", ""))
        try:
            y = int(float(year))
            if 1900 <= y <= 2100:
                e["year"].append(f"y:{y}")
        except Exception:
            pass

        # Deduplicate per type.
        ent[aid] = {k: sorted(set(v)) for k, v in e.items() if v}

    return sorted(set(awards)), ent


def add_weighted_edge(adj: Dict[str, Dict[str, float]], a: str, b: str, w: float):
    if a == b:
        return
    adj[a][b] = adj[a].get(b, 0.0) + w
    adj[b][a] = adj[b].get(a, 0.0) + w


def build_award_projection(
    awards: List[str],
    entities_by_award: Dict[str, Dict[str, List[str]]],
    sources: Sequence[str],
    weights: Dict[str, float],
    max_group_size: int,
    max_edges_per_large_group: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    rng = random.Random(seed)
    adj: Dict[str, Dict[str, float]] = defaultdict(dict)
    for aid in awards:
        adj[aid]  # ensure isolated nodes exist

    group_to_awards: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for aid, typed in entities_by_award.items():
        for source in sources:
            for node in typed.get(source, []):
                group_to_awards[(source, node)].append(aid)

    skipped_large = Counter()
    sampled_large = Counter()
    for (source, node), group in group_to_awards.items():
        group = sorted(set(group))
        if len(group) < 2:
            continue
        w = weights.get(source, 1.0)
        n = len(group)
        if n <= max_group_size:
            for i in range(n):
                ai = group[i]
                for j in range(i + 1, n):
                    add_weighted_edge(adj, ai, group[j], w)
        else:
            # Avoid giant cliques. Sample sparse pairs from large groups.
            if max_edges_per_large_group <= 0:
                skipped_large[source] += 1
                continue
            possible = n * (n - 1) // 2
            m = min(max_edges_per_large_group, possible)
            seen_pairs = set()
            tries = 0
            while len(seen_pairs) < m and tries < m * 10:
                i, j = rng.sample(range(n), 2)
                if i > j:
                    i, j = j, i
                if (i, j) in seen_pairs:
                    tries += 1
                    continue
                seen_pairs.add((i, j))
                add_weighted_edge(adj, group[i], group[j], w)
                tries += 1
            sampled_large[source] += 1

    return adj


def weighted_choice(neighbors: List[str], weights: List[float], rng: random.Random) -> str:
    total = sum(weights)
    if total <= 0:
        return rng.choice(neighbors)
    r = rng.random() * total
    upto = 0.0
    for n, w in zip(neighbors, weights):
        upto += w
        if upto >= r:
            return n
    return neighbors[-1]


def node2vec_walk(adj: Dict[str, Dict[str, float]], start: str, length: int, p: float, q: float, rng: random.Random) -> List[str]:
    walk = [start]
    if start not in adj or not adj[start]:
        return walk
    for _ in range(length - 1):
        cur = walk[-1]
        nbrs = list(adj.get(cur, {}).keys())
        if not nbrs:
            break
        if len(walk) == 1:
            weights = [adj[cur][x] for x in nbrs]
        else:
            prev = walk[-2]
            prev_neighbors = set(adj.get(prev, {}).keys())
            weights = []
            for x in nbrs:
                base = adj[cur][x]
                if x == prev:
                    weights.append(base / max(p, 1e-9))
                elif x in prev_neighbors:
                    weights.append(base)
                else:
                    weights.append(base / max(q, 1e-9))
        walk.append(weighted_choice(nbrs, weights, rng))
    return walk


def generate_node2vec_walks(adj, num_walks, walk_length, p, q, seed) -> List[List[str]]:
    rng = random.Random(seed)
    nodes = list(adj.keys())
    walks = []
    for i in range(num_walks):
        rng.shuffle(nodes)
        for n in nodes:
            walks.append(node2vec_walk(adj, n, walk_length, p, q, rng))
    return walks


def build_hetero_adjacency(awards: List[str], entities_by_award: Dict[str, Dict[str, List[str]]], sources: Sequence[str]):
    """Build typed undirected adjacency for metapath walks."""
    adj_by_type: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    node_type: Dict[str, str] = {}

    for aid in awards:
        a_node = f"a:{aid}"
        node_type[a_node] = "award"
        for source in sources:
            for ent_node in entities_by_award.get(aid, {}).get(source, []):
                # ent_node already has type prefix, but source is the semantic type.
                node_type[ent_node] = source
                adj_by_type[("award", source)][a_node].append(ent_node)
                adj_by_type[(source, "award")][ent_node].append(a_node)

    for key in list(adj_by_type.keys()):
        for n in list(adj_by_type[key].keys()):
            adj_by_type[key][n] = sorted(set(adj_by_type[key][n]))
    return adj_by_type, node_type


def parse_metapaths(s: str) -> List[List[str]]:
    paths = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        parts = [p.strip().lower() for p in re.split(r"[-:>]", item) if p.strip()]
        if len(parts) >= 2:
            paths.append(parts)
    return paths


def metapath_walk(adj_by_type, start_award_id: str, metapath: List[str], length: int, rng: random.Random) -> List[str]:
    # metapath is type sequence, e.g. award-person-award. Repeat its transitions.
    start = f"a:{start_award_id}"
    if metapath[0] != "award":
        metapath = ["award"] + metapath
    walk = [start]
    if len(metapath) < 2:
        return walk
    for step in range(length - 1):
        cur = walk[-1]
        from_type = metapath[step % (len(metapath) - 1)]
        to_type = metapath[(step + 1) % (len(metapath) - 1)]
        # If path ends in award and repeats, next transition should follow first edge again.
        if step % (len(metapath) - 1) == len(metapath) - 2 and metapath[-1] == "award":
            from_type = metapath[-2]
            to_type = "award"
        nbrs = adj_by_type.get((from_type, to_type), {}).get(cur, [])
        if not nbrs:
            break
        walk.append(rng.choice(nbrs))
    return walk


def generate_metapath_walks(awards, adj_by_type, metapaths, num_walks, walk_length, seed) -> List[List[str]]:
    rng = random.Random(seed)
    starts = list(awards)
    walks = []
    for path in metapaths:
        for i in range(num_walks):
            rng.shuffle(starts)
            for aid in starts:
                w = metapath_walk(adj_by_type, aid, path, walk_length, rng)
                if len(w) > 1:
                    walks.append(w)
    return walks


def train_word2vec(walks: List[List[str]], dim: int, window: int, workers: int, epochs: int, seed: int) -> Word2Vec:
    if not walks:
        raise ValueError("No walks generated; cannot train embeddings.")
    model = Word2Vec(
        sentences=walks,
        vector_size=dim,
        window=window,
        min_count=1,
        sg=1,
        negative=10,
        workers=workers,
        epochs=epochs,
        seed=seed,
    )
    return model


def save_award_embeddings(model: Word2Vec, award_ids: Sequence[str], outpath: Path, dim: int, prefix: str = ""):
    rows = []
    for aid in award_ids:
        key = f"{prefix}{aid}" if prefix else aid
        if key in model.wv:
            vec = model.wv[key]
            missing = 0
        else:
            vec = np.zeros(dim, dtype=float)
            missing = 1
        row = {"award_id": aid, "embedding_missing": missing}
        for j, v in enumerate(vec):
            row[f"emb_{j:03d}"] = float(v)
        rows.append(row)
    pd.DataFrame(rows).to_csv(outpath, index=False)


def graph_stats(adj) -> Dict[str, float]:
    degs = [len(v) for v in adj.values()]
    edge_count = sum(degs) // 2
    return {
        "nodes": len(adj),
        "edges": edge_count,
        "degree_mean": float(np.mean(degs)) if degs else 0.0,
        "degree_median": float(np.median(degs)) if degs else 0.0,
        "degree_max": int(max(degs)) if degs else 0,
        "isolated_nodes": int(sum(1 for d in degs if d == 0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=None, help="Repository root. Defaults to auto-detection.")
    ap.add_argument("--awards", default=None, help="cleaned_nsf_awards_2000_2025.csv. Defaults to data/processed discovery.")
    ap.add_argument("--benchmark", default="", help="methodkg_labeled_benchmark_v3_modeling.csv or benchmark_v3.zip; defaults to data/benchmark discovery and is used for coverage report")
    ap.add_argument("--outdir", default=None, help="Output directory. Defaults to artifacts/features/walk_embeddings_v1")
    ap.add_argument("--overwrite", action="store_true", help="Delete the output directory before writing new embedding artifacts.")
    ap.add_argument("--method", choices=["node2vec", "metapath2vec", "both"], default="both")
    ap.add_argument("--embedding_dim", type=int, default=64)
    ap.add_argument("--walk_length", type=int, default=20)
    ap.add_argument("--num_walks", type=int, default=8)
    ap.add_argument("--window_size", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--node2vec_p", type=float, default=1.0)
    ap.add_argument("--node2vec_q", type=float, default=1.0)
    ap.add_argument("--projection_sources", nargs="+", default=["person", "institution", "program"],
                    choices=["person", "institution", "program", "nsforg", "directorate", "state", "year"])
    ap.add_argument("--metapath_sources", nargs="+", default=["person", "institution", "program", "nsforg"],
                    choices=["person", "institution", "program", "nsforg", "directorate", "state", "year"])
    ap.add_argument("--metapaths", default="award-person-award,award-institution-award,award-program-award,award-nsforg-award")
    ap.add_argument("--max_group_size", type=int, default=250,
                    help="Max group size for complete award projection cliques.")
    ap.add_argument("--max_edges_per_large_group", type=int, default=5000,
                    help="Sparse sampled pair edges for groups larger than max_group_size. Set 0 to skip large groups.")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    repo_root = find_repo_root(args.repo_root or Path(__file__).resolve())
    awards_path = resolve_existing_path(args.awards, repo_root) if args.awards else discover_awards(repo_root)
    benchmark_path = resolve_existing_path(args.benchmark, repo_root, required=False) if args.benchmark else discover_benchmark(repo_root)
    outdir = resolve_output_path(args.outdir, repo_root, repo_root / "artifacts" / "features" / "walk_embeddings_v1")
    reset_dir_if_overwrite(outdir, args.overwrite)
    write_resolved_paths(repo_root=repo_root, awards=awards_path, benchmark=benchmark_path, outdir=outdir)

    t0 = time.time()
    awards_df = pd.read_csv(awards_path, low_memory=False)
    awards_df["award_id"] = awards_df.get("award_id", awards_df.get("AwardNumber", "")).apply(clean_award_id)
    awards_df = awards_df[awards_df["award_id"].astype(str) != ""].drop_duplicates("award_id")

    awards, entities = extract_award_entities(awards_df)
    report_rows = []
    report_rows.append({"section": "input", "metric": "award_rows", "value": len(awards_df)})
    report_rows.append({"section": "input", "metric": "unique_awards", "value": len(awards)})

    benchmark_awards = []
    if benchmark_path:
        bench = load_csv_maybe_zip(benchmark_path)
        if "award_id" in bench.columns:
            benchmark_awards = [clean_award_id(x) for x in bench["award_id"].tolist()]
            report_rows.append({"section": "benchmark", "metric": "benchmark_rows", "value": len(benchmark_awards)})
            report_rows.append({"section": "benchmark", "metric": "benchmark_unique_awards", "value": len(set(benchmark_awards))})

    # Entity coverage.
    for typ in ["person", "institution", "program", "nsforg", "directorate", "state", "year"]:
        count = sum(1 for aid in awards if entities.get(aid, {}).get(typ))
        report_rows.append({"section": "entity_coverage", "metric": f"awards_with_{typ}", "value": count})

    summary = {
        "award_rows": int(len(awards_df)),
        "unique_awards": int(len(awards)),
        "method": args.method,
        "embedding_dim": args.embedding_dim,
        "walk_length": args.walk_length,
        "num_walks": args.num_walks,
        "window_size": args.window_size,
        "epochs": args.epochs,
        "projection_sources": args.projection_sources,
        "metapath_sources": args.metapath_sources,
        "metapaths": args.metapaths,
    }

    weights = {
        "person": 3.0,
        "institution": 1.0,
        "program": 1.0,
        "nsforg": 0.5,
        "directorate": 0.25,
        "state": 0.25,
        "year": 0.10,
    }

    if args.method in {"node2vec", "both"}:
        print("Building award-award projection for node2vec...")
        adj = build_award_projection(
            awards, entities, args.projection_sources, weights,
            args.max_group_size, args.max_edges_per_large_group, args.seed,
        )
        stats = graph_stats(adj)
        summary["node2vec_graph"] = stats
        for k, v in stats.items():
            report_rows.append({"section": "node2vec_graph", "metric": k, "value": v})
        print("Generating node2vec walks...")
        walks = generate_node2vec_walks(adj, args.num_walks, args.walk_length, args.node2vec_p, args.node2vec_q, args.seed)
        summary["node2vec_walk_count"] = len(walks)
        report_rows.append({"section": "node2vec", "metric": "walk_count", "value": len(walks)})
        print("Training node2vec Word2Vec model...")
        model = train_word2vec(walks, args.embedding_dim, args.window_size, args.workers, args.epochs, args.seed)
        model.save(str(outdir / "node2vec_word2vec.model"))
        save_award_embeddings(model, awards, outdir / "node2vec_award_embeddings.csv", args.embedding_dim, prefix="")
        if benchmark_awards:
            missing = sum(1 for aid in benchmark_awards if aid not in model.wv)
            report_rows.append({"section": "node2vec", "metric": "benchmark_missing_embeddings", "value": missing})

    if args.method in {"metapath2vec", "both"}:
        print("Building heterogeneous adjacency for metapath2vec...")
        adj_by_type, node_type = build_hetero_adjacency(awards, entities, args.metapath_sources)
        metapaths = parse_metapaths(args.metapaths)
        summary["parsed_metapaths"] = metapaths
        report_rows.append({"section": "metapath2vec", "metric": "metapath_count", "value": len(metapaths)})
        for key, mapping in adj_by_type.items():
            edges = sum(len(v) for v in mapping.values())
            report_rows.append({"section": "hetero_edges", "metric": f"{key[0]}_to_{key[1]}", "value": edges})
        print("Generating metapath2vec walks...")
        walks = generate_metapath_walks(awards, adj_by_type, metapaths, args.num_walks, args.walk_length, args.seed)
        summary["metapath2vec_walk_count"] = len(walks)
        report_rows.append({"section": "metapath2vec", "metric": "walk_count", "value": len(walks)})
        print("Training metapath2vec Word2Vec model...")
        model = train_word2vec(walks, args.embedding_dim, args.window_size, args.workers, args.epochs, args.seed)
        model.save(str(outdir / "metapath2vec_word2vec.model"))
        save_award_embeddings(model, awards, outdir / "metapath2vec_award_embeddings.csv", args.embedding_dim, prefix="a:")
        if benchmark_awards:
            missing = sum(1 for aid in benchmark_awards if f"a:{aid}" not in model.wv)
            report_rows.append({"section": "metapath2vec", "metric": "benchmark_missing_embeddings", "value": missing})

    summary["elapsed_seconds"] = round(time.time() - t0, 3)
    pd.DataFrame(report_rows).to_csv(outdir / "embedding_build_report.csv", index=False)
    with open(outdir / "walk_build_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Done. Outputs written to", outdir)


if __name__ == "__main__":
    main()

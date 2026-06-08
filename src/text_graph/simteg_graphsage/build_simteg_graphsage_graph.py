#!/usr/bin/env python3
"""
Build a MethodKG SimTeG-style GraphSAGE award graph with text embeddings.

This script creates a PyTorch Geometric Data object with one node per NSF award.
The node set is built from the UNION of the full cleaned NSF corpus and the
labeled benchmark so all benchmark rows are retained for evaluation.
Edges are temporal award-projection edges: prior awards connect to later awards
when they share a PI/Co-PI, institution, program element code, NSF organization,
or directorate. The default graph is directed from older awards to newer awards,
which is a safer default for temporal/generalization experiments than a fully
undirected projection.

Inputs:
  cleaned_nsf_awards_2000_2025.csv
  methodkg_labeled_benchmark_v2_modeling.csv
  optional award_pi_edges.csv

Outputs:
  simteg_graphsage_award_graph.pt
  simteg_graphsage_node_index.csv
  simteg_graphsage_edge_summary.csv
  simteg_graphsage_feature_manifest.csv
  simteg_graphsage_build_summary.json
  simteg_benchmark_award_match_report.csv
  simteg_benchmark_added_from_benchmark_only.csv
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from path_utils import default_simteg_graph_dir, default_simteg_text_embeddings, discover_award_pi_edges, discover_awards, discover_benchmark, find_repo_root, read_csv_or_zip, reset_dir_if_overwrite, resolve_existing_path, resolve_output_path, write_resolved_paths

try:
    from torch_geometric.data import Data
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "torch_geometric is required. Install PyTorch Geometric for your CUDA/PyTorch version. "
        "See README.md in this package. Original import error: " + repr(exc)
    )


TARGET_COLUMNS_DEFAULT = [
    "target_integration_binary",
    "target_design_binary",
    "target_mmr_binary",
    "target_mmr_multiclass",
    "target_qual_binary",
    "target_quant_binary",
    "target_method_signal_binary",
]

SPLIT_COLUMNS_DEFAULT = [
    "split_random_cluster_stratified",
    "split_temporal_cluster_safe",
    "split_cross_program_cluster_safe",
    "split_cold_start_pi_cluster_safe",
    "split_cold_start_institution_cluster_safe",
    "split_edu_to_eng_cluster_safe",
]


def clean_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_key(x) -> str:
    s = clean_str(x).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def split_program_codes(value) -> List[str]:
    s = clean_str(value)
    if not s:
        return []
    parts = re.split(r"[,;|/\s]+", s)
    out = []
    seen = set()
    for p in parts:
        p = re.sub(r"[^A-Za-z0-9]", "", p).upper()
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def read_csv_flexible(path: str | Path) -> pd.DataFrame:
    return read_csv_or_zip(path, dtype=str, encoding="utf-8-sig")


def clean_award_id_value(x) -> str:
    """Return a safe string award id without corrupting decimal-like ids.

    Older versions used str.replace(r"\\D", ""), which turns values like
    "1922666.0" into "19226660". This function removes Excel wrappers and
    decimal .0 suffixes before keeping digits.
    """
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.startswith('="') and s.endswith('"'):
        s = s[2:-1].strip()
    elif s.startswith("='") and s.endswith("'"):
        s = s[2:-1].strip()
    s = s.strip().strip('"').strip("'").strip()

    # Common CSV artifact: integer IDs read as floats.
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".", 1)[0]
    # Scientific notation artifact, e.g. 1.922666e6.
    elif re.fullmatch(r"\d+(?:\.\d+)?[eE][+-]?\d+", s):
        try:
            val = float(s)
            if math.isfinite(val) and abs(val - round(val)) < 1e-6:
                s = str(int(round(val)))
        except Exception:
            pass

    digits = re.sub(r"\D", "", s)
    return digits.strip()


def award_id_match_key(award_id: str) -> str:
    """Match key that ignores leading zeros but preserves the display award_id."""
    aid = clean_award_id_value(award_id)
    if not aid:
        return ""
    return aid.lstrip("0") or "0"


def ensure_award_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "award_id" not in out.columns:
        if "AwardNumber" not in out.columns:
            raise ValueError("Expected either award_id or AwardNumber column.")
        out["award_id"] = out["AwardNumber"].apply(clean_award_id_value)
    else:
        out["award_id"] = out["award_id"].apply(clean_award_id_value)
    out["award_id_key"] = out["award_id"].apply(award_id_match_key)
    return out


def add_basic_columns(awards: pd.DataFrame) -> pd.DataFrame:
    out = ensure_award_id(awards)
    if "start_year" not in out.columns:
        if "StartDate" in out.columns:
            out["start_year"] = pd.to_datetime(out["StartDate"], errors="coerce").dt.year
        else:
            raise ValueError("Awards file needs start_year or StartDate.")
    out["start_year"] = pd.to_numeric(out["start_year"], errors="coerce")

    for col in ["NSFDirectorate", "NSFOrganization", "AwardInstrument", "ProgramElementCode(s)"]:
        if col not in out.columns:
            out[col] = ""
    if "institution_id" not in out.columns:
        if "organization_clean" in out.columns:
            out["institution_id"] = out["organization_clean"].map(lambda s: "inst_" + normalize_key(s))
        elif "Organization" in out.columns:
            out["institution_id"] = out["Organization"].map(lambda s: "inst_" + normalize_key(s))
        else:
            out["institution_id"] = ""
    if "organization_clean" not in out.columns:
        out["organization_clean"] = out.get("Organization", "")
    if "person_id" not in out.columns:
        out["person_id"] = ""
    for col in ["team_size", "num_co_pis", "abstract_word_count", "title_word_count"]:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    return out


def merge_benchmark_metadata(
    awards: pd.DataFrame,
    benchmark: pd.DataFrame,
    outdir: Optional[Path] = None,
) -> pd.DataFrame:
    """Attach benchmark labels/splits and retain all benchmark awards.

    v1 used the cleaned awards file as the only node source, so benchmark awards
    absent from the cleaned corpus were silently dropped. v2 builds from the
    union of cleaned awards and benchmark rows. Benchmark-only awards are kept as
    valid nodes; they may have fewer graph edges if their corpus metadata is
    limited, but they remain in train/validation/test evaluation.
    """
    awards = ensure_award_id(awards)
    bench = ensure_award_id(benchmark)

    awards = awards[awards["award_id_key"] != ""].copy()
    bench = bench[bench["award_id_key"] != ""].copy()

    # De-duplicate by match key, preferring the latest/first cleaned award row.
    awards_unique = awards.drop_duplicates(subset=["award_id_key"], keep="first").copy()
    bench_unique = bench.drop_duplicates(subset=["award_id_key"], keep="first").copy()

    keep_cols = ["award_id", "award_id_key"]
    for col in TARGET_COLUMNS_DEFAULT + SPLIT_COLUMNS_DEFAULT + [
        "benchmark_id", "annotation_id", "project_cluster_id", "primary_program_key",
        "target_mmr_multiclass", "label_mmr_class",
        "title_clean", "abstract_clean", "start_year", "NSFDirectorate", "NSFOrganization",
        "Program(s)", "ProgramElementCode(s)", "AwardInstrument", "person_id", "pi_clean",
        "organization_clean", "institution_id", "State", "OrganizationState", "team_size",
        "num_co_pis", "abstract_word_count", "title_word_count",
    ]:
        if col in bench_unique.columns and col not in keep_cols:
            keep_cols.append(col)
    bench_small = bench_unique[keep_cols].copy()

    # Match report before union.
    award_keys = set(awards_unique["award_id_key"])
    bench_keys = set(bench_small["award_id_key"])
    matched_keys = award_keys & bench_keys
    missing_keys = bench_keys - award_keys
    report_rows = [
        {"metric": "cleaned_awards_unique_keys", "value": len(award_keys)},
        {"metric": "benchmark_unique_keys", "value": len(bench_keys)},
        {"metric": "benchmark_matched_in_cleaned_awards", "value": len(matched_keys)},
        {"metric": "benchmark_missing_from_cleaned_awards_added_as_nodes", "value": len(missing_keys)},
    ]

    # Left merge labels/splits into full cleaned corpus.
    bench_meta_cols = [c for c in bench_small.columns if c not in {"award_id"}]
    out = awards_unique.merge(
        bench_small[bench_meta_cols],
        on="award_id_key",
        how="left",
        suffixes=("", "_bench"),
    )

    # For matched benchmark rows, prefer benchmark award_id spelling/display.
    if "award_id_bench" in out.columns:
        out["award_id"] = out["award_id_bench"].fillna(out["award_id"])
        out = out.drop(columns=["award_id_bench"])

    # Add benchmark-only rows that were not in cleaned awards. Ensure all columns exist.
    missing_bench = bench_small[bench_small["award_id_key"].isin(missing_keys)].copy()
    missing_added = pd.DataFrame(columns=out.columns)
    if len(missing_bench):
        # Start with all current output columns so concatenation is stable.
        for col in out.columns:
            if col not in missing_bench.columns:
                missing_bench[col] = np.nan
        missing_added = missing_bench[out.columns].copy()
        # Fill institution_id from organization if needed.
        if "institution_id" in missing_added.columns:
            inst_missing = missing_added["institution_id"].fillna("").astype(str).str.strip() == ""
            if "organization_clean" in missing_added.columns:
                missing_added.loc[inst_missing, "institution_id"] = missing_added.loc[inst_missing, "organization_clean"].map(lambda s: "inst_" + normalize_key(s))
        out = pd.concat([out, missing_added], ignore_index=True)

    # Fill cleaned-awards columns from benchmark versions where pandas created suffixes.
    for col in [
        "title_clean", "abstract_clean", "start_year", "NSFDirectorate", "NSFOrganization",
        "Program(s)", "ProgramElementCode(s)", "AwardInstrument", "person_id", "pi_clean",
        "organization_clean", "institution_id", "State", "OrganizationState", "team_size",
        "num_co_pis", "abstract_word_count", "title_word_count",
    ]:
        bench_col = col + "_bench"
        if bench_col in out.columns:
            if col in out.columns:
                empty = out[col].isna() | (out[col].astype(str).str.strip() == "")
                out.loc[empty, col] = out.loc[empty, bench_col]
                out = out.drop(columns=[bench_col])
            else:
                out = out.rename(columns={bench_col: col})

    out["is_benchmark"] = out["award_id_key"].isin(bench_keys).astype(int)
    out["node_source"] = np.where(out["award_id_key"].isin(missing_keys), "benchmark_only_added", "cleaned_awards")

    if outdir is not None:
        pd.DataFrame(report_rows).to_csv(outdir / "simteg_benchmark_award_match_report.csv", index=False, encoding="utf-8-sig")
        missing_cols = [c for c in [
            "award_id", "award_id_key", "benchmark_id", "annotation_id", "title_clean", "start_year",
            "NSFDirectorate", "NSFOrganization", "ProgramElementCode(s)", "organization_clean", "person_id",
        ] if c in missing_added.columns]
        missing_added[missing_cols].to_csv(outdir / "simteg_benchmark_added_from_benchmark_only.csv", index=False, encoding="utf-8-sig")

    return out


def top_categories(series: pd.Series, top_k: int) -> List[str]:
    if top_k <= 0:
        return []
    s = series.fillna("").astype(str).str.strip()
    s = s[s != ""]
    return s.value_counts().head(top_k).index.tolist()


def make_features(df: pd.DataFrame, top_program_k: int, top_instrument_k: int) -> Tuple[np.ndarray, List[Dict[str, str]]]:
    """Create non-text structural/categorical node features for every award."""
    feature_parts = []
    manifest = []

    year = pd.to_numeric(df["start_year"], errors="coerce")
    min_year = float(np.nanmin(year)) if year.notna().any() else 2000.0
    max_year = float(np.nanmax(year)) if year.notna().any() else 2025.0
    denom = max(max_year - min_year, 1.0)
    numeric = pd.DataFrame(index=df.index)
    numeric["start_year_scaled"] = (year.fillna(min_year) - min_year) / denom
    numeric["team_size_log1p"] = np.log1p(pd.to_numeric(df.get("team_size", 0), errors="coerce").fillna(0).clip(lower=0))
    numeric["num_co_pis_log1p"] = np.log1p(pd.to_numeric(df.get("num_co_pis", 0), errors="coerce").fillna(0).clip(lower=0))
    numeric["has_abstract"] = (pd.to_numeric(df.get("abstract_word_count", 0), errors="coerce").fillna(0) > 0).astype(float)
    numeric["abstract_word_count_log1p"] = np.log1p(pd.to_numeric(df.get("abstract_word_count", 0), errors="coerce").fillna(0).clip(lower=0))
    numeric["title_word_count_log1p"] = np.log1p(pd.to_numeric(df.get("title_word_count", 0), errors="coerce").fillna(0).clip(lower=0))
    feature_parts.append(numeric.astype(float).values)
    for col in numeric.columns:
        manifest.append({"feature": col, "type": "numeric", "source": "awards"})

    categorical_specs = [
        ("NSFDirectorate", sorted([v for v in df["NSFDirectorate"].fillna("").astype(str).unique() if v])),
        ("NSFOrganization", sorted([v for v in df["NSFOrganization"].fillna("").astype(str).unique() if v])),
        ("AwardInstrument", top_categories(df["AwardInstrument"], top_instrument_k)),
    ]

    for col, values in categorical_specs:
        for val in values:
            arr = (df[col].fillna("").astype(str) == val).astype(float).values.reshape(-1, 1)
            feature_parts.append(arr)
            manifest.append({"feature": f"{col}={val}", "type": "one_hot", "source": col})

    # Program one-hot using top program element codes. Awards can have multiple codes.
    program_lists = df["ProgramElementCode(s)"].apply(split_program_codes)
    counts: Dict[str, int] = defaultdict(int)
    for codes in program_lists:
        for code in codes:
            counts[code] += 1
    top_programs = [k for k, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_program_k]]
    top_set = set(top_programs)
    for code in top_programs:
        arr = program_lists.apply(lambda xs, c=code: float(c in xs)).values.reshape(-1, 1)
        feature_parts.append(arr)
        manifest.append({"feature": f"ProgramElementCode={code}", "type": "multi_hot", "source": "ProgramElementCode(s)"})
    if top_program_k > 0:
        arr_other = program_lists.apply(lambda xs: float(any(c not in top_set for c in xs))).values.reshape(-1, 1)
        feature_parts.append(arr_other)
        manifest.append({"feature": "ProgramElementCode=OTHER", "type": "multi_hot", "source": "ProgramElementCode(s)"})

    x = np.concatenate(feature_parts, axis=1).astype(np.float32) if feature_parts else np.zeros((len(df), 1), dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, manifest




def read_text_embeddings(path: str | Path) -> pd.DataFrame:
    emb = read_csv_flexible(path)
    emb = ensure_award_id(emb)
    emb_cols = [c for c in emb.columns if c.startswith("text_emb_") or c.startswith("emb_")]
    if not emb_cols:
        # Fallback: all numeric columns except ids.
        emb_cols = [c for c in emb.columns if c not in {"award_id", "award_id_key"}]
    if not emb_cols:
        raise ValueError(f"No text embedding columns found in {path}")
    emb = emb.drop_duplicates(subset=["award_id_key"], keep="first").copy()
    for c in emb_cols:
        emb[c] = pd.to_numeric(emb[c], errors="coerce").fillna(0.0)
    return emb[["award_id_key"] + emb_cols]


def make_simteg_features(df: pd.DataFrame, text_embeddings_path: str | Path, top_program_k: int, top_instrument_k: int, include_structural: bool = True) -> Tuple[np.ndarray, List[Dict[str, str]]]:
    """Create SimTeG node features: text embeddings plus optional structural metadata.

    Text embeddings are the key SimTeG feature. Missing embeddings are filled with zeros
    and tracked by a has_text_embedding feature/manifest entry.
    """
    emb = read_text_embeddings(text_embeddings_path)
    emb_cols = [c for c in emb.columns if c != "award_id_key"]
    merged = df[["award_id_key"]].merge(emb, on="award_id_key", how="left")
    has_emb = merged[emb_cols].notna().all(axis=1).astype(float).values.reshape(-1, 1)
    emb_arr = merged[emb_cols].fillna(0.0).astype(np.float32).values
    feature_parts = [emb_arr, has_emb.astype(np.float32)]
    manifest = []
    for c in emb_cols:
        manifest.append({"feature": c, "type": "text_embedding", "source": str(text_embeddings_path)})
    manifest.append({"feature": "has_text_embedding", "type": "indicator", "source": str(text_embeddings_path)})

    if include_structural:
        struct_arr, struct_manifest = make_features(df, top_program_k, top_instrument_k)
        feature_parts.append(struct_arr)
        for row in struct_manifest:
            row = dict(row)
            row["source"] = "structural_" + str(row.get("source", "awards"))
            manifest.append(row)

    x = np.concatenate(feature_parts, axis=1).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, manifest

def add_edges_from_groups(
    edge_set: set,
    groups: Dict[str, List[int]],
    years: np.ndarray,
    max_prior: int,
    bidirectional: bool,
    source_name: str,
    summary_rows: List[Dict[str, object]],
) -> None:
    before = len(edge_set)
    group_count = 0
    skipped_singletons = 0
    for _, nodes in groups.items():
        uniq = sorted(set(nodes), key=lambda i: (years[i] if not math.isnan(years[i]) else -1, i))
        if len(uniq) < 2:
            skipped_singletons += 1
            continue
        group_count += 1
        for pos, dst in enumerate(uniq):
            prior = uniq[max(0, pos - max_prior):pos] if max_prior > 0 else uniq[:pos]
            for src in prior:
                if src == dst:
                    continue
                edge_set.add((src, dst))
                if bidirectional:
                    edge_set.add((dst, src))
    summary_rows.append({
        "edge_source": source_name,
        "groups_used": group_count,
        "singleton_groups_skipped": skipped_singletons,
        "edges_added": len(edge_set) - before,
        "max_prior_per_node_per_group": max_prior,
        "bidirectional": int(bidirectional),
    })


def build_group_indices(
    df: pd.DataFrame,
    award_to_idx: Dict[str, int],
    award_key_to_idx: Dict[str, int],
    award_pi_edges: Optional[pd.DataFrame],
) -> Dict[str, Dict[str, List[int]]]:
    groups_by_source: Dict[str, Dict[str, List[int]]] = {}

    if award_pi_edges is not None and len(award_pi_edges):
        e = ensure_award_id(award_pi_edges)
        person_col = "person_id" if "person_id" in e.columns else "pi_id" if "pi_id" in e.columns else None
        if person_col:
            person_groups: Dict[str, List[int]] = defaultdict(list)
            for _, r in e.iterrows():
                aid = clean_award_id_value(r.get("award_id", ""))
                aid_key = award_id_match_key(aid)
                pid = clean_str(r.get(person_col, ""))
                idx = award_to_idx.get(aid, award_key_to_idx.get(aid_key))
                if idx is not None and pid:
                    person_groups[pid].append(idx)
            groups_by_source["person"] = person_groups

    # Fallback lead PI grouping if no edge file or no useful person groups.
    if "person" not in groups_by_source or len(groups_by_source["person"]) == 0:
        person_groups = defaultdict(list)
        for i, r in df.iterrows():
            pid = clean_str(r.get("person_id", ""))
            if pid:
                person_groups[pid].append(i)
        groups_by_source["person"] = person_groups

    inst_groups = defaultdict(list)
    for i, r in df.iterrows():
        inst = clean_str(r.get("institution_id", "")) or normalize_key(r.get("organization_clean", ""))
        if inst:
            inst_groups[inst].append(i)
    groups_by_source["institution"] = inst_groups

    program_groups = defaultdict(list)
    for i, r in df.iterrows():
        for code in split_program_codes(r.get("ProgramElementCode(s)", "")):
            program_groups[code].append(i)
    groups_by_source["program"] = program_groups

    org_groups = defaultdict(list)
    for i, r in df.iterrows():
        org = clean_str(r.get("NSFOrganization", ""))
        if org:
            org_groups[org].append(i)
    groups_by_source["nsf_org"] = org_groups

    directorate_groups = defaultdict(list)
    for i, r in df.iterrows():
        d = clean_str(r.get("NSFDirectorate", ""))
        if d:
            directorate_groups[d].append(i)
    groups_by_source["directorate"] = directorate_groups

    return groups_by_source


def encode_targets_and_splits(df: pd.DataFrame, outdir: Path) -> Dict[str, Dict[str, int]]:
    label_maps: Dict[str, Dict[str, int]] = {}
    for col in TARGET_COLUMNS_DEFAULT:
        if col not in df.columns:
            continue
        if col == "target_mmr_multiclass":
            vals = sorted(v for v in df.loc[df["is_benchmark"] == 1, col].dropna().astype(str).unique() if v != "")
            label_maps[col] = {v: i for i, v in enumerate(vals)}
            df[col + "__encoded"] = df[col].map(label_maps[col]).fillna(-1).astype(int)
        else:
            df[col + "__encoded"] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)
            label_maps[col] = {"0": 0, "1": 1}
    with open(outdir / "simteg_graphsage_label_maps.json", "w", encoding="utf-8") as f:
        json.dump(label_maps, f, indent=2)
    return label_maps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", default=None, help="Repository root. Defaults to auto-detection.")
    parser.add_argument("--awards", default=None, help="cleaned_nsf_awards_2000_2025.csv. Defaults to data/processed/.../cleaned_nsf_awards_2000_2025.csv")
    parser.add_argument("--benchmark", default=None, help="Benchmark CSV/zip. Defaults to data/benchmark v3.")
    parser.add_argument("--award_pi_edges", default=None, help="Optional award_pi_edges.csv from cleaning pipeline. Defaults to data/edges/award_pi_edges.csv if present.")
    parser.add_argument("--outdir", default=None, help="Output directory. Defaults to artifacts/graphs/simteg_graphsage_data_scibert_v1")
    parser.add_argument("--text_embeddings", default=None, help="methodkg_simteg_text_embeddings.csv. Defaults to artifacts/features/simteg_text_embeddings_scibert_v1/methodkg_simteg_text_embeddings.csv")
    parser.add_argument("--no_structural_features", action="store_true", help="Use text embeddings only as node features")
    parser.add_argument("--bidirectional", action="store_true", help="Use undirected/bidirectional edges. Default is temporal older->newer only.")
    parser.add_argument("--max_prior_person", type=int, default=10)
    parser.add_argument("--max_prior_institution", type=int, default=5)
    parser.add_argument("--max_prior_program", type=int, default=8)
    parser.add_argument("--max_prior_org", type=int, default=2)
    parser.add_argument("--max_prior_directorate", type=int, default=1)
    parser.add_argument("--top_program_k", type=int, default=100)
    parser.add_argument("--top_instrument_k", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true", help="Delete the output directory before rebuilding graph artifacts.")
    args = parser.parse_args()

    repo_root = find_repo_root(args.repo_root or Path(__file__).resolve())
    args.awards = str(resolve_existing_path(args.awards, repo_root) if args.awards else discover_awards(repo_root))
    args.benchmark = str(resolve_existing_path(args.benchmark, repo_root) if args.benchmark else discover_benchmark(repo_root))
    edge_path = resolve_existing_path(args.award_pi_edges, repo_root, required=False) if args.award_pi_edges else discover_award_pi_edges(repo_root)
    args.award_pi_edges = str(edge_path) if edge_path else ""
    args.text_embeddings = str(resolve_existing_path(args.text_embeddings, repo_root) if args.text_embeddings else resolve_existing_path(default_simteg_text_embeddings(repo_root), repo_root))
    outdir = resolve_output_path(args.outdir, repo_root, default_simteg_graph_dir(repo_root))
    reset_dir_if_overwrite(outdir, args.overwrite)
    write_resolved_paths(repo_root=repo_root, awards=args.awards, benchmark=args.benchmark, award_pi_edges=args.award_pi_edges, text_embeddings=args.text_embeddings, outdir=outdir)

    awards = add_basic_columns(read_csv_flexible(args.awards))
    awards = awards.drop_duplicates(subset=["award_id_key"]).reset_index(drop=True)
    benchmark = add_basic_columns(read_csv_flexible(args.benchmark))
    df = merge_benchmark_metadata(awards, benchmark, outdir=outdir)
    df = add_basic_columns(df)
    df = df.sort_values(["start_year", "award_id"], na_position="last").reset_index(drop=True)

    # Use both display award_id and leading-zero-insensitive key for robust edge matching.
    award_to_idx = {aid: i for i, aid in enumerate(df["award_id"].astype(str))}
    award_key_to_idx = {key: i for i, key in enumerate(df["award_id_key"].astype(str)) if key}
    award_pi_edges = None
    if args.award_pi_edges:
        p = Path(args.award_pi_edges)
        if p.exists():
            award_pi_edges = read_csv_flexible(p)
        else:
            print(f"Warning: award_pi_edges path does not exist: {p}. Falling back to lead PI only.")

    x_np, feature_manifest = make_simteg_features(df, args.text_embeddings, args.top_program_k, args.top_instrument_k, include_structural=(not args.no_structural_features))
    years = pd.to_numeric(df["start_year"], errors="coerce").astype(float).values

    groups_by_source = build_group_indices(df, award_to_idx, award_key_to_idx, award_pi_edges)
    edge_set = set()
    edge_summary = []
    add_edges_from_groups(edge_set, groups_by_source.get("person", {}), years, args.max_prior_person, args.bidirectional, "person", edge_summary)
    add_edges_from_groups(edge_set, groups_by_source.get("institution", {}), years, args.max_prior_institution, args.bidirectional, "institution", edge_summary)
    add_edges_from_groups(edge_set, groups_by_source.get("program", {}), years, args.max_prior_program, args.bidirectional, "program", edge_summary)
    add_edges_from_groups(edge_set, groups_by_source.get("nsf_org", {}), years, args.max_prior_org, args.bidirectional, "nsf_org", edge_summary)
    add_edges_from_groups(edge_set, groups_by_source.get("directorate", {}), years, args.max_prior_directorate, args.bidirectional, "directorate", edge_summary)

    if not edge_set:
        raise RuntimeError("No edges were created. Check input columns and award_pi_edges.")
    edge_index_np = np.array(sorted(edge_set), dtype=np.int64).T

    # Encode targets after merge and before saving node index.
    label_maps = encode_targets_and_splits(df, outdir)
    y_dict = {}
    for target in label_maps:
        y_dict[target] = torch.tensor(df[target + "__encoded"].values, dtype=torch.long)

    data = Data(
        x=torch.tensor(x_np, dtype=torch.float32),
        edge_index=torch.tensor(edge_index_np, dtype=torch.long),
    )
    data.num_nodes = len(df)
    data.award_id = df["award_id"].astype(str).tolist()
    data.is_benchmark = torch.tensor(df["is_benchmark"].values.astype(bool), dtype=torch.bool)
    data.y_dict = y_dict

    torch.save(data, outdir / "simteg_graphsage_award_graph.pt")

    node_cols = [
        "award_id", "award_id_key", "node_source", "start_year", "is_benchmark", "benchmark_id", "annotation_id", "project_cluster_id",
        "NSFDirectorate", "NSFOrganization", "AwardInstrument", "ProgramElementCode(s)",
        "institution_id", "organization_clean", "person_id",
    ]
    node_cols += [c for c in TARGET_COLUMNS_DEFAULT + [t + "__encoded" for t in TARGET_COLUMNS_DEFAULT] + SPLIT_COLUMNS_DEFAULT if c in df.columns]
    node_cols = [c for c in node_cols if c in df.columns]
    node_index = df[node_cols].copy()
    node_index.insert(0, "node_idx", np.arange(len(node_index)))
    node_index.to_csv(outdir / "simteg_graphsage_node_index.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(edge_summary).to_csv(outdir / "simteg_graphsage_edge_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(feature_manifest).to_csv(outdir / "simteg_graphsage_feature_manifest.csv", index=False, encoding="utf-8-sig")

    summary = {
        "num_award_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.shape[1]),
        "feature_dim": int(data.x.shape[1]),
        "benchmark_nodes": int(data.is_benchmark.sum().item()),
        "benchmark_nodes_expected": int(benchmark["award_id_key"].nunique()),
        "benchmark_only_nodes_added": int((df.get("node_source", pd.Series(dtype=str)) == "benchmark_only_added").sum()),
        "bidirectional": bool(args.bidirectional),
        "edge_direction": "bidirectional/undirected" if args.bidirectional else "temporal older_to_newer",
        "label_maps": label_maps,
        "edge_summary": edge_summary,
        "input_awards": str(args.awards),
        "input_benchmark": str(args.benchmark),
        "input_award_pi_edges": str(args.award_pi_edges),
    }
    with open(outdir / "simteg_graphsage_build_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Wrote GraphSAGE data to", outdir.resolve())
    print("Nodes:", summary["num_award_nodes"])
    print("Edges:", summary["num_edges"])
    print("Features:", summary["feature_dim"])
    print("Benchmark nodes:", summary["benchmark_nodes"])
    print("Expected benchmark nodes:", summary["benchmark_nodes_expected"])
    print("Benchmark-only nodes added:", summary["benchmark_only_nodes_added"])
    if summary["benchmark_nodes"] != summary["benchmark_nodes_expected"]:
        print("WARNING: benchmark_nodes does not equal expected benchmark unique ids. Check match reports.")


if __name__ == "__main__":
    main()

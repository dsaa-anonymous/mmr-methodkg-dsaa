#!/usr/bin/env python3
"""Build a MethodKG heterogeneous graph for structural-only HGT.

This script creates a PyTorch Geometric HeteroData object with node types:
  award, person, institution, program, nsf_org, directorate, state, year

It is designed for MethodKG Graph Phase 4/graph-only HGT experiments.
The model consumes metadata/structural features only; it does not encode award
text. Labels are attached only to award nodes from the benchmark file.

Important: this graph is a transductive heterogeneous-GNN baseline. It uses the
unlabeled full MethodKG structure. Use historical-feature baselines for the
strictest leakage-safe temporal claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
import sys
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from path_utils import discover_award_pi_edges, discover_awards, discover_benchmark, find_repo_root, read_csv_or_zip, reset_dir_if_overwrite, resolve_existing_path, resolve_output_path, write_resolved_paths
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch

try:
    from torch_geometric.data import HeteroData
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Could not import torch_geometric. Install PyTorch Geometric in your TIDE environment.\n"
        "See README.md for installation notes. Original error: " + repr(exc)
    )

TARGET_COLS = [
    "target_integration_binary",
    "target_design_binary",
    "target_mmr_binary",
    "target_mmr_multiclass",
    "target_method_signal_binary",
    "target_qual_binary",
    "target_quant_binary",
]

SPLIT_COLS = [
    "split_random_cluster_stratified",
    "split_temporal_cluster_safe",
    "split_cross_program_cluster_safe",
    "split_cold_start_pi_cluster_safe",
    "split_cold_start_institution_cluster_safe",
    "split_edu_to_eng_cluster_safe",
]

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def normalize_award_id(x) -> str:
    """Normalize NSF award IDs robustly without turning 1922666.0 into 19226660."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.startswith('="') and s.endswith('"'):
        s = s[2:-1]
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".")[0]
    s = re.sub(r"[^0-9]", "", s)
    return s.strip()


def norm_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def norm_for_id(x) -> str:
    s = norm_text(x).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def make_id(prefix: str, value: str) -> str:
    value = norm_for_id(value)
    if not value:
        value = "unknown"
    h = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{h}"


def split_codes(value) -> List[str]:
    s = norm_text(value)
    if not s:
        return []
    parts = re.split(r"[,;|/\s]+", s)
    out, seen = [], set()
    for p in parts:
        p = re.sub(r"[^A-Za-z0-9]", "", p).upper().strip()
        if p and p not in seen:
            out.append(p)
            seen.add(p)
    return out


def parse_year(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float")
    if np.issubdtype(series.dtype, np.number):
        return pd.to_numeric(series, errors="coerce")
    dt = pd.to_datetime(series, errors="coerce")
    year_from_date = dt.dt.year
    numeric = pd.to_numeric(series, errors="coerce")
    return year_from_date.fillna(numeric)


def safe_numeric(series, default=0.0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def safe_log1p(series):
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    s = s.clip(lower=0)
    return np.log1p(s)


def minmax(values: Iterable[float], default: float = 0.0) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size == 0:
        return arr
    mask = np.isfinite(arr)
    if not mask.any():
        return np.full_like(arr, default, dtype=np.float32)
    mn, mx = float(arr[mask].min()), float(arr[mask].max())
    out = np.full_like(arr, default, dtype=np.float32)
    if mx <= mn:
        out[mask] = 0.0
    else:
        out[mask] = (arr[mask] - mn) / (mx - mn)
    return out.astype(np.float32)


def read_csv(path: str | Path) -> pd.DataFrame:
    return read_csv_or_zip(path, preferred_name="methodkg_labeled_benchmark_v3_modeling.csv", dtype=str, encoding="utf-8-sig")


def prepare_awards(awards: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    awards = awards.copy()
    benchmark = benchmark.copy()
    awards["award_id"] = awards.get("award_id", awards.get("AwardNumber", "")).apply(normalize_award_id)
    benchmark["award_id"] = benchmark["award_id"].apply(normalize_award_id)

    # Add benchmark-only rows if needed. Keep benchmark metadata for all benchmark awards.
    award_ids = set(awards["award_id"].astype(str))
    missing_bench = benchmark[~benchmark["award_id"].astype(str).isin(award_ids)].copy()
    # Ensure columns present before concat.
    for c in awards.columns:
        if c not in missing_bench.columns:
            missing_bench[c] = ""
    for c in missing_bench.columns:
        if c not in awards.columns:
            awards[c] = ""
    missing_bench = missing_bench[awards.columns]
    combined = pd.concat([awards, missing_bench], ignore_index=True)

    combined = combined.drop_duplicates(subset=["award_id"], keep="first").copy()

    # Normalize key columns used downstream.
    if "title_clean" not in combined.columns:
        combined["title_clean"] = combined.get("Title", "")
    if "abstract_clean" not in combined.columns:
        combined["abstract_clean"] = combined.get("Abstract", "")
    combined["title_clean"] = combined["title_clean"].apply(norm_text)
    combined["abstract_clean"] = combined["abstract_clean"].apply(norm_text)

    if "start_year" not in combined.columns or combined["start_year"].fillna("").eq("").all():
        combined["start_year"] = parse_year(combined.get("StartDate", pd.Series(index=combined.index)))
    else:
        combined["start_year"] = pd.to_numeric(combined["start_year"], errors="coerce")

    if "institution_id" not in combined.columns:
        if "organization_clean" not in combined.columns:
            combined["organization_clean"] = combined.get("Organization", "").apply(norm_text)
        combined["institution_id"] = combined["organization_clean"].apply(lambda x: make_id("inst", x))

    if "organization_clean" not in combined.columns:
        combined["organization_clean"] = combined.get("Organization", "").apply(norm_text)

    if "person_id" not in combined.columns:
        if "pi_id" in combined.columns:
            combined["person_id"] = combined["pi_id"].fillna("")
        else:
            combined["person_id"] = combined.get("PrincipalInvestigator", "").apply(lambda x: make_id("person", x))

    # Fallback values.
    for col in ["NSFOrganization", "NSFDirectorate", "State", "OrganizationState", "ProgramElementCode(s)", "Program(s)"]:
        if col not in combined.columns:
            combined[col] = ""
        combined[col] = combined[col].apply(norm_text)

    combined["award_id"] = combined["award_id"].astype(str)
    return combined


def build_node_maps(values: Iterable[str]) -> Tuple[Dict[str, int], List[str]]:
    items = sorted({str(v) for v in values if str(v) and str(v).lower() != "nan"})
    return {v: i for i, v in enumerate(items)}, items


def add_edges(edge_store: Dict[Tuple[str, str, str], List[Tuple[int, int]]],
              src_type: str, rel: str, dst_type: str, src_idx: int, dst_idx: int,
              add_reverse: bool = True):
    edge_store[(src_type, rel, dst_type)].append((src_idx, dst_idx))
    if add_reverse:
        edge_store[(dst_type, "rev_" + rel, src_type)].append((dst_idx, src_idx))


def tensorize_edges(edge_pairs: List[Tuple[int, int]]) -> torch.Tensor:
    if not edge_pairs:
        return torch.empty((2, 0), dtype=torch.long)
    arr = np.asarray(edge_pairs, dtype=np.int64)
    # Deduplicate for memory efficiency.
    arr = np.unique(arr, axis=0)
    return torch.from_numpy(arr.T).contiguous()


def build_person_edges_from_file(path: Path, award_map: Dict[str, int], person_map: Dict[str, int]) -> Dict[str, List[Tuple[int, str, str]]]:
    out: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)
    if not path or not path.exists():
        return out
    df = read_csv(path)
    if "award_id" not in df.columns:
        return out
    df["award_id"] = df["award_id"].apply(normalize_award_id)
    if "person_id" not in df.columns:
        if "pi_id" in df.columns:
            df["person_id"] = df["pi_id"]
        else:
            df["person_id"] = df.get("pi_name", "").apply(lambda x: make_id("person", x))
    if "role" not in df.columns:
        df["role"] = "PI"
    for _, r in df.iterrows():
        aid = str(r.get("award_id", ""))
        pid = str(r.get("person_id", ""))
        role = str(r.get("role", "PI")).lower()
        if aid in award_map and pid in person_map:
            rel = "has_pi" if role == "pi" else "has_copi"
            out[aid].append((person_map[pid], rel, pid))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=None, help="Repository root. Defaults to auto-detection.")
    ap.add_argument("--awards", default=None, help="cleaned_nsf_awards_2000_2025.csv. Defaults to data/processed discovery.")
    ap.add_argument("--benchmark", default=None, help="methodkg_labeled_benchmark_v3_modeling.csv or benchmark_v3.zip. Defaults to data/benchmark discovery.")
    ap.add_argument("--award_pi_edges", default="", help="Optional award_pi_edges.csv")
    ap.add_argument("--outdir", default=None, help="Output directory. Defaults to artifacts/graphs/hgt_data_v1")
    ap.add_argument("--overwrite", action="store_true", help="Delete the output directory before writing new graph artifacts.")
    ap.add_argument("--include_award_amount", action="store_true", help="Include award_amount feature; off by default to avoid amendment/leakage concerns")
    ap.add_argument("--drop_text_length_features", action="store_true", help="Remove title/abstract word-count features for a stricter structural-only baseline")
    args = ap.parse_args()

    repo_root = find_repo_root(args.repo_root or Path(__file__).resolve())
    awards_path = resolve_existing_path(args.awards, repo_root) if args.awards else discover_awards(repo_root)
    benchmark_path = resolve_existing_path(args.benchmark, repo_root) if args.benchmark else discover_benchmark(repo_root)
    award_pi_path = resolve_existing_path(args.award_pi_edges, repo_root, required=False) if args.award_pi_edges else discover_award_pi_edges(repo_root)
    outdir = resolve_output_path(args.outdir, repo_root, repo_root / "artifacts" / "graphs" / "hgt_data_v1")
    reset_dir_if_overwrite(outdir, args.overwrite)
    write_resolved_paths(repo_root=repo_root, awards=awards_path, benchmark=benchmark_path, award_pi_edges=award_pi_path or "<not found; using lead PI only>", outdir=outdir)

    awards_raw = read_csv(awards_path)
    benchmark = read_csv(benchmark_path)
    benchmark["award_id"] = benchmark["award_id"].apply(normalize_award_id)

    awards = prepare_awards(awards_raw, benchmark)
    awards["award_id"] = awards["award_id"].apply(normalize_award_id)
    awards = awards[awards["award_id"] != ""].copy()

    # Merge benchmark target/split columns onto awards for output metadata.
    keep_bench_cols = ["award_id", "benchmark_id", "project_cluster_id"]
    keep_bench_cols += [c for c in TARGET_COLS + SPLIT_COLS if c in benchmark.columns]
    bench_meta = benchmark[[c for c in keep_bench_cols if c in benchmark.columns]].drop_duplicates("award_id")
    awards = awards.merge(bench_meta, on="award_id", how="left", suffixes=("", "_bench"))
    awards["is_benchmark"] = awards["award_id"].isin(set(benchmark["award_id"])).astype(int)

    # Build entity keys.
    awards["inst_key"] = awards["institution_id"].fillna("").astype(str)
    awards.loc[awards["inst_key"].eq("") | awards["inst_key"].str.lower().eq("nan"), "inst_key"] = awards.loc[
        awards["inst_key"].eq("") | awards["inst_key"].str.lower().eq("nan"), "organization_clean"
    ].apply(lambda x: make_id("inst", x))

    awards["nsf_org_key"] = awards["NSFOrganization"].fillna("").astype(str).str.upper().str.strip()
    awards["directorate_key"] = awards["NSFDirectorate"].fillna("").astype(str).str.upper().str.strip()
    awards["state_key"] = awards["OrganizationState"].fillna("").astype(str).str.upper().str.strip()
    awards.loc[awards["state_key"].eq("") | awards["state_key"].str.lower().eq("nan"), "state_key"] = awards["State"].fillna("").astype(str).str.upper().str.strip()
    awards["year_key"] = pd.to_numeric(awards["start_year"], errors="coerce").fillna(-1).astype(int).astype(str)
    awards.loc[awards["year_key"].eq("-1"), "year_key"] = "unknown_year"

    awards["program_codes"] = awards["ProgramElementCode(s)"].apply(split_codes)
    # Fallback to program-name hash if no code.
    def program_keys(row):
        codes = row["program_codes"]
        if codes:
            return [f"program_code:{c}" for c in codes]
        name = norm_text(row.get("Program(s)", ""))
        if name:
            return ["program_name:" + make_id("program", name)]
        return []
    awards["program_keys"] = awards.apply(program_keys, axis=1)

    # Person IDs from award_pi_edges if available, otherwise lead PI person_id.
    award_pi_path = Path(award_pi_path) if award_pi_path else None
    person_values = set()
    if award_pi_path and award_pi_path.exists():
        pi_df = read_csv(award_pi_path)
        if "person_id" not in pi_df.columns:
            pi_df["person_id"] = pi_df.get("pi_id", pi_df.get("pi_name", "")).apply(lambda x: make_id("person", x))
        pi_df["award_id"] = pi_df["award_id"].apply(normalize_award_id)
        person_values.update(pi_df["person_id"].fillna("").astype(str).tolist())
    person_values.update(awards["person_id"].fillna("").astype(str).tolist())

    award_map, award_items = build_node_maps(awards["award_id"].tolist())
    person_map, person_items = build_node_maps(person_values)
    inst_map, inst_items = build_node_maps(awards["inst_key"].tolist())
    prog_map, prog_items = build_node_maps([p for vals in awards["program_keys"] for p in vals])
    org_map, org_items = build_node_maps(awards["nsf_org_key"].tolist())
    dir_map, dir_items = build_node_maps(awards["directorate_key"].tolist())
    state_map, state_items = build_node_maps(awards["state_key"].tolist())
    year_map, year_items = build_node_maps(awards["year_key"].tolist())

    data = HeteroData()

    # Award features.
    n_awards = len(award_items)
    award_df = awards.set_index("award_id").reindex(award_items).reset_index()
    year_vals = pd.to_numeric(award_df["start_year"], errors="coerce")
    feat_cols = []
    award_feats = []
    award_feats.append(minmax(year_vals.fillna(year_vals.median() if year_vals.notna().any() else 0)))
    feat_cols.append("start_year_minmax")
    # word counts are non-content metadata; optional.
    if not args.drop_text_length_features:
        if "title_word_count" in award_df.columns:
            title_wc = safe_log1p(award_df["title_word_count"])
        else:
            title_wc = np.log1p(award_df["title_clean"].fillna("").astype(str).str.split().str.len())
        if "abstract_word_count" in award_df.columns:
            abs_wc = safe_log1p(award_df["abstract_word_count"])
        else:
            abs_wc = np.log1p(award_df["abstract_clean"].fillna("").astype(str).str.split().str.len())
        award_feats.extend([minmax(title_wc), minmax(abs_wc)])
        feat_cols.extend(["title_word_count_log_minmax", "abstract_word_count_log_minmax"])
        if "has_abstract" in award_df.columns:
            award_feats.append(safe_numeric(award_df["has_abstract"]).clip(0, 1).to_numpy(dtype=np.float32))
        else:
            award_feats.append((award_df["abstract_clean"].fillna("").astype(str).str.len() > 0).astype(float).to_numpy(dtype=np.float32))
        feat_cols.append("has_abstract")
    if "team_size" in award_df.columns:
        award_feats.append(minmax(safe_log1p(award_df["team_size"])))
        feat_cols.append("team_size_log_minmax")
    if "num_co_pis" in award_df.columns:
        award_feats.append(minmax(safe_log1p(award_df["num_co_pis"])))
        feat_cols.append("num_co_pis_log_minmax")
    if args.include_award_amount and "award_amount" in award_df.columns:
        award_feats.append(minmax(safe_log1p(award_df["award_amount"])))
        feat_cols.append("award_amount_log_minmax")
    if "legacy_nsf_org_flag" in award_df.columns:
        award_feats.append(safe_numeric(award_df["legacy_nsf_org_flag"]).clip(0, 1).to_numpy(dtype=np.float32))
        feat_cols.append("legacy_nsf_org_flag")

    if not award_feats:
        award_feats = [np.zeros(n_awards, dtype=np.float32)]
        feat_cols = ["constant"]
    data["award"].x = torch.tensor(np.vstack(award_feats).T, dtype=torch.float32)

    # Entity features from counts. These are structural metadata, transductive by design.
    def entity_features(keys: List[str], award_lists_by_key: Dict[str, List[str]], extra_counts: Dict[str, int] | None = None):
        feats = []
        counts = [len(award_lists_by_key.get(k, [])) for k in keys]
        feats.append(minmax(np.log1p(counts)))
        first_years, last_years = [], []
        for k in keys:
            aids = award_lists_by_key.get(k, [])
            yrs = pd.to_numeric(awards[awards["award_id"].isin(aids)]["start_year"], errors="coerce")
            if yrs.notna().any():
                first_years.append(float(yrs.min()))
                last_years.append(float(yrs.max()))
            else:
                first_years.append(np.nan)
                last_years.append(np.nan)
        feats.append(minmax(first_years))
        feats.append(minmax(last_years))
        if extra_counts is not None:
            feats.append(minmax(np.log1p([extra_counts.get(k, 0) for k in keys])))
        return torch.tensor(np.vstack(feats).T, dtype=torch.float32)

    # Count awards by entity.
    awards_by_person = defaultdict(list)
    awards_by_inst = defaultdict(list)
    awards_by_prog = defaultdict(list)
    awards_by_org = defaultdict(list)
    awards_by_dir = defaultdict(list)
    awards_by_state = defaultdict(list)
    awards_by_year = defaultdict(list)

    for _, r in awards.iterrows():
        aid = str(r["award_id"])
        if aid not in award_map:
            continue
        pid = str(r.get("person_id", ""))
        if pid:
            awards_by_person[pid].append(aid)
        inst = str(r.get("inst_key", ""))
        if inst:
            awards_by_inst[inst].append(aid)
        for p in r.get("program_keys", []):
            awards_by_prog[p].append(aid)
        org = str(r.get("nsf_org_key", ""))
        if org:
            awards_by_org[org].append(aid)
        d = str(r.get("directorate_key", ""))
        if d:
            awards_by_dir[d].append(aid)
        st = str(r.get("state_key", ""))
        if st:
            awards_by_state[st].append(aid)
        y = str(r.get("year_key", ""))
        if y:
            awards_by_year[y].append(aid)

    # Add additional person awards from edge file.
    if award_pi_path and award_pi_path.exists():
        pi_df = read_csv(award_pi_path)
        pi_df["award_id"] = pi_df["award_id"].apply(normalize_award_id)
        if "person_id" not in pi_df.columns:
            pi_df["person_id"] = pi_df.get("pi_id", pi_df.get("pi_name", "")).apply(lambda x: make_id("person", x))
        for _, r in pi_df.iterrows():
            aid, pid = str(r["award_id"]), str(r["person_id"])
            if aid in award_map and pid in person_map and aid not in awards_by_person[pid]:
                awards_by_person[pid].append(aid)

    data["person"].x = entity_features(person_items, awards_by_person)
    data["institution"].x = entity_features(inst_items, awards_by_inst)
    data["program"].x = entity_features(prog_items, awards_by_prog)
    data["nsf_org"].x = entity_features(org_items, awards_by_org)
    data["directorate"].x = entity_features(dir_items, awards_by_dir)
    data["state"].x = entity_features(state_items, awards_by_state)
    # Year feature: normalized year and log award count.
    years_numeric = []
    for y in year_items:
        try:
            years_numeric.append(float(y))
        except Exception:
            years_numeric.append(np.nan)
    year_feat = np.vstack([
        minmax(years_numeric),
        minmax(np.log1p([len(awards_by_year.get(y, [])) for y in year_items])),
    ]).T
    data["year"].x = torch.tensor(year_feat, dtype=torch.float32)

    # Build edges.
    edge_store: Dict[Tuple[str, str, str], List[Tuple[int, int]]] = defaultdict(list)
    award_person_edges = build_person_edges_from_file(award_pi_path, award_map, person_map) if award_pi_path else {}

    for _, r in awards.iterrows():
        aid = str(r["award_id"])
        if aid not in award_map:
            continue
        aidx = award_map[aid]
        # Person edges: from file if available for this award, else lead person.
        person_recs = award_person_edges.get(aid, [])
        if not person_recs:
            pid = str(r.get("person_id", ""))
            if pid and pid in person_map:
                person_recs = [(person_map[pid], "has_pi", pid)]
        for pidx, rel, _ in person_recs:
            add_edges(edge_store, "award", rel, "person", aidx, pidx, add_reverse=True)

        inst = str(r.get("inst_key", ""))
        if inst in inst_map:
            add_edges(edge_store, "award", "at_institution", "institution", aidx, inst_map[inst], add_reverse=True)
        for p in r.get("program_keys", []):
            if p in prog_map:
                add_edges(edge_store, "award", "funded_by_program", "program", aidx, prog_map[p], add_reverse=True)
                org = str(r.get("nsf_org_key", ""))
                if org in org_map:
                    add_edges(edge_store, "program", "program_in_org", "nsf_org", prog_map[p], org_map[org], add_reverse=True)
        org = str(r.get("nsf_org_key", ""))
        if org in org_map:
            add_edges(edge_store, "award", "in_nsf_org", "nsf_org", aidx, org_map[org], add_reverse=True)
            d = str(r.get("directorate_key", ""))
            if d in dir_map:
                add_edges(edge_store, "nsf_org", "org_in_directorate", "directorate", org_map[org], dir_map[d], add_reverse=True)
        d = str(r.get("directorate_key", ""))
        if d in dir_map:
            add_edges(edge_store, "award", "in_directorate", "directorate", aidx, dir_map[d], add_reverse=True)
        st = str(r.get("state_key", ""))
        if st in state_map:
            if inst in inst_map:
                add_edges(edge_store, "institution", "institution_in_state", "state", inst_map[inst], state_map[st], add_reverse=True)
            add_edges(edge_store, "award", "in_state", "state", aidx, state_map[st], add_reverse=True)
        y = str(r.get("year_key", ""))
        if y in year_map:
            add_edges(edge_store, "award", "has_year", "year", aidx, year_map[y], add_reverse=True)

    for etype, pairs in edge_store.items():
        data[etype].edge_index = tensorize_edges(pairs)

    # Save target metadata on award nodes where possible.
    benchmark_award_set = set(benchmark["award_id"].astype(str))
    is_benchmark = np.array([1 if aid in benchmark_award_set else 0 for aid in award_items], dtype=np.int64)
    data["award"].is_benchmark = torch.tensor(is_benchmark, dtype=torch.bool)

    # Output tables.
    award_node_index = award_df[["award_id"]].copy()
    award_node_index.insert(0, "award_node_idx", np.arange(len(award_node_index)))
    award_node_index["is_benchmark"] = award_node_index["award_id"].isin(benchmark_award_set).astype(int)
    # Attach target/split info for runners.
    bench_cols = ["award_id", "benchmark_id", "project_cluster_id"] + [c for c in TARGET_COLS + SPLIT_COLS if c in benchmark.columns]
    bench_cols = [c for c in bench_cols if c in benchmark.columns]
    award_node_index = award_node_index.merge(benchmark[bench_cols].drop_duplicates("award_id"), on="award_id", how="left")
    award_node_index.to_csv(outdir / "hgt_award_node_index.csv", index=False, encoding="utf-8-sig")

    # Entity maps.
    def save_map(ntype, items):
        pd.DataFrame({"node_idx": range(len(items)), "node_id": items}).to_csv(outdir / f"hgt_{ntype}_node_index.csv", index=False, encoding="utf-8-sig")
    for ntype, items in [("person", person_items), ("institution", inst_items), ("program", prog_items), ("nsf_org", org_items), ("directorate", dir_items), ("state", state_items), ("year", year_items)]:
        save_map(ntype, items)

    edge_rows = []
    for etype, pairs in edge_store.items():
        edge_rows.append({
            "src_type": etype[0], "relation": etype[1], "dst_type": etype[2],
            "raw_edges": len(pairs),
            "dedup_edges": int(data[etype].edge_index.size(1)),
        })
    pd.DataFrame(edge_rows).sort_values(["src_type", "relation", "dst_type"]).to_csv(outdir / "hgt_edge_summary.csv", index=False, encoding="utf-8-sig")

    feature_manifest = []
    feature_manifest.append({"node_type": "award", "feature_dim": int(data["award"].x.size(1)), "features": "|".join(feat_cols)})
    for ntype in ["person", "institution", "program", "nsf_org", "directorate", "state"]:
        feature_manifest.append({"node_type": ntype, "feature_dim": int(data[ntype].x.size(1)), "features": "log_award_count_minmax|first_year_minmax|last_year_minmax"})
    feature_manifest.append({"node_type": "year", "feature_dim": int(data["year"].x.size(1)), "features": "year_minmax|log_award_count_minmax"})
    pd.DataFrame(feature_manifest).to_csv(outdir / "hgt_feature_manifest.csv", index=False, encoding="utf-8-sig")

    match_report = pd.DataFrame([
        {"metric": "benchmark_rows_expected", "value": int(len(benchmark))},
        {"metric": "benchmark_unique_awards", "value": int(benchmark["award_id"].nunique())},
        {"metric": "benchmark_nodes_in_graph", "value": int(award_node_index["is_benchmark"].sum())},
        {"metric": "benchmark_awards_missing_from_raw_awards", "value": int(len(set(benchmark["award_id"]) - set(awards_raw.get("award_id", awards_raw.get("AwardNumber", pd.Series(dtype=str))).apply(normalize_award_id).astype(str))) if len(awards_raw) else 0)},
    ])
    match_report.to_csv(outdir / "hgt_benchmark_award_match_report.csv", index=False, encoding="utf-8-sig")

    summary = {
        "award_nodes": int(data["award"].num_nodes),
        "person_nodes": int(data["person"].num_nodes),
        "institution_nodes": int(data["institution"].num_nodes),
        "program_nodes": int(data["program"].num_nodes),
        "nsf_org_nodes": int(data["nsf_org"].num_nodes),
        "directorate_nodes": int(data["directorate"].num_nodes),
        "state_nodes": int(data["state"].num_nodes),
        "year_nodes": int(data["year"].num_nodes),
        "edge_types": len(edge_store),
        "total_dedup_edges": int(sum(data[etype].edge_index.size(1) for etype in data.edge_types)),
        "benchmark_nodes": int(award_node_index["is_benchmark"].sum()),
        "benchmark_nodes_expected": int(len(benchmark)),
        "award_feature_dim": int(data["award"].x.size(1)),
        "include_award_amount": bool(args.include_award_amount),
        "drop_text_length_features": bool(args.drop_text_length_features),
        "input_awards": str(awards_path),
        "input_benchmark": str(benchmark_path),
        "input_award_pi_edges": str(award_pi_path or ""),
        "note": "Transductive heterogeneous structural HGT graph. No award text embeddings are used.",
    }
    with open(outdir / "hgt_build_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    torch.save(data, outdir / "hgt_heterodata.pt")
    # Save benchmark table used by runner.
    award_node_index.to_csv(outdir / "hgt_benchmark_table.csv", index=False, encoding="utf-8-sig")

    print("Wrote HGT graph to", outdir.resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build lightweight graph-only / structured-context features for MethodKG.

This script creates leakage-safe historical graph/context features for each labeled
award in the benchmark. It does NOT use title/abstract text, candidate flags, or
labels as inputs. Historical features use only awards with start_year strictly
before the target award's start_year.

Inputs:
  --awards cleaned_nsf_awards_2000_2025.csv
  --benchmark methodkg_labeled_benchmark_v2_modeling.csv OR benchmark_v2.zip
  --award_pi_edges optional award_pi_edges.csv from the MethodKG cleaning pipeline

Outputs:
  methodkg_graph_only_features.csv
  methodkg_graph_only_feature_manifest.csv
  methodkg_graph_only_feature_summary.csv

Example:
  python build_graph_only_features.py \
    --awards cleaned_nsf_awards_2000_2025.csv \
    --benchmark benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv \
    --outdir graph_features_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import zipfile
from collections import defaultdict
from pathlib import Path
import sys
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from path_utils import discover_award_pi_edges, discover_awards, discover_benchmark, find_repo_root, read_csv_or_zip as read_csv_or_zip_shared, reset_dir_if_overwrite, resolve_existing_path, resolve_output_path, write_resolved_paths
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

TARGET_COL_PREFIXES = ("target_",)
LABEL_COL_PREFIXES = ("label_",)
SPLIT_COL_PREFIXES = ("split_",)
TEXT_COLS = {"title_clean", "abstract_clean"}


def read_csv_or_zip(path: str | Path, preferred_name: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """Read CSV directly or choose the right CSV inside a zip, preferring v3 benchmark files."""
    return read_csv_or_zip_shared(path, preferred_name=preferred_name, **kwargs)


def normalize_for_id(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def safe_float(x, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def split_codes(value) -> List[str]:
    s = safe_str(value)
    if not s:
        return []
    parts = re.split(r"[,;|/\s]+", s)
    out = []
    seen = set()
    for p in parts:
        p = re.sub(r"[^A-Za-z0-9]", "", p).upper().strip()
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def primary_program_key(row: Mapping) -> str:
    codes = split_codes(row.get("ProgramElementCode(s)", ""))
    if codes:
        return "element_" + codes[0]
    name = normalize_for_id(row.get("Program(s)", ""))
    if name:
        return "name_" + name[:80]
    return "unknown_program"


def stable_hash(value: str, prefix: str = "id") -> str:
    value = safe_str(value)
    h = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{h}"


def make_edge_participants_from_awards(awards: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in awards.iterrows():
        award_id = safe_str(r.get("award_id", ""))
        person_id = safe_str(r.get("person_id", ""))
        if award_id and person_id:
            rows.append({
                "award_id": award_id,
                "person_id": person_id,
                "role": "PI",
                "start_year": r.get("start_year", np.nan),
                "institution_id": r.get("institution_id", ""),
                "organization_clean": r.get("organization_clean", ""),
                "NSFDirectorate": r.get("NSFDirectorate", ""),
                "NSFOrganization": r.get("NSFOrganization", ""),
            })
    return pd.DataFrame(rows)


def load_award_pi_edges(path: Optional[str | Path], awards: pd.DataFrame) -> pd.DataFrame:
    if path:
        edges = pd.read_csv(path, low_memory=False)
        if "person_id" not in edges.columns and "pi_id" in edges.columns:
            edges["person_id"] = edges["pi_id"]
        if "role" not in edges.columns:
            edges["role"] = "participant"
        keep = [c for c in [
            "award_id", "person_id", "role", "start_year", "institution_id",
            "organization_clean", "NSFDirectorate", "NSFOrganization"
        ] if c in edges.columns]
        edges = edges[keep].copy()
    else:
        print("[INFO] No --award_pi_edges provided; using lead PI only from awards table.")
        edges = make_edge_participants_from_awards(awards)

    edges["award_id"] = edges["award_id"].astype(str).str.strip()
    edges["person_id"] = edges["person_id"].astype(str).str.strip()
    edges = edges[(edges["award_id"] != "") & (edges["person_id"] != "")].copy()
    edges = edges.drop_duplicates(subset=["award_id", "person_id", "role"])
    return edges


def set_len(d: Mapping, key: str) -> int:
    return len(d.get(key, set()))


def list_len(d: Mapping, key: str) -> int:
    return len(d.get(key, []))


def years_since(current_year: int, year_value) -> float:
    if year_value is None or (isinstance(year_value, float) and math.isnan(year_value)):
        return 0.0
    try:
        return float(current_year - int(year_value))
    except Exception:
        return 0.0


def summarize(values: Sequence[float], prefix: str) -> Dict[str, float]:
    if not values:
        return {
            f"{prefix}_sum": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_nonzero_count": 0.0,
        }
    arr = np.asarray(values, dtype=float)
    return {
        f"{prefix}_sum": float(arr.sum()),
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_max": float(arr.max()),
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_nonzero_count": float((arr > 0).sum()),
    }


def add_log1p_features(feat: Dict[str, object], numeric_prefixes: Optional[Sequence[str]] = None) -> None:
    """Add log1p versions of selected count-like numeric features."""
    prefixes = tuple(numeric_prefixes or [
        "g_", "m_team_size", "m_num_co_pis"
    ])
    add = {}
    for k, v in feat.items():
        if not k.startswith(prefixes):
            continue
        try:
            val = float(v)
        except Exception:
            continue
        if val >= 0 and ("count" in k or "prior" in k or "degree" in k or "awards" in k or k.endswith("_sum") or k.endswith("_max")):
            add[f"{k}_log1p"] = float(np.log1p(val))
    feat.update(add)


def build_features(awards: pd.DataFrame, benchmark: pd.DataFrame, award_pi_edges: pd.DataFrame) -> pd.DataFrame:
    required_awards = {"award_id", "start_year", "person_id", "organization_clean", "NSFOrganization", "NSFDirectorate"}
    missing = sorted(required_awards - set(awards.columns))
    if missing:
        raise ValueError(f"Awards table is missing required columns: {missing}")
    if "award_id" not in benchmark.columns:
        raise ValueError("Benchmark table must contain award_id")

    awards = awards.copy()
    benchmark = benchmark.copy()
    awards["award_id"] = awards["award_id"].astype(str).str.strip()
    benchmark["award_id"] = benchmark["award_id"].astype(str).str.strip()
    awards["start_year"] = pd.to_numeric(awards["start_year"], errors="coerce").astype("Int64")
    benchmark["start_year"] = pd.to_numeric(benchmark["start_year"], errors="coerce").astype("Int64")

    # Add normalized graph keys.
    awards["g_primary_program_key"] = awards.apply(primary_program_key, axis=1)
    if "institution_id" not in awards.columns:
        awards["institution_id"] = awards["organization_clean"].apply(lambda x: stable_hash(x, "inst"))
    awards["g_institution_key"] = awards["institution_id"].fillna("").astype(str)
    awards["g_state_key"] = awards.get("OrganizationState", awards.get("State", "")).fillna("").astype(str).str.upper()
    awards["g_award_instrument"] = awards.get("AwardInstrument", "").fillna("").astype(str)
    awards["m_team_size"] = pd.to_numeric(awards.get("team_size", 1), errors="coerce").fillna(1).astype(float)
    awards["m_num_co_pis"] = pd.to_numeric(awards.get("num_co_pis", 0), errors="coerce").fillna(0).astype(float)

    meta_cols_to_merge = [
        "award_id", "g_primary_program_key", "g_institution_key", "g_state_key", "g_award_instrument",
        "m_team_size", "m_num_co_pis", "person_id", "NSFDirectorate", "NSFOrganization",
        "organization_clean", "start_year"
    ]
    meta_cols_to_merge = [c for c in meta_cols_to_merge if c in awards.columns]
    benchmark = benchmark.merge(
        awards[meta_cols_to_merge].drop_duplicates("award_id"),
        on="award_id",
        how="left",
        suffixes=("", "_awards")
    )
    # If benchmark already had columns, prefer benchmark values but fill missing from awards.
    for base in ["person_id", "NSFDirectorate", "NSFOrganization", "organization_clean", "start_year"]:
        aw = f"{base}_awards"
        if aw in benchmark.columns:
            if base in benchmark.columns:
                benchmark[base] = benchmark[base].where(benchmark[base].notna() & (benchmark[base].astype(str) != ""), benchmark[aw])
            else:
                benchmark[base] = benchmark[aw]
            benchmark = benchmark.drop(columns=[aw])

    # Participants by award.
    award_pi_edges = award_pi_edges.copy()
    award_pi_edges["award_id"] = award_pi_edges["award_id"].astype(str).str.strip()
    award_pi_edges["person_id"] = award_pi_edges["person_id"].astype(str).str.strip()
    participants_by_award: Dict[str, List[str]] = defaultdict(list)
    for award_id, group in award_pi_edges.groupby("award_id"):
        people = sorted(set([p for p in group["person_id"].astype(str) if p and p != "nan"]))
        participants_by_award[award_id] = people

    # Award metadata records indexed by award_id.
    award_records = awards.drop_duplicates("award_id").set_index("award_id").to_dict("index")
    labeled_ids = set(benchmark["award_id"].astype(str))
    labeled_by_year: Dict[int, List[Mapping]] = defaultdict(list)
    for _, r in benchmark.iterrows():
        if pd.isna(r.get("start_year")):
            continue
        labeled_by_year[int(r["start_year"])].append(r.to_dict())

    awards_by_year: Dict[int, List[Tuple[str, Mapping]]] = defaultdict(list)
    for award_id, r in award_records.items():
        y = r.get("start_year")
        if pd.isna(y):
            continue
        awards_by_year[int(y)].append((award_id, r))

    # Historical structures.
    person_award_count = defaultdict(int)
    person_programs = defaultdict(set)
    person_institutions = defaultdict(set)
    person_orgs = defaultdict(set)
    person_collaborators = defaultdict(set)
    person_first_year = {}
    person_last_year = {}

    inst_award_count = defaultdict(int)
    inst_people = defaultdict(set)
    inst_programs = defaultdict(set)
    inst_years = defaultdict(set)
    inst_first_year = {}
    inst_last_year = {}

    program_award_count = defaultdict(int)
    program_people = defaultdict(set)
    program_institutions = defaultdict(set)
    program_years = defaultdict(set)
    program_first_year = {}
    program_last_year = {}

    org_award_count = defaultdict(int)
    org_people = defaultdict(set)
    org_institutions = defaultdict(set)
    org_years = defaultdict(set)

    dir_award_count = defaultdict(int)
    dir_people = defaultdict(set)
    dir_institutions = defaultdict(set)
    dir_years = defaultdict(set)

    state_award_count = defaultdict(int)
    state_people = defaultdict(set)
    state_programs = defaultdict(set)
    state_years = defaultdict(set)

    pair_collab_count = defaultdict(int)

    all_years = sorted(set(awards_by_year.keys()) | set(labeled_by_year.keys()))
    out_rows: List[Dict[str, object]] = []

    def get_row_value(row: Mapping, key: str, default=""):
        v = row.get(key, default)
        return default if pd.isna(v) else v

    for year in all_years:
        # Compute features for labeled awards in this year BEFORE adding same-year awards to history.
        for br in labeled_by_year.get(year, []):
            award_id = safe_str(br.get("award_id", ""))
            ar = award_records.get(award_id, {})
            participants = participants_by_award.get(award_id, [])
            lead_pid = safe_str(get_row_value(br, "person_id", ar.get("person_id", "")))
            if not participants and lead_pid:
                participants = [lead_pid]

            program = safe_str(get_row_value(br, "g_primary_program_key", ar.get("g_primary_program_key", "unknown_program")))
            inst = safe_str(get_row_value(br, "g_institution_key", ar.get("g_institution_key", "")))
            org = safe_str(get_row_value(br, "NSFOrganization", ar.get("NSFOrganization", "")))
            direc = safe_str(get_row_value(br, "NSFDirectorate", ar.get("NSFDirectorate", "")))
            state = safe_str(get_row_value(br, "g_state_key", ar.get("g_state_key", "")))
            award_instrument = safe_str(get_row_value(br, "g_award_instrument", ar.get("g_award_instrument", "")))
            team_size = safe_float(get_row_value(br, "m_team_size", ar.get("m_team_size", len(participants) or 1)), 1.0)
            num_co_pis = safe_float(get_row_value(br, "m_num_co_pis", ar.get("m_num_co_pis", max(0, len(participants) - 1))), 0.0)

            person_counts = [person_award_count[p] for p in participants]
            person_collab_counts = [len(person_collaborators.get(p, set())) for p in participants]
            person_program_counts = [len(person_programs.get(p, set())) for p in participants]
            person_inst_counts = [len(person_institutions.get(p, set())) for p in participants]
            person_seen = [1 if person_award_count[p] > 0 else 0 for p in participants]

            pair_counts = []
            if len(participants) >= 2:
                people_sorted = sorted(set(participants))
                for i in range(len(people_sorted)):
                    for j in range(i + 1, len(people_sorted)):
                        pair_counts.append(pair_collab_count[frozenset((people_sorted[i], people_sorted[j]))])

            feat: Dict[str, object] = {
                "award_id": award_id,
                "start_year": int(year),
                # Static graph/metadata known at award time. Prefix m_ or cat_ so the trainer can select them safely.
                "cat_NSFDirectorate": direc or "unknown",
                "cat_NSFOrganization": org or "unknown",
                "cat_primary_program_key": program or "unknown_program",
                "cat_institution_key": inst or "unknown_institution",
                "cat_state_key": state or "unknown_state",
                "cat_award_instrument": award_instrument or "unknown",
                "m_start_year": int(year),
                "m_team_size": team_size,
                "m_num_co_pis": num_co_pis,
                "m_has_copi": 1.0 if num_co_pis > 0 else 0.0,
                "m_num_current_participants": float(len(set(participants))),
                # Lead PI history.
                "g_lead_pi_prior_award_count": float(person_award_count[lead_pid]) if lead_pid else 0.0,
                "g_lead_pi_prior_collaborator_count": float(len(person_collaborators.get(lead_pid, set()))) if lead_pid else 0.0,
                "g_lead_pi_prior_program_count": float(len(person_programs.get(lead_pid, set()))) if lead_pid else 0.0,
                "g_lead_pi_prior_institution_count": float(len(person_institutions.get(lead_pid, set()))) if lead_pid else 0.0,
                "g_lead_pi_seen_before": 1.0 if lead_pid and person_award_count[lead_pid] > 0 else 0.0,
                "g_lead_pi_years_since_first": years_since(year, person_first_year.get(lead_pid)),
                "g_lead_pi_years_since_last": years_since(year, person_last_year.get(lead_pid)),
                # Institution history.
                "g_institution_prior_award_count": float(inst_award_count[inst]),
                "g_institution_prior_person_count": float(len(inst_people.get(inst, set()))),
                "g_institution_prior_program_count": float(len(inst_programs.get(inst, set()))),
                "g_institution_prior_year_count": float(len(inst_years.get(inst, set()))),
                "g_institution_seen_before": 1.0 if inst_award_count[inst] > 0 else 0.0,
                "g_institution_age_years": years_since(year, inst_first_year.get(inst)),
                "g_institution_years_since_last": years_since(year, inst_last_year.get(inst)),
                # Program history.
                "g_program_prior_award_count": float(program_award_count[program]),
                "g_program_prior_person_count": float(len(program_people.get(program, set()))),
                "g_program_prior_institution_count": float(len(program_institutions.get(program, set()))),
                "g_program_prior_year_count": float(len(program_years.get(program, set()))),
                "g_program_seen_before": 1.0 if program_award_count[program] > 0 else 0.0,
                "g_program_age_years": years_since(year, program_first_year.get(program)),
                "g_program_years_since_last": years_since(year, program_last_year.get(program)),
                # NSF org/directorate/state history.
                "g_nsforg_prior_award_count": float(org_award_count[org]),
                "g_nsforg_prior_person_count": float(len(org_people.get(org, set()))),
                "g_nsforg_prior_institution_count": float(len(org_institutions.get(org, set()))),
                "g_nsforg_prior_year_count": float(len(org_years.get(org, set()))),
                "g_directorate_prior_award_count": float(dir_award_count[direc]),
                "g_directorate_prior_person_count": float(len(dir_people.get(direc, set()))),
                "g_directorate_prior_institution_count": float(len(dir_institutions.get(direc, set()))),
                "g_directorate_prior_year_count": float(len(dir_years.get(direc, set()))),
                "g_state_prior_award_count": float(state_award_count[state]),
                "g_state_prior_person_count": float(len(state_people.get(state, set()))),
                "g_state_prior_program_count": float(len(state_programs.get(state, set()))),
                "g_state_prior_year_count": float(len(state_years.get(state, set()))),
                # Cold-start flags.
                "g_cold_start_lead_pi": 1.0 if not lead_pid or person_award_count[lead_pid] == 0 else 0.0,
                "g_cold_start_institution": 1.0 if inst_award_count[inst] == 0 else 0.0,
                "g_cold_start_program": 1.0 if program_award_count[program] == 0 else 0.0,
            }
            feat.update(summarize(person_counts, "g_team_prior_award_count"))
            feat.update(summarize(person_collab_counts, "g_team_prior_collaborator_count"))
            feat.update(summarize(person_program_counts, "g_team_prior_program_count"))
            feat.update(summarize(person_inst_counts, "g_team_prior_institution_count"))
            feat.update(summarize(pair_counts, "g_current_team_prior_pair_collaboration_count"))
            feat["g_team_seen_before_fraction"] = float(np.mean(person_seen)) if person_seen else 0.0
            feat["g_current_team_any_prior_pair_collaboration"] = 1.0 if any(v > 0 for v in pair_counts) else 0.0
            add_log1p_features(feat)
            out_rows.append(feat)

        # After all target rows for this year are featurized, update history with all awards from this year.
        for award_id, ar in awards_by_year.get(year, []):
            participants = participants_by_award.get(award_id, [])
            lead_pid = safe_str(ar.get("person_id", ""))
            if not participants and lead_pid:
                participants = [lead_pid]
            participants = sorted(set([p for p in participants if p and p != "nan"]))
            program = safe_str(ar.get("g_primary_program_key", "unknown_program"))
            inst = safe_str(ar.get("g_institution_key", ""))
            org = safe_str(ar.get("NSFOrganization", ""))
            direc = safe_str(ar.get("NSFDirectorate", ""))
            state = safe_str(ar.get("g_state_key", ""))

            for p in participants:
                person_award_count[p] += 1
                if p not in person_first_year:
                    person_first_year[p] = year
                person_last_year[p] = year
                if program:
                    person_programs[p].add(program)
                if inst:
                    person_institutions[p].add(inst)
                if org:
                    person_orgs[p].add(org)
                for q in participants:
                    if q != p:
                        person_collaborators[p].add(q)

            if len(participants) >= 2:
                for i, p in enumerate(participants):
                    for q in participants[i + 1:]:
                        pair_collab_count[frozenset((p, q))] += 1

            inst_award_count[inst] += 1
            inst_years[inst].add(year)
            if inst not in inst_first_year:
                inst_first_year[inst] = year
            inst_last_year[inst] = year
            for p in participants:
                inst_people[inst].add(p)
            if program:
                inst_programs[inst].add(program)

            program_award_count[program] += 1
            program_years[program].add(year)
            if program not in program_first_year:
                program_first_year[program] = year
            program_last_year[program] = year
            for p in participants:
                program_people[program].add(p)
            if inst:
                program_institutions[program].add(inst)

            org_award_count[org] += 1
            org_years[org].add(year)
            for p in participants:
                org_people[org].add(p)
            if inst:
                org_institutions[org].add(inst)

            dir_award_count[direc] += 1
            dir_years[direc].add(year)
            for p in participants:
                dir_people[direc].add(p)
            if inst:
                dir_institutions[direc].add(inst)

            state_award_count[state] += 1
            state_years[state].add(year)
            for p in participants:
                state_people[state].add(p)
            if program:
                state_programs[state].add(program)

    feat_df = pd.DataFrame(out_rows)
    # Merge benchmark labels/splits/ids back in. Do not include text columns in the features output unless explicitly requested.
    keep_bench_cols = []
    for c in benchmark.columns:
        if c == "award_id" or c == "benchmark_id" or c == "annotation_id" or c.startswith(TARGET_COL_PREFIXES) or c.startswith(LABEL_COL_PREFIXES) or c.startswith(SPLIT_COL_PREFIXES) or c in {"project_cluster_id", "benchmark_version", "label_source"}:
            keep_bench_cols.append(c)
    keep_bench_cols = list(dict.fromkeys(keep_bench_cols))
    merged = benchmark[keep_bench_cols].merge(feat_df, on="award_id", how="left")
    return merged


def build_manifest(feature_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in feature_df.columns:
        if c.startswith("g_"):
            role = "graph_historical_feature"
        elif c.startswith("m_"):
            role = "known_static_numeric_metadata"
        elif c.startswith("cat_"):
            role = "known_static_categorical_metadata"
        elif c.startswith("target_"):
            role = "target"
        elif c.startswith("label_"):
            role = "label"
        elif c.startswith("split_"):
            role = "split"
        else:
            role = "identifier_or_metadata"
        dtype = str(feature_df[c].dtype)
        missing = int(feature_df[c].isna().sum())
        unique = int(feature_df[c].nunique(dropna=True))
        rows.append({"column": c, "role": role, "dtype": dtype, "missing": missing, "unique_non_null": unique})
    return pd.DataFrame(rows)


def build_summary(feature_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    def add(section, metric, value):
        rows.append({"section": section, "metric": metric, "value": value})
    add("dataset", "rows", len(feature_df))
    add("dataset", "unique_award_ids", feature_df["award_id"].nunique())
    for col in [c for c in feature_df.columns if c.startswith("split_")]:
        for val, count in feature_df[col].value_counts(dropna=False).items():
            add(f"split:{col}", str(val), int(count))
    for col in [c for c in feature_df.columns if c.startswith("target_")][:20]:
        for val, count in feature_df[col].value_counts(dropna=False).items():
            add(f"target:{col}", str(val), int(count))
    for col in ["g_cold_start_lead_pi", "g_cold_start_institution", "g_cold_start_program"]:
        if col in feature_df.columns:
            add("graph_feature", f"{col}_sum", int(pd.to_numeric(feature_df[col], errors="coerce").fillna(0).sum()))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", default=None, help="Repository root. Defaults to auto-detection.")
    parser.add_argument("--awards", default=None, help="Cleaned full awards CSV. Defaults to data/processed discovery.")
    parser.add_argument("--benchmark", default=None, help="Benchmark modeling CSV or benchmark_v3.zip. Defaults to v3 discovery under data/benchmark/.")
    parser.add_argument("--award_pi_edges", default=None, help="Optional award_pi_edges.csv. Recommended for Co-PI/team features.")
    parser.add_argument("--outdir", default=None, help="Output directory. Defaults to artifacts/features/graph_features_v1")
    parser.add_argument("--overwrite", action="store_true", help="Delete the output directory before writing new artifacts.")
    parser.add_argument("--output_prefix", default="methodkg_graph_only")
    args = parser.parse_args()

    repo_root = find_repo_root(args.repo_root or Path(__file__).resolve())
    awards_path = resolve_existing_path(args.awards, repo_root) if args.awards else discover_awards(repo_root)
    benchmark_path = resolve_existing_path(args.benchmark, repo_root) if args.benchmark else discover_benchmark(repo_root)
    award_pi_path = resolve_existing_path(args.award_pi_edges, repo_root, required=False) if args.award_pi_edges else discover_award_pi_edges(repo_root)
    outdir = resolve_output_path(args.outdir, repo_root, repo_root / "artifacts" / "features" / "graph_features_v1")
    reset_dir_if_overwrite(outdir, args.overwrite)
    write_resolved_paths(repo_root=repo_root, awards=awards_path, benchmark=benchmark_path, award_pi_edges=award_pi_path or "<not found; using lead PI only>", outdir=outdir)

    print("[INFO] Reading awards:", awards_path)
    awards = pd.read_csv(awards_path, low_memory=False)
    print("[INFO] Reading benchmark:", benchmark_path)
    benchmark = read_csv_or_zip(benchmark_path, preferred_name="methodkg_labeled_benchmark_v3_modeling.csv")
    print("[INFO] Loading award-person edges")
    award_pi_edges = load_award_pi_edges(str(award_pi_path) if award_pi_path else None, awards)

    print("[INFO] Building leakage-safe historical graph features")
    features = build_features(awards, benchmark, award_pi_edges)
    manifest = build_manifest(features)
    summary = build_summary(features)

    features_path = outdir / f"{args.output_prefix}_features.csv"
    manifest_path = outdir / f"{args.output_prefix}_feature_manifest.csv"
    summary_path = outdir / f"{args.output_prefix}_feature_summary.csv"
    features.to_csv(features_path, index=False, encoding="utf-8-sig")
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("[DONE] Wrote:")
    print(" -", features_path)
    print(" -", manifest_path)
    print(" -", summary_path)
    print("Rows:", len(features), "Columns:", features.shape[1])


if __name__ == "__main__":
    main()

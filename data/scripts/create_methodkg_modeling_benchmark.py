#!/usr/bin/env python3
"""
Create the official MethodKG labeled benchmark v3 modeling files from
``final_gold_labels_adjudicated.csv``.

Reviewer-facing default usage from the repository root:

  python data/scripts/create_methodkg_modeling_benchmark.py --overwrite

Default input:
  data/processed/final_gold_labels_adjudicated.csv

Default output directory:
  data/benchmark/

Main outputs:
  methodkg_labeled_benchmark_v3_modeling.csv
  methodkg_labeled_benchmark_v3_audit.csv
  methodkg_benchmark_v3_label_quality_report.csv
  methodkg_benchmark_v3_label_issues.csv
  methodkg_benchmark_v3_split_summary.csv
  methodkg_benchmark_v3_split_leakage_report.csv
  methodkg_benchmark_v3_duplicate_cluster_report.csv
  methodkg_benchmark_v3_reliability_report.csv
  methodkg_benchmark_v3_feature_manifest.csv
  methodkg_benchmark_v3_summary.json


  python data/scripts/create_methodkg_modeling_benchmark.py --overwrite

Notes:
  * This script is intentionally deterministic. Given the same final gold
    labels and seed, it reproduces the official v3 modeling benchmark.
  * The final gold file stores both numeric code columns and human-readable
    ``*_name`` columns. The human-readable adjudicated labels are used as the
    source of truth for modeling targets.
  * The script writes modeling-safe exports separately from audit/provenance
    outputs so that training code uses only approved columns.
"""

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LABEL_COLS = [
    "label_mmr_class",
    "label_qual_signal",
    "label_quant_signal",
    "label_design_present",
    "label_design_type",
    "label_integration_present",
    "label_integration_type",
    "label_reporting_completeness",
]

ALLOWED = {
    "label_mmr_class": {
        "explicit_mmr", "implicit_mmr", "qual_only", "quant_only",
        "multi_method_not_mmr", "unclear", "no_method_signal"
    },
    "label_qual_signal": {"yes", "no", "unclear"},
    "label_quant_signal": {"yes", "no", "unclear"},
    "label_design_present": {"yes", "no", "unclear"},
    "label_design_type": {
        "convergent", "explanatory_sequential", "exploratory_sequential",
        "embedded", "transformative", "multiphase", "triangulation",
        "other", "none", "unclear",
        "mixed_methods_explicit", "mixed_methods_implicit",
        "program_evaluation", "case_study", "quantitative_general",
        "qualitative_general", "ethnographic_or_observational", "survey",
        "learning_analytics_or_computational", "longitudinal",
        "instrument_development_validation", "participatory_or_community_based",
        "experimental", "design_based_research", "quasi_experimental"
    },
    "label_integration_present": {"yes", "no", "unclear"},
    "label_integration_type": {
        "merging", "connecting", "embedding", "triangulation",
        "joint_display", "meta_inference", "other", "none", "unclear",
        "explicit_mixed_methods", "implicit_unspecified", "evaluation_integration",
        "explanatory_sequential_quant_to_qual",
        "exploratory_sequential_qual_to_quant",
        "instrument_development", "embedded", "design_iteration",
        "convergent_parallel"
    },
    "label_reporting_completeness": {
        "low", "medium", "high", "unclear",
        "none_no_report", "minimal", "partial", "adequate", "complete"
    },
}

MMR_POSITIVE = {"explicit_mmr", "implicit_mmr"}
METHOD_SIGNAL_CLASSES = {
    "explicit_mmr", "implicit_mmr", "qual_only", "quant_only", "multi_method_not_mmr"
}
REPORTING_ORDINAL = {
    "none_no_report": 0, "minimal": 1, "partial": 2, "adequate": 3, "complete": 4,
    "low": 0, "medium": 1, "high": 2, "unclear": np.nan
}
YES_NO_UNCLEAR = {"yes": 1, "no": 0, "unclear": np.nan}

DESIGN_SPECIFIC_TYPES = {
    "program_evaluation", "case_study", "ethnographic_or_observational",
    "survey", "learning_analytics_or_computational", "longitudinal",
    "instrument_development_validation", "participatory_or_community_based",
    "experimental", "design_based_research", "quasi_experimental",
    "convergent", "explanatory_sequential", "exploratory_sequential",
    "embedded", "transformative", "multiphase", "triangulation", "other"
}
INTEGRATION_STRICT_TYPES = {
    "evaluation_integration", "explanatory_sequential_quant_to_qual",
    "exploratory_sequential_qual_to_quant", "instrument_development",
    "embedded", "design_iteration", "triangulation", "convergent_parallel",
    "merging", "connecting", "joint_display", "meta_inference", "other"
}

GENERATED_PREFIXES = (
    "target_",
    "split_",
)
GENERATED_COLUMNS = {
    "benchmark_id", "benchmark_version", "label_source", "project_cluster_id",
    "cluster_size", "cluster_min_year", "cluster_max_year", "cluster_text_duplicate_flag",
    "primary_program_key", "primary_program_source", "label_final_status",
    "annotation_status_v3", "adjudication_status_v3", "label_issue_count",
    "hard_label_issue_count", "review_label_issue_count", "warning_label_issue_count",
    "stratify_label_v3",
}

AUDIT_OR_PROVENANCE_PATTERNS = (
    "candidate", "annotation_guidance", "review_priority", "annotation_phase",
    "annotation_status", "double_annotation_required", "adjudication_status",
)

LEAKAGE_RISK_COLUMNS = {
    "award_amount",
    "AwardedAmountToDate",
    "LastAmendmentDate",
    "last_amendment_date",
    "award_status_derived",
}

MODEL_SAFE_BASE_COLUMNS = [
    "benchmark_id", "benchmark_version", "label_source", "label_final_status",
    "annotation_id", "award_id", "project_cluster_id", "cluster_size",
    "title_clean", "abstract_clean", "start_year", "NSFDirectorate", "NSFOrganization",
    "Program(s)", "ProgramElementCode(s)", "primary_program_key", "primary_program_source",
    "AwardInstrument", "pi_clean", "person_id", "organization_clean", "State",
    "label_mmr_class", "label_qual_signal", "label_quant_signal",
    "label_design_present", "label_design_type", "label_integration_present",
    "label_integration_type", "label_reporting_completeness",
    "target_mmr_binary", "target_mmr_multiclass", "target_explicit_mmr_binary",
    "target_implicit_mmr_binary", "target_method_signal_binary", "target_qual_binary",
    "target_quant_binary", "target_design_binary", "target_design_type",
    "target_integration_binary", "target_integration_type", "target_reporting_quality",
    "target_reporting_quality_ordinal",
    "target_design_binary_broad", "target_design_specific_binary",
    "target_integration_binary_broad", "target_integration_strict_binary",
    "split_random_cluster_stratified", "split_temporal_cluster_safe",
    "split_edu_to_eng_cluster_safe", "split_cross_program_cluster_safe",
    "split_cold_start_pi_cluster_safe", "split_cold_start_institution_cluster_safe",
]

_SPLIT_ORDER = ["train", "validation", "test"]
_SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def stable_hash_hex(value: str, n: int = 12) -> str:
    value = "" if pd.isna(value) else str(value)
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:n]


def stable_hash_int(value: str) -> int:
    value = "" if pd.isna(value) else str(value)
    return int(hashlib.sha1(value.encode("utf-8")).hexdigest(), 16)


def normalize_label_series(s: pd.Series) -> pd.Series:
    """Normalize labels from either string labels or *_name columns.

    This version is intentionally stricter than v2 because the final gold
    adjudicated file may contain labels such as "none/no report" and
    broad design/integration type names.
    """
    out = (
        s.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    return out


def value_counts_count_then_first(series: pd.Series) -> List[Tuple[object, int]]:
    """Return value counts sorted by count desc, then first appearance.

    This reproduces the order used in the released v3 quality report for tied
    counts, avoiding harmless but noisy byte-level diffs across pandas versions.
    """
    counts = series.value_counts(dropna=False).to_dict()
    first_pos = {}
    for i, value in enumerate(series.tolist()):
        key = "__nan__" if pd.isna(value) else value
        if key not in first_pos:
            first_pos[key] = i
    def sort_key(item):
        value, count = item
        key = "__nan__" if pd.isna(value) else value
        return (-int(count), first_pos.get(key, 10**12), str(value))
    return sorted(counts.items(), key=sort_key)


def canonicalize_final_gold_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Make final_gold_labels_adjudicated.csv compatible with modeling targets.

    The final gold file may store numeric code columns such as label_mmr_class
    and human-readable columns such as label_mmr_class_name. The modeling
    pipeline expects canonical string labels, so when a *_name column exists we
    use it as the source of truth for the corresponding base label.
    """
    out = df.copy()
    name_to_base = {
        "label_mmr_class_name": "label_mmr_class",
        "label_qual_signal_name": "label_qual_signal",
        "label_quant_signal_name": "label_quant_signal",
        "label_design_present_name": "label_design_present",
        "label_integration_present_name": "label_integration_present",
        "label_reporting_completeness_name": "label_reporting_completeness",
    }
    for name_col, base_col in name_to_base.items():
        if name_col in out.columns:
            out[base_col] = out[name_col]
    if "project_text_id" in out.columns and "project_cluster_id" not in out.columns:
        out["project_cluster_id"] = out["project_text_id"]
    return out


def normalize_text_for_cluster(value: str) -> str:
    if pd.isna(value):
        return ""
    value = str(value).lower().strip()
    value = re.sub(r"\s+", " ", value)
    return value


def normalize_group_value(value: str) -> str:
    if pd.isna(value):
        return "unknown"
    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value if value else "unknown"


def split_program_codes(value: str) -> List[str]:
    if pd.isna(value):
        return []
    value = str(value).strip()
    if not value:
        return []
    parts = re.split(r"[,;|/\s]+", value)
    out = []
    seen = set()
    for p in parts:
        p = re.sub(r"[^A-Za-z0-9]", "", p).upper().strip()
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def normalize_program_name(value: str) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value


def first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def mode_with_tiebreak(values: Iterable) -> str:
    vals = [str(v) for v in values if not pd.isna(v)]
    if not vals:
        return "unknown"
    counts = Counter(vals)
    max_count = max(counts.values())
    winners = sorted([k for k, v in counts.items() if v == max_count])
    return winners[0]


class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, x: str):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: str) -> str:
        self.add(x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def clean_generated_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = []
    for col in df.columns:
        if col in GENERATED_COLUMNS or any(col.startswith(p) for p in GENERATED_PREFIXES):
            drop_cols.append(col)
    return df.drop(columns=drop_cols, errors="ignore")


def add_identifiers_and_clusters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "benchmark_id" not in out.columns:
        out.insert(0, "benchmark_id", [f"methodkg_v3_{i + 1:05d}" for i in range(len(out))])
    else:
        out["benchmark_id"] = [f"methodkg_v3_{i + 1:05d}" for i in range(len(out))]

    out["benchmark_version"] = "methodkg_labeled_benchmark_v3"
    out["label_source"] = "human_verified_ra"

    if "project_text_id" in out.columns:
        out["project_cluster_id"] = out["project_text_id"].fillna("").astype(str)
        missing_cluster = out["project_cluster_id"].eq("")
        if missing_cluster.any():
            title = out.get("title_clean", pd.Series([""] * len(out))).map(normalize_text_for_cluster)
            abstract = out.get("abstract_clean", pd.Series([""] * len(out))).map(normalize_text_for_cluster)
            cluster_key = title + " || " + abstract
            out.loc[missing_cluster, "project_cluster_id"] = cluster_key[missing_cluster].map(lambda x: "cluster_" + stable_hash_hex(x))
    else:
        title = out.get("title_clean", pd.Series([""] * len(out))).map(normalize_text_for_cluster)
        abstract = out.get("abstract_clean", pd.Series([""] * len(out))).map(normalize_text_for_cluster)
        cluster_key = title + " || " + abstract
        out["project_cluster_id"] = cluster_key.map(lambda x: "cluster_" + stable_hash_hex(x))

    cluster_stats = out.groupby("project_cluster_id", dropna=False).agg(
        cluster_size=("award_id", "size") if "award_id" in out.columns else ("project_cluster_id", "size"),
        cluster_min_year=("start_year", lambda s: pd.to_numeric(s, errors="coerce").min()),
        cluster_max_year=("start_year", lambda s: pd.to_numeric(s, errors="coerce").max()),
    ).reset_index()
    out = out.merge(cluster_stats, on="project_cluster_id", how="left")
    out["cluster_text_duplicate_flag"] = (out["cluster_size"] > 1).astype(int)
    return out


def add_primary_program_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    program_keys = []
    sources = []
    for _, row in out.iterrows():
        codes = split_program_codes(row.get("ProgramElementCode(s)", ""))
        if codes:
            program_keys.append("element:" + codes[0])
            sources.append("ProgramElementCode")
            continue
        name = normalize_program_name(row.get("Program(s)", ""))
        if name:
            program_keys.append("name:" + name)
            sources.append("ProgramNameFallback")
        else:
            program_keys.append("missing_program")
            sources.append("MissingProgram")
    out["primary_program_key"] = program_keys
    out["primary_program_source"] = sources
    return out


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in LABEL_COLS:
        if col not in out.columns:
            raise ValueError(f"Missing required label column: {col}")
        out[col] = normalize_label_series(out[col])

    out["target_mmr_binary"] = out["label_mmr_class"].isin(MMR_POSITIVE).astype(int)
    out["target_mmr_multiclass"] = out["label_mmr_class"]
    out["target_explicit_mmr_binary"] = (out["label_mmr_class"] == "explicit_mmr").astype(int)
    out["target_implicit_mmr_binary"] = (out["label_mmr_class"] == "implicit_mmr").astype(int)
    out["target_method_signal_binary"] = out["label_mmr_class"].isin(METHOD_SIGNAL_CLASSES).astype(int)

    out["target_qual_binary"] = out["label_qual_signal"].map(YES_NO_UNCLEAR)
    out["target_quant_binary"] = out["label_quant_signal"].map(YES_NO_UNCLEAR)
    out["target_design_binary"] = out["label_design_present"].map(YES_NO_UNCLEAR)
    out["target_design_binary_broad"] = out["target_design_binary"]
    out["target_design_specific_binary"] = out["label_design_type"].isin(DESIGN_SPECIFIC_TYPES).astype(int)
    out["target_design_type"] = out["label_design_type"]
    out["target_integration_binary"] = out["label_integration_present"].map(YES_NO_UNCLEAR)
    out["target_integration_binary_broad"] = out["target_integration_binary"]
    out["target_integration_strict_binary"] = out["label_integration_type"].isin(INTEGRATION_STRICT_TYPES).astype(int)
    out["target_integration_type"] = out["label_integration_type"]
    out["target_reporting_quality"] = out["label_reporting_completeness"]
    out["target_reporting_quality_ordinal"] = out["label_reporting_completeness"].map(REPORTING_ORDINAL)

    design_token = out["target_design_binary"].fillna(-1).astype(int).astype(str)
    integration_token = out["target_integration_binary"].fillna(-1).astype(int).astype(str)
    out["stratify_label_v3"] = (
        out["target_mmr_multiclass"].astype(str)
        + "|design=" + design_token
        + "|integration=" + integration_token
    )
    return out


def split_bucket_from_hash(value: str) -> str:
    bucket = stable_hash_int(value) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def assign_cluster_stratified_split(
    df: pd.DataFrame,
    cluster_col: str,
    stratify_col: str,
    seed: int,
    out_col: str,
) -> pd.Series:
    rng = np.random.default_rng(seed)
    cluster_df = df.groupby(cluster_col, dropna=False).agg(
        row_count=(cluster_col, "size"),
        stratum=(stratify_col, mode_with_tiebreak),
    ).reset_index()

    cluster_to_split = {}
    for stratum, sub in cluster_df.groupby("stratum", dropna=False):
        clusters = sub[[cluster_col, "row_count"]].copy()
        order = np.arange(len(clusters))
        rng.shuffle(order)
        clusters = clusters.iloc[order].reset_index(drop=True)

        total_rows = int(clusters["row_count"].sum())
        targets = {
            "train": total_rows * _SPLIT_RATIOS["train"],
            "validation": total_rows * _SPLIT_RATIOS["validation"],
            "test": total_rows * _SPLIT_RATIOS["test"],
        }
        current = {k: 0 for k in _SPLIT_ORDER}

        # For very small strata, deterministic hash gives better coverage than
        # forcing impossible 70/15/15 row counts.
        if len(clusters) < 3:
            for _, row in clusters.iterrows():
                cluster_to_split[row[cluster_col]] = split_bucket_from_hash(f"{out_col}:{row[cluster_col]}")
            continue

        for _, row in clusters.iterrows():
            # Greedy weighted assignment: place the next cluster in the split with
            # the largest remaining deficit relative to its target.
            deficits = {k: targets[k] - current[k] for k in _SPLIT_ORDER}
            chosen = max(_SPLIT_ORDER, key=lambda k: (deficits[k], -current[k]))
            cluster_to_split[row[cluster_col]] = chosen
            current[chosen] += int(row["row_count"])

    return df[cluster_col].map(cluster_to_split).fillna("unknown")


def assign_temporal_cluster_safe_split(df: pd.DataFrame) -> pd.Series:
    # Conservative rule: use the cluster max year. If any duplicate/collaborative
    # abstract appears in a future period, the whole cluster is assigned to that
    # future period so the same text cannot appear in earlier training data.
    max_year = pd.to_numeric(df["cluster_max_year"], errors="coerce")
    return pd.Series(
        np.select(
            [max_year <= 2016, (max_year >= 2017) & (max_year <= 2019), max_year >= 2020],
            ["train", "validation", "test"],
            default="unknown",
        ),
        index=df.index,
    )


def assign_edu_to_eng_cluster_safe_split(df: pd.DataFrame) -> pd.Series:
    dir_upper = df.get("NSFDirectorate", pd.Series([""] * len(df))).fillna("").astype(str).str.upper()
    tmp = pd.DataFrame({"project_cluster_id": df["project_cluster_id"], "is_eng": dir_upper.eq("ENG")})
    cluster_has_eng = tmp.groupby("project_cluster_id")["is_eng"].any().to_dict()
    return df["project_cluster_id"].map(lambda c: "test" if cluster_has_eng.get(c, False) else "train")


def assign_component_hash_split(
    df: pd.DataFrame,
    group_col: str,
    out_col: str,
    cluster_col: str = "project_cluster_id",
) -> pd.Series:
    uf = UnionFind()
    for _, row in df[[cluster_col, group_col]].iterrows():
        cluster_node = f"cluster:{row[cluster_col]}"
        group_value = normalize_group_value(row[group_col])
        group_node = f"{group_col}:{group_value}"
        uf.union(cluster_node, group_node)

    comp_to_split = {}
    row_splits = []
    for _, row in df[[cluster_col]].iterrows():
        comp = uf.find(f"cluster:{row[cluster_col]}")
        if comp not in comp_to_split:
            comp_to_split[comp] = split_bucket_from_hash(f"{out_col}:{comp}")
        row_splits.append(comp_to_split[comp])
    return pd.Series(row_splits, index=df.index)


def add_splits(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    out = df.copy()
    out["split_random_cluster_stratified"] = assign_cluster_stratified_split(
        out, "project_cluster_id", "stratify_label_v3", seed, "split_random_cluster_stratified"
    )
    out["split_temporal_cluster_safe"] = assign_temporal_cluster_safe_split(out)
    out["split_edu_to_eng_cluster_safe"] = assign_edu_to_eng_cluster_safe_split(out)
    out["split_cold_start_pi_cluster_safe"] = assign_component_hash_split(
        out, "person_id", "split_cold_start_pi_cluster_safe"
    )
    out["split_cold_start_institution_cluster_safe"] = assign_component_hash_split(
        out, "organization_clean", "split_cold_start_institution_cluster_safe"
    )
    out["split_cross_program_cluster_safe"] = assign_component_hash_split(
        out, "primary_program_key", "split_cross_program_cluster_safe"
    )
    return out


def build_label_issues(df: pd.DataFrame) -> pd.DataFrame:
    issues = []

    def add_issue(mask, issue_type, severity, message):
        cols = [
            "benchmark_id", "annotation_id", "award_id", "title_clean", "label_mmr_class",
            "label_qual_signal", "label_quant_signal", "label_design_present",
            "label_design_type", "label_integration_present", "label_integration_type",
        ]
        cols = [c for c in cols if c in df.columns]
        sub = df.loc[mask, cols].copy()
        if len(sub) == 0:
            return
        sub.insert(0, "issue_type", issue_type)
        sub.insert(1, "severity", severity)
        sub.insert(2, "message", message)
        issues.append(sub)

    for col in LABEL_COLS:
        add_issue(
            df[col].fillna("").eq(""),
            f"missing_{col}", "error", f"Missing label value for {col}."
        )
        add_issue(
            ~df[col].isin(ALLOWED[col]),
            f"invalid_{col}", "error", f"Invalid label value for {col}."
        )

    add_issue(
        (df["label_design_present"] == "yes") & (df["label_design_type"] == "none"),
        "design_yes_type_none", "error",
        "Design is marked present, but design type is none."
    )
    add_issue(
        (df["label_design_present"] == "no") & (~df["label_design_type"].isin(["none", "unclear"])),
        "design_no_type_real", "error",
        "Design is marked absent, but a design type is provided."
    )
    add_issue(
        (df["label_integration_present"] == "yes") & (df["label_integration_type"] == "none"),
        "integration_yes_type_none", "error",
        "Integration is marked present, but integration type is none."
    )
    add_issue(
        (df["label_integration_present"] == "no") & (~df["label_integration_type"].isin(["none", "unclear"])),
        "integration_no_type_real", "error",
        "Integration is marked absent, but an integration type is provided."
    )
    add_issue(
        (df["label_mmr_class"] == "qual_only") & (df["label_quant_signal"] == "yes"),
        "qual_only_with_quant_yes", "error",
        "Class is qual_only, but quantitative signal is yes."
    )
    add_issue(
        (df["label_mmr_class"] == "quant_only") & (df["label_qual_signal"] == "yes"),
        "quant_only_with_qual_yes", "error",
        "Class is quant_only, but qualitative signal is yes."
    )
    add_issue(
        (df["label_mmr_class"] == "no_method_signal") & (
            (df["label_qual_signal"] == "yes") |
            (df["label_quant_signal"] == "yes") |
            (df["label_design_present"] == "yes") |
            (df["label_integration_present"] == "yes")
        ),
        "no_method_signal_with_method_label", "error",
        "Class is no_method_signal, but a method-related label is yes."
    )
    add_issue(
        (df["label_mmr_class"] == "implicit_mmr") & (df["label_qual_signal"] != "yes"),
        "implicit_mmr_without_qual_yes", "error",
        "implicit_mmr should normally have qualitative signal = yes. Review this row."
    )
    add_issue(
        (df["label_mmr_class"] == "implicit_mmr") & (df["label_quant_signal"] != "yes"),
        "implicit_mmr_without_quant_yes", "error",
        "implicit_mmr should normally have quantitative signal = yes. Review this row."
    )
    add_issue(
        (df["label_mmr_class"] == "explicit_mmr") & (df["label_qual_signal"] != "yes"),
        "explicit_mmr_without_qual_yes", "review",
        "explicit_mmr may be valid without detailed qualitative evidence, but should be checked."
    )
    add_issue(
        (df["label_mmr_class"] == "explicit_mmr") & (df["label_quant_signal"] != "yes"),
        "explicit_mmr_without_quant_yes", "review",
        "explicit_mmr may be valid without detailed quantitative evidence, but should be checked."
    )
    add_issue(
        (df["label_reporting_completeness"] == "high") & (df["label_mmr_class"].isin(["no_method_signal", "unclear"])),
        "high_reporting_without_method_signal", "review",
        "Reporting completeness is high even though no method signal/unclear class is assigned."
    )

    if issues:
        return pd.concat(issues, ignore_index=True)
    return pd.DataFrame(columns=[
        "issue_type", "severity", "message", "benchmark_id", "annotation_id", "award_id",
        "title_clean", "label_mmr_class", "label_qual_signal", "label_quant_signal",
        "label_design_present", "label_design_type", "label_integration_present", "label_integration_type",
    ])


def add_label_workflow_status(
    df: pd.DataFrame,
    issues: pd.DataFrame,
    mark_double_annotation_as_adjudicated: bool = False,
) -> pd.DataFrame:
    out = df.copy()
    issue_counts = issues.groupby(["benchmark_id", "severity"]).size().unstack(fill_value=0) if len(issues) else pd.DataFrame()

    for severity in ["error", "review", "warning"]:
        col = f"{severity}_label_issue_count"
        if len(issue_counts) and severity in issue_counts.columns:
            out[col] = out["benchmark_id"].map(issue_counts[severity]).fillna(0).astype(int)
        else:
            out[col] = 0
    out["hard_label_issue_count"] = out["error_label_issue_count"]
    out["label_issue_count"] = (
        out["error_label_issue_count"] + out["review_label_issue_count"] + out["warning_label_issue_count"]
    )
    out["label_final_status"] = np.where(
        out["hard_label_issue_count"] > 0,
        "needs_label_review",
        "ready_for_modeling"
    )

    # v3 workflow metadata: do not overwrite original status columns. These fields
    # reflect what can be inferred from the file itself.
    labels_complete = out[LABEL_COLS].notna().all(axis=1) & ~out[LABEL_COLS].astype(str).eq("").any(axis=1)
    out["annotation_status_v3"] = np.where(labels_complete, "complete_labels_present", "incomplete_labels")

    if "double_annotation_required" in out.columns:
        double_required = out["double_annotation_required"].fillna("no").astype(str).str.lower().isin(["yes", "true", "1"])
    else:
        double_required = pd.Series(False, index=out.index)

    if mark_double_annotation_as_adjudicated:
        out["adjudication_status_v3"] = np.where(double_required, "adjudicated_by_user_flag", "not_required")
    else:
        out["adjudication_status_v3"] = np.where(double_required, "needs_adjudication_metadata", "not_required")
    return out


def build_duplicate_cluster_report(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("project_cluster_id", dropna=False).agg(
        rows=("award_id", "size") if "award_id" in df.columns else ("project_cluster_id", "size"),
        award_ids=("award_id", lambda s: "|".join(sorted(set(s.astype(str)))) if "award_id" in df.columns else ("project_cluster_id", "size")),
        years=("start_year", lambda s: "|".join(str(int(y)) for y in sorted(set(pd.to_numeric(s, errors="coerce").dropna())))),
        titles=("title_clean", lambda s: " || ".join(list(dict.fromkeys(s.fillna("").astype(str).head(3))))),
    ).reset_index()
    return agg[agg["rows"] > 1].sort_values(["rows", "project_cluster_id"], ascending=[False, True]).copy()


def build_split_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    split_cols = [c for c in df.columns if c.startswith("split_")]
    for split_col in split_cols:
        for split_name, sub in df.groupby(split_col, dropna=False):
            row = {
                "split_column": split_col,
                "split": split_name,
                "rows": len(sub),
                "clusters": sub["project_cluster_id"].nunique(),
                "mmr_positive": int(sub["target_mmr_binary"].sum()),
                "design_positive": int(sub["target_design_binary"].fillna(0).sum()),
                "integration_positive": int(sub["target_integration_binary"].fillna(0).sum()),
                "explicit_mmr": int((sub["label_mmr_class"] == "explicit_mmr").sum()),
                "implicit_mmr": int((sub["label_mmr_class"] == "implicit_mmr").sum()),
                "qual_only": int((sub["label_mmr_class"] == "qual_only").sum()),
                "quant_only": int((sub["label_mmr_class"] == "quant_only").sum()),
                "multi_method_not_mmr": int((sub["label_mmr_class"] == "multi_method_not_mmr").sum()),
                "no_method_signal": int((sub["label_mmr_class"] == "no_method_signal").sum()),
                "unclear": int((sub["label_mmr_class"] == "unclear").sum()),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def _crossing_groups(df: pd.DataFrame, split_col: str, group_col: str) -> pd.DataFrame:
    sub = df.groupby(group_col, dropna=False).agg(
        rows=(group_col, "size"),
        split_count=(split_col, "nunique"),
        splits=(split_col, lambda s: "|".join(sorted(set(s.astype(str))))),
    ).reset_index()
    return sub[(sub["rows"] > 1) & (sub["split_count"] > 1)].copy()


def build_split_leakage_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    split_cols = [c for c in df.columns if c.startswith("split_")]
    group_checks = {
        "project_cluster_id": "duplicate_text_cluster",
        "person_id": "same_pi",
        "organization_clean": "same_institution",
        "primary_program_key": "same_program",
    }
    for split_col in split_cols:
        for group_col, check_name in group_checks.items():
            if group_col not in df.columns:
                continue
            crossing = _crossing_groups(df, split_col, group_col)
            rows.append({
                "split_column": split_col,
                "check": check_name,
                "group_column": group_col,
                "crossing_group_count": int(len(crossing)),
                "crossing_row_count": int(crossing["rows"].sum()) if len(crossing) else 0,
            })
    return pd.DataFrame(rows)


def build_quality_report(df: pd.DataFrame, issues: pd.DataFrame, duplicate_clusters: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(section, metric, value):
        rows.append({"section": section, "metric": metric, "value": value})

    add("dataset", "rows", len(df))
    add("dataset", "unique_award_ids", df["award_id"].nunique() if "award_id" in df.columns else "")
    add("dataset", "unique_annotation_ids", df["annotation_id"].nunique() if "annotation_id" in df.columns else "")
    add("dataset", "unique_project_clusters", df["project_cluster_id"].nunique())
    add("dataset", "duplicate_project_clusters", len(duplicate_clusters))
    add("dataset", "rows_in_duplicate_project_clusters", int(duplicate_clusters["rows"].sum()) if len(duplicate_clusters) else 0)
    add("dataset", "missing_abstracts", int(df["abstract_clean"].fillna("").eq("").sum()) if "abstract_clean" in df.columns else "")
    add("dataset", "missing_titles", int(df["title_clean"].fillna("").eq("").sum()) if "title_clean" in df.columns else "")
    add("dataset", "start_year_min", int(pd.to_numeric(df["start_year"], errors="coerce").min()))
    add("dataset", "start_year_max", int(pd.to_numeric(df["start_year"], errors="coerce").max()))

    for col in LABEL_COLS:
        add("missing_labels", col, int(df[col].fillna("").eq("").sum()))
        invalid = sorted(set(df[col].dropna().astype(str)) - ALLOWED[col])
        add("invalid_label_values", col, "|".join(invalid) if invalid else "")
        for val, count in value_counts_count_then_first(df[col]):
            add(f"label_distribution:{col}", str(val), int(count))

    target_cols = [
        "target_mmr_binary", "target_method_signal_binary", "target_design_binary",
        "target_integration_binary", "target_reporting_quality_ordinal"
    ]
    for col in target_cols:
        if col in df.columns:
            for val, count in value_counts_count_then_first(df[col]):
                add(f"target_distribution:{col}", str(val), int(count))

    for status, count in value_counts_count_then_first(df["label_final_status"]):
        add("label_final_status", str(status), int(count))
    for status, count in value_counts_count_then_first(df["annotation_status_v3"]):
        add("annotation_status_v3", str(status), int(count))
    for status, count in value_counts_count_then_first(df["adjudication_status_v3"]):
        add("adjudication_status_v3", str(status), int(count))

    if len(issues):
        for issue_type, count in issues["issue_type"].value_counts().items():
            add("label_issues", issue_type, int(count))
        for severity, count in issues["severity"].value_counts().items():
            add("label_issue_severity", severity, int(count))
    else:
        add("label_issues", "total", 0)

    return pd.DataFrame(rows)


def build_reliability_report(df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple agreement if paired annotator columns exist; otherwise report unavailable.

    Expected optional naming conventions:
      annotator1_label_mmr_class / annotator2_label_mmr_class
      label_mmr_class_a1 / label_mmr_class_a2
      ann1_label_mmr_class / ann2_label_mmr_class
    """
    rows = []

    def add(label_col, metric, value, note=""):
        rows.append({"label": label_col, "metric": metric, "value": value, "note": note})

    for label_col in LABEL_COLS:
        possible_pairs = [
            (f"annotator1_{label_col}", f"annotator2_{label_col}"),
            (f"{label_col}_annotator1", f"{label_col}_annotator2"),
            (f"{label_col}_a1", f"{label_col}_a2"),
            (f"ann1_{label_col}", f"ann2_{label_col}"),
        ]
        pair = next((p for p in possible_pairs if p[0] in df.columns and p[1] in df.columns), None)
        if pair is None:
            add(label_col, "status", "unavailable", "No paired annotator-specific columns found.")
            continue

        a = normalize_label_series(df[pair[0]])
        b = normalize_label_series(df[pair[1]])
        valid = a.ne("") & b.ne("")
        n = int(valid.sum())
        if n == 0:
            add(label_col, "status", "unavailable", "Paired columns exist but have no overlapping labels.")
            continue

        agreement = float((a[valid] == b[valid]).mean())
        add(label_col, "paired_rows", n)
        add(label_col, "percent_agreement", round(agreement, 4))

        # Cohen's kappa for nominal labels.
        labels = sorted(set(a[valid]) | set(b[valid]))
        if labels:
            obs = agreement
            pa = a[valid].value_counts(normalize=True).reindex(labels, fill_value=0)
            pb = b[valid].value_counts(normalize=True).reindex(labels, fill_value=0)
            exp = float((pa * pb).sum())
            kappa = (obs - exp) / (1 - exp) if not math.isclose(1 - exp, 0.0) else np.nan
            add(label_col, "cohens_kappa", round(float(kappa), 4) if not pd.isna(kappa) else "nan")

    return pd.DataFrame(rows)


def build_feature_manifest(df: pd.DataFrame, modeling_cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        if col in modeling_cols:
            role = "modeling_safe_export"
        elif col in LEAKAGE_RISK_COLUMNS:
            role = "excluded_leakage_risk"
        elif any(pattern in col for pattern in AUDIT_OR_PROVENANCE_PATTERNS):
            role = "audit_only_annotation_or_sampling_provenance"
        elif col in LABEL_COLS or col.startswith("target_"):
            role = "label_or_target"
        elif col.startswith("split_"):
            role = "evaluation_split"
        else:
            role = "audit_only_or_unused_metadata"
        rows.append({"column": col, "role": role})
    return pd.DataFrame(rows)


def make_modeling_safe_df(df: pd.DataFrame, include_award_amount: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    cols = [c for c in MODEL_SAFE_BASE_COLUMNS if c in df.columns]
    if include_award_amount and "award_amount" in df.columns:
        insert_after = "AwardInstrument" if "AwardInstrument" in cols else cols[-1]
        idx = cols.index(insert_after) + 1
        cols.insert(idx, "award_amount")

    # Never include candidate/provenance/generated audit columns beyond explicit split/status fields.
    cols = list(dict.fromkeys(cols))
    return df[cols].copy(), cols


def reorder_audit_columns(df: pd.DataFrame) -> pd.DataFrame:
    first_cols = [
        "benchmark_id", "benchmark_version", "label_source", "label_final_status",
        "annotation_status_v3", "adjudication_status_v3", "label_issue_count",
        "hard_label_issue_count", "review_label_issue_count", "warning_label_issue_count",
        "annotation_id", "award_id", "project_cluster_id", "cluster_size",
        "cluster_min_year", "cluster_max_year", "cluster_text_duplicate_flag",
        "title_clean", "abstract_clean", "start_year", "NSFDirectorate", "NSFOrganization",
        "Program(s)", "ProgramElementCode(s)", "primary_program_key", "primary_program_source",
        "AwardInstrument", "award_amount", "pi_clean", "person_id", "organization_clean", "State",
    ] + LABEL_COLS + [
        "target_mmr_binary", "target_mmr_multiclass", "target_explicit_mmr_binary",
        "target_implicit_mmr_binary", "target_method_signal_binary", "target_qual_binary",
        "target_quant_binary", "target_design_binary", "target_design_type",
        "target_integration_binary", "target_integration_type", "target_reporting_quality",
        "target_reporting_quality_ordinal",
        "target_design_binary_broad", "target_design_specific_binary",
        "target_integration_binary_broad", "target_integration_strict_binary",
        "stratify_label_v3",
        "split_random_cluster_stratified", "split_temporal_cluster_safe",
        "split_edu_to_eng_cluster_safe", "split_cross_program_cluster_safe",
        "split_cold_start_pi_cluster_safe", "split_cold_start_institution_cluster_safe",
    ]
    first_cols = [c for c in first_cols if c in df.columns]
    remaining = [c for c in df.columns if c not in first_cols]
    return df[first_cols + remaining]


def write_json_summary(path: Path, report: pd.DataFrame, split_leakage: pd.DataFrame):
    summary = {
        "created_by": "create_methodkg_labeled_benchmark_v3_final_gold.py",
        "key_counts": {},
        "split_leakage_checks": split_leakage.to_dict(orient="records"),
    }
    for _, row in report.iterrows():
        if row["section"] == "dataset":
            summary["key_counts"][row["metric"]] = row["value"]
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


OUTPUT_FILENAMES = [
    "methodkg_labeled_benchmark_v3_modeling.csv",
    "methodkg_labeled_benchmark_v3_audit.csv",
    "methodkg_benchmark_v3_label_quality_report.csv",
    "methodkg_benchmark_v3_label_issues.csv",
    "methodkg_benchmark_v3_split_summary.csv",
    "methodkg_benchmark_v3_split_leakage_report.csv",
    "methodkg_benchmark_v3_duplicate_cluster_report.csv",
    "methodkg_benchmark_v3_reliability_report.csv",
    "methodkg_benchmark_v3_feature_manifest.csv",
    "methodkg_benchmark_v3_summary.json",
]


def resolve_repo_path(path_value: Optional[str], repo_root: Path, default_relative: str) -> Path:
    """Resolve CLI paths consistently from the repository root."""
    path = Path(path_value) if path_value else repo_root / default_relative
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compare_output_dirs(generated_dir: Path, reference_dir: Path, filenames: Sequence[str]) -> pd.DataFrame:
    """Compare generated outputs against a reference benchmark directory."""
    rows = []
    for name in filenames:
        generated = generated_dir / name
        reference = reference_dir / name
        row = {
            "file": name,
            "generated_exists": generated.exists(),
            "reference_exists": reference.exists(),
            "generated_size": generated.stat().st_size if generated.exists() else None,
            "reference_size": reference.stat().st_size if reference.exists() else None,
            "sha256_match": False,
        }
        if generated.exists() and reference.exists():
            row["sha256_match"] = file_sha256(generated) == file_sha256(reference)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_root",
        default=None,
        help="Repository root. Defaults to current working directory.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Final adjudicated gold CSV. Default: data/processed/final_gold_labels_adjudicated.csv",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory. Default: data/benchmark",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for cluster-stratified split")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing benchmark output files.",
    )
    parser.add_argument(
        "--compare_to_dir",
        default=None,
        help="Optional reference benchmark directory to compare generated outputs against.",
    )
    parser.add_argument(
        "--from_v1",
        action="store_true",
        help="Use if the input is methodkg_labeled_benchmark_v1.csv. Generated v1 columns will be dropped and rebuilt.",
    )
    parser.add_argument(
        "--include_award_amount_in_modeling_file",
        action="store_true",
        help="Include award_amount in the modeling-safe file. Default excludes it because AwardedAmountToDate can be leakage-prone.",
    )
    parser.add_argument(
        "--mark_double_annotation_as_adjudicated",
        action="store_true",
        help="Only use this if double-annotated rows have already been adjudicated outside this file.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path.cwd().resolve()
    input_path = resolve_repo_path(args.input, repo_root, "data/processed/final_gold_labels_adjudicated.csv")
    outdir = resolve_repo_path(args.outdir, repo_root, "data/benchmark")

    if not input_path.exists():
        raise FileNotFoundError(f"Input final gold file not found: {input_path}")

    existing_outputs = [name for name in OUTPUT_FILENAMES if (outdir / name).exists()]
    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            f"Output directory already contains {len(existing_outputs)} benchmark files: {outdir}. "
            "Pass --overwrite, or write to a temporary --outdir first."
        )

    outdir.mkdir(parents=True, exist_ok=True)

    print("Resolved paths:")
    print("  repo_root:", repo_root)
    print("  input:", input_path)
    print("  outdir:", outdir)

    df = pd.read_csv(input_path, low_memory=False, encoding="utf-8-sig")
    df = canonicalize_final_gold_schema(df)
    if args.from_v1:
        df = clean_generated_columns(df)

    missing_cols = [c for c in LABEL_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing label columns: {missing_cols}")
    for required in ["award_id", "title_clean", "abstract_clean", "start_year"]:
        if required not in df.columns:
            raise ValueError(f"Missing required benchmark column: {required}")

    benchmark = add_identifiers_and_clusters(df)
    benchmark = add_primary_program_key(benchmark)
    benchmark = add_targets(benchmark)
    benchmark = add_splits(benchmark, seed=args.seed)

    issues = build_label_issues(benchmark)
    benchmark = add_label_workflow_status(
        benchmark,
        issues,
        mark_double_annotation_as_adjudicated=args.mark_double_annotation_as_adjudicated,
    )
    duplicate_clusters = build_duplicate_cluster_report(benchmark)
    quality_report = build_quality_report(benchmark, issues, duplicate_clusters)
    split_summary = build_split_summary(benchmark)
    split_leakage = build_split_leakage_report(benchmark)
    reliability_report = build_reliability_report(benchmark)

    audit = reorder_audit_columns(benchmark)
    modeling, modeling_cols = make_modeling_safe_df(
        benchmark,
        include_award_amount=args.include_award_amount_in_modeling_file,
    )
    feature_manifest = build_feature_manifest(benchmark, modeling_cols)

    modeling.to_csv(outdir / "methodkg_labeled_benchmark_v3_modeling.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(outdir / "methodkg_labeled_benchmark_v3_audit.csv", index=False, encoding="utf-8-sig")
    quality_report.to_csv(outdir / "methodkg_benchmark_v3_label_quality_report.csv", index=False, encoding="utf-8-sig")
    issues.to_csv(outdir / "methodkg_benchmark_v3_label_issues.csv", index=False, encoding="utf-8-sig")
    split_summary.to_csv(outdir / "methodkg_benchmark_v3_split_summary.csv", index=False, encoding="utf-8-sig")
    split_leakage.to_csv(outdir / "methodkg_benchmark_v3_split_leakage_report.csv", index=False, encoding="utf-8-sig")
    duplicate_clusters.to_csv(outdir / "methodkg_benchmark_v3_duplicate_cluster_report.csv", index=False, encoding="utf-8-sig")
    reliability_report.to_csv(outdir / "methodkg_benchmark_v3_reliability_report.csv", index=False, encoding="utf-8-sig")
    feature_manifest.to_csv(outdir / "methodkg_benchmark_v3_feature_manifest.csv", index=False, encoding="utf-8-sig")
    write_json_summary(outdir / "methodkg_benchmark_v3_summary.json", quality_report, split_leakage)

    print("Wrote MethodKG benchmark v3 outputs to", outdir.resolve())
    print("Rows:", len(benchmark))
    print("Unique project clusters:", benchmark["project_cluster_id"].nunique())
    print("Duplicate text clusters:", len(duplicate_clusters))
    print("Label issue rows:", len(issues))
    print("Modeling-safe file columns:", len(modeling.columns))
    print("Files:")
    for name in OUTPUT_FILENAMES:
        path = outdir / name
        print(f" - {name}: {path.stat().st_size:,} bytes")

    if args.compare_to_dir:
        reference_dir = resolve_repo_path(args.compare_to_dir, repo_root, args.compare_to_dir)
        if not reference_dir.exists():
            raise FileNotFoundError(f"Reference benchmark directory not found: {reference_dir}")
        comparison = compare_output_dirs(outdir, reference_dir, OUTPUT_FILENAMES)
        comparison_path = outdir / "methodkg_benchmark_v3_output_comparison.csv"
        comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
        mismatches = comparison[~comparison["sha256_match"]]
        print("Comparison against:", reference_dir)
        print(comparison.to_string(index=False))
        print(f"Wrote comparison report: {comparison_path}")
        if len(mismatches):
            raise SystemExit(
                f"Output comparison failed: {len(mismatches)} files differ or are missing. "
                "Inspect methodkg_benchmark_v3_output_comparison.csv before promoting outputs."
            )
        print("Output comparison passed: all benchmark files match the reference directory.")


if __name__ == "__main__":
    main()

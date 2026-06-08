#!/usr/bin/env python3
"""
Run lightweight graph-only baselines for multiple MethodKG targets and splits.

Example:
  python run_graph_all_splits.py \
    --features graph_features_v1/methodkg_graph_only_features.csv \
    --outdir results/graph_only_all \
    --targets target_integration_binary target_design_binary target_mmr_multiclass
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from path_utils import find_repo_root, reset_dir_if_overwrite, resolve_existing_path, resolve_output_path, write_resolved_paths

DEFAULT_SPLITS = [
    "split_random_cluster_stratified",
    "split_temporal_cluster_safe",
    "split_cross_program_cluster_safe",
    "split_cold_start_pi_cluster_safe",
    "split_cold_start_institution_cluster_safe",
    "split_edu_to_eng_cluster_safe",
]

DEFAULT_TARGETS = [
    "target_integration_binary",
    "target_design_binary",
    "target_mmr_multiclass",
]

DEFAULT_MODELS = ["dummy", "graph_lr", "graph_rf", "graph_extra_trees"]

TARGET_SHORT = {
    "target_integration_binary": "integration_binary",
    "target_design_binary": "design_binary",
    "target_mmr_multiclass": "mmr_multiclass",
    "target_mmr_binary": "mmr_binary",
    "target_qual_binary": "qual_binary",
    "target_quant_binary": "quant_binary",
    "target_method_signal_binary": "method_signal_binary",
}


def short_name(value: str) -> str:
    if value in TARGET_SHORT:
        return TARGET_SHORT[value]
    value = value.replace("target_", "").replace("split_", "")
    return value


def run_command(cmd, dry_run=False):
    print("\n$", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def write_metrics_rollup(results_root: Path, paper_outputs: Path, summary_name: str, table_name: str) -> None:
    try:
        import pandas as pd
    except Exception:
        return
    rows = []
    for path in results_root.rglob("metrics.csv"):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        rel = path.parent.relative_to(results_root)
        parts = rel.parts
        df.insert(0, "result_dir", str(rel))
        if len(parts) >= 1 and "target" not in df.columns:
            df.insert(1, "target_folder", parts[0])
        if len(parts) >= 2 and "split_folder" not in df.columns:
            df.insert(2, "split_folder", parts[1])
        if len(parts) >= 3 and "model_family" not in df.columns:
            df.insert(3, "model_family", parts[2])
        rows.append(df)
    if not rows:
        return
    summary = pd.concat(rows, ignore_index=True)
    summaries = paper_outputs / "summaries"
    tables = paper_outputs / "tables"
    summaries.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summaries / summary_name, index=False, encoding="utf-8-sig")
    split_cols = [c for c in ["split", "split_folder"] if c in summary.columns]
    if split_cols:
        mask = summary[split_cols[0]].astype(str).str.lower().eq("test") if split_cols[0] == "split" else summary[split_cols[0]].astype(str).str.contains("test", case=False, na=False)
        test_summary = summary[mask].copy()
    else:
        test_summary = summary.copy()
    test_summary.to_csv(tables / table_name, index=False, encoding="utf-8-sig")
    print("Wrote paper rollups:", summaries / summary_name, "and", tables / table_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", default=None, help="Repository root. Defaults to auto-detection.")
    parser.add_argument("--features", default=None, help="Feature CSV. Defaults to artifacts/features/graph_features_v1/methodkg_graph_only_features.csv")
    parser.add_argument("--outdir", default=None, help="Output directory. Defaults to experiments/graph_only/historical_features/graph_only_primary_all")
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        choices=["dummy", "dummy_stratified", "graph_lr", "graph_rf", "graph_extra_trees", "graph_hgb"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_static_metadata", action="store_true")
    parser.add_argument("--keep_unclear", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Delete the experiment output root before rerunning.")
    args = parser.parse_args()

    repo_root = find_repo_root(args.repo_root or Path(__file__).resolve())
    features_path = resolve_existing_path(args.features, repo_root) if args.features else resolve_existing_path(repo_root / "artifacts" / "features" / "graph_features_v1" / "methodkg_graph_only_features.csv", repo_root)
    out_root = resolve_output_path(args.outdir, repo_root, repo_root / "experiments" / "graph_only" / "historical_features" / "graph_only_primary_all")
    reset_dir_if_overwrite(out_root, args.overwrite)
    paper_outputs = repo_root / "paper_outputs"
    write_resolved_paths(repo_root=repo_root, features=features_path, experiments_outdir=out_root, paper_outputs=paper_outputs)

    script_dir = Path(__file__).resolve().parent
    runner = script_dir / "run_graph_baselines.py"

    for target in args.targets:
        for split in args.splits:
            outdir = out_root / short_name(target) / short_name(split)
            cmd = [
                sys.executable,
                str(runner),
                "--features", str(features_path),
                "--outdir", str(outdir),
                "--target", target,
                "--split_col", split,
                "--models", *args.models,
                "--seed", str(args.seed),
            ]
            if args.no_static_metadata:
                cmd.append("--no_static_metadata")
            if args.keep_unclear:
                cmd.append("--keep_unclear")
            run_command(cmd, dry_run=args.dry_run)

    print("\n[DONE] Completed graph-only baseline runs.")
    print("Results root:", out_root)
    write_metrics_rollup(out_root, paper_outputs, "graph_only_historical_features_metrics_summary.csv", "graph_only_historical_features_test_metrics.csv")


if __name__ == "__main__":
    main()

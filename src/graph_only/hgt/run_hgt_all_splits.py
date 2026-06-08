#!/usr/bin/env python3
"""Run MethodKG structural-only HGT over multiple targets and split columns."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from path_utils import find_repo_root, reset_dir_if_overwrite, resolve_existing_path, resolve_output_path, write_resolved_paths

TARGET_ALIASES = {
    "target_integration_binary": "integration_binary",
    "target_design_binary": "design_binary",
    "target_mmr_multiclass": "mmr_multiclass",
    "target_mmr_binary": "mmr_binary",
    "target_qual_binary": "qual_binary",
    "target_quant_binary": "quant_binary",
    "target_method_signal_binary": "method_signal_binary",
}

SPLIT_ALIASES = {
    "split_random_cluster_stratified": "random_cluster_stratified",
    "split_temporal_cluster_safe": "temporal_cluster_safe",
    "split_cross_program_cluster_safe": "cross_program_cluster_safe",
    "split_cold_start_pi_cluster_safe": "cold_start_pi_cluster_safe",
    "split_cold_start_institution_cluster_safe": "cold_start_institution_cluster_safe",
    "split_edu_to_eng_cluster_safe": "edu_to_eng_cluster_safe",
}

DEFAULT_TARGETS = [
    "target_integration_binary",
    "target_design_binary",
    "target_mmr_multiclass",
]

DEFAULT_SPLITS = [
    "split_random_cluster_stratified",
    "split_temporal_cluster_safe",
    "split_cross_program_cluster_safe",
    "split_cold_start_pi_cluster_safe",
    "split_cold_start_institution_cluster_safe",
    "split_edu_to_eng_cluster_safe",
]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=None, help="Repository root. Defaults to auto-detection.")
    ap.add_argument("--graph_dir", default=None, help="Graph directory. Defaults to artifacts/graphs/hgt_data_v1")
    ap.add_argument("--outdir", default=None, help="Output directory. Defaults to experiments/graph_only/hgt_structural/hgt_structural_primary_all_h256")
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    ap.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    ap.add_argument("--hidden_channels", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.35)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--use_amp", action="store_true")
    ap.add_argument("--tune_threshold", action="store_true")
    ap.add_argument("--threshold_objective", choices=["positive_f1", "macro_f1"], default="positive_f1")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="Delete the experiment output root before rerunning.")
    args = ap.parse_args()

    repo_root = find_repo_root(args.repo_root or Path(__file__).resolve())
    graph_dir = resolve_existing_path(args.graph_dir, repo_root) if args.graph_dir else resolve_existing_path(repo_root / "artifacts" / "graphs" / "hgt_data_v1", repo_root)
    outroot = resolve_output_path(args.outdir, repo_root, repo_root / "experiments" / "graph_only" / "hgt_structural" / "hgt_structural_primary_all_h256")
    reset_dir_if_overwrite(outroot, args.overwrite)
    paper_outputs = repo_root / "paper_outputs"
    write_resolved_paths(repo_root=repo_root, graph_dir=graph_dir, experiments_outdir=outroot, paper_outputs=paper_outputs)

    script = Path(__file__).with_name("run_hgt_structural.py")

    for target in args.targets:
        target_alias = TARGET_ALIASES.get(target, target.replace("target_", ""))
        for split in args.splits:
            split_alias = SPLIT_ALIASES.get(split, split.replace("split_", ""))
            outdir = outroot / target_alias / split_alias / "hgt_structural"
            cmd = [
                sys.executable, str(script),
                "--graph_dir", str(graph_dir),
                "--outdir", str(outdir),
                "--target", target,
                "--split_col", split,
                "--hidden_channels", str(args.hidden_channels),
                "--num_layers", str(args.num_layers),
                "--heads", str(args.heads),
                "--dropout", str(args.dropout),
                "--lr", str(args.lr),
                "--weight_decay", str(args.weight_decay),
                "--epochs", str(args.epochs),
                "--patience", str(args.patience),
                "--seed", str(args.seed),
                "--device", args.device,
                "--threshold_objective", args.threshold_objective,
            ]
            if args.use_amp:
                cmd.append("--use_amp")
            if args.tune_threshold:
                cmd.append("--tune_threshold")
            print("$", " ".join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, check=True)

    write_metrics_rollup(outroot, paper_outputs, "hgt_structural_metrics_summary.csv", "hgt_structural_test_metrics.csv")


if __name__ == "__main__":
    main()

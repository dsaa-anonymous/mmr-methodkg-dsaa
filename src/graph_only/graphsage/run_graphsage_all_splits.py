#!/usr/bin/env python3
"""Run structural GraphSAGE over multiple MethodKG targets and benchmark splits."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from path_utils import find_repo_root, reset_dir_if_overwrite, resolve_existing_path, resolve_output_path, write_resolved_paths

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


def short_target_name(target: str) -> str:
    s = target
    for prefix in ["target_", "label_"]:
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s.replace("_binary", "_binary").replace("_multiclass", "_multiclass")


def short_split_name(split: str) -> str:
    s = split
    for prefix in ["split_"]:
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s


def run(cmd, dry_run=False):
    print("\n$", " ".join(cmd), flush=True)
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
    parser.add_argument("--graph_dir", default=None, help="Graph directory. Defaults to artifacts/graphs/graphsage_data_v2")
    parser.add_argument("--outdir", default=None, help="Output directory. Defaults to experiments/graph_only/graphsage_structural/graphsage_structural_primary_all_v2")
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--tune_threshold", action="store_true")
    parser.add_argument("--threshold_metric", default="positive_f1", choices=["positive_f1", "macro_f1"])
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Delete the experiment output root before rerunning.")
    args = parser.parse_args()

    repo_root = find_repo_root(args.repo_root or Path(__file__).resolve())
    graph_dir = resolve_existing_path(args.graph_dir, repo_root) if args.graph_dir else resolve_existing_path(repo_root / "artifacts" / "graphs" / "graphsage_data_v2", repo_root)
    out_root = resolve_output_path(args.outdir, repo_root, repo_root / "experiments" / "graph_only" / "graphsage_structural" / "graphsage_structural_primary_all_v2")
    reset_dir_if_overwrite(out_root, args.overwrite)
    paper_outputs = repo_root / "paper_outputs"
    write_resolved_paths(repo_root=repo_root, graph_dir=graph_dir, experiments_outdir=out_root, paper_outputs=paper_outputs)

    script = Path(__file__).resolve().parent / "run_graphsage_structural.py"
    for target in args.targets:
        for split in args.splits:
            outdir = out_root / short_target_name(target) / short_split_name(split) / "graphsage_structural"
            cmd = [
                sys.executable,
                str(script),
                "--graph_dir", str(graph_dir),
                "--outdir", str(outdir),
                "--target", target,
                "--split_col", split,
                "--hidden_channels", str(args.hidden_channels),
                "--num_layers", str(args.num_layers),
                "--dropout", str(args.dropout),
                "--lr", str(args.lr),
                "--weight_decay", str(args.weight_decay),
                "--epochs", str(args.epochs),
                "--patience", str(args.patience),
                "--seed", str(args.seed),
                "--device", args.device,
                "--threshold_metric", args.threshold_metric,
            ]
            if args.tune_threshold:
                cmd.append("--tune_threshold")
            if args.use_amp:
                cmd.append("--use_amp")
            run(cmd, dry_run=args.dry_run)

    write_metrics_rollup(out_root, paper_outputs, "graphsage_structural_metrics_summary.csv", "graphsage_structural_test_metrics.csv")


if __name__ == "__main__":
    main()


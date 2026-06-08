#!/usr/bin/env python3
"""Run SimTeG-GraphSAGE over multiple MethodKG targets and benchmark splits."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from path_utils import default_simteg_graph_dir, find_repo_root, reset_dir_if_overwrite, resolve_existing_path, resolve_output_path, write_metrics_rollup, write_resolved_paths

DEFAULT_TARGETS = ["target_integration_binary", "target_design_binary", "target_mmr_multiclass"]
DEFAULT_SPLITS = [
    "split_random_cluster_stratified",
    "split_temporal_cluster_safe",
    "split_cross_program_cluster_safe",
    "split_cold_start_pi_cluster_safe",
    "split_cold_start_institution_cluster_safe",
    "split_edu_to_eng_cluster_safe",
]


def short_target_name(target: str) -> str:
    for prefix in ["target_", "label_"]:
        if target.startswith(prefix):
            target = target[len(prefix):]
    return target


def short_split_name(split: str) -> str:
    return split[len("split_"):] if split.startswith("split_") else split


def run(cmd, dry_run=False):
    print("\n$", " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", default=None, help="Repository root. Defaults to auto-detection.")
    parser.add_argument("--graph_dir", default=None, help="Graph directory. Defaults to artifacts/graphs/simteg_graphsage_data_scibert_v1")
    parser.add_argument("--outdir", default=None, help="Output root. Defaults to experiments/text_graph/simteg_graphsage/primary")
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
    graph_dir = resolve_existing_path(args.graph_dir, repo_root) if args.graph_dir else resolve_existing_path(default_simteg_graph_dir(repo_root), repo_root)
    out_root = resolve_output_path(args.outdir, repo_root, repo_root / "experiments" / "text_graph" / "simteg_graphsage" / "primary")
    reset_dir_if_overwrite(out_root, args.overwrite)
    paper_outputs = repo_root / "paper_outputs"
    write_resolved_paths(repo_root=repo_root, graph_dir=graph_dir, experiments_outdir=out_root, paper_outputs=paper_outputs)

    script = Path(__file__).resolve().parent / "run_simteg_graphsage.py"
    for target in args.targets:
        for split in args.splits:
            outdir = out_root / short_target_name(target) / short_split_name(split) / "simteg_graphsage"
            cmd = [
                sys.executable, str(script),
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

    write_metrics_rollup(out_root, paper_outputs, "simteg_graphsage_metrics_summary.csv", "simteg_graphsage_test_metrics.csv")


if __name__ == "__main__":
    main()

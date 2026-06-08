#!/usr/bin/env python3
"""Run Text-HGT over multiple MethodKG targets and benchmark splits."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from path_utils import default_text_hgt_graph_dir, find_repo_root, reset_dir_if_overwrite, resolve_existing_path, resolve_output_path, write_metrics_rollup, write_resolved_paths

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
    return target[len("target_"):] if target.startswith("target_") else target


def short_split_name(split: str) -> str:
    return split[len("split_"):] if split.startswith("split_") else split


def run(cmd, dry_run=False):
    print("\n$", " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=None, help="Repository root. Defaults to auto-detection.")
    ap.add_argument("--graph_dir", default=None, help="Graph directory. Defaults to artifacts/graphs/text_hgt_data_scibert_v1")
    ap.add_argument("--outdir", default=None, help="Output root. Defaults to experiments/text_graph/text_hgt/primary")
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
    graph_dir = resolve_existing_path(args.graph_dir, repo_root) if args.graph_dir else resolve_existing_path(default_text_hgt_graph_dir(repo_root), repo_root)
    outroot = resolve_output_path(args.outdir, repo_root, repo_root / "experiments" / "text_graph" / "text_hgt" / "primary")
    reset_dir_if_overwrite(outroot, args.overwrite)
    paper_outputs = repo_root / "paper_outputs"
    write_resolved_paths(repo_root=repo_root, graph_dir=graph_dir, experiments_outdir=outroot, paper_outputs=paper_outputs)

    script = Path(__file__).resolve().parent / "run_text_hgt.py"
    for target in args.targets:
        target_alias = short_target_name(target)
        for split in args.splits:
            split_alias = short_split_name(split)
            outdir = outroot / target_alias / split_alias / "text_hgt"
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
            run(cmd, dry_run=args.dry_run)

    write_metrics_rollup(outroot, paper_outputs, "text_hgt_metrics_summary.csv", "text_hgt_test_metrics.csv")


if __name__ == "__main__":
    main()

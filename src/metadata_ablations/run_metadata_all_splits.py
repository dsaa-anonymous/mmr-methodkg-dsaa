#!/usr/bin/env python3
"""Run MethodKG metadata-only or text+metadata ablations across targets and splits."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from path_utils import (
    default_text_embeddings_scibert,
    discover_benchmark,
    find_repo_root,
    reset_dir_if_overwrite,
    resolve_existing_path,
    resolve_output_path,
    write_metrics_rollup,
    write_resolved_paths,
)

DEFAULT_SPLITS = [
    "split_random_cluster_stratified",
    "split_temporal_cluster_safe",
    "split_cross_program_cluster_safe",
    "split_cold_start_pi_cluster_safe",
    "split_cold_start_institution_cluster_safe",
    "split_edu_to_eng_cluster_safe",
]

TARGET_SLUG = {
    "target_integration_binary": "integration_binary",
    "target_design_binary": "design_binary",
    "target_mmr_multiclass": "mmr_multiclass",
    "target_mmr_binary": "mmr_binary",
    "target_qual_binary": "qual_binary",
    "target_quant_binary": "quant_binary",
    "target_method_signal_binary": "method_signal_binary",
}

SPLIT_SLUG = {
    "split_random_cluster_stratified": "random_cluster_stratified",
    "split_temporal_cluster_safe": "temporal_cluster_safe",
    "split_cross_program_cluster_safe": "cross_program_cluster_safe",
    "split_cold_start_pi_cluster_safe": "cold_start_pi_cluster_safe",
    "split_cold_start_institution_cluster_safe": "cold_start_institution_cluster_safe",
    "split_edu_to_eng_cluster_safe": "edu_to_eng_cluster_safe",
}


def run(cmd, dry_run=False):
    print("\n$ " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=None, help="Repository root. Defaults to auto-detection.")
    ap.add_argument("--benchmark", default=None, help="Benchmark CSV/zip. Defaults to data/benchmark v3.")
    ap.add_argument("--outdir", default=None, help="Output root. Defaults to experiments/text_graph/text_metadata_scibert_v1")
    ap.add_argument("--targets", nargs="+", default=["target_integration_binary", "target_design_binary", "target_mmr_multiclass"])
    ap.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    ap.add_argument("--text_embeddings", default=None, help="Optional text embedding CSV. Omit for metadata-only; use --use_default_text_embeddings for SciBERT defaults.")
    ap.add_argument("--use_default_text_embeddings", action="store_true", help="Use artifacts/features/text_embeddings_scibert_mean_v1/methodkg_text_embeddings.csv if --text_embeddings is omitted.")
    ap.add_argument("--metadata_cols", nargs="*", default=None)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--class_weight", default="balanced", choices=["balanced", "none"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tune_threshold", action="store_true")
    ap.add_argument("--threshold_metric", default="macro_f1", choices=["macro_f1", "positive_f1"])
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="Delete the experiment output root before rerunning.")
    args = ap.parse_args()

    repo_root = find_repo_root(args.repo_root or Path(__file__).resolve())
    benchmark = resolve_existing_path(args.benchmark, repo_root) if args.benchmark else discover_benchmark(repo_root)
    text_embeddings = None
    if args.text_embeddings:
        text_embeddings = resolve_existing_path(args.text_embeddings, repo_root)
    elif args.use_default_text_embeddings and default_text_embeddings_scibert(repo_root).exists():
        text_embeddings = default_text_embeddings_scibert(repo_root).resolve()
    default_out = repo_root / "experiments" / "text_graph" / ("text_metadata_scibert_v1" if text_embeddings else "metadata_only")
    outroot = resolve_output_path(args.outdir, repo_root, default_out)
    reset_dir_if_overwrite(outroot, args.overwrite)
    paper_outputs = repo_root / "paper_outputs"
    write_resolved_paths(repo_root=repo_root, benchmark=benchmark, experiments_outdir=outroot, text_embeddings=text_embeddings, paper_outputs=paper_outputs)

    script = Path(__file__).resolve().parent / "run_metadata_ablations.py"
    mode_name = "text_metadata" if text_embeddings else "metadata_only"

    default_models = ["dummy", "metadata_lr", "metadata_svm", "metadata_mlp", "metadata_extra_trees"]
    models = args.models if args.models else default_models

    for target in args.targets:
        tslug = TARGET_SLUG.get(target, target.replace("target_", ""))
        for split in args.splits:
            sslug = SPLIT_SLUG.get(split, split.replace("split_", ""))
            od = outroot / tslug / sslug / mode_name
            cmd = [
                sys.executable, str(script),
                "--repo_root", str(repo_root),
                "--benchmark", str(benchmark),
                "--outdir", str(od),
                "--target", target,
                "--split_col", split,
                "--models", *models,
                "--class_weight", args.class_weight,
                "--seed", str(args.seed),
            ]
            if text_embeddings:
                cmd += ["--text_embeddings", str(text_embeddings)]
            if args.metadata_cols:
                cmd += ["--metadata_cols", *args.metadata_cols]
            if args.tune_threshold:
                cmd += ["--tune_threshold", "--threshold_metric", args.threshold_metric]
            run(cmd, dry_run=args.dry_run)

    if text_embeddings:
        write_metrics_rollup(outroot, paper_outputs, "text_metadata_scibert_metrics_summary.csv", "text_metadata_scibert_test_metrics.csv")
    else:
        write_metrics_rollup(outroot, paper_outputs, "metadata_only_metrics_summary.csv", "metadata_only_test_metrics.csv")


if __name__ == "__main__":
    main()

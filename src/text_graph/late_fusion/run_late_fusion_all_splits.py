#!/usr/bin/env python3
"""Run MethodKG late-fusion baselines over multiple targets and splits."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from path_utils import (
    default_graph_features,
    default_metapath2vec_embeddings,
    default_node2vec_embeddings,
    default_text_embeddings_minilm,
    default_text_embeddings_scibert,
    discover_benchmark,
    find_repo_root,
    reset_dir_if_overwrite,
    resolve_existing_path,
    resolve_output_path,
    write_metrics_rollup,
    write_resolved_paths,
)

TARGET_NAME_MAP = {
    "target_integration_binary": "integration_binary",
    "target_design_binary": "design_binary",
    "target_mmr_multiclass": "mmr_multiclass",
    "target_mmr_binary": "mmr_binary",
    "target_qual_binary": "qual_binary",
    "target_quant_binary": "quant_binary",
    "target_method_signal_binary": "method_signal_binary",
}
SPLIT_NAME_MAP = {
    "split_random_cluster_stratified": "random_cluster_stratified",
    "split_temporal_cluster_safe": "temporal_cluster_safe",
    "split_cross_program_cluster_safe": "cross_program_cluster_safe",
    "split_cold_start_pi_cluster_safe": "cold_start_pi_cluster_safe",
    "split_cold_start_institution_cluster_safe": "cold_start_institution_cluster_safe",
    "split_edu_to_eng_cluster_safe": "edu_to_eng_cluster_safe",
}
DEFAULT_TARGETS = ["target_integration_binary", "target_design_binary", "target_mmr_multiclass"]
DEFAULT_SPLITS = list(SPLIT_NAME_MAP.keys())
DEFAULT_MODELS = ["dummy", "fusion_lr", "fusion_svm", "fusion_mlp", "fusion_extra_trees"]


def run(cmd, dry_run=False):
    print("\n$", " ".join(str(c) for c in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def maybe_resolve(value, repo_root, default_path, required=False):
    if value:
        return resolve_existing_path(value, repo_root, required=required)
    if default_path.exists():
        return default_path.resolve()
    if required:
        return resolve_existing_path(default_path, repo_root, required=True)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=None, help="Repository root. Defaults to auto-detection.")
    ap.add_argument("--benchmark", default=None, help="Benchmark CSV/zip. Defaults to data/benchmark v3.")
    ap.add_argument("--outdir", default=None, help="Output root. Defaults to experiments/text_graph/late_fusion_<embedding_family>/primary")
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    ap.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    ap.add_argument("--embedding_family", choices=["minilm", "scibert"], default="minilm", help="Which late-fusion text embedding family to use when --text_embeddings/--outdir are not supplied.")
    ap.add_argument("--text_embeddings", default=None, help="Text embedding CSV. Defaults to the artifact for --embedding_family.")
    ap.add_argument("--graph_features", default=None, help="Defaults to artifacts/features/graph_features_v1/methodkg_graph_only_features.csv if present.")
    ap.add_argument("--node2vec_embeddings", default=None, help="Defaults to artifacts/features/walk_embeddings_v1/node2vec_award_embeddings.csv if present.")
    ap.add_argument("--metapath2vec_embeddings", default=None, help="Defaults to artifacts/features/walk_embeddings_v1/metapath2vec_award_embeddings.csv if present.")
    ap.add_argument("--include_metadata", action="store_true")
    ap.add_argument("--feature_groups", nargs="*", default=None)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tune_threshold", action="store_true")
    ap.add_argument("--threshold_metric", choices=["macro_f1", "positive_f1"], default="macro_f1")
    ap.add_argument("--class_weight", choices=["balanced", "none"], default="balanced")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="Delete the experiment output root before rerunning.")
    args = ap.parse_args()

    repo_root = find_repo_root(args.repo_root or Path(__file__).resolve())
    benchmark = resolve_existing_path(args.benchmark, repo_root) if args.benchmark else discover_benchmark(repo_root)
    default_outroot = repo_root / "experiments" / "text_graph" / f"late_fusion_{args.embedding_family}" / "primary"
    outroot = resolve_output_path(args.outdir, repo_root, default_outroot)
    reset_dir_if_overwrite(outroot, args.overwrite)

    default_text_embeddings = default_text_embeddings_scibert(repo_root) if args.embedding_family == "scibert" else default_text_embeddings_minilm(repo_root)
    text_embeddings = maybe_resolve(args.text_embeddings, repo_root, default_text_embeddings, required=False)
    graph_features = maybe_resolve(args.graph_features, repo_root, default_graph_features(repo_root), required=False)
    node2vec_embeddings = maybe_resolve(args.node2vec_embeddings, repo_root, default_node2vec_embeddings(repo_root), required=False)
    metapath2vec_embeddings = maybe_resolve(args.metapath2vec_embeddings, repo_root, default_metapath2vec_embeddings(repo_root), required=False)
    paper_outputs = repo_root / "paper_outputs"

    write_resolved_paths(
        repo_root=repo_root,
        benchmark=benchmark,
        experiments_outdir=outroot,
        text_embeddings=text_embeddings,
        graph_features=graph_features,
        node2vec_embeddings=node2vec_embeddings,
        metapath2vec_embeddings=metapath2vec_embeddings,
        paper_outputs=paper_outputs,
    )

    script = Path(__file__).resolve().parent / "run_late_fusion_baselines.py"
    for target in args.targets:
        target_name = TARGET_NAME_MAP.get(target, target.replace("target_", ""))
        for split in args.splits:
            split_name = SPLIT_NAME_MAP.get(split, split.replace("split_", ""))
            outdir = outroot / target_name / split_name / "late_fusion"
            cmd = [
                sys.executable, str(script),
                "--repo_root", str(repo_root),
                "--benchmark", str(benchmark),
                "--outdir", str(outdir),
                "--target", target,
                "--split_col", split,
                "--models", *args.models,
                "--seed", str(args.seed),
                "--class_weight", args.class_weight,
            ]
            if text_embeddings:
                cmd += ["--text_embeddings", str(text_embeddings)]
            if graph_features:
                cmd += ["--graph_features", str(graph_features)]
            if node2vec_embeddings:
                cmd += ["--node2vec_embeddings", str(node2vec_embeddings)]
            if metapath2vec_embeddings:
                cmd += ["--metapath2vec_embeddings", str(metapath2vec_embeddings)]
            if args.include_metadata:
                cmd += ["--include_metadata"]
            if args.feature_groups:
                cmd += ["--feature_groups", *args.feature_groups]
            if args.tune_threshold:
                cmd += ["--tune_threshold", "--threshold_metric", args.threshold_metric]
            run(cmd, dry_run=args.dry_run)

    summary_name = f"late_fusion_{args.embedding_family}_metrics_summary.csv"
    table_name = f"late_fusion_{args.embedding_family}_test_metrics.csv"
    write_metrics_rollup(outroot, paper_outputs, summary_name, table_name)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run MethodKG text-only models across the recommended benchmark v2 splits.

This wrapper calls:
  - run_text_baselines.py for regex / TF-IDF / frozen embedding baselines
  - train_transformer_text.py optionally for SciBERT/BERT fine-tuning

Repo-aware defaults in this patched version:
  --input defaults to data/benchmark/**/methodkg_labeled_benchmark_v2_modeling.csv
  --outdir defaults to experiments/text_only/text_only_all
  --embedding_cache_dir defaults to artifacts/features/text_embeddings_minilm_v1
  --paper_outputs_dir defaults to paper_outputs

Example from repo root:
  python src/text_only/run_text_all_splits.py --overwrite

Example with embeddings and transformer fine-tuning:
  python src/text_only/run_text_all_splits.py \
    --overwrite \
    --include_embeddings \
    --include_transformer \
    --transformer_splits split_random_cluster_stratified split_temporal_cluster_safe
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

RECOMMENDED_SPLITS = [
    "split_random_cluster_stratified",
    "split_temporal_cluster_safe",
    "split_cross_program_cluster_safe",
    "split_cold_start_pi_cluster_safe",
    "split_cold_start_institution_cluster_safe",
    "split_edu_to_eng_cluster_safe",
]

RECOMMENDED_TARGETS = [
    "target_integration_binary",
    "target_design_binary",
    "target_mmr_multiclass",
]

DEFAULT_CLASSICAL_MODELS = [
    "regex",
    "tfidf_lr",
    "tfidf_svm",
]


def safe_name(value: str) -> str:
    return value.replace("target_", "").replace("split_", "")


def find_repo_root(start: Path) -> Path:
    """Find the repo root from a script path or current working directory."""
    start = start.resolve()
    candidates = [start] + list(start.parents)
    for p in candidates:
        if (p / ".git").exists():
            return p
        marker_dirs = [p / "data", p / "src", p / "experiments", p / "artifacts", p / "paper_outputs"]
        if sum(x.exists() for x in marker_dirs) >= 2:
            return p
    return Path.cwd().resolve()


def resolve_path(path: Optional[str], repo_root: Path) -> Optional[Path]:
    if not path:
        return None
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return (repo_root / p).resolve()




def is_sidecar_or_hidden_path(path: Path) -> bool:
    """Return True for macOS AppleDouble/resource-fork and hidden metadata files."""
    return any(part.startswith("._") or part == "__MACOSX" for part in path.parts) or path.name.startswith(".")


def find_default_input(repo_root: Path) -> Path:
    """Locate the frozen labeled benchmark after the data folder move."""
    exact_candidates = [
        repo_root / "data" / "benchmark" / "methodkg_labeled_benchmark_v2_modeling.csv",
        repo_root / "data" / "benchmark" / "benchmark_v2" / "methodkg_labeled_benchmark_v2_modeling.csv",
        repo_root / "data" / "benchmark" / "benchmark_v2.zip",
        repo_root / "data" / "benchmark" / "methodkg_labeled_benchmark_v1.csv",
    ]
    for p in exact_candidates:
        if p.exists() and not is_sidecar_or_hidden_path(p):
            return p.resolve()

    search_roots = [
        repo_root / "data" / "benchmark",
        repo_root / "data" / "processed" / "methodkg_outputs_v7_clustered_from_cleaned",
        repo_root / "data" / "processed",
    ]
    patterns = [
        "**/methodkg_labeled_benchmark_v2_modeling.csv",
        "**/*benchmark*v2*modeling*.csv",
        "**/benchmark_v2.zip",
        "**/methodkg_labeled_benchmark_v1.csv",
    ]
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            matches = sorted(p for p in root.glob(pattern) if not is_sidecar_or_hidden_path(p))
            if matches:
                return matches[0].resolve()

    raise FileNotFoundError(
        "Could not find the labeled modeling dataset. Expected one of:\n"
        "  data/benchmark/methodkg_labeled_benchmark_v2_modeling.csv\n"
        "  data/benchmark/benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv\n"
        "  data/benchmark/benchmark_v2.zip\n"
        "Pass --input explicitly if your file has a different name."
    )


def run_command(cmd: list[str], dry_run: bool = False) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def collect_metrics(outdir: Path) -> pd.DataFrame:
    rows = []
    for csv_path in sorted(outdir.glob("**/metrics_summary.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"Warning: failed to read {csv_path}: {exc}")
            continue
        rel_parts = csv_path.relative_to(outdir).parts
        target_name = rel_parts[0] if len(rel_parts) > 0 else "unknown_target"
        split_name = rel_parts[1] if len(rel_parts) > 1 else "unknown_split"
        run_family = rel_parts[2] if len(rel_parts) > 2 else "unknown_family"
        df.insert(0, "run_family", run_family)
        df.insert(0, "split_col", split_name)
        df.insert(0, "target", target_name)
        df.insert(0, "source_metrics_path", str(csv_path))
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


def write_paper_outputs(outdir: Path, paper_outputs_dir: Path) -> None:
    metrics = collect_metrics(outdir)
    if metrics.empty:
        print("No metrics_summary.csv files found; paper_outputs were not updated.")
        return

    summaries_dir = paper_outputs_dir / "summaries"
    tables_dir = paper_outputs_dir / "tables"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary_path = summaries_dir / "text_only_all_metrics_summary.csv"
    table_path = tables_dir / "text_only_all_test_metrics.csv"
    metrics.to_csv(summary_path, index=False)

    test_col = metrics.get("split")
    if test_col is not None:
        test_metrics = metrics[test_col.astype(str).str.lower().eq("test")].copy()
    else:
        test_metrics = metrics.copy()
    sort_cols = [c for c in ["target", "split_col", "run_family", "model"] if c in test_metrics.columns]
    if sort_cols:
        test_metrics = test_metrics.sort_values(sort_cols)
    test_metrics.to_csv(table_path, index=False)

    print("\nUpdated paper outputs:")
    print("  ", summary_path.resolve())
    print("  ", table_path.resolve())


def maybe_clean_dir(path: Path, overwrite: bool, protected_root: Path) -> None:
    if not overwrite or not path.exists():
        return
    path_resolved = path.resolve()
    protected_root = protected_root.resolve()
    if path_resolved == protected_root:
        raise ValueError(f"Refusing to delete protected root directory: {path_resolved}")
    if protected_root not in path_resolved.parents:
        raise ValueError(f"Refusing to delete directory outside protected root {protected_root}: {path_resolved}")
    shutil.rmtree(path_resolved)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", default=None, help="Repo root. Defaults to auto-detection from this script path.")
    parser.add_argument("--input", default=None, help="Benchmark v2 modeling CSV or benchmark_v2.zip. Defaults to data/benchmark discovery.")
    parser.add_argument("--outdir", default=None, help="Output directory for all runs. Defaults to experiments/text_only/text_only_all.")
    parser.add_argument("--paper_outputs_dir", default=None, help="Defaults to paper_outputs under repo root.")
    parser.add_argument("--embedding_cache_dir", default=None, help="Defaults to artifacts/features/text_embeddings_minilm_v1 under repo root.")
    parser.add_argument("--targets", nargs="+", default=RECOMMENDED_TARGETS, help="Target columns to run")
    parser.add_argument("--splits", nargs="+", default=RECOMMENDED_SPLITS, help="Split columns to run")
    parser.add_argument("--classical_models", nargs="+", default=DEFAULT_CLASSICAL_MODELS,
                        help="Models passed to run_text_baselines.py")
    parser.add_argument("--include_embeddings", action="store_true",
                        help="Also run frozen embedding baselines. Requires sentence-transformers.")
    parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Sentence-transformers model for frozen embedding baselines")
    parser.add_argument("--include_transformer", action="store_true",
                        help="Also run train_transformer_text.py. This is slower.")
    parser.add_argument("--transformer_splits", nargs="+", default=None,
                        help="Subset of splits for transformer fine-tuning. Defaults to --splits.")
    parser.add_argument("--transformer_model_name", default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Delete each target/split run folder before rerunning it.")
    parser.add_argument("--no_paper_outputs", action="store_true", help="Do not aggregate metrics into paper_outputs.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running them")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    repo_root = resolve_path(args.repo_root, Path.cwd()) if args.repo_root else find_repo_root(here)
    assert repo_root is not None

    input_path = resolve_path(args.input, repo_root) if args.input else find_default_input(repo_root)
    if input_path is None or not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    outdir = resolve_path(args.outdir, repo_root) if args.outdir else repo_root / "experiments" / "text_only" / "text_only_all"
    paper_outputs_dir = resolve_path(args.paper_outputs_dir, repo_root) if args.paper_outputs_dir else repo_root / "paper_outputs"
    embedding_cache_dir = resolve_path(args.embedding_cache_dir, repo_root) if args.embedding_cache_dir else repo_root / "artifacts" / "features" / "text_embeddings_minilm_v1"
    assert outdir is not None and paper_outputs_dir is not None and embedding_cache_dir is not None

    baseline_script = here / "run_text_baselines.py"
    transformer_script = here / "train_transformer_text.py"

    if not baseline_script.exists():
        raise FileNotFoundError(f"Missing {baseline_script}")
    if args.include_transformer and not transformer_script.exists():
        raise FileNotFoundError(f"Missing {transformer_script}")

    outdir.mkdir(parents=True, exist_ok=True)
    embedding_cache_dir.mkdir(parents=True, exist_ok=True)

    print("Resolved paths:")
    print("  repo_root:", repo_root)
    print("  input:", input_path)
    print("  experiments outdir:", outdir)
    print("  artifact embedding cache:", embedding_cache_dir)
    print("  paper_outputs:", paper_outputs_dir)

    classical_models = list(args.classical_models)
    if args.include_embeddings:
        for model in ["frozen_embedding_lr", "frozen_embedding_mlp"]:
            if model not in classical_models:
                classical_models.append(model)

    for target in args.targets:
        for split in args.splits:
            run_outdir = outdir / safe_name(target) / safe_name(split) / "classical"
            maybe_clean_dir(run_outdir, args.overwrite, outdir)
            cmd = [
                sys.executable,
                str(baseline_script),
                "--repo_root", str(repo_root),
                "--input", str(input_path),
                "--outdir", str(run_outdir),
                "--target", target,
                "--split_col", split,
                "--models", *classical_models,
                "--embedding_cache_dir", str(embedding_cache_dir),
                "--seed", str(args.seed),
            ]
            if args.overwrite:
                cmd.append("--overwrite")
            if args.include_embeddings:
                cmd.extend(["--embedding_model", args.embedding_model, "--cache_embeddings"])
            run_command(cmd, dry_run=args.dry_run)

    if args.include_transformer:
        transformer_splits = args.transformer_splits or args.splits
        for target in args.targets:
            for split in transformer_splits:
                run_outdir = outdir / safe_name(target) / safe_name(split) / "transformer"
                maybe_clean_dir(run_outdir, args.overwrite, outdir)
                cmd = [
                    sys.executable,
                    str(transformer_script),
                    "--repo_root", str(repo_root),
                    "--input", str(input_path),
                    "--outdir", str(run_outdir),
                    "--target", target,
                    "--split_col", split,
                    "--model_name", args.transformer_model_name,
                    "--epochs", str(args.epochs),
                    "--batch_size", str(args.batch_size),
                    "--max_length", str(args.max_length),
                    "--seed", str(args.seed),
                ]
                if args.overwrite:
                    cmd.append("--overwrite")
                run_command(cmd, dry_run=args.dry_run)

    if not args.dry_run and not args.no_paper_outputs:
        write_paper_outputs(outdir, paper_outputs_dir)

    print("\nDone. Results directory:", outdir.resolve())


if __name__ == "__main__":
    main()

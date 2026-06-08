#!/usr/bin/env python3
"""Run MethodKG graph-embedding baselines across targets and splits."""

import argparse
import subprocess
import sys
from pathlib import Path
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from path_utils import discover_benchmark, find_repo_root, reset_dir_if_overwrite, resolve_existing_path, resolve_output_path, write_resolved_paths

DEFAULT_SPLITS = [
    "split_random_cluster_stratified",
    "split_temporal_cluster_safe",
    "split_cross_program_cluster_safe",
    "split_cold_start_pi_cluster_safe",
    "split_cold_start_institution_cluster_safe",
    "split_edu_to_eng_cluster_safe",
]

SHORT_TARGET = {
    "target_integration_binary": "integration_binary",
    "target_design_binary": "design_binary",
    "target_mmr_multiclass": "mmr_multiclass",
    "target_mmr_binary": "mmr_binary",
    "target_qual_binary": "qual_binary",
    "target_quant_binary": "quant_binary",
    "target_method_signal_binary": "method_signal_binary",
}


def short_split(s: str) -> str:
    return s.replace("split_", "")


def emb_name(path: str) -> str:
    p = Path(path).name
    if "node2vec" in p:
        return "node2vec"
    if "metapath" in p:
        return "metapath2vec"
    return Path(path).stem


def run(cmd, dry_run=False):
    print("$", " ".join(cmd), flush=True)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=None, help="Repository root. Defaults to auto-detection.")
    ap.add_argument("--benchmark", default=None, help="Benchmark CSV/zip. Defaults to v3 discovery under data/benchmark/.")
    ap.add_argument("--embeddings", nargs="+", default=None,
                    help="One or more embedding CSVs, e.g. node2vec_award_embeddings.csv metapath2vec_award_embeddings.csv")
    ap.add_argument("--outdir", default=None, help="Output directory. Defaults to experiments/graph_only/node2vec_metapath2vec/walk_embedding_primary_all")
    ap.add_argument("--targets", nargs="+", default=["target_integration_binary", "target_design_binary", "target_mmr_multiclass"])
    ap.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    ap.add_argument("--models", nargs="+", default=["dummy", "emb_lr", "emb_svm", "emb_rf", "emb_extra_trees"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tune_threshold", action="store_true")
    ap.add_argument("--threshold_objective", choices=["macro_f1", "positive_f1"], default="macro_f1")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="Delete the experiment output root before rerunning.")
    args = ap.parse_args()

    repo_root = find_repo_root(args.repo_root or Path(__file__).resolve())
    benchmark_path = resolve_existing_path(args.benchmark, repo_root) if args.benchmark else discover_benchmark(repo_root)
    if args.embeddings:
        embedding_paths = [resolve_existing_path(e, repo_root) for e in args.embeddings]
    else:
        candidate_dir = repo_root / "artifacts" / "features" / "walk_embeddings_v1"
        embedding_paths = [p for p in [candidate_dir / "node2vec_award_embeddings.csv", candidate_dir / "metapath2vec_award_embeddings.csv"] if p.exists()]
        if not embedding_paths:
            raise FileNotFoundError(f"No walk embedding CSVs found in {candidate_dir}. Run build_walk_graph_embeddings.py first or pass --embeddings.")
    outbase = resolve_output_path(args.outdir, repo_root, repo_root / "experiments" / "graph_only" / "node2vec_metapath2vec" / "walk_embedding_primary_all")
    reset_dir_if_overwrite(outbase, args.overwrite)
    paper_outputs = repo_root / "paper_outputs"
    write_resolved_paths(repo_root=repo_root, benchmark=benchmark_path, embeddings=", ".join(str(p) for p in embedding_paths), experiments_outdir=outbase, paper_outputs=paper_outputs)

    here = Path(__file__).resolve().parent
    runner = here / "run_embedding_baselines.py"

    for emb in embedding_paths:
        e_short = emb_name(emb)
        for target in args.targets:
            t_short = SHORT_TARGET.get(target, target.replace("target_", ""))
            for split in args.splits:
                s_short = short_split(split)
                outdir = outbase / e_short / t_short / s_short
                cmd = [
                    sys.executable, str(runner),
                    "--benchmark", str(benchmark_path),
                    "--embeddings", str(emb),
                    "--outdir", str(outdir),
                    "--target", target,
                    "--split_col", split,
                    "--models", *args.models,
                    "--seed", str(args.seed),
                    "--threshold_objective", args.threshold_objective,
                ]
                if args.tune_threshold:
                    cmd.append("--tune_threshold")
                run(cmd, dry_run=args.dry_run)

    write_metrics_rollup(outbase, paper_outputs, "walk_embedding_metrics_summary.csv", "walk_embedding_test_metrics.csv")


if __name__ == "__main__":
    main()

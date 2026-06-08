#!/usr/bin/env python3
"""Collect metrics.csv files from graph embedding baseline runs."""

import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = []
    for p in Path(args.results_dir).rglob("metrics.csv"):
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print("Skipping", p, e)
            continue
        rel = p.relative_to(args.results_dir)
        df.insert(0, "metrics_path", str(rel))
        rows.append(df)
    if not rows:
        raise SystemExit(f"No metrics.csv files found under {args.results_dir}")
    out = pd.concat(rows, ignore_index=True)

    # Nice ordering if columns exist.
    first = [
        "metrics_path", "embedding_file", "target", "split_col", "model", "task_type",
        "accuracy", "macro_f1", "weighted_f1", "positive_f1", "positive_precision", "positive_recall",
        "roc_auc", "pr_auc", "threshold", "train_rows", "validation_rows", "test_rows",
        "embedding_missing_train", "embedding_missing_test",
    ]
    cols = [c for c in first if c in out.columns] + [c for c in out.columns if c not in first]
    out = out[cols]
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output}")


if __name__ == "__main__":
    main()

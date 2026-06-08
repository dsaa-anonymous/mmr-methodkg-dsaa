#!/usr/bin/env python3
"""Summarize MethodKG metadata ablation metrics recursively."""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = Path(args.results_dir)
    rows = []
    for metrics_path in root.rglob("metrics.csv"):
        rel_parts = metrics_path.relative_to(root).parts
        # Prefer the combined metrics.csv at target/split/mode/metrics.csv.
        # Skip nested per-model metrics at target/split/mode/model/metrics.csv.
        if len(rel_parts) > 4:
            continue
        try:
            df = pd.read_csv(metrics_path)
        except Exception as e:
            print(f"Skipping unreadable metrics file {metrics_path}: {e}")
            continue
        # expected: target/split/mode/metrics.csv
        df["metrics_path"] = str(metrics_path)
        if len(rel_parts) >= 4:
            df["target_slug"] = rel_parts[0]
            df["split_slug"] = rel_parts[1]
            df["ablation_mode"] = rel_parts[2]
        rows.append(df)

    if rows:
        out = pd.concat(rows, ignore_index=True)
        # Remove duplicate rows from per-model metrics if combined metrics rows are present.
        subset = [c for c in ["target", "split_col", "model", "metrics_path"] if c in out.columns]
        if "metrics_path" in subset:
            subset.remove("metrics_path")
        if subset:
            out = out.drop_duplicates(subset=subset, keep="first")
    else:
        out = pd.DataFrame()

    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with {len(out)} rows")


if __name__ == "__main__":
    main()

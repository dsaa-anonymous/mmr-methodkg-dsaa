#!/usr/bin/env python3
"""Summarize metrics.csv files from MethodKG SimTeG-GraphSAGE runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output", default="simteg_structural_summary.csv")
    args = parser.parse_args()

    root = Path(args.results_dir)
    rows = []
    for path in root.rglob("metrics.csv"):
        try:
            df = pd.read_csv(path)
            if len(df):
                row = df.iloc[0].to_dict()
                row["metrics_path"] = str(path)
                rows.append(row)
        except Exception as exc:
            rows.append({"metrics_path": str(path), "error": repr(exc)})
    out = pd.DataFrame(rows)
    if len(out):
        sort_cols = [c for c in ["target", "split_col", "model"] if c in out.columns]
        if sort_cols:
            out = out.sort_values(sort_cols)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Collect metrics.csv files from graph-only runs into one summary CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output", default="graph_only_all_results_summary.csv")
    args = parser.parse_args()

    root = Path(args.results_dir)
    frames = []
    for path in root.rglob("metrics.csv"):
        try:
            df = pd.read_csv(path)
            df.insert(0, "metrics_path", str(path.relative_to(root)))
            frames.append(df)
        except Exception as e:
            print("[WARN] Could not read", path, e)
    if not frames:
        raise SystemExit(f"No metrics.csv files found under {root}")
    out = pd.concat(frames, ignore_index=True)
    sort_cols = [c for c in ["target", "split_col", "macro_f1", "positive_f1", "weighted_f1"] if c in out.columns]
    if {"target", "split_col", "macro_f1"}.issubset(out.columns):
        out = out.sort_values(["target", "split_col", "macro_f1"], ascending=[True, True, False])
    out_path = root / args.output
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("[DONE] Wrote", out_path)
    print("Rows:", len(out))


if __name__ == "__main__":
    main()

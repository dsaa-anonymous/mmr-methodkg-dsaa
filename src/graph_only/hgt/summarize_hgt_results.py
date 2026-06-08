#!/usr/bin/env python3
"""Summarize MethodKG HGT result metrics recursively."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--output", default="hgt_structural_summary.csv")
    args = ap.parse_args()

    root = Path(args.results_dir)
    rows = []
    for path in root.rglob("metrics.csv"):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        rel = path.relative_to(root)
        parts = rel.parts
        if len(parts) >= 4:
            df.insert(0, "target_folder", parts[0])
            df.insert(1, "split_folder", parts[1])
            df.insert(2, "model_folder", parts[2])
        df.insert(0, "metrics_path", str(path))
        rows.append(df)
    if rows:
        out = pd.concat(rows, ignore_index=True)
    else:
        out = pd.DataFrame()
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Wrote {args.output} with {len(out)} rows")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize MethodKG late-fusion metrics files recursively."""
import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    rows = []
    root = Path(args.results_dir)
    for p in root.rglob("metrics.csv"):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        for _, r in df.iterrows():
            row = r.to_dict()
            row["metrics_path"] = str(p)
            parts = p.relative_to(root).parts
            if len(parts) >= 4:
                row.setdefault("target_dir", parts[0])
                row.setdefault("split_dir", parts[1])
                row.setdefault("run_dir", parts[2])
                row.setdefault("model_dir", parts[3])
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with {len(out)} rows")


if __name__ == "__main__":
    main()

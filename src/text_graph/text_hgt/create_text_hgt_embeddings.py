#!/usr/bin/env python3
"""
Create MethodKG award text embeddings for SimTeG-style GraphSAGE.

This script encodes the UNION of the full cleaned award corpus and the labeled
benchmark so every award node used by the graph can receive a text feature.
It supports either sentence-transformers or vanilla Hugging Face transformers
(mean pooling). It can also encode with a local fine-tuned checkpoint if you
pass --model_name /path/to/checkpoint.

Recommended safe/default use for MethodKG:
  pretrained SciBERT mean-pooled embeddings for all 32K awards.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from path_utils import default_text_hgt_embeddings, discover_awards, discover_benchmark, find_repo_root, read_csv_or_zip, reset_dir_if_overwrite, resolve_existing_path, resolve_output_path, write_resolved_paths
from tqdm import tqdm


def clean_award_id_value(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.startswith('="') and s.endswith('"'):
        s = s[2:-1].strip()
    elif s.startswith("='") and s.endswith("'"):
        s = s[2:-1].strip()
    s = s.strip().strip('"').strip("'").strip()
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".", 1)[0]
    elif re.fullmatch(r"\d+(?:\.\d+)?[eE][+-]?\d+", s):
        try:
            val = float(s)
            if math.isfinite(val) and abs(val - round(val)) < 1e-6:
                s = str(int(round(val)))
        except Exception:
            pass
    return re.sub(r"\D", "", s).strip()


def award_id_match_key(award_id: str) -> str:
    aid = clean_award_id_value(award_id)
    if not aid:
        return ""
    return aid.lstrip("0") or "0"


def read_csv(path: str | Path) -> pd.DataFrame:
    return read_csv_or_zip(path, dtype=str, encoding="utf-8-sig")


def ensure_award_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "award_id" in out.columns:
        out["award_id"] = out["award_id"].apply(clean_award_id_value)
    elif "AwardNumber" in out.columns:
        out["award_id"] = out["AwardNumber"].apply(clean_award_id_value)
    else:
        raise ValueError("Input needs award_id or AwardNumber column")
    out["award_id_key"] = out["award_id"].apply(award_id_match_key)
    return out


def make_text(row: pd.Series, text_cols: List[str], sep: str = " [SEP] ") -> str:
    parts = []
    for c in text_cols:
        if c in row.index:
            v = row[c]
            if pd.notna(v) and str(v).strip():
                parts.append(str(v).strip())
    return sep.join(parts).strip()


def build_award_text_table(awards_path: str, benchmark_path: str | None, text_cols: List[str]) -> pd.DataFrame:
    awards = ensure_award_id(read_csv(awards_path))
    tables = [awards]
    if benchmark_path:
        bench = ensure_award_id(read_csv(benchmark_path))
        tables.append(bench)
    all_df = pd.concat(tables, ignore_index=True, sort=False)
    all_df = all_df[all_df["award_id_key"].fillna("") != ""].copy()

    # Prefer rows with nonempty abstract/title and benchmark text when duplicates exist.
    for c in text_cols:
        if c not in all_df.columns:
            all_df[c] = ""
    all_df["_text_len"] = all_df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.len()
    all_df = all_df.sort_values(["award_id_key", "_text_len"], ascending=[True, False])
    all_df = all_df.drop_duplicates(subset=["award_id_key"], keep="first").copy()
    all_df["text"] = all_df.apply(lambda r: make_text(r, text_cols), axis=1)
    return all_df[["award_id", "award_id_key", "text"]].reset_index(drop=True)


def encode_sentence_transformers(texts: List[str], model_name: str, batch_size: int, device: str, normalize: bool) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=None if device == "auto" else device)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return emb.astype(np.float32)


def mean_pool(last_hidden_state, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def encode_transformers(texts: List[str], model_name: str, batch_size: int, device: str, max_length: int, normalize: bool) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    outs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
            if normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            outs.append(pooled.detach().cpu().numpy().astype(np.float32))
    return np.vstack(outs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", default=None, help="Repository root. Defaults to auto-detection.")
    parser.add_argument("--awards", default=None, help="cleaned_nsf_awards_2000_2025.csv. Defaults to data/processed/.../cleaned_nsf_awards_2000_2025.csv")
    parser.add_argument("--benchmark", default=None, help="Benchmark CSV/zip. Defaults to data/benchmark v3.")
    parser.add_argument("--outdir", default=None, help="Output directory. Defaults to artifacts/features/text_hgt_embeddings_scibert_v1")
    parser.add_argument("--model_name", default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--backend", choices=["transformers", "sentence_transformers"], default="transformers")
    parser.add_argument("--text_cols", nargs="+", default=["title_clean", "abstract_clean"], help="Text columns to concatenate")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Delete the output directory before writing new embeddings.")
    args = parser.parse_args()

    repo_root = find_repo_root(args.repo_root or Path(__file__).resolve())
    args.awards = str(resolve_existing_path(args.awards, repo_root) if args.awards else discover_awards(repo_root))
    args.benchmark = str(resolve_existing_path(args.benchmark, repo_root) if args.benchmark else discover_benchmark(repo_root))
    outdir = resolve_output_path(args.outdir, repo_root, default_text_hgt_embeddings(repo_root).parent)
    reset_dir_if_overwrite(outdir, args.overwrite)
    write_resolved_paths(repo_root=repo_root, awards=args.awards, benchmark=args.benchmark, outdir=outdir)
    table = build_award_text_table(args.awards, args.benchmark or None, args.text_cols)
    table.to_csv(outdir / "simteg_award_text_table.csv", index=False, encoding="utf-8-sig")

    texts = table["text"].fillna("").astype(str).tolist()
    if args.backend == "sentence_transformers":
        emb = encode_sentence_transformers(texts, args.model_name, args.batch_size, args.device, args.normalize)
    else:
        emb = encode_transformers(texts, args.model_name, args.batch_size, args.device, args.max_length, args.normalize)

    emb_cols = [f"text_emb_{i:04d}" for i in range(emb.shape[1])]
    out = pd.DataFrame(emb, columns=emb_cols)
    out.insert(0, "award_id_key", table["award_id_key"].values)
    out.insert(0, "award_id", table["award_id"].values)
    out.to_csv(outdir / "methodkg_simteg_text_embeddings.csv", index=False, encoding="utf-8-sig")

    summary = {
        "num_awards_encoded": int(len(out)),
        "embedding_dim": int(emb.shape[1]),
        "model_name": args.model_name,
        "backend": args.backend,
        "text_cols": args.text_cols,
        "normalize": bool(args.normalize),
        "max_length": int(args.max_length),
        "input_awards": args.awards,
        "input_benchmark": args.benchmark,
    }
    with open(outdir / "simteg_text_embedding_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Wrote", outdir / "methodkg_simteg_text_embeddings.csv")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create frozen text embeddings for MethodKG awards.

Default uses sentence-transformers/all-MiniLM-L6-v2 for speed. You can also use
scientific/stronger embedding models if installed and available.
"""
import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from path_utils import (
    default_text_embeddings_minilm,
    default_text_embeddings_scibert,
    discover_benchmark,
    find_repo_root,
    read_csv_or_zip,
    reset_dir_if_overwrite,
    resolve_existing_path,
    resolve_output_path,
    write_resolved_paths,
)


def normalize_award_id(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0") and re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    return re.sub(r"\D", "", s)


def make_text(df: pd.DataFrame) -> list:
    title = df.get("title_clean", pd.Series([""] * len(df))).fillna("").astype(str)
    abstract = df.get("abstract_clean", pd.Series([""] * len(df))).fillna("").astype(str)
    return (title + " [SEP] " + abstract).str.strip().tolist()


def encode_sentence_transformers(texts, model_name, batch_size, device, normalize):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device if device != "auto" else None)
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )


def encode_transformers(texts, model_name, batch_size, device, max_length, normalize):
    import torch
    from transformers import AutoModel, AutoTokenizer
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            # Mean pooling over attention mask.
            last = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            if normalize:
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            outs.append(emb.cpu().numpy())
    return np.vstack(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=None, help="Repository root. Defaults to auto-detection.")
    ap.add_argument("--input", default=None, help="CSV/zip with award_id, title_clean, abstract_clean. Defaults to data/benchmark v3.")
    ap.add_argument("--outdir", default=None, help="Output directory. Defaults to artifacts/features/text_embeddings_<family>_v1.")
    ap.add_argument("--embedding_family", choices=["minilm", "scibert"], default="minilm", help="Convenience preset for output directory/model/backend defaults.")
    ap.add_argument("--model_name", default=None, help="Embedding model. Defaults from --embedding_family.")
    ap.add_argument("--backend", choices=["sentence_transformers", "transformers"], default=None, help="Encoding backend. Defaults from --embedding_family.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--prefix", default="text_emb")
    ap.add_argument("--overwrite", action="store_true", help="Delete the output directory before writing new embeddings.")
    args = ap.parse_args()

    repo_root = find_repo_root(args.repo_root or Path(__file__).resolve())
    input_path = resolve_existing_path(args.input, repo_root) if args.input else discover_benchmark(repo_root)
    if args.embedding_family == "scibert":
        default_embedding_path = default_text_embeddings_scibert(repo_root)
        model_name = args.model_name or "allenai/scibert_scivocab_uncased"
        backend = args.backend or "transformers"
    else:
        default_embedding_path = default_text_embeddings_minilm(repo_root)
        model_name = args.model_name or "sentence-transformers/all-MiniLM-L6-v2"
        backend = args.backend or "sentence_transformers"
    outdir = resolve_output_path(args.outdir, repo_root, default_embedding_path.parent)
    reset_dir_if_overwrite(outdir, args.overwrite)
    write_resolved_paths(repo_root=repo_root, input=input_path, outdir=outdir)
    df = read_csv_or_zip(input_path, encoding="utf-8-sig")
    if "award_id" not in df.columns:
        raise ValueError("Input must contain award_id")
    df = df.drop_duplicates(subset=["award_id"], keep="first").copy()
    df["award_id"] = df["award_id"].apply(normalize_award_id)
    texts = make_text(df)

    if backend == "sentence_transformers":
        emb = encode_sentence_transformers(texts, model_name, args.batch_size, args.device, args.normalize)
    else:
        emb = encode_transformers(texts, model_name, args.batch_size, args.device, args.max_length, args.normalize)

    cols = [f"{args.prefix}_{i:04d}" for i in range(emb.shape[1])]
    out = pd.DataFrame(emb, columns=cols)
    out.insert(0, "award_id", df["award_id"].values)
    out.to_csv(outdir / "methodkg_text_embeddings.csv", index=False, encoding="utf-8-sig")
    with open(outdir / "text_embedding_summary.json", "w") as f:
        json.dump({
            "input": args.input,
            "embedding_family": args.embedding_family,
            "model_name": model_name,
            "backend": backend,
            "rows": int(len(out)),
            "embedding_dim": int(emb.shape[1]),
            "normalize": bool(args.normalize),
        }, f, indent=2)
    print(f"Wrote {outdir / 'methodkg_text_embeddings.csv'} with shape {out.shape}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Fine-tune a transformer text classifier for MethodKG-Labeled.

Recommended default model:
  allenai/scibert_scivocab_uncased

Recommended input:
  methodkg_labeled_benchmark_v2_modeling.csv

Examples:
  python train_transformer_text.py \
    --input benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv \
    --outdir scibert_integration_random \
    --target target_integration_binary \
    --split_col split_random_cluster_stratified \
    --model_name allenai/scibert_scivocab_uncased \
    --epochs 5 \
    --batch_size 8 \
    --max_length 512

  python train_transformer_text.py \
    --input benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv \
    --outdir scibert_mmr_temporal \
    --target target_mmr_multiclass \
    --split_col split_temporal_cluster_safe \
    --task_type multiclass \
    --epochs 5 \
    --batch_size 8
"""

from __future__ import annotations

import argparse
import json
import os
import re
import inspect
import zipfile
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )
except Exception as e:  # pragma: no cover
    raise ImportError(
        "This script requires transformers and torch. Install with: "
        "pip install transformers torch accelerate scikit-learn pandas numpy"
    ) from e

DEFAULT_SPLIT_COL = "split_random_cluster_stratified"
DEFAULT_CLUSTER_COL = "project_cluster_id"
RANDOM_SEED = 42


def safe_name(value: str) -> str:
    return value.replace("target_", "").replace("split_", "")


def find_repo_root(start: Path) -> Path:
    """Find the repo root from a script path or current working directory."""
    start = start.resolve()
    candidates = [start] + list(start.parents)
    for root in candidates:
        if (root / ".git").exists():
            return root
        marker_dirs = [root / "data", root / "src", root / "experiments", root / "artifacts", root / "paper_outputs"]
        if sum(x.exists() for x in marker_dirs) >= 2:
            return root
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
    for candidate in exact_candidates:
        if candidate.exists() and not is_sidecar_or_hidden_path(candidate):
            return candidate.resolve()

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


def default_outdir(repo_root: Path, target: str, split_col: str) -> Path:
    return repo_root / "experiments" / "text_only" / "scibert_finetuned" / safe_name(target) / safe_name(split_col) / "transformer"


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def read_input_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p, "r") as zf:
            candidates = [
                name for name in zf.namelist()
                if (
                    name.endswith("methodkg_labeled_benchmark_v2_modeling.csv")
                    or name.endswith("methodkg_labeled_benchmark_v1.csv")
                    or name.endswith(".csv")
                )
                and not Path(name).name.startswith("._")
                and not Path(name).name.startswith(".")
                and "__MACOSX" not in Path(name).parts
            ]
            if not candidates:
                raise ValueError("No CSV file found inside zip input.")
            candidates = sorted(
                candidates,
                key=lambda x: (
                    0 if x.endswith("methodkg_labeled_benchmark_v2_modeling.csv") else 1,
                    len(x),
                ),
            )
            with zf.open(candidates[0]) as f:
                return pd.read_csv(f, low_memory=False)
    return pd.read_csv(p, low_memory=False, encoding="utf-8-sig")


def build_text(df: pd.DataFrame, mode: str) -> pd.Series:
    title = df.get("title_clean", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
    abstract = df.get("abstract_clean", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
    if mode == "title":
        return title
    if mode == "abstract":
        return abstract
    if mode == "title_abstract":
        return (title + " [SEP] " + abstract).str.strip()
    raise ValueError(f"Unsupported text_mode: {mode}")


def infer_task_type(y: pd.Series, requested: str) -> str:
    if requested != "auto":
        return requested
    if y.dtype.kind in "biufc":
        vals = sorted(pd.Series(y.dropna().unique()).astype(float).tolist())
        if set(vals).issubset({0.0, 1.0}) and len(vals) <= 2:
            return "binary"
    unique = sorted(y.dropna().astype(str).unique().tolist())
    if len(unique) <= 2:
        return "binary"
    return "multiclass"


def prepare_y(y_raw: pd.Series, task_type: str, drop_unclear: bool) -> Tuple[pd.Series, Optional[LabelEncoder], Dict[str, Any]]:
    y = y_raw.copy()
    if y.dtype.kind not in "biufc":
        y = y.fillna("").astype(str).str.strip().str.lower().str.replace(r"[\s-]+", "_", regex=True)
        y = y.replace({"": np.nan, "nan": np.nan, "none": np.nan})
        if drop_unclear:
            y = y.replace({"unclear": np.nan})

    if task_type == "binary":
        if y.dtype.kind in "biufc":
            y_num = pd.to_numeric(y, errors="coerce")
        else:
            y_num = y.map({"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0})
            if y_num.isna().all():
                labels = sorted(y.dropna().astype(str).unique().tolist())
                if len(labels) != 2:
                    raise ValueError(f"Cannot coerce target to binary. Values: {labels}")
                mapping = {labels[0]: 0, labels[1]: 1}
                y_num = y.map(mapping)
        return y_num.astype("float"), None, {"label_mapping": {"0": 0, "1": 1}}

    le = LabelEncoder()
    nonnull = y.dropna().astype(str)
    le.fit(nonnull)
    y_encoded = pd.Series(np.nan, index=y.index, dtype="float")
    y_encoded.loc[nonnull.index] = le.transform(nonnull).astype(float)
    return y_encoded, le, {"classes": le.classes_.tolist()}


def add_internal_validation_if_needed(
    df: pd.DataFrame,
    split_col: str,
    y_col_internal: str,
    cluster_col: str = DEFAULT_CLUSTER_COL,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    out = df.copy()
    split = out[split_col].fillna("unknown").astype(str).str.lower()
    has_val = split.isin(["validation", "val", "dev"]).any()
    if has_val:
        return out

    train_mask = split.eq("train")
    if train_mask.sum() < 10:
        return out

    if cluster_col in out.columns:
        # Build an aligned cluster key; fill missing or blank clusters with row ids.
        split_key = out[cluster_col].astype("object").copy()
        missing_cluster = split_key.isna() | split_key.astype(str).str.strip().eq("")
        if missing_cluster.any():
            fallback = pd.Series(out.index.astype(str), index=out.index)
            split_key.loc[missing_cluster] = fallback.loc[missing_cluster]
        split_key = split_key.astype(str)

        cluster_df = pd.DataFrame({
            cluster_col: split_key.loc[train_mask],
            y_col_internal: out.loc[train_mask, y_col_internal],
        })
        cluster_labels = cluster_df.groupby(cluster_col)[y_col_internal].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0])
        clusters = cluster_labels.index.to_numpy()
        labels = cluster_labels.to_numpy()
        stratify = labels if len(pd.Series(labels).dropna().unique()) > 1 and pd.Series(labels).value_counts().min() >= 2 else None
        _, val_clusters = train_test_split(clusters, test_size=0.15, random_state=seed, stratify=stratify)
        out.loc[train_mask & split_key.isin(val_clusters), split_col] = "validation"
        return out

    train_idx = out.index[train_mask].to_numpy()
    y_train = out.loc[train_idx, y_col_internal]
    stratify = y_train if y_train.nunique(dropna=True) > 1 and y_train.value_counts().min() >= 2 else None
    _, val_idx = train_test_split(train_idx, test_size=0.15, random_state=seed, stratify=stratify)
    out.loc[val_idx, split_col] = "validation"
    return out


def prepare_dataframe(args: argparse.Namespace) -> Tuple[pd.DataFrame, Optional[LabelEncoder], Dict[str, Any], str]:
    df = read_input_table(args.input)
    if args.target not in df.columns:
        raise ValueError(f"Target not found: {args.target}")
    if args.split_col not in df.columns:
        raise ValueError(f"Split column not found: {args.split_col}")

    task_type = infer_task_type(df[args.target], args.task_type)
    y, label_encoder, label_info = prepare_y(df[args.target], task_type, args.drop_unclear)
    df = df.copy()
    df["__y__"] = y
    df["__text__"] = build_text(df, args.text_mode)
    df = df[df["__y__"].notna()].copy()
    df = df[df["__text__"].fillna("").astype(str).str.len() > 0].copy()
    df = add_internal_validation_if_needed(df, args.split_col, "__y__", seed=args.seed)
    df["__split__"] = df[args.split_col].fillna("unknown").astype(str).str.lower().replace({"val": "validation", "dev": "validation"})

    if not df["__split__"].eq("train").any() or not df["__split__"].eq("test").any():
        raise ValueError(f"Split must contain train and test rows. Counts: {df['__split__'].value_counts().to_dict()}")
    if not df["__split__"].eq("validation").any():
        raise ValueError("Transformer training requires validation rows. Use a split with validation or allow internal validation.")

    return df, label_encoder, label_info, task_type


class MethodKGTextDataset(torch.utils.data.Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer: Any, max_length: int):
        self.texts = texts
        self.labels = labels.astype(int)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
        )
        enc["labels"] = int(self.labels[idx])
        return enc


class WeightedTrainer(Trainer):
    def __init__(self, *args: Any, class_weights: Optional[torch.Tensor] = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model: Any, inputs: Dict[str, Any], return_outputs: bool = False, **kwargs: Any):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_class_weights(y_train: np.ndarray, num_labels: int) -> torch.Tensor:
    counts = np.bincount(y_train.astype(int), minlength=num_labels).astype(float)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_labels * counts)
    return torch.tensor(weights, dtype=torch.float)


def build_compute_metrics(task_type: str):
    def compute_metrics(eval_pred: Any) -> Dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        if task_type == "binary":
            probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
            out = {
                "accuracy": accuracy_score(labels, preds),
                "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
                "positive_f1": f1_score(labels, preds, pos_label=1, zero_division=0),
                "precision": precision_score(labels, preds, pos_label=1, zero_division=0),
                "recall": recall_score(labels, preds, pos_label=1, zero_division=0),
            }
            if len(np.unique(labels)) == 2:
                try:
                    out["roc_auc"] = roc_auc_score(labels, probs)
                except Exception:
                    pass
                try:
                    out["pr_auc"] = average_precision_score(labels, probs)
                except Exception:
                    pass
            return {k: float(v) for k, v in out.items()}
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        }
    return compute_metrics


def predict_and_save(
    trainer: Trainer,
    dataset: MethodKGTextDataset,
    rows: pd.DataFrame,
    split_name: str,
    outdir: Path,
    task_type: str,
    label_encoder: Optional[LabelEncoder],
) -> Dict[str, Any]:
    pred_output = trainer.predict(dataset)
    logits = pred_output.predictions
    y_true = pred_output.label_ids
    y_pred = np.argmax(logits, axis=-1)
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()

    if task_type == "binary":
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "positive_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        }
        if len(np.unique(y_true)) == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))
            metrics["pr_auc"] = float(average_precision_score(y_true, probs[:, 1]))
    else:
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }

    with open(outdir / f"{split_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    labels_for_report = None
    target_names = None
    if label_encoder is not None:
        labels_for_report = list(range(len(label_encoder.classes_)))
        target_names = label_encoder.classes_.tolist()

    report = classification_report(
        y_true,
        y_pred,
        labels=labels_for_report,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )
    with open(outdir / f"{split_name}_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels_for_report)).to_csv(outdir / f"{split_name}_confusion_matrix.csv", index=False)

    pred_df = rows[[c for c in ["benchmark_id", "award_id", "project_cluster_id", "title_clean", "start_year"] if c in rows.columns]].copy()
    pred_df["y_true"] = y_true
    pred_df["y_pred"] = y_pred
    if label_encoder is not None:
        pred_df["y_true_label"] = label_encoder.inverse_transform(y_true)
        pred_df["y_pred_label"] = label_encoder.inverse_transform(y_pred)
    if task_type == "binary":
        pred_df["score_positive"] = probs[:, 1]
    pred_df.to_csv(outdir / f"{split_name}_predictions.csv", index=False, encoding="utf-8-sig")

    return {"split": split_name, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a transformer text classifier for MethodKG.")
    parser.add_argument("--repo_root", default=None, help="Repo root. Defaults to auto-detection from this script path.")
    parser.add_argument("--input", default=None, help="Path to methodkg_labeled_benchmark_v2_modeling.csv or benchmark_v2.zip. Defaults to data/benchmark discovery.")
    parser.add_argument("--outdir", default=None, help="Output directory. Defaults to experiments/text_only/scibert_finetuned/<target>/<split>/transformer.")
    parser.add_argument("--target", default="target_integration_binary")
    parser.add_argument("--split_col", default=DEFAULT_SPLIT_COL)
    parser.add_argument("--task_type", default="auto", choices=["auto", "binary", "multiclass"])
    parser.add_argument("--text_mode", default="title_abstract", choices=["title", "abstract", "title_abstract"])
    parser.add_argument("--drop_unclear", action="store_true", default=True)
    parser.add_argument("--keep_unclear", dest="drop_unclear", action="store_false")
    parser.add_argument("--model_name", default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--no_class_weights", action="store_true", help="Disable inverse-frequency class weights")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 if supported by your GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true", help="Delete --outdir before writing new outputs")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    repo_root = resolve_path(args.repo_root, Path.cwd()) if args.repo_root else find_repo_root(here)
    assert repo_root is not None

    input_path = resolve_path(args.input, repo_root) if args.input else find_default_input(repo_root)
    if input_path is None or not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    args.input = str(input_path)

    outdir = resolve_path(args.outdir, repo_root) if args.outdir else default_outdir(repo_root, args.target, args.split_col)
    assert outdir is not None
    if args.overwrite:
        clean_dir(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Resolved paths:")
    print("  repo_root:", repo_root)
    print("  input:", args.input)
    print("  outdir:", outdir)

    set_seed(args.seed)

    df, label_encoder, label_info, task_type = prepare_dataframe(args)
    train_df = df[df["__split__"] == "train"].copy()
    val_df = df[df["__split__"] == "validation"].copy()
    test_df = df[df["__split__"] == "test"].copy()

    num_labels = 2 if task_type == "binary" else len(label_encoder.classes_) if label_encoder is not None else int(df["__y__"].nunique())

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)

    train_ds = MethodKGTextDataset(train_df["__text__"].astype(str).tolist(), train_df["__y__"].astype(int).to_numpy(), tokenizer, args.max_length)
    val_ds = MethodKGTextDataset(val_df["__text__"].astype(str).tolist(), val_df["__y__"].astype(int).to_numpy(), tokenizer, args.max_length)
    test_ds = MethodKGTextDataset(test_df["__text__"].astype(str).tolist(), test_df["__y__"].astype(int).to_numpy(), tokenizer, args.max_length)

    metric_for_best = "eval_macro_f1"

    # Hugging Face Transformers changed the TrainingArguments keyword from
    # evaluation_strategy to eval_strategy in newer releases. Build the kwargs
    # dynamically so the script works across both older and newer versions.
    ta_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    training_kwargs = {
        "output_dir": str(outdir / "hf_checkpoints"),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "save_strategy": "epoch",
        "save_safetensors": False,
        "logging_strategy": "steps",
        "logging_steps": 25,
        "load_best_model_at_end": True,
        "metric_for_best_model": metric_for_best,
        "greater_is_better": True,
        "report_to": "none",
        "fp16": args.fp16,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "seed": args.seed,
    }
    if "eval_strategy" in ta_params:
        training_kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in ta_params:
        training_kwargs["evaluation_strategy"] = "epoch"
    else:
        print("Warning: this transformers version exposes neither eval_strategy nor evaluation_strategy; evaluation scheduling will use defaults.")

    # Some older releases may not support a subset of these optional kwargs.
    training_kwargs = {k: v for k, v in training_kwargs.items() if k in ta_params}
    training_args = TrainingArguments(**training_kwargs)

    class_weights = None
    if not args.no_class_weights:
        class_weights = compute_class_weights(train_df["__y__"].astype(int).to_numpy(), num_labels)

    # Hugging Face Transformers also changed Trainer.__init__: recent
    # releases replaced tokenizer= with processing_class=, and some wrappers
    # accept neither. Build kwargs dynamically for local version compatibility.
    trainer_params = set(inspect.signature(Trainer.__init__).parameters.keys())
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": build_compute_metrics(task_type),
        "class_weights": class_weights,
    }
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = WeightedTrainer(**trainer_kwargs)

    config = vars(args).copy()
    config.update({
        "task_type_resolved": task_type,
        "label_info": label_info,
        "num_labels": num_labels,
        "n_train": len(train_df),
        "n_validation": len(val_df),
        "n_test": len(test_df),
    })
    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    trainer.train()
    trainer.save_model(str(outdir / "best_model"))
    tokenizer.save_pretrained(str(outdir / "best_model"))

    rows = []
    rows.append(predict_and_save(trainer, train_ds, train_df, "train", outdir, task_type, label_encoder))
    rows.append(predict_and_save(trainer, val_ds, val_df, "validation", outdir, task_type, label_encoder))
    rows.append(predict_and_save(trainer, test_ds, test_df, "test", outdir, task_type, label_encoder))
    pd.DataFrame(rows).to_csv(outdir / "metrics_summary.csv", index=False)

    print("Wrote outputs to:", outdir.resolve())
    print(pd.DataFrame(rows))


if __name__ == "__main__":
    main()

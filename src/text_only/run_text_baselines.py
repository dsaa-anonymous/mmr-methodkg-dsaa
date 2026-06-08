#!/usr/bin/env python3
"""
Run text-only baselines for MethodKG-Labeled.

Recommended input:
  methodkg_labeled_benchmark_v2_modeling.csv

Supported models:
  regex
  tfidf_lr
  tfidf_svm
  frozen_embedding_lr
  frozen_embedding_mlp

Examples:
  python run_text_baselines.py \
    --input benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv \
    --outdir results_text_integration_random \
    --target target_integration_binary \
    --split_col split_random_cluster_stratified \
    --models regex tfidf_lr tfidf_svm

  python run_text_baselines.py \
    --input benchmark_v2/methodkg_labeled_benchmark_v2_modeling.csv \
    --outdir results_text_mmr_temporal \
    --target target_mmr_multiclass \
    --split_col split_temporal_cluster_safe \
    --models regex tfidf_lr tfidf_svm frozen_embedding_lr \
    --embedding_model sentence-transformers/all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import zipfile
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

DEFAULT_TEXT_COLS = ["title_clean", "abstract_clean"]
DEFAULT_SPLIT_COL = "split_random_cluster_stratified"
DEFAULT_CLUSTER_COL = "project_cluster_id"
RANDOM_SEED = 42

CANDIDATE_OR_AUDIT_PATTERNS = [
    r"^candidate_",
    r"_candidate$",
    r"^annotation_",
    r"^annotator",
    r"review_priority",
    r"annotation_guidance",
]


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
    return repo_root / "experiments" / "text_only" / "text_only_all" / safe_name(target) / safe_name(split_col) / "classical"


def default_embedding_cache_dir(repo_root: Path, embedding_model: str) -> Path:
    model_lower = embedding_model.lower()
    if "scibert" in model_lower:
        leaf = "text_embeddings_scibert_mean_v1"
    elif "minilm" in model_lower or "all-minilm" in model_lower:
        leaf = "text_embeddings_minilm_v1"
    else:
        leaf = "text_embeddings_frozen_v1"
    return repo_root / "artifacts" / "features" / leaf


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def read_input_table(path: str) -> pd.DataFrame:
    """Read CSV directly or find the modeling CSV inside a zip archive."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if is_sidecar_or_hidden_path(p):
        sibling = p.with_name(p.name[2:]) if p.name.startswith("._") else None
        if sibling is not None and sibling.exists() and not is_sidecar_or_hidden_path(sibling):
            print(f"Warning: ignoring sidecar file {p.name}; using {sibling.name} instead.", flush=True)
            p = sibling
        else:
            raise ValueError(
                f"Input path points to a hidden/sidecar metadata file, not a dataset CSV: {p}. "
                "Pass the real methodkg_labeled_benchmark_v2_modeling.csv file."
            )

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
            # Prefer the v2 modeling file if present.
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


def is_audit_or_candidate_col(col: str) -> bool:
    return any(re.search(pattern, col) for pattern in CANDIDATE_OR_AUDIT_PATTERNS)


def infer_task_type(y: pd.Series, target: str, requested: str) -> str:
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
    # Normalize text labels, but preserve numeric binary targets.
    if y.dtype.kind not in "biufc":
        y = y.fillna("").astype(str).str.strip().str.lower().str.replace(r"[\s-]+", "_", regex=True)
        y = y.replace({"": np.nan, "nan": np.nan, "none": np.nan})
        if drop_unclear:
            y = y.replace({"unclear": np.nan})

    if task_type == "binary":
        if y.dtype.kind in "biufc":
            y_num = pd.to_numeric(y, errors="coerce")
        else:
            # Accept common yes/no labels in case the user points to label_* columns.
            y_num = y.map({"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0})
            if y_num.isna().all():
                # If there are exactly two string classes, map sorted labels to 0/1.
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
    """Create a validation split from train rows when the split has no validation rows."""
    out = df.copy()
    split = out[split_col].fillna("unknown").astype(str).str.lower()
    has_val = split.isin(["validation", "val", "dev"]).any()
    if has_val:
        return out

    train_mask = split.eq("train")
    if train_mask.sum() < 10:
        return out

    # Cluster-level internal validation to avoid putting duplicate text clusters in both train and validation.
    if cluster_col in out.columns:
        # Pandas fillna cannot take an Index as the replacement value. Build an
        # aligned temporary split key instead, using row ids only when cluster ids
        # are missing or blank. This path is triggered for splits such as EDU->ENG
        # that have train/test but no explicit validation set.
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
        # Use the most common label per cluster for stratification when feasible.
        cluster_labels = cluster_df.groupby(cluster_col)[y_col_internal].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0])
        clusters = cluster_labels.index.to_numpy()
        labels = cluster_labels.to_numpy()
        stratify = labels if len(pd.Series(labels).dropna().unique()) > 1 and pd.Series(labels).value_counts().min() >= 2 else None
        train_clusters, val_clusters = train_test_split(
            clusters,
            test_size=0.15,
            random_state=seed,
            stratify=stratify,
        )
        out.loc[train_mask & split_key.isin(val_clusters), split_col] = "validation"
        return out

    train_idx = out.index[train_mask].to_numpy()
    y_train = out.loc[train_idx, y_col_internal]
    stratify = y_train if y_train.nunique(dropna=True) > 1 and y_train.value_counts().min() >= 2 else None
    _, val_idx = train_test_split(train_idx, test_size=0.15, random_state=seed, stratify=stratify)
    out.loc[val_idx, split_col] = "validation"
    return out


def split_data(
    df: pd.DataFrame,
    target: str,
    split_col: str,
    task_type: str,
    text_mode: str,
    drop_unclear: bool,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    if target not in df.columns:
        raise ValueError(f"Target column not found: {target}")
    if split_col not in df.columns:
        raise ValueError(f"Split column not found: {split_col}")
    if any(is_audit_or_candidate_col(c) for c in [target, split_col]):
        raise ValueError("Target/split column appears to be an audit/candidate column. Use target_* and split_* columns.")

    y_tmp, label_encoder, label_info = prepare_y(df[target], task_type, drop_unclear=drop_unclear)
    df2 = df.copy()
    df2["__y__"] = y_tmp
    df2["__text__"] = build_text(df2, text_mode)
    df2 = df2[df2["__y__"].notna()].copy()
    df2 = df2[df2["__text__"].fillna("").astype(str).str.len() > 0].copy()

    df2 = add_internal_validation_if_needed(df2, split_col, "__y__", seed=seed)

    split = df2[split_col].fillna("unknown").astype(str).str.lower().replace({"val": "validation", "dev": "validation"})
    train = df2[split == "train"].copy()
    val = df2[split == "validation"].copy()
    test = df2[split == "test"].copy()

    if len(train) == 0 or len(test) == 0:
        raise ValueError(f"Split {split_col} must contain train and test rows after filtering. Counts: {split.value_counts().to_dict()}")

    return {
        "df": df2,
        "train": train,
        "val": val,
        "test": test,
        "label_encoder": label_encoder,
        "label_info": label_info,
        "task_type": task_type,
        "target": target,
        "split_col": split_col,
    }


def get_text_y(part: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    return part["__text__"].astype(str).tolist(), part["__y__"].astype(int).to_numpy()


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: Optional[np.ndarray]) -> Dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "positive_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            out["roc_auc"] = float("nan")
        try:
            out["pr_auc"] = float(average_precision_score(y_true, y_score))
        except Exception:
            out["pr_auc"] = float("nan")
    return out


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def get_scores(model: Any, X: Any, task_type: str) -> Optional[np.ndarray]:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if task_type == "binary":
            if proba.ndim == 2 and proba.shape[1] > 1:
                return proba[:, 1]
            return proba.ravel()
        return proba
    if hasattr(model, "decision_function"):
        score = model.decision_function(X)
        if task_type == "binary" and getattr(score, "ndim", 1) > 1:
            return score[:, 1]
        return score
    return None


def save_evaluation(
    outdir: Path,
    model_name: str,
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray],
    task_type: str,
    label_encoder: Optional[LabelEncoder],
    rows: pd.DataFrame,
) -> Dict[str, Any]:
    if task_type == "binary":
        metrics = binary_metrics(y_true, y_pred, y_score if y_score is not None and np.ndim(y_score) == 1 else None)
    else:
        metrics = multiclass_metrics(y_true, y_pred)

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
    cm = confusion_matrix(y_true, y_pred, labels=labels_for_report)

    prefix = f"{model_name}_{split_name}"
    with open(outdir / f"{prefix}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(outdir / f"{prefix}_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    pd.DataFrame(cm).to_csv(outdir / f"{prefix}_confusion_matrix.csv", index=False)

    pred_df = rows[[c for c in ["benchmark_id", "award_id", "project_cluster_id", "title_clean", "start_year"] if c in rows.columns]].copy()
    pred_df["y_true"] = y_true
    pred_df["y_pred"] = y_pred
    if label_encoder is not None:
        pred_df["y_true_label"] = label_encoder.inverse_transform(y_true)
        pred_df["y_pred_label"] = label_encoder.inverse_transform(y_pred)
    if y_score is not None and task_type == "binary" and np.ndim(y_score) == 1:
        pred_df["score_positive"] = y_score
    pred_df.to_csv(outdir / f"{prefix}_predictions.csv", index=False, encoding="utf-8-sig")

    metrics_prefixed = {"model": model_name, "split": split_name, **metrics}
    return metrics_prefixed


def regex_predict(texts: Iterable[str], target: str, task_type: str, label_encoder: Optional[LabelEncoder]) -> np.ndarray:
    """Rule baseline. Intended as a weak baseline, not as final labels."""
    patterns = {
        "explicit_mmr": re.compile(r"\b(mixed[-\s]?methods?|mixed[-\s]?methodolog(?:y|ies)|mixed\s+research|multi[-\s]?methods?|multimethods?|mmr)\b", re.I),
        "qual": re.compile(r"\b(qualitative|interviews?|focus\s+groups?|case\s+stud(?:y|ies)|ethnograph\w*|observations?|field\s+notes?|open[-\s]?ended|thematic\s+analysis|coding|grounded\s+theory|narrative\s+analysis)\b", re.I),
        "quant": re.compile(r"\b(quantitative|surveys?|questionnaires?|statistical|statistics|regression|anova|ancova|t[-\s]?test|chi[-\s]?square|experiment\w*|randomi[sz]ed|quasi[-\s]?experimental|pre[-\s]?post|assessment|scale|factor\s+analysis|structural\s+equation|sem|bayesian|data\s+mining|learning\s+analytics|modeling|modelling)\b", re.I),
        "design": re.compile(r"\b(convergent\s+parallel|convergent\s+design|explanatory\s+sequential|sequential\s+explanatory|exploratory\s+sequential|sequential\s+exploratory|embedded\s+design|transformative\s+design|multiphase\s+design|triangulation\s+design)\b", re.I),
        "integration": re.compile(r"\b(joint\s+displays?|meta[-\s]?inferences?|integrat\w*.{0,100}(qualitative|quantitative|mixed[-\s]?methods?|findings|results|analysis)|triangulat\w*.{0,100}(qualitative|quantitative|findings|results)|merg\w*.{0,100}(qualitative|quantitative|findings|results))\b", re.I),
        "method": re.compile(r"\b(methodolog\w*|methods?|data\s+collection|data\s+analysis|evaluation|assess\w*|measure\w*|sample|participants?|instruments?|protocol)\b", re.I),
    }

    if task_type == "binary":
        out = []
        for t in texts:
            t = t or ""
            if "integration" in target:
                pred = bool(patterns["integration"].search(t))
            elif "design" in target:
                pred = bool(patterns["design"].search(t))
            elif "qual" in target:
                pred = bool(patterns["qual"].search(t))
            elif "quant" in target:
                pred = bool(patterns["quant"].search(t))
            elif "explicit" in target:
                pred = bool(patterns["explicit_mmr"].search(t))
            elif "implicit" in target:
                pred = bool(patterns["qual"].search(t) and patterns["quant"].search(t) and not patterns["explicit_mmr"].search(t))
            elif "method_signal" in target:
                pred = bool(patterns["method"].search(t) or patterns["qual"].search(t) or patterns["quant"].search(t) or patterns["explicit_mmr"].search(t))
            else:
                pred = bool(patterns["explicit_mmr"].search(t) or (patterns["qual"].search(t) and patterns["quant"].search(t)))
            out.append(int(pred))
        return np.array(out, dtype=int)

    # Rule baseline for target_mmr_multiclass.
    class_preds = []
    for t in texts:
        t = t or ""
        explicit = bool(patterns["explicit_mmr"].search(t))
        qual = bool(patterns["qual"].search(t))
        quant = bool(patterns["quant"].search(t))
        method = bool(patterns["method"].search(t))
        if explicit:
            lab = "explicit_mmr"
        elif qual and quant:
            lab = "implicit_mmr"
        elif qual:
            lab = "qual_only"
        elif quant:
            lab = "quant_only"
        elif method:
            lab = "multi_method_not_mmr"
        else:
            lab = "no_method_signal"
        class_preds.append(lab)

    if label_encoder is None:
        raise ValueError("Multiclass regex baseline requires a label encoder.")
    # Map labels not present in training to the first known class to avoid crashing.
    known = set(label_encoder.classes_.tolist())
    safe = [lab if lab in known else label_encoder.classes_[0] for lab in class_preds]
    return label_encoder.transform(safe)


def run_regex(data: Dict[str, Any], outdir: Path) -> List[Dict[str, Any]]:
    results = []
    for split_name in ["train", "validation", "test"]:
        part = data["train"] if split_name == "train" else data["val"] if split_name == "validation" else data["test"]
        if len(part) == 0:
            continue
        X, y = get_text_y(part)
        pred = regex_predict(X, data["target"], data["task_type"], data["label_encoder"])
        score = pred.astype(float) if data["task_type"] == "binary" else None
        results.append(save_evaluation(outdir, "regex", split_name, y, pred, score, data["task_type"], data["label_encoder"], part))
    return results


def make_tfidf_pipeline(model_name: str, task_type: str, max_features: int, min_df: int, ngram_max: int, calibrate_svm: bool, seed: int = RANDOM_SEED) -> Pipeline:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, ngram_max),
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
        norm="l2",
    )

    if model_name == "tfidf_lr":
        clf = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=-1,
        )
    elif model_name == "tfidf_svm":
        base = LinearSVC(class_weight="balanced", random_state=seed, max_iter=5000)
        if calibrate_svm:
            clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        else:
            clf = base
    else:
        raise ValueError(model_name)

    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def run_tfidf_model(data: Dict[str, Any], outdir: Path, model_name: str, args: argparse.Namespace) -> List[Dict[str, Any]]:
    X_train, y_train = get_text_y(data["train"])
    model = make_tfidf_pipeline(model_name, data["task_type"], args.max_features, args.min_df, args.ngram_max, args.calibrate_svm, seed=args.seed)
    model.fit(X_train, y_train)
    joblib.dump(model, outdir / f"{model_name}_model.joblib")

    results = []
    for split_name in ["train", "validation", "test"]:
        part = data["train"] if split_name == "train" else data["val"] if split_name == "validation" else data["test"]
        if len(part) == 0:
            continue
        X, y = get_text_y(part)
        pred = model.predict(X)
        score = get_scores(model, X, data["task_type"])
        results.append(save_evaluation(outdir, model_name, split_name, y, pred, score, data["task_type"], data["label_encoder"], part))
    return results


def maybe_prefix_e5(texts: List[str], model_name: str) -> List[str]:
    if "e5" in model_name.lower():
        return ["passage: " + t for t in texts]
    return texts


def encode_with_sentence_transformer(texts: List[str], model_name: str, batch_size: int, cache_path: Optional[Path] = None) -> np.ndarray:
    if cache_path is not None and cache_path.exists():
        return np.load(cache_path)
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise ImportError(
            "sentence-transformers is required for frozen embedding baselines. "
            "Install with: pip install sentence-transformers"
        ) from e

    model = SentenceTransformer(model_name)
    emb = model.encode(
        maybe_prefix_e5(texts, model_name),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if cache_path is not None:
        np.save(cache_path, emb)
    return emb


def run_frozen_embedding_model(data: Dict[str, Any], outdir: Path, model_name: str, args: argparse.Namespace) -> List[Dict[str, Any]]:
    emb_dir = Path(args.embedding_cache_dir) if args.embedding_cache_dir else outdir / "embedding_cache"
    emb_dir.mkdir(parents=True, exist_ok=True)
    safe_embedding_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.embedding_model)
    cache_stub = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        f"{safe_embedding_name}_{data['target']}_{data['split_col']}_{args.text_mode}",
    )

    X_train_text, y_train = get_text_y(data["train"])
    X_train = encode_with_sentence_transformer(
        X_train_text,
        args.embedding_model,
        args.embedding_batch_size,
        emb_dir / f"{cache_stub}_train.npy" if args.cache_embeddings else None,
    )

    if model_name == "frozen_embedding_lr":
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs", n_jobs=-1)),
        ])
    elif model_name == "frozen_embedding_mlp":
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(256, 64), alpha=1e-4, learning_rate_init=1e-3, max_iter=500, early_stopping=True, random_state=args.seed)),
        ])
    else:
        raise ValueError(model_name)

    clf.fit(X_train, y_train)
    joblib.dump(clf, outdir / f"{model_name}_classifier.joblib")

    results = []
    for split_name in ["train", "validation", "test"]:
        part = data["train"] if split_name == "train" else data["val"] if split_name == "validation" else data["test"]
        if len(part) == 0:
            continue
        X_text, y = get_text_y(part)
        X_emb = encode_with_sentence_transformer(
            X_text,
            args.embedding_model,
            args.embedding_batch_size,
            emb_dir / f"{cache_stub}_{split_name}.npy" if args.cache_embeddings else None,
        )
        pred = clf.predict(X_emb)
        score = get_scores(clf, X_emb, data["task_type"])
        results.append(save_evaluation(outdir, model_name, split_name, y, pred, score, data["task_type"], data["label_encoder"], part))
    return results


def write_run_config(outdir: Path, args: argparse.Namespace, data: Dict[str, Any]) -> None:
    config = vars(args).copy()
    config.update({
        "task_type_resolved": data["task_type"],
        "label_info": data["label_info"],
        "n_train": len(data["train"]),
        "n_validation": len(data["val"]),
        "n_test": len(data["test"]),
    })
    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MethodKG text-only baselines.")
    parser.add_argument("--repo_root", default=None, help="Repo root. Defaults to auto-detection from this script path.")
    parser.add_argument("--input", default=None, help="Path to methodkg_labeled_benchmark_v2_modeling.csv or benchmark_v2.zip. Defaults to data/benchmark discovery.")
    parser.add_argument("--outdir", default=None, help="Directory for model outputs. Defaults to experiments/text_only/text_only_all/<target>/<split>/classical.")
    parser.add_argument("--target", default="target_integration_binary", help="Target column")
    parser.add_argument("--split_col", default=DEFAULT_SPLIT_COL, help="Split column")
    parser.add_argument("--text_mode", default="title_abstract", choices=["title", "abstract", "title_abstract"])
    parser.add_argument("--task_type", default="auto", choices=["auto", "binary", "multiclass"])
    parser.add_argument("--drop_unclear", action="store_true", default=True, help="Drop unclear labels for string targets; default true")
    parser.add_argument("--keep_unclear", dest="drop_unclear", action="store_false", help="Keep unclear as a class for multiclass targets")
    parser.add_argument("--models", nargs="+", default=["regex", "tfidf_lr", "tfidf_svm"], choices=["regex", "tfidf_lr", "tfidf_svm", "frozen_embedding_lr", "frozen_embedding_mlp", "all"])

    parser.add_argument("--max_features", type=int, default=100000)
    parser.add_argument("--min_df", type=int, default=2)
    parser.add_argument("--ngram_max", type=int, default=2)
    parser.add_argument("--calibrate_svm", action="store_true", default=True)
    parser.add_argument("--no_calibrate_svm", dest="calibrate_svm", action="store_false")

    parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding_batch_size", type=int, default=32)
    parser.add_argument("--cache_embeddings", action="store_true", default=True)
    parser.add_argument("--no_cache_embeddings", dest="cache_embeddings", action="store_false")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for validation split and stochastic models")
    parser.add_argument("--embedding_cache_dir", default=None, help="Optional artifacts/features directory for frozen embedding caches")
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

    if args.embedding_cache_dir is None:
        args.embedding_cache_dir = str(default_embedding_cache_dir(repo_root, args.embedding_model))
    else:
        args.embedding_cache_dir = str(resolve_path(args.embedding_cache_dir, repo_root))
    Path(args.embedding_cache_dir).mkdir(parents=True, exist_ok=True)

    print("Resolved paths:")
    print("  repo_root:", repo_root)
    print("  input:", args.input)
    print("  outdir:", outdir)
    print("  embedding_cache_dir:", args.embedding_cache_dir)

    df = read_input_table(args.input)
    requested_task_type = infer_task_type(df[args.target], args.target, args.task_type)
    data = split_data(
        df=df,
        target=args.target,
        split_col=args.split_col,
        task_type=requested_task_type,
        text_mode=args.text_mode,
        drop_unclear=args.drop_unclear,
        seed=args.seed,
    )
    write_run_config(outdir, args, data)

    models = args.models
    if "all" in models:
        models = ["regex", "tfidf_lr", "tfidf_svm", "frozen_embedding_lr", "frozen_embedding_mlp"]

    all_results: List[Dict[str, Any]] = []
    for m in models:
        print(f"\n=== Running {m} ===", flush=True)
        if m == "regex":
            all_results.extend(run_regex(data, outdir))
        elif m in {"tfidf_lr", "tfidf_svm"}:
            all_results.extend(run_tfidf_model(data, outdir, m, args))
        elif m in {"frozen_embedding_lr", "frozen_embedding_mlp"}:
            all_results.extend(run_frozen_embedding_model(data, outdir, m, args))
        else:
            raise ValueError(m)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(outdir / "metrics_summary.csv", index=False)
    print("\nWrote outputs to:", outdir.resolve())
    print(results_df)


if __name__ == "__main__":
    main()

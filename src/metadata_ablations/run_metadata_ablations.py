#!/usr/bin/env python3
"""
Run MethodKG metadata-only and text+metadata ablation baselines.

This script is intentionally graph-free: it uses structured metadata from the
benchmark file, optionally concatenated with precomputed text embeddings.
It does NOT use graph features, node2vec/metapath2vec embeddings, candidate flags,
annotation fields, labels, targets as inputs, or award text itself.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from path_utils import discover_benchmark, find_repo_root, read_csv_or_zip, reset_dir_if_overwrite, resolve_existing_path, resolve_output_path, write_resolved_paths
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
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
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.svm import LinearSVC


DEFAULT_METADATA_COLUMNS = [
    # temporal / award metadata
    "start_year",
    "year",
    "award_year",
    "AwardInstrument",
    "award_instrument",
    "AwardInstrumentName",
    # NSF hierarchy / program metadata
    "NSFDirectorate",
    "directorate",
    "NSFOrganization",
    "division",
    "Program(s)",
    "Program",
    "program",
    "ProgramElementCode(s)",
    "ProgramElementCode",
    "primary_program_key",
    # institution / geography metadata
    "organization_clean",
    "institution_clean",
    "Organization",
    "Institution",
    "State",
    "OrganizationState",
    "state",
    # simple non-graph team metadata if already available in benchmark
    "team_size",
    "num_pis",
    "num_copis",
]

LEAKY_PREFIXES = (
    "label_",
    "target_",
    "candidate_",
    "annotation_",
)

LEAKY_EXACT = {
    "benchmark_id",  # identifier, not a predictive feature
    "award_id",     # join key only
    "AwardID",
    "project_cluster_id",
    "review_priority",
    "annotation_guidance",
    "explicit_mmr_candidate",
    "implicit_mmr_candidate",
    "qual_signal_candidate",
    "quant_signal_candidate",
    "integration_candidate",
    "design_label_candidate",
    "candidate_stratum",
    "award_amount",  # potentially post-award amendment leakage
    "AwardedAmountToDate",
    "last_amendment_date",
    "abstract_clean",
    "title_clean",
    "AbstractNarration",
    "Title",
}

SPLIT_VALUES = {"train", "val", "valid", "validation", "dev", "test"}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def normalize_award_id(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    # preserve pure strings where possible but remove whitespace and obvious .0
    return s


def load_table(path: str) -> pd.DataFrame:
    return read_csv_or_zip(path, encoding="utf-8-sig")


def _stringify_label_value(v):
    """Normalize common CSV label representations.

    Pandas sometimes reads binary target columns as strings like "0.0" or
    "1.0" after zip/copy/export steps.  Direct astype(int) fails on those
    strings, so we normalize them explicitly before label-type inference.
    """
    if pd.isna(v):
        return np.nan
    s = str(v).strip()
    if s == "":
        return np.nan
    sl = s.lower()
    if sl in {"true", "false"}:
        return "1" if sl == "true" else "0"
    # Handle "0.0", "1.0", numpy floats, and accidental float formatting.
    try:
        f = float(s)
        if np.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
    except Exception:
        pass
    return s


def infer_label_type(y: pd.Series) -> str:
    vals = sorted(pd.Series(y).dropna().map(_stringify_label_value).dropna().astype(str).unique().tolist())
    if set(vals).issubset({"0", "1"}) or len(vals) == 2:
        return "binary"
    return "multiclass"


def clean_target(y: pd.Series) -> pd.Series:
    # Keep multiclass strings. Convert binary-looking labels to int.
    cleaned = pd.Series(y).map(_stringify_label_value)
    vals = cleaned.dropna().astype(str).unique().tolist()
    if set(vals).issubset({"0", "1"}):
        return cleaned.astype(int)
    return cleaned.astype(str)


def labels_to_int_array(y: pd.Series) -> np.ndarray:
    """Return binary labels as int, robust to strings like '0.0'."""
    return pd.Series(y).map(_stringify_label_value).astype(int).values


def load_text_embeddings(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    emb = pd.read_csv(path)
    if "award_id" not in emb.columns:
        raise ValueError("Text embeddings file must contain award_id")
    emb = emb.copy()
    emb["award_id"] = emb["award_id"].map(normalize_award_id)
    return emb


def find_embedding_cols(df: pd.DataFrame) -> List[str]:
    emb_cols = [c for c in df.columns if re.match(r"^(emb|text_emb|x)_\d+$", str(c))]
    if emb_cols:
        return emb_cols
    # fallback: all numeric non-id columns except obvious metadata
    bad = {"award_id", "benchmark_id"}
    nums = [c for c in df.columns if c not in bad and pd.api.types.is_numeric_dtype(df[c])]
    return nums


def select_metadata_columns(df: pd.DataFrame, user_cols: Optional[List[str]]) -> Tuple[List[str], List[str]]:
    if user_cols:
        candidates = user_cols
    else:
        candidates = DEFAULT_METADATA_COLUMNS

    cols = []
    missing = []
    for c in candidates:
        if c in df.columns:
            if c in LEAKY_EXACT or any(c.startswith(p) for p in LEAKY_PREFIXES) or c.startswith("split_"):
                continue
            cols.append(c)
        else:
            missing.append(c)

    # De-duplicate while preserving order
    seen = set()
    cols2 = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            cols2.append(c)
    return cols2, missing


def build_feature_frame(
    benchmark: pd.DataFrame,
    text_embeddings: Optional[pd.DataFrame],
    metadata_cols: List[str],
) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    df = benchmark.copy()
    if "award_id" not in df.columns:
        raise ValueError("Benchmark file must contain award_id")
    df["award_id"] = df["award_id"].map(normalize_award_id)

    feature_cols: List[str] = []
    text_cols: List[str] = []

    if text_embeddings is not None:
        emb = text_embeddings.copy()
        text_cols = find_embedding_cols(emb)
        if not text_cols:
            raise ValueError("No embedding columns found in text embeddings file")
        keep = ["award_id"] + text_cols
        df = df.merge(emb[keep], on="award_id", how="left", validate="one_to_one")
        missing_emb = int(df[text_cols].isna().all(axis=1).sum())
        if missing_emb > 0:
            print(f"WARNING: {missing_emb} benchmark rows are missing text embeddings; filling with 0.")
            df[text_cols] = df[text_cols].fillna(0.0)
        feature_cols.extend(text_cols)

    # use metadata cols that survived merge/selection
    meta_cols = [c for c in metadata_cols if c in df.columns]
    feature_cols.extend(meta_cols)

    if not feature_cols:
        raise ValueError("No input features selected. Provide metadata columns or text embeddings.")

    return df, feature_cols, text_cols, meta_cols


def make_preprocessor(df: pd.DataFrame, feature_cols: List[str]) -> ColumnTransformer:
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=2)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore")

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", ohe),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipe, categorical_cols))

    return ColumnTransformer(transformers=transformers, sparse_threshold=0.3)


def canonical_split_values(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().replace({"validation": "val", "valid": "val", "dev": "val"})


def add_internal_validation_if_needed(df: pd.DataFrame, split_col: str, y_col: str, seed: int) -> pd.DataFrame:
    out = df.copy()
    split = canonical_split_values(out[split_col])
    if "val" in set(split):
        out[split_col] = split
        return out

    train_mask = split.eq("train")
    if train_mask.sum() < 10:
        out[split_col] = split
        return out

    cluster_col = "project_cluster_id" if "project_cluster_id" in out.columns else None
    y_train = out.loc[train_mask, y_col]

    if cluster_col:
        groups = out.loc[train_mask, cluster_col].fillna(out.loc[train_mask, "award_id"]).astype(str)
        # group-level validation split
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        train_idx_local = np.arange(train_mask.sum())
        try:
            _, val_local = next(gss.split(train_idx_local, y_train, groups=groups))
            train_indices = out.index[train_mask]
            val_indices = train_indices[val_local]
        except Exception:
            _, val_indices = train_test_split(out.index[train_mask], test_size=0.15, random_state=seed, stratify=None)
    else:
        stratify = y_train if y_train.value_counts().min() >= 2 else None
        _, val_indices = train_test_split(out.index[train_mask], test_size=0.15, random_state=seed, stratify=stratify)

    split2 = split.copy()
    split2.loc[val_indices] = "val"
    out[split_col] = split2
    return out


def get_model(name: str, label_type: str, seed: int, class_weight: str) -> Any:
    cw = class_weight if class_weight != "none" else None
    if name == "dummy":
        return DummyClassifier(strategy="most_frequent")
    if name in {"metadata_lr", "text_metadata_lr", "meta_lr", "tm_lr"}:
        return LogisticRegression(max_iter=3000, class_weight=cw, solver="saga", n_jobs=-1)
    if name in {"metadata_svm", "text_metadata_svm", "meta_svm", "tm_svm"}:
        return LinearSVC(class_weight=cw, random_state=seed, max_iter=5000)
    if name in {"metadata_rf", "text_metadata_rf", "meta_rf", "tm_rf"}:
        return RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1, class_weight=cw)
    if name in {"metadata_extra_trees", "text_metadata_extra_trees", "meta_extra_trees", "tm_extra_trees"}:
        return ExtraTreesClassifier(n_estimators=500, random_state=seed, n_jobs=-1, class_weight=cw)
    if name in {"metadata_mlp", "text_metadata_mlp", "meta_mlp", "tm_mlp"}:
        return MLPClassifier(hidden_layer_sizes=(256, 64), alpha=1e-4, learning_rate_init=1e-3,
                             max_iter=400, early_stopping=True, random_state=seed)
    raise ValueError(f"Unknown model: {name}")


def decision_scores(model: Any, X: Any, classes: np.ndarray) -> Optional[np.ndarray]:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        if s.ndim == 1:
            return s
        return s
    return None


def fitted_score_classes(pipe: Pipeline, scores: Optional[np.ndarray], fallback_classes: np.ndarray) -> Optional[np.ndarray]:
    """Return the class labels corresponding to score columns when available.

    For multiclass problems with rare labels, a class may be absent from the
    training fold. In that case sklearn estimators expose only the fitted
    classes_ and return score matrices with fewer columns than the global
    label set.
    """
    if scores is None:
        return None
    arr = np.asarray(scores)
    if arr.ndim == 1:
        return None
    try:
        model = pipe.named_steps.get("model")
        cls = getattr(model, "classes_", None)
        if cls is not None and len(cls) == arr.shape[1]:
            return np.asarray(cls)
    except Exception:
        pass
    if len(fallback_classes) == arr.shape[1]:
        return fallback_classes
    return np.array([f"col{j}" for j in range(arr.shape[1])])


def tune_binary_threshold(y_val: np.ndarray, scores_val: np.ndarray, metric: str) -> float:
    if scores_val.ndim > 1:
        scores = scores_val[:, 1]
    else:
        scores = scores_val
    qs = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 99)))
    best_t, best_m = 0.5, -1.0
    for t in qs:
        pred = (scores >= t).astype(int)
        if metric == "positive_f1":
            m = f1_score(y_val, pred, pos_label=1, zero_division=0)
        else:
            m = f1_score(y_val, pred, average="macro", zero_division=0)
        if m > best_m:
            best_m, best_t = m, float(t)
    return best_t


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: Optional[np.ndarray]) -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    d["accuracy"] = accuracy_score(y_true, y_pred)
    d["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    d["weighted_f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    d["positive_f1"] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    d["precision"] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    d["recall"] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    if scores is not None:
        s = scores[:, 1] if getattr(scores, "ndim", 1) > 1 else scores
        try:
            d["roc_auc"] = roc_auc_score(y_true, s)
        except Exception:
            d["roc_auc"] = np.nan
        try:
            d["pr_auc"] = average_precision_score(y_true, s)
        except Exception:
            d["pr_auc"] = np.nan
    return d


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: Optional[np.ndarray], classes: np.ndarray) -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    d["accuracy"] = accuracy_score(y_true, y_pred)
    d["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    d["weighted_f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    # AUC metrics are only meaningful when all classes are present in y_true and scores are probabilities.
    d["roc_auc_macro_ovr"] = np.nan
    d["pr_auc_macro"] = np.nan
    return d


def save_outputs(
    outdir: Path,
    model_name: str,
    df: pd.DataFrame,
    idx_test: pd.Index,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: Optional[np.ndarray],
    classes: np.ndarray,
    metrics: Dict[str, Any],
    label_type: str,
    threshold: Optional[float],
    score_classes: Optional[np.ndarray] = None,
) -> None:
    od = ensure_dir(outdir / model_name)
    metrics2 = dict(metrics)
    metrics2["model"] = model_name
    metrics2["label_type"] = label_type
    metrics2["threshold"] = threshold if threshold is not None else np.nan
    pd.DataFrame([metrics2]).to_csv(od / "metrics.csv", index=False)

    pred_df = df.loc[idx_test, [c for c in ["award_id", "benchmark_id", "project_cluster_id"] if c in df.columns]].copy()
    pred_df["y_true"] = y_true
    pred_df["y_pred"] = y_pred
    if scores is not None:
        scores_arr = np.asarray(scores)
        if getattr(scores_arr, "ndim", 1) == 1:
            pred_df["score"] = scores_arr
        else:
            # In multiclass splits with very rare labels, some classes may be absent
            # from the training fold. scikit-learn then returns score columns only
            # for the fitted estimator's classes_ (e.g., 6 columns for a 7-class
            # global label set). Write score columns aligned to the fitted classes
            # instead of assuming every global class has a score column.
            if score_classes is not None and len(score_classes) == scores_arr.shape[1]:
                cols = list(score_classes)
            elif len(classes) == scores_arr.shape[1]:
                cols = list(classes)
            else:
                cols = [f"col{j}" for j in range(scores_arr.shape[1])]
            for j, cls in enumerate(cols):
                pred_df[f"score_{cls}"] = scores_arr[:, j]
    pred_df.to_csv(od / "predictions.csv", index=False)

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    pd.DataFrame(cm, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes]).to_csv(od / "confusion_matrix.csv")

    rep = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
    with open(od / "classification_report.json", "w") as f:
        json.dump(rep, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=None, help="Repository root. Defaults to auto-detection.")
    ap.add_argument("--benchmark", default=None, help="Benchmark CSV/zip. Defaults to data/benchmark v3.")
    ap.add_argument("--outdir", default=None, help="Output directory. Defaults to experiments/text_graph/metadata_only/single_run")
    ap.add_argument("--target", required=True)
    ap.add_argument("--split_col", required=True)
    ap.add_argument("--text_embeddings", default=None, help="Optional text embedding CSV. If omitted, metadata-only.")
    ap.add_argument("--metadata_cols", nargs="*", default=None, help="Optional explicit metadata columns to use.")
    ap.add_argument("--models", nargs="+", default=["dummy", "metadata_lr", "metadata_svm", "metadata_mlp", "metadata_extra_trees"])
    ap.add_argument("--class_weight", default="balanced", choices=["balanced", "none"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tune_threshold", action="store_true")
    ap.add_argument("--threshold_metric", default="macro_f1", choices=["macro_f1", "positive_f1"])
    ap.add_argument("--overwrite", action="store_true", help="Delete the output directory before rerunning.")
    args = ap.parse_args()

    repo_root = find_repo_root(args.repo_root or Path(__file__).resolve())
    args.benchmark = str(resolve_existing_path(args.benchmark, repo_root) if args.benchmark else discover_benchmark(repo_root))
    args.outdir = str(resolve_output_path(args.outdir, repo_root, repo_root / "experiments" / "text_graph" / "metadata_only" / "single_run"))
    if args.text_embeddings:
        args.text_embeddings = str(resolve_existing_path(args.text_embeddings, repo_root))
    reset_dir_if_overwrite(Path(args.outdir), args.overwrite)
    write_resolved_paths(repo_root=repo_root, benchmark=args.benchmark, outdir=args.outdir, text_embeddings=args.text_embeddings)

    outdir = ensure_dir(args.outdir)
    df0 = load_table(args.benchmark)
    if args.target not in df0.columns:
        raise ValueError(f"Target {args.target} not found")
    if args.split_col not in df0.columns:
        raise ValueError(f"Split column {args.split_col} not found")

    df0["award_id"] = df0["award_id"].map(normalize_award_id)
    y_col = "__y__"
    df0[y_col] = clean_target(df0[args.target])
    label_type = infer_label_type(df0[y_col])

    metadata_cols, missing_metadata_cols = select_metadata_columns(df0, args.metadata_cols)
    text_emb = load_text_embeddings(args.text_embeddings)
    df, feature_cols, text_cols, meta_cols = build_feature_frame(df0, text_emb, metadata_cols)

    # Add internal validation only if the split lacks one.
    df = add_internal_validation_if_needed(df, args.split_col, y_col, args.seed)
    split = canonical_split_values(df[args.split_col])
    train_mask = split.eq("train")
    val_mask = split.eq("val")
    test_mask = split.eq("test")

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError(f"Split {args.split_col} must contain train and test rows")
    if val_mask.sum() == 0:
        print("WARNING: No validation rows; threshold tuning will be skipped.")

    classes = np.array(sorted(pd.Series(df[y_col]).dropna().unique().tolist(), key=lambda x: str(x)))
    if label_type == "binary":
        # Ensure binary classes are [0,1] if possible.
        try:
            classes = np.array([0, 1])
        except Exception:
            pass

    preproc = make_preprocessor(df, feature_cols)
    X_all = df[feature_cols]
    y_all = df[y_col]

    rows = []
    for model_name in args.models:
        model = get_model(model_name, label_type, args.seed, args.class_weight)
        pipe = Pipeline([("preprocess", preproc), ("model", model)])
        pipe.fit(X_all.loc[train_mask], y_all.loc[train_mask])

        threshold = None
        if label_type == "binary" and args.tune_threshold and val_mask.sum() > 0:
            scores_val = decision_scores(pipe, X_all.loc[val_mask], classes)
            if scores_val is not None:
                yv = labels_to_int_array(y_all.loc[val_mask])
                threshold = tune_binary_threshold(yv, np.asarray(scores_val), args.threshold_metric)

        scores_test = decision_scores(pipe, X_all.loc[test_mask], classes)
        if label_type == "binary" and threshold is not None and scores_test is not None:
            st = np.asarray(scores_test)
            s = st[:, 1] if st.ndim > 1 else st
            y_pred = (s >= threshold).astype(int)
        else:
            y_pred = pipe.predict(X_all.loc[test_mask])

        y_true = y_all.loc[test_mask].values
        if label_type == "binary":
            try:
                y_true_eval = labels_to_int_array(pd.Series(y_true))
                y_pred_eval = labels_to_int_array(pd.Series(y_pred))
            except Exception:
                y_true_eval = y_true
                y_pred_eval = y_pred
            metrics = binary_metrics(y_true_eval, y_pred_eval, np.asarray(scores_test) if scores_test is not None else None)
            save_classes = classes
            score_classes = fitted_score_classes(pipe, np.asarray(scores_test) if scores_test is not None else None, save_classes)
            save_outputs(outdir, model_name, df, df.index[test_mask], y_true_eval, y_pred_eval,
                         np.asarray(scores_test) if scores_test is not None else None, save_classes, metrics, label_type, threshold, score_classes)
        else:
            metrics = multiclass_metrics(y_true, y_pred, np.asarray(scores_test) if scores_test is not None else None, classes)
            score_classes = fitted_score_classes(pipe, np.asarray(scores_test) if scores_test is not None else None, classes)
            save_outputs(outdir, model_name, df, df.index[test_mask], y_true, y_pred,
                         np.asarray(scores_test) if scores_test is not None else None, classes, metrics, label_type, threshold, score_classes)

        row = dict(metrics)
        row.update({
            "model": model_name,
            "target": args.target,
            "split_col": args.split_col,
            "label_type": label_type,
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
            "n_test": int(test_mask.sum()),
            "n_features_raw": len(feature_cols),
            "n_text_embedding_cols": len(text_cols),
            "n_metadata_cols": len(meta_cols),
            "threshold": threshold if threshold is not None else np.nan,
        })
        rows.append(row)

    pd.DataFrame(rows).to_csv(outdir / "metrics.csv", index=False)

    manifest = {
        "benchmark": args.benchmark,
        "target": args.target,
        "split_col": args.split_col,
        "mode": "text_plus_metadata" if text_cols else "metadata_only",
        "text_embeddings": args.text_embeddings,
        "feature_cols": feature_cols,
        "text_embedding_cols_count": len(text_cols),
        "metadata_cols": meta_cols,
        "missing_default_metadata_cols": missing_metadata_cols,
        "models": args.models,
        "leakage_excluded_prefixes": LEAKY_PREFIXES,
        "leakage_excluded_exact": sorted(LEAKY_EXACT),
        "n_rows": int(len(df)),
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
    }
    with open(outdir / "run_summary.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote metadata ablation outputs to {outdir}")


if __name__ == "__main__":
    main()

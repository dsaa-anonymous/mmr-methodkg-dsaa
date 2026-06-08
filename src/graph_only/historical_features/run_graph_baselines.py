#!/usr/bin/env python3
"""
Run lightweight graph-only baselines on MethodKG graph/context features.

This script trains scikit-learn models using only graph/history/static metadata
features produced by build_graph_only_features.py. It never uses award text,
candidate flags, annotation guidance, or labels as input features.

Example:
  python run_graph_baselines.py \
    --features graph_features_v1/methodkg_graph_only_features.csv \
    --outdir results/graph_integration_random \
    --target target_integration_binary \
    --split_col split_random_cluster_stratified \
    --models dummy graph_lr graph_rf graph_extra_trees
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize

DEFAULT_MODELS = ["dummy", "graph_lr", "graph_rf", "graph_extra_trees"]

ID_COLS = {"benchmark_id", "annotation_id", "award_id", "project_cluster_id", "benchmark_version", "label_source"}
TEXT_COLS = {"title_clean", "abstract_clean"}
PROVENANCE_PREFIXES = (
    "candidate_", "annotation_", "review_", "explicit_mmr_candidate", "implicit_mmr_candidate",
    "qual_signal_candidate", "quant_signal_candidate", "design_label_candidate", "integration_candidate"
)


def onehot_encoder():
    """Create a OneHotEncoder compatible with recent and older scikit-learn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def infer_task_type(y: pd.Series, target: str) -> str:
    if target.endswith("_multiclass") or y.dtype == object:
        return "multiclass"
    vals = sorted(pd.Series(y).dropna().unique().tolist())
    if len(vals) <= 2:
        return "binary"
    return "multiclass"


def clean_target(df: pd.DataFrame, target: str, drop_unclear: bool = True) -> pd.DataFrame:
    out = df.copy()
    if target not in out.columns:
        raise ValueError(f"Target column not found: {target}")
    out = out[out[target].notna()].copy()
    if drop_unclear:
        out = out[out[target].astype(str).str.lower() != "unclear"].copy()
    # Numeric-looking binary targets should be int.
    if out[target].dtype != object:
        vals = sorted(out[target].dropna().unique().tolist())
        if set(vals).issubset({0, 1, 0.0, 1.0}):
            out[target] = out[target].astype(int)
    else:
        # If string values are 0/1, coerce.
        normalized = out[target].astype(str).str.strip().str.lower()
        if set(normalized.unique()).issubset({"0", "1"}):
            out[target] = normalized.astype(int)
        else:
            out[target] = normalized
    return out


def select_feature_columns(df: pd.DataFrame, include_static_metadata: bool = True) -> tuple[list[str], list[str]]:
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for c in df.columns:
        if c in ID_COLS or c in TEXT_COLS:
            continue
        if c.startswith("target_") or c.startswith("label_") or c.startswith("split_"):
            continue
        if any(c.startswith(p) for p in PROVENANCE_PREFIXES):
            continue
        if c.startswith("g_") or (include_static_metadata and c.startswith("m_")):
            numeric_cols.append(c)
        elif include_static_metadata and c.startswith("cat_"):
            categorical_cols.append(c)
    return numeric_cols, categorical_cols


def make_preprocessor(numeric_cols: Sequence[str], categorical_cols: Sequence[str]) -> ColumnTransformer:
    transformers = []
    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            list(numeric_cols),
        ))
    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", onehot_encoder()),
            ]),
            list(categorical_cols),
        ))
    if not transformers:
        raise ValueError("No graph-only feature columns were selected.")
    return ColumnTransformer(transformers)


def make_model(name: str, seed: int, task_type: str):
    if name == "dummy":
        return DummyClassifier(strategy="most_frequent")
    if name == "dummy_stratified":
        return DummyClassifier(strategy="stratified", random_state=seed)
    if name == "graph_lr":
        return LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="saga" if task_type == "multiclass" else "liblinear",
            n_jobs=-1 if task_type == "multiclass" else None,
            random_state=seed,
        )
    if name == "graph_rf":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "graph_extra_trees":
        return ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "graph_hgb":
        return HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            l2_regularization=0.01,
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {name}")


def predict_scores(model: Pipeline, X_test: pd.DataFrame, task_type: str, positive_label=1):
    y_score = None
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)
        except Exception:
            proba = None
    if task_type == "binary" and proba is not None:
        classes = list(model.named_steps["model"].classes_)
        if positive_label in classes:
            y_score = proba[:, classes.index(positive_label)]
        elif str(positive_label) in classes:
            y_score = proba[:, classes.index(str(positive_label))]
        elif proba.shape[1] == 2:
            y_score = proba[:, 1]
    return y_score, proba


def compute_metrics(y_true, y_pred, y_score=None, proba=None, task_type="binary") -> Dict[str, object]:
    metrics: Dict[str, object] = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    if task_type == "binary":
        positive_label = 1
        labels = sorted(pd.Series(y_true).unique().tolist())
        if positive_label not in labels and len(labels) == 2:
            positive_label = labels[-1]
        metrics["positive_label"] = str(positive_label)
        metrics["positive_f1"] = float(f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0))
        metrics["positive_precision"] = float(precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0))
        metrics["positive_recall"] = float(recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0))
        if y_score is not None and len(set(y_true)) == 2:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
            except Exception:
                metrics["roc_auc"] = np.nan
            try:
                metrics["pr_auc"] = float(average_precision_score(y_true, y_score, pos_label=positive_label))
            except Exception:
                metrics["pr_auc"] = np.nan
        else:
            metrics["roc_auc"] = np.nan
            metrics["pr_auc"] = np.nan
    else:
        metrics["positive_f1"] = np.nan
        metrics["positive_precision"] = np.nan
        metrics["positive_recall"] = np.nan
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan
        if proba is not None:
            try:
                classes = sorted(pd.Series(y_true).unique().tolist())
                y_bin = label_binarize(y_true, classes=classes)
                if y_bin.shape[1] == proba.shape[1]:
                    metrics["macro_ovr_roc_auc"] = float(roc_auc_score(y_bin, proba, average="macro", multi_class="ovr"))
            except Exception:
                metrics["macro_ovr_roc_auc"] = np.nan
    return metrics


def run_one(features_path: str | Path, outdir: str | Path, target: str, split_col: str,
            models: Sequence[str], seed: int, include_static_metadata: bool, drop_unclear: bool) -> pd.DataFrame:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(features_path, low_memory=False)
    if split_col not in df.columns:
        raise ValueError(f"Split column not found: {split_col}")
    df = clean_target(df, target, drop_unclear=drop_unclear)
    df = df[df[split_col].isin(["train", "validation", "test"])].copy()
    if not {"train", "test"}.issubset(set(df[split_col].unique())):
        raise ValueError(f"Split {split_col} must contain at least train and test rows")

    train_df = df[df[split_col] == "train"].copy()
    test_df = df[df[split_col] == "test"].copy()
    val_df = df[df[split_col] == "validation"].copy()

    y_train = train_df[target]
    y_test = test_df[target]
    task_type = infer_task_type(y_train, target)
    numeric_cols, categorical_cols = select_feature_columns(df, include_static_metadata=include_static_metadata)
    feature_cols = numeric_cols + categorical_cols

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    results = []
    for model_name in models:
        print(f"[RUN] target={target} split={split_col} model={model_name}")
        preprocessor = make_preprocessor(numeric_cols, categorical_cols)
        clf = make_model(model_name, seed=seed, task_type=task_type)
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", clf),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_score, proba = predict_scores(pipe, X_test, task_type=task_type)
        metrics = compute_metrics(y_test, y_pred, y_score=y_score, proba=proba, task_type=task_type)
        metrics.update({
            "target": target,
            "split_col": split_col,
            "model": model_name,
            "task_type": task_type,
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "n_numeric_features": int(len(numeric_cols)),
            "n_categorical_features": int(len(categorical_cols)),
        })
        results.append(metrics)

        model_out = outdir / model_name
        model_out.mkdir(parents=True, exist_ok=True)
        pred_df = test_df[[c for c in ["benchmark_id", "annotation_id", "award_id", "project_cluster_id", target] if c in test_df.columns]].copy()
        pred_df["y_true"] = y_test.values
        pred_df["y_pred"] = y_pred
        if y_score is not None:
            pred_df["y_score_positive"] = y_score
        pred_df.to_csv(model_out / "predictions.csv", index=False, encoding="utf-8-sig")

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        with open(model_out / "classification_report.json", "w") as f:
            json.dump(report, f, indent=2)
        labels = sorted(pd.Series(list(y_test) + list(y_pred)).unique().tolist())
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])
        cm_df.to_csv(model_out / "confusion_matrix.csv", encoding="utf-8-sig")
        with open(model_out / "run_config.json", "w") as f:
            json.dump({
                "features_path": str(features_path),
                "target": target,
                "split_col": split_col,
                "model": model_name,
                "seed": seed,
                "include_static_metadata": include_static_metadata,
                "drop_unclear": drop_unclear,
                "numeric_cols": numeric_cols,
                "categorical_cols": categorical_cols,
                "metrics": metrics,
            }, f, indent=2)

    metrics_df = pd.DataFrame(results)
    metrics_df = metrics_df[[
        "target", "split_col", "model", "task_type", "train_rows", "validation_rows", "test_rows",
        "accuracy", "macro_f1", "weighted_f1", "positive_f1", "positive_precision", "positive_recall",
        "roc_auc", "pr_auc", "n_numeric_features", "n_categorical_features"
    ] + [c for c in metrics_df.columns if c not in {
        "target", "split_col", "model", "task_type", "train_rows", "validation_rows", "test_rows",
        "accuracy", "macro_f1", "weighted_f1", "positive_f1", "positive_precision", "positive_recall",
        "roc_auc", "pr_auc", "n_numeric_features", "n_categorical_features"
    }]]
    metrics_df.to_csv(outdir / "metrics.csv", index=False, encoding="utf-8-sig")
    print("[DONE] Wrote", outdir / "metrics.csv")
    return metrics_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="methodkg_graph_only_features.csv from build_graph_only_features.py")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--split_col", required=True)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        choices=["dummy", "dummy_stratified", "graph_lr", "graph_rf", "graph_extra_trees", "graph_hgb"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_static_metadata", action="store_true", help="Use historical graph features only; exclude static categorical/numeric metadata.")
    parser.add_argument("--keep_unclear", action="store_true", help="Keep target value 'unclear' for multiclass tasks.")
    args = parser.parse_args()

    run_one(
        features_path=args.features,
        outdir=args.outdir,
        target=args.target,
        split_col=args.split_col,
        models=args.models,
        seed=args.seed,
        include_static_metadata=not args.no_static_metadata,
        drop_unclear=not args.keep_unclear,
    )


if __name__ == "__main__":
    main()

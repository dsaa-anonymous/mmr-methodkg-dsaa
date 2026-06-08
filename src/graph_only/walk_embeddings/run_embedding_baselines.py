#!/usr/bin/env python3
"""
Train graph-embedding-only classifiers on MethodKG benchmark splits.

Inputs:
  methodkg_labeled_benchmark_v2_modeling.csv or benchmark_v2.zip
  node2vec_award_embeddings.csv or metapath2vec_award_embeddings.csv

Outputs:
  metrics.csv
  predictions.csv
  classification_report.json
  confusion_matrix.csv
  feature_importance.csv where available
"""

import argparse
import json
import zipfile
from pathlib import Path
import sys
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from path_utils import discover_benchmark, find_repo_root, read_csv_or_zip, resolve_existing_path, resolve_output_path, write_resolved_paths
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def clean_award_id(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    return "".join(ch for ch in s if ch.isdigit())


def load_csv_maybe_zip(path: str | Path, preferred_name="methodkg_labeled_benchmark_v3_modeling.csv") -> pd.DataFrame:
    return read_csv_or_zip(path, preferred_name=preferred_name)


def infer_task_type(y: pd.Series) -> str:
    vals = sorted(set(y.dropna().astype(str)))
    if len(vals) <= 2 and set(vals).issubset({"0", "1", "0.0", "1.0"}):
        return "binary"
    if len(vals) <= 2:
        return "binary"
    return "multiclass"


def normalize_labels(y: pd.Series):
    # Convert numeric binary columns to int if possible, otherwise leave strings.
    nonnull = y.dropna()
    try:
        yn = pd.to_numeric(nonnull)
        if set(yn.unique()).issubset({0, 1, 0.0, 1.0}):
            return pd.to_numeric(y).astype(int)
    except Exception:
        pass
    return y.astype(str)


def ensure_validation_split(df: pd.DataFrame, split_col: str, y_col: str, seed: int) -> pd.DataFrame:
    out = df.copy()
    vals = set(out[split_col].dropna().astype(str))
    if "validation" in vals:
        return out
    if "train" not in vals:
        raise ValueError(f"Split column {split_col} must contain train rows.")
    train_idx = out.index[out[split_col].astype(str) == "train"].to_numpy()
    if len(train_idx) < 10:
        return out
    y_train = out.loc[train_idx, y_col]
    # Use stratified split when possible.
    try:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        tr_rel, val_rel = next(splitter.split(np.zeros(len(train_idx)), y_train))
        val_idx = train_idx[val_rel]
    except Exception:
        rng = np.random.default_rng(seed)
        val_size = max(1, int(round(0.15 * len(train_idx))))
        val_idx = rng.choice(train_idx, size=val_size, replace=False)
    out.loc[val_idx, split_col] = "validation"
    return out


def get_splits(df: pd.DataFrame, split_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = df[split_col].astype(str)
    train_idx = df.index[s == "train"].to_numpy()
    val_idx = df.index[s == "validation"].to_numpy()
    test_idx = df.index[s == "test"].to_numpy()
    if len(test_idx) == 0:
        raise ValueError(f"Split {split_col} has no test rows.")
    if len(val_idx) == 0:
        raise ValueError(f"Split {split_col} has no validation rows after internal validation creation.")
    return train_idx, val_idx, test_idx


def build_model(name: str, seed: int, task_type: str):
    if name == "dummy":
        return DummyClassifier(strategy="most_frequent")
    if name == "emb_lr":
        max_iter = 3000
        if task_type == "binary":
            clf = LogisticRegression(max_iter=max_iter, class_weight="balanced", solver="liblinear")
        else:
            clf = LogisticRegression(max_iter=max_iter, class_weight="balanced", solver="lbfgs", multi_class="auto")
        return Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=0.0)), ("scale", StandardScaler()), ("clf", clf)])
    if name == "emb_svm":
        clf = LinearSVC(class_weight="balanced", random_state=seed, max_iter=10000)
        return Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=0.0)), ("scale", StandardScaler()), ("clf", clf)])
    if name == "emb_rf":
        return Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("clf", RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample", random_state=seed, n_jobs=-1, min_samples_leaf=2)),
        ])
    if name == "emb_extra_trees":
        return Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("clf", ExtraTreesClassifier(n_estimators=400, class_weight="balanced", random_state=seed, n_jobs=-1, min_samples_leaf=2)),
        ])
    if name == "emb_mlp":
        return Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4, learning_rate_init=1e-3, max_iter=300, early_stopping=True, random_state=seed)),
        ])
    raise ValueError(f"Unknown model: {name}")


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        try:
            p = model.predict_proba(X)
            if p.shape[1] == 2:
                return p[:, 1]
            return p
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            return model.decision_function(X)
        except Exception:
            pass
    return None


def tune_threshold(y_val, scores, objective="macro_f1"):
    if scores is None:
        return 0.5, None
    scores = np.asarray(scores)
    if scores.ndim != 1:
        return 0.5, None
    # Candidate thresholds from score quantiles plus 0 for margin-based SVM.
    qs = np.linspace(0.02, 0.98, 97)
    candidates = sorted(set(np.quantile(scores, qs).tolist() + [0.0, 0.5]))
    best_t, best_score = candidates[0], -1.0
    y_val = np.asarray(y_val).astype(int)
    for t in candidates:
        pred = (scores >= t).astype(int)
        if objective == "positive_f1":
            score = f1_score(y_val, pred, pos_label=1, zero_division=0)
        else:
            score = f1_score(y_val, pred, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t, best_score


def evaluate(y_true, y_pred, scores, task_type: str) -> Dict[str, float]:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if task_type == "binary":
        y_true_int = np.asarray(y_true).astype(int)
        y_pred_int = np.asarray(y_pred).astype(int)
        out.update({
            "positive_precision": precision_score(y_true_int, y_pred_int, pos_label=1, zero_division=0),
            "positive_recall": recall_score(y_true_int, y_pred_int, pos_label=1, zero_division=0),
            "positive_f1": f1_score(y_true_int, y_pred_int, pos_label=1, zero_division=0),
        })
        if scores is not None:
            sc = np.asarray(scores)
            if sc.ndim == 1 and len(set(y_true_int)) == 2:
                try:
                    out["roc_auc"] = roc_auc_score(y_true_int, sc)
                except Exception:
                    out["roc_auc"] = np.nan
                try:
                    out["pr_auc"] = average_precision_score(y_true_int, sc)
                except Exception:
                    out["pr_auc"] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=None, help="Repository root. Defaults to auto-detection.")
    ap.add_argument("--benchmark", default=None, help="methodkg_labeled_benchmark_v3_modeling.csv or benchmark_v3.zip. Defaults to data/benchmark discovery.")
    ap.add_argument("--embeddings", required=True, help="node2vec_award_embeddings.csv or metapath2vec_award_embeddings.csv")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--split_col", required=True)
    ap.add_argument("--models", nargs="+", default=["dummy", "emb_lr", "emb_svm", "emb_rf", "emb_extra_trees"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tune_threshold", action="store_true", help="Tune binary threshold on validation set.")
    ap.add_argument("--threshold_objective", choices=["macro_f1", "positive_f1"], default="macro_f1")
    args = ap.parse_args()

    repo_root = find_repo_root(args.repo_root or Path(__file__).resolve())
    benchmark_path = resolve_existing_path(args.benchmark, repo_root) if args.benchmark else discover_benchmark(repo_root)
    embeddings_path = resolve_existing_path(args.embeddings, repo_root)
    outdir = resolve_output_path(args.outdir, repo_root, args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_resolved_paths(repo_root=repo_root, benchmark=benchmark_path, embeddings=embeddings_path, outdir=outdir)

    bench = load_csv_maybe_zip(benchmark_path)
    emb = pd.read_csv(embeddings_path)
    bench["award_id"] = bench["award_id"].apply(clean_award_id)
    emb["award_id"] = emb["award_id"].apply(clean_award_id)

    if args.target not in bench.columns:
        raise ValueError(f"Target {args.target} not found in benchmark.")
    if args.split_col not in bench.columns:
        raise ValueError(f"Split column {args.split_col} not found in benchmark.")

    df = bench.merge(emb, on="award_id", how="left")
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise ValueError("No embedding columns found. Expected columns named emb_000, emb_001, ...")
    if "embedding_missing" not in df.columns:
        df["embedding_missing"] = df[emb_cols].isna().all(axis=1).astype(int)
    # Use missing indicator as a feature too.
    feature_cols = emb_cols + ["embedding_missing"]
    df[feature_cols] = df[feature_cols].fillna(0.0)

    y = normalize_labels(df[args.target])
    df["__y__"] = y
    task_type = infer_task_type(y)
    df = ensure_validation_split(df, args.split_col, "__y__", args.seed)
    train_idx, val_idx, test_idx = get_splits(df, args.split_col)

    X = df[feature_cols]
    y_all = df["__y__"]
    X_train, y_train = X.loc[train_idx], y_all.loc[train_idx]
    X_val, y_val = X.loc[val_idx], y_all.loc[val_idx]
    X_test, y_test = X.loc[test_idx], y_all.loc[test_idx]

    metrics_rows = []
    pred_frames = []
    reports = {}
    confusions = {}

    for model_name in args.models:
        print(f"Training {model_name} on {args.target} / {args.split_col}...")
        model = build_model(model_name, args.seed, task_type)
        model.fit(X_train, y_train)

        val_scores = get_scores(model, X_val)
        test_scores = get_scores(model, X_test)
        threshold = None
        threshold_val_score = None

        if task_type == "binary" and args.tune_threshold and model_name != "dummy":
            threshold, threshold_val_score = tune_threshold(y_val, val_scores, args.threshold_objective)
            if test_scores is not None and np.asarray(test_scores).ndim == 1:
                y_pred = (np.asarray(test_scores) >= threshold).astype(int)
            else:
                y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test)

        metric = evaluate(y_test, y_pred, test_scores, task_type)
        metric.update({
            "model": model_name,
            "target": args.target,
            "split_col": args.split_col,
            "task_type": task_type,
            "train_rows": len(train_idx),
            "validation_rows": len(val_idx),
            "test_rows": len(test_idx),
            "embedding_file": str(args.embeddings),
            "embedding_missing_train": int(df.loc[train_idx, "embedding_missing"].sum()),
            "embedding_missing_test": int(df.loc[test_idx, "embedding_missing"].sum()),
            "threshold": threshold if threshold is not None else "",
            "threshold_validation_score": threshold_val_score if threshold_val_score is not None else "",
        })
        metrics_rows.append(metric)

        pred = df.loc[test_idx, ["benchmark_id", "award_id", "title_clean", args.target, args.split_col]].copy()
        pred["model"] = model_name
        pred["y_true"] = list(y_test)
        pred["y_pred"] = list(y_pred)
        if test_scores is not None:
            sc = np.asarray(test_scores)
            if sc.ndim == 1:
                pred["score"] = sc
        pred_frames.append(pred)

        labels_sorted = sorted(set(list(y_train) + list(y_val) + list(y_test)), key=lambda x: str(x))
        reports[model_name] = classification_report(y_test, y_pred, labels=labels_sorted, output_dict=True, zero_division=0)
        confusions[model_name] = {
            "labels": [str(x) for x in labels_sorted],
            "matrix": confusion_matrix(y_test, y_pred, labels=labels_sorted).tolist(),
        }

    pd.DataFrame(metrics_rows).to_csv(outdir / "metrics.csv", index=False)
    pd.concat(pred_frames, ignore_index=True).to_csv(outdir / "predictions.csv", index=False)
    with open(outdir / "classification_report.json", "w") as f:
        json.dump(reports, f, indent=2)
    with open(outdir / "confusion_matrix.json", "w") as f:
        json.dump(confusions, f, indent=2)

    run_info = {
        "target": args.target,
        "split_col": args.split_col,
        "task_type": task_type,
        "feature_count": len(feature_cols),
        "train_rows": len(train_idx),
        "validation_rows": len(val_idx),
        "test_rows": len(test_idx),
        "models": args.models,
        "tune_threshold": args.tune_threshold,
        "threshold_objective": args.threshold_objective,
    }
    with open(outdir / "run_summary.json", "w") as f:
        json.dump(run_info, f, indent=2)
    print("Done. Results written to", outdir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run MethodKG Text+Graph late-fusion baselines.

This script trains supervised classifiers on the 2,500-row MethodKG benchmark by
combining optional feature groups:
  - text embeddings, e.g. MiniLM/SciBERT/SPECTER/e5 embeddings
  - historical graph features
  - node2vec embeddings
  - metapath2vec embeddings
  - simple award metadata

It never uses label_*, target_*, split_*, candidate_*, annotation_* or review/guidance
columns as input features.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from path_utils import (
    default_graph_features,
    default_metapath2vec_embeddings,
    default_node2vec_embeddings,
    default_text_embeddings_minilm,
    discover_benchmark,
    find_repo_root,
    read_csv_or_zip,
    reset_dir_if_overwrite,
    resolve_existing_path,
    resolve_output_path,
    write_resolved_paths,
)
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
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

ID_COLS = {"award_id", "benchmark_id", "annotation_id", "project_cluster_id"}
TEXT_COLS = {"title_clean", "abstract_clean", "text", "combined_text"}
LEAKY_PREFIXES = (
    "label_", "target_", "split_", "candidate_", "annotation_",
    "explicit_mmr_candidate", "implicit_mmr_candidate", "qual_signal_candidate",
    "quant_signal_candidate", "integration_candidate", "design_label_candidate",
)
LEAKY_SUBSTRINGS = (
    "guidance", "review_priority", "annotator", "adjudication", "quality_report",
)
DEFAULT_METADATA_COLS = [
    "start_year", "NSFDirectorate", "NSFOrganization", "ProgramElementCode(s)",
    "AwardInstrument", "State", "OrganizationState",
]

SPLIT_VALUE_MAP = {
    "val": "validation",
    "valid": "validation",
    "dev": "validation",
}


def normalize_award_id(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0") and re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    # Keep digits only, but do not accidentally append the decimal zero.
    s = re.sub(r"\D", "", s)
    return s


def read_csv(path: str) -> pd.DataFrame:
    return read_csv_or_zip(path, encoding="utf-8-sig")


def safe_col_name(prefix: str, col: str) -> str:
    col = str(col).replace(" ", "_")
    col = re.sub(r"[^A-Za-z0-9_]+", "_", col)
    col = re.sub(r"_+", "_", col).strip("_")
    return f"{prefix}__{col}"


def is_leaky_col(col: str) -> bool:
    c = str(col)
    if c in ID_COLS or c in TEXT_COLS:
        return True
    if c.startswith(LEAKY_PREFIXES):
        return True
    lc = c.lower()
    if any(s in lc for s in LEAKY_SUBSTRINGS):
        return True
    if lc in {"awardnumber", "title", "abstract"}:
        return True
    return False


def load_feature_file(path: str, prefix: str) -> Tuple[pd.DataFrame, List[str]]:
    df = read_csv(path)
    if "award_id" not in df.columns:
        raise ValueError(f"Feature file {path} must contain award_id")
    df = df.copy()
    df["award_id"] = df["award_id"].apply(normalize_award_id)
    keep = [c for c in df.columns if c == "award_id" or not is_leaky_col(c)]
    df = df[keep].copy()
    rename = {c: safe_col_name(prefix, c) for c in df.columns if c != "award_id"}
    df = df.rename(columns=rename)
    feature_cols = [rename[c] for c in rename]
    # Drop duplicate award_id rows safely.
    df = df.drop_duplicates(subset=["award_id"], keep="first")
    return df, feature_cols


def load_and_merge_features(args) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    benchmark = read_csv(args.benchmark)
    if "award_id" not in benchmark.columns:
        raise ValueError("Benchmark file must contain award_id")
    benchmark = benchmark.copy()
    benchmark["award_id"] = benchmark["award_id"].apply(normalize_award_id)

    df = benchmark.copy()
    groups: Dict[str, List[str]] = {}

    if args.text_embeddings:
        feat, cols = load_feature_file(args.text_embeddings, "text")
        df = df.merge(feat, on="award_id", how="left")
        groups["text_embeddings"] = cols

    if args.graph_features:
        feat, cols = load_feature_file(args.graph_features, "graphhist")
        df = df.merge(feat, on="award_id", how="left")
        groups["historical_graph_features"] = cols

    if args.node2vec_embeddings:
        feat, cols = load_feature_file(args.node2vec_embeddings, "node2vec")
        df = df.merge(feat, on="award_id", how="left")
        groups["node2vec"] = cols

    if args.metapath2vec_embeddings:
        feat, cols = load_feature_file(args.metapath2vec_embeddings, "metapath2vec")
        df = df.merge(feat, on="award_id", how="left")
        groups["metapath2vec"] = cols

    if args.include_metadata:
        meta_cols = [c for c in DEFAULT_METADATA_COLS if c in df.columns]
        groups["metadata"] = meta_cols

    selected_groups = set(args.feature_groups) if args.feature_groups else set(groups.keys())
    feature_cols: List[str] = []
    active_groups: Dict[str, List[str]] = {}
    for g, cols in groups.items():
        if g in selected_groups:
            cols2 = [c for c in cols if c in df.columns and not is_leaky_col(c)]
            if cols2:
                active_groups[g] = cols2
                feature_cols.extend(cols2)

    if not feature_cols:
        raise ValueError(
            "No feature columns selected. Provide at least one feature file or use --include_metadata."
        )

    # Drop columns that are entirely missing. This can happen when a feature file is incomplete.
    kept = []
    dropped_all_missing = []
    for c in feature_cols:
        if df[c].notna().sum() == 0:
            dropped_all_missing.append(c)
        else:
            kept.append(c)
    feature_cols = kept
    for g in list(active_groups):
        active_groups[g] = [c for c in active_groups[g] if c in feature_cols]
        if not active_groups[g]:
            del active_groups[g]

    if not feature_cols:
        raise ValueError("All selected feature columns were entirely missing after merge.")

    df.attrs["feature_cols"] = feature_cols
    df.attrs["feature_groups"] = active_groups
    df.attrs["dropped_all_missing"] = dropped_all_missing
    return df, active_groups


def normalize_split_values(s: pd.Series) -> pd.Series:
    return s.fillna("unknown").astype(str).str.strip().str.lower().replace(SPLIT_VALUE_MAP)


def prepare_target(df: pd.DataFrame, target: str) -> Tuple[pd.Series, Dict[str, int], Dict[int, str]]:
    if target not in df.columns:
        raise ValueError(f"Target column not found: {target}")
    y_raw = df[target]
    # Drop missing targets handled by caller via non-null mask.
    if pd.api.types.is_numeric_dtype(y_raw):
        vals = sorted([v for v in pd.unique(y_raw.dropna())])
        # If 0/1 floats, map to ints.
        label_to_id = {str(int(v)) if float(v).is_integer() else str(v): i for i, v in enumerate(vals)}
        # Keep class id equal to integer if binary 0/1.
        if set(vals).issubset({0, 1, 0.0, 1.0}):
            y = y_raw.astype(int)
            label_to_id = {"0": 0, "1": 1}
            id_to_label = {0: "0", 1: "1"}
        else:
            mapping = {v: i for i, v in enumerate(vals)}
            y = y_raw.map(mapping).astype(int)
            id_to_label = {i: str(v) for v, i in mapping.items()}
    else:
        s = y_raw.fillna("missing").astype(str).str.strip()
        labels = sorted(pd.unique(s))
        label_to_id = {lab: i for i, lab in enumerate(labels)}
        id_to_label = {i: lab for lab, i in label_to_id.items()}
        y = s.map(label_to_id).astype(int)
    return y, label_to_id, id_to_label


def add_internal_validation(df: pd.DataFrame, split_col: str, y_col: str, seed: int) -> pd.Series:
    split = normalize_split_values(df[split_col])
    if (split == "validation").sum() > 0:
        return split
    train_idx = df.index[split == "train"].to_numpy()
    if len(train_idx) < 10:
        raise ValueError("Not enough training rows to create an internal validation split")

    # Cluster-safe when project_cluster_id exists: split clusters, not rows.
    if "project_cluster_id" in df.columns:
        tmp = df.loc[train_idx, ["project_cluster_id", y_col]].copy()
        tmp["cluster_key"] = tmp["project_cluster_id"].fillna(pd.Series(tmp.index.astype(str), index=tmp.index)).astype(str)
        # Majority label per cluster for optional stratification.
        cluster_df = tmp.groupby("cluster_key", dropna=False)[y_col].agg(lambda x: x.value_counts().idxmax()).reset_index()
        stratify = cluster_df[y_col] if cluster_df[y_col].value_counts().min() >= 2 and cluster_df[y_col].nunique() > 1 else None
        try:
            tr_clusters, val_clusters = train_test_split(
                cluster_df["cluster_key"], test_size=0.15, random_state=seed, stratify=stratify
            )
        except Exception:
            tr_clusters, val_clusters = train_test_split(
                cluster_df["cluster_key"], test_size=0.15, random_state=seed
            )
        val_set = set(val_clusters.astype(str))
        val_mask = df.index.isin(tmp.index[tmp["cluster_key"].isin(val_set)])
    else:
        y = df.loc[train_idx, y_col]
        stratify = y if y.value_counts().min() >= 2 and y.nunique() > 1 else None
        _, val_idx = train_test_split(train_idx, test_size=0.15, random_state=seed, stratify=stratify)
        val_mask = df.index.isin(val_idx)

    split2 = split.copy()
    split2.loc[val_mask & (split == "train")] = "validation"
    return split2


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_cols = []
    categorical_cols = []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            numeric_cols.append(c)
        else:
            # Try numeric coercion if mostly numeric strings.
            coerced = pd.to_numeric(X[c], errors="coerce")
            if coerced.notna().mean() > 0.95:
                X[c] = coerced
                numeric_cols.append(c)
            else:
                categorical_cols.append(c)

    transformers = []
    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
            numeric_cols,
        ))
    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
            categorical_cols,
        ))
    if not transformers:
        raise ValueError("No numeric or categorical feature columns available")
    return ColumnTransformer(transformers), numeric_cols, categorical_cols


def make_model(name: str, num_classes: int, seed: int, class_weight=None):
    if name == "dummy":
        return DummyClassifier(strategy="most_frequent")
    if name == "fusion_lr":
        if num_classes == 2:
            return LogisticRegression(max_iter=5000, class_weight=class_weight, C=1.0, solver="liblinear")
        return LogisticRegression(max_iter=5000, class_weight=class_weight, C=1.0, solver="lbfgs", multi_class="auto")
    if name == "fusion_svm":
        # Use uncalibrated LinearSVC instead of CalibratedClassifierCV.
        # Calibration with cv=3 fails for MethodKG multiclass labels because
        # the rare "unclear" class may have fewer than 3 examples in train.
        # get_scores() below converts decision_function outputs into
        # sigmoid/softmax pseudo-probabilities for threshold tuning and metrics.
        return LinearSVC(class_weight=class_weight, C=1.0, random_state=seed, max_iter=10000)
    if name == "fusion_mlp":
        return MLPClassifier(
            hidden_layer_sizes=(256, 64), activation="relu", alpha=1e-4,
            batch_size=64, learning_rate_init=1e-3, max_iter=300,
            early_stopping=True, random_state=seed,
        )
    if name == "fusion_rf":
        return RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=2,
            class_weight=class_weight, n_jobs=-1, random_state=seed,
        )
    if name == "fusion_extra_trees":
        return ExtraTreesClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=2,
            class_weight=class_weight, n_jobs=-1, random_state=seed,
        )
    raise ValueError(f"Unknown model: {name}")


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim == 1:
            # Convert to pseudo-probability via sigmoid.
            p1 = 1 / (1 + np.exp(-scores))
            return np.vstack([1 - p1, p1]).T
        e = np.exp(scores - scores.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    return None


def tune_binary_threshold(y_val, score_val, metric="macro_f1") -> Tuple[float, float]:
    best_t = 0.5
    best_v = -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pred = (score_val >= t).astype(int)
        if metric == "positive_f1":
            v = f1_score(y_val, pred, pos_label=1, zero_division=0)
        else:
            v = f1_score(y_val, pred, average="macro", zero_division=0)
        if v > best_v:
            best_v = v
            best_t = float(t)
    return best_t, best_v


def compute_metrics(y_true, y_pred, score, id_to_label: Dict[int, str]) -> Dict[str, float]:
    labels = sorted(id_to_label.keys())
    num_classes = len(labels)
    out = {
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if num_classes == 2:
        out.update({
            "positive_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "positive_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "positive_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        })
        if score is not None:
            s1 = score[:, 1] if score.ndim == 2 else score
            try:
                out["roc_auc"] = float(roc_auc_score(y_true, s1))
            except Exception:
                out["roc_auc"] = math.nan
            try:
                out["pr_auc"] = float(average_precision_score(y_true, s1))
            except Exception:
                out["pr_auc"] = math.nan
    else:
        if score is not None and score.ndim == 2 and score.shape[1] == num_classes:
            # Multiclass ROC/PR-AUC are undefined or misleading when a class is
            # absent from y_true. This happens in MethodKG for the very rare
            # "unclear" class. In that case, report NaN and rely on macro-F1,
            # weighted-F1, and the confusion matrix.
            present = set(np.unique(y_true).tolist())
            all_present = set(labels).issubset(present)
            if all_present:
                try:
                    y_bin = label_binarize(y_true, classes=labels)
                    out["macro_ovr_roc_auc"] = float(roc_auc_score(y_bin, score, average="macro", multi_class="ovr"))
                except Exception:
                    out["macro_ovr_roc_auc"] = math.nan
                try:
                    y_bin = label_binarize(y_true, classes=labels)
                    out["macro_pr_auc"] = float(average_precision_score(y_bin, score, average="macro"))
                except Exception:
                    out["macro_pr_auc"] = math.nan
            else:
                out["macro_ovr_roc_auc"] = math.nan
                out["macro_pr_auc"] = math.nan
    return out


def save_outputs(outdir: Path, model_name: str, target: str, split_col: str, y_test, y_pred, score, ids, id_to_label, metrics, run_info):
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_row = {"model": model_name, "target": target, "split_col": split_col, **metrics}
    pd.DataFrame([metrics_row]).to_csv(outdir / "metrics.csv", index=False)

    pred_df = pd.DataFrame({"award_id": ids, "y_true": y_test, "y_pred": y_pred})
    pred_df["y_true_label"] = pred_df["y_true"].map(id_to_label)
    pred_df["y_pred_label"] = pred_df["y_pred"].map(id_to_label)
    if score is not None:
        if score.ndim == 1:
            pred_df["score"] = score
        else:
            for i in range(score.shape[1]):
                pred_df[f"score_{id_to_label.get(i, i)}"] = score[:, i]
    pred_df.to_csv(outdir / "predictions.csv", index=False)

    cm = confusion_matrix(y_test, y_pred, labels=sorted(id_to_label.keys()))
    pd.DataFrame(cm, index=[id_to_label[i] for i in sorted(id_to_label)], columns=[id_to_label[i] for i in sorted(id_to_label)]).to_csv(outdir / "confusion_matrix.csv")
    report = classification_report(y_test, y_pred, labels=sorted(id_to_label.keys()), target_names=[id_to_label[i] for i in sorted(id_to_label)], zero_division=0, output_dict=True)
    with open(outdir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(outdir / "run_summary.json", "w") as f:
        json.dump(run_info, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", default=None, help="Repository root. Defaults to auto-detection.")
    parser.add_argument("--benchmark", default=None, help="Benchmark CSV/zip. Defaults to data/benchmark v3.")
    parser.add_argument("--outdir", default=None, help="Output directory. Defaults to experiments/text_graph/late_fusion_minilm/single_run")
    parser.add_argument("--target", required=True)
    parser.add_argument("--split_col", required=True)
    parser.add_argument("--text_embeddings", default=None)
    parser.add_argument("--graph_features", default=None)
    parser.add_argument("--node2vec_embeddings", default=None)
    parser.add_argument("--metapath2vec_embeddings", default=None)
    parser.add_argument("--include_metadata", action="store_true")
    parser.add_argument("--feature_groups", nargs="*", default=None,
                        help="Subset of feature groups: text_embeddings historical_graph_features node2vec metapath2vec metadata")
    parser.add_argument("--models", nargs="+", default=["dummy", "fusion_lr", "fusion_svm", "fusion_mlp", "fusion_extra_trees"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tune_threshold", action="store_true")
    parser.add_argument("--threshold_metric", choices=["macro_f1", "positive_f1"], default="macro_f1")
    parser.add_argument("--class_weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--overwrite", action="store_true", help="Delete the output directory before rerunning.")
    args = parser.parse_args()

    repo_root = find_repo_root(args.repo_root or Path(__file__).resolve())
    args.benchmark = str(resolve_existing_path(args.benchmark, repo_root) if args.benchmark else discover_benchmark(repo_root))
    args.outdir = str(resolve_output_path(args.outdir, repo_root, repo_root / "experiments" / "text_graph" / "late_fusion_minilm" / "single_run"))
    if args.text_embeddings is None and default_text_embeddings_minilm(repo_root).exists():
        args.text_embeddings = str(default_text_embeddings_minilm(repo_root))
    elif args.text_embeddings:
        args.text_embeddings = str(resolve_existing_path(args.text_embeddings, repo_root))
    if args.graph_features is None and default_graph_features(repo_root).exists():
        args.graph_features = str(default_graph_features(repo_root))
    elif args.graph_features:
        args.graph_features = str(resolve_existing_path(args.graph_features, repo_root))
    if args.node2vec_embeddings is None and default_node2vec_embeddings(repo_root).exists():
        args.node2vec_embeddings = str(default_node2vec_embeddings(repo_root))
    elif args.node2vec_embeddings:
        args.node2vec_embeddings = str(resolve_existing_path(args.node2vec_embeddings, repo_root))
    if args.metapath2vec_embeddings is None and default_metapath2vec_embeddings(repo_root).exists():
        args.metapath2vec_embeddings = str(default_metapath2vec_embeddings(repo_root))
    elif args.metapath2vec_embeddings:
        args.metapath2vec_embeddings = str(resolve_existing_path(args.metapath2vec_embeddings, repo_root))
    reset_dir_if_overwrite(Path(args.outdir), args.overwrite)
    write_resolved_paths(repo_root=repo_root, benchmark=args.benchmark, outdir=args.outdir, text_embeddings=args.text_embeddings, graph_features=args.graph_features, node2vec_embeddings=args.node2vec_embeddings, metapath2vec_embeddings=args.metapath2vec_embeddings)

    df, groups = load_and_merge_features(args)
    feature_cols: List[str] = df.attrs["feature_cols"]
    dropped_all_missing: List[str] = df.attrs.get("dropped_all_missing", [])

    if args.split_col not in df.columns:
        raise ValueError(f"Split column not found: {args.split_col}")
    if args.target not in df.columns:
        raise ValueError(f"Target column not found: {args.target}")

    # Drop missing target rows.
    work = df[df[args.target].notna()].copy()
    y, label_to_id, id_to_label = prepare_target(work, args.target)
    work["__y__"] = y.values
    split = add_internal_validation(work, args.split_col, "__y__", seed=args.seed)
    work["__split__"] = split

    train_mask = work["__split__"] == "train"
    val_mask = work["__split__"] == "validation"
    test_mask = work["__split__"] == "test"
    if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError(
            f"Need train/validation/test rows. Got train={train_mask.sum()} val={val_mask.sum()} test={test_mask.sum()}"
        )

    X_all = work[feature_cols].copy()
    y_all = work["__y__"].astype(int)
    num_classes = y_all.nunique()
    class_weight = "balanced" if args.class_weight == "balanced" else None

    manifest_rows = []
    for g, cols in groups.items():
        manifest_rows.append({"feature_group": g, "n_features": len(cols), "columns_preview": "|".join(cols[:25])})
    manifest = pd.DataFrame(manifest_rows)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    manifest.to_csv(Path(args.outdir) / "feature_manifest.csv", index=False)

    all_metrics = []
    for model_name in args.models:
        X_for_preproc = X_all.copy()
        preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_for_preproc)
        model = make_model(model_name, num_classes=num_classes, seed=args.seed, class_weight=class_weight)
        pipe = Pipeline([("prep", preprocessor), ("model", model)])

        pipe.fit(X_for_preproc.loc[train_mask], y_all.loc[train_mask])
        score_val = get_scores(pipe, X_for_preproc.loc[val_mask])
        score_test = get_scores(pipe, X_for_preproc.loc[test_mask])
        threshold = None
        threshold_val_metric = None
        if num_classes == 2 and args.tune_threshold and score_val is not None:
            threshold, threshold_val_metric = tune_binary_threshold(
                y_all.loc[val_mask].to_numpy(), score_val[:, 1], metric=args.threshold_metric
            )
            y_pred = (score_test[:, 1] >= threshold).astype(int)
        else:
            y_pred = pipe.predict(X_for_preproc.loc[test_mask])

        y_test = y_all.loc[test_mask].to_numpy()
        metrics = compute_metrics(y_test, y_pred, score_test, id_to_label)
        metrics["n_train"] = int(train_mask.sum())
        metrics["n_validation"] = int(val_mask.sum())
        metrics["n_classes"] = int(num_classes)
        if threshold is not None:
            metrics["threshold"] = float(threshold)
            metrics["threshold_validation_metric"] = float(threshold_val_metric)

        model_outdir = Path(args.outdir) / model_name
        run_info = {
            "model": model_name,
            "target": args.target,
            "split_col": args.split_col,
            "feature_groups": {g: len(cols) for g, cols in groups.items()},
            "n_features_total": len(feature_cols),
            "numeric_cols": len(numeric_cols),
            "categorical_cols": len(categorical_cols),
            "dropped_all_missing_features": len(dropped_all_missing),
            "label_to_id": label_to_id,
            "id_to_label": id_to_label,
            "class_weight": class_weight,
            "seed": args.seed,
            "threshold_metric": args.threshold_metric if args.tune_threshold else None,
        }
        save_outputs(
            model_outdir, model_name, args.target, args.split_col,
            y_test, y_pred, score_test,
            work.loc[test_mask, "award_id"].astype(str).tolist(),
            id_to_label, metrics, run_info,
        )
        all_metrics.append({"model": model_name, "target": args.target, "split_col": args.split_col, **metrics})

    pd.DataFrame(all_metrics).to_csv(Path(args.outdir) / "metrics_all_models.csv", index=False)
    print(f"Wrote late fusion results to {Path(args.outdir).resolve()}")
    print(pd.DataFrame(all_metrics).to_string(index=False))


if __name__ == "__main__":
    main()

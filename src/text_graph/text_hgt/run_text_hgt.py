#!/usr/bin/env python3
"""Train TG4 text+HGT on MethodKG heterogeneous graph.

Predicts labels for award nodes from the 2,500-row benchmark using text embeddings plus heterogeneous graph message passing.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    average_precision_score,
    roc_auc_score,
)

try:
    from torch_geometric.nn import HGTConv
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Could not import torch_geometric.nn.HGTConv. Install PyTorch Geometric.\n"
        "Original error: " + repr(exc)
    )

TARGET_ALIASES = {
    "target_integration_binary": "integration_binary",
    "target_design_binary": "design_binary",
    "target_mmr_binary": "mmr_binary",
    "target_mmr_multiclass": "mmr_multiclass",
    "target_method_signal_binary": "method_signal_binary",
    "target_qual_binary": "qual_binary",
    "target_quant_binary": "quant_binary",
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def move_heterodata(data, device):
    return data.to(device)


class TextHGT(torch.nn.Module):
    def __init__(self, metadata, in_dims: Dict[str, int], hidden_channels: int, out_channels: int,
                 num_layers: int = 2, heads: int = 4, dropout: float = 0.35):
        super().__init__()
        self.dropout = dropout
        self.lin_dict = torch.nn.ModuleDict()
        for node_type, dim in in_dims.items():
            self.lin_dict[node_type] = torch.nn.Linear(dim, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HGTConv(hidden_channels, hidden_channels, metadata, heads=heads))
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x_dict, edge_index_dict):
        h = {}
        for node_type, x in x_dict.items():
            h[node_type] = F.relu(self.lin_dict[node_type](x))
            h[node_type] = F.dropout(h[node_type], p=self.dropout, training=self.training)
        for conv in self.convs:
            out = conv(h, edge_index_dict)
            new_h = {}
            for node_type in h.keys():
                if node_type in out and out[node_type] is not None:
                    z = F.relu(out[node_type])
                    z = F.dropout(z, p=self.dropout, training=self.training)
                    new_h[node_type] = z
                else:
                    # Some node types may not receive messages in a given schema.
                    new_h[node_type] = h[node_type]
            h = new_h
        return self.classifier(h["award"])


def add_internal_validation_if_needed(df: pd.DataFrame, split_col: str, target_col: str, seed: int) -> pd.DataFrame:
    """If a split has train/test only, carve validation from train, cluster-safe when possible."""
    out = df.copy()
    vals = set(out[split_col].dropna().astype(str).str.lower())
    if "validation" in vals or "val" in vals:
        out["__split__"] = out[split_col].astype(str).str.lower().replace({"val": "validation"})
        return out
    out["__split__"] = out[split_col].astype(str).str.lower()
    train_mask = out["__split__"].eq("train")
    train_df = out.loc[train_mask].copy()
    if len(train_df) == 0:
        return out
    cluster_col = "project_cluster_id" if "project_cluster_id" in out.columns else "award_id"
    tmp = train_df[[cluster_col, target_col]].copy()
    tmp[cluster_col] = tmp[cluster_col].fillna(train_df["award_id"]).astype(str)
    # Cluster-level target for stratification.
    cluster_df = tmp.groupby(cluster_col, dropna=False)[target_col].agg(lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0]).reset_index()
    rng = np.random.default_rng(seed)
    val_clusters = []
    for _, sub in cluster_df.groupby(target_col, dropna=False):
        clusters = sub[cluster_col].astype(str).to_numpy()
        rng.shuffle(clusters)
        n_val = max(1, int(round(0.15 * len(clusters)))) if len(clusters) >= 5 else max(0, int(round(0.15 * len(clusters))))
        val_clusters.extend(clusters[:n_val].tolist())
    val_set = set(val_clusters)
    row_clusters = out[cluster_col].fillna(out["award_id"]).astype(str)
    out.loc[train_mask & row_clusters.isin(val_set), "__split__"] = "validation"
    return out


def prepare_labels_and_masks(bench: pd.DataFrame, target: str, split_col: str, num_award_nodes: int, seed: int):
    bench = bench.copy()
    if target not in bench.columns:
        raise ValueError(f"Target column not found in benchmark table: {target}")
    if split_col not in bench.columns:
        raise ValueError(f"Split column not found in benchmark table: {split_col}")
    bench = bench[bench["is_benchmark"].fillna(0).astype(int) == 1].copy()
    bench[target] = bench[target].astype(str).str.strip()
    # Drop unclear/nan labels.
    invalid_values = {"", "nan", "none", "unclear", "<na>"}
    bench = bench[~bench[target].str.lower().isin(invalid_values)].copy()
    bench = add_internal_validation_if_needed(bench, split_col, target, seed)

    labels = sorted(bench[target].dropna().astype(str).unique().tolist())
    # For binary numeric targets, force 0/1 order.
    if set(labels).issubset({"0", "1", "0.0", "1.0"}):
        def norm_bin(v):
            return str(int(float(v)))
        bench[target] = bench[target].apply(norm_bin)
        labels = ["0", "1"]
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    y = torch.full((num_award_nodes,), -1, dtype=torch.long)
    train_mask = torch.zeros(num_award_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_award_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_award_nodes, dtype=torch.bool)

    for _, r in bench.iterrows():
        idx = int(r["award_node_idx"])
        lab = str(r[target])
        if lab not in label_to_id:
            continue
        y[idx] = label_to_id[lab]
        split = str(r["__split__"]).lower()
        if split == "train":
            train_mask[idx] = True
        elif split in {"validation", "val"}:
            val_mask[idx] = True
        elif split == "test":
            test_mask[idx] = True
    return y, train_mask, val_mask, test_mask, label_to_id, bench


def compute_class_weights(y: torch.Tensor, train_mask: torch.Tensor, num_classes: int, device):
    labels = y[train_mask]
    counts = torch.bincount(labels[labels >= 0], minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights.to(device)


def evaluate_from_logits(y_true: np.ndarray, logits: np.ndarray, label_names: List[str], threshold: float | None = None):
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    if len(label_names) == 2:
        pos_probs = probs[:, 1]
        if threshold is None:
            y_pred = probs.argmax(axis=1)
        else:
            y_pred = (pos_probs >= threshold).astype(int)
    else:
        y_pred = probs.argmax(axis=1)
        pos_probs = None

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if len(label_names) == 2:
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1], average="binary", zero_division=0)
        metrics.update({
            "positive_precision": float(p),
            "positive_recall": float(r),
            "positive_f1": float(f),
        })
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, pos_probs))
        except Exception:
            metrics["roc_auc"] = float("nan")
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, pos_probs))
        except Exception:
            metrics["pr_auc"] = float("nan")
        metrics["threshold"] = float(threshold) if threshold is not None else float("nan")
    return metrics, y_pred, probs


def tune_threshold_on_validation(y_val: np.ndarray, logits_val: np.ndarray, objective: str = "positive_f1") -> float:
    probs = torch.softmax(torch.tensor(logits_val), dim=1).numpy()[:, 1]
    thresholds = np.unique(np.quantile(probs, np.linspace(0.01, 0.99, 99)))
    best_t, best_score = 0.5, -1.0
    for t in thresholds:
        pred = (probs >= t).astype(int)
        if objective == "macro_f1":
            score = f1_score(y_val, pred, average="macro", zero_division=0)
        else:
            score = f1_score(y_val, pred, pos_label=1, zero_division=0)
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--split_col", required=True)
    ap.add_argument("--hidden_channels", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.35)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--use_amp", action="store_true")
    ap.add_argument("--tune_threshold", action="store_true")
    ap.add_argument("--threshold_objective", choices=["positive_f1", "macro_f1"], default="positive_f1")
    args = ap.parse_args()

    set_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    graph_dir = Path(args.graph_dir)

    data = torch.load(graph_dir / "text_hgt_heterodata.pt", map_location="cpu", weights_only=False)
    bench = pd.read_csv(graph_dir / "text_hgt_benchmark_table.csv", dtype=str, low_memory=False, encoding="utf-8-sig")
    bench["award_node_idx"] = pd.to_numeric(bench["award_node_idx"], errors="coerce").astype(int)

    y, train_mask, val_mask, test_mask, label_to_id, bench_used = prepare_labels_and_masks(
        bench, args.target, args.split_col, data["award"].num_nodes, args.seed
    )
    id_to_label = {v: k for k, v in label_to_id.items()}
    label_names = [id_to_label[i] for i in range(len(id_to_label))]

    device = pick_device(args.device)
    data = move_heterodata(data, device)
    y = y.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    in_dims = {ntype: int(data[ntype].x.size(1)) for ntype in data.node_types}
    model = TextHGT(
        metadata=data.metadata(),
        in_dims=in_dims,
        hidden_channels=args.hidden_channels,
        out_channels=len(label_names),
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_weights = compute_class_weights(y, train_mask, len(label_names), device)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and device.type == "cuda")

    best_state = None
    best_val = -1.0
    best_epoch = -1
    patience_left = args.patience
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.use_amp and device.type == "cuda"):
            logits = model(data.x_dict, data.edge_index_dict)
            loss = F.cross_entropy(logits[train_mask], y[train_mask], weight=class_weights)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        model.eval()
        with torch.no_grad():
            logits_eval = model(data.x_dict, data.edge_index_dict).detach().cpu().numpy()
        y_cpu = y.detach().cpu().numpy()
        val_idx = val_mask.detach().cpu().numpy().astype(bool)
        train_idx = train_mask.detach().cpu().numpy().astype(bool)
        if val_idx.sum() > 0:
            val_metrics, _, _ = evaluate_from_logits(y_cpu[val_idx], logits_eval[val_idx], label_names)
            val_score = val_metrics["macro_f1"]
        else:
            val_metrics, _, _ = evaluate_from_logits(y_cpu[train_idx], logits_eval[train_idx], label_names)
            val_score = val_metrics["macro_f1"]
        history.append({"epoch": epoch, "train_loss": float(loss.detach().cpu()), "val_macro_f1": float(val_score), **{f"val_{k}": v for k, v in val_metrics.items()}})
        if val_score > best_val:
            best_val = val_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(data.x_dict, data.edge_index_dict).detach().cpu().numpy()
    y_cpu = y.detach().cpu().numpy()
    masks = {
        "train": train_mask.detach().cpu().numpy().astype(bool),
        "validation": val_mask.detach().cpu().numpy().astype(bool),
        "test": test_mask.detach().cpu().numpy().astype(bool),
    }

    threshold = None
    if args.tune_threshold and len(label_names) == 2 and masks["validation"].sum() > 0:
        threshold = tune_threshold_on_validation(y_cpu[masks["validation"]], logits[masks["validation"]], args.threshold_objective)

    metric_rows = []
    pred_rows = []
    reports = {}
    cms = {}
    for split_name, mask in masks.items():
        if mask.sum() == 0:
            continue
        metrics, y_pred, probs = evaluate_from_logits(y_cpu[mask], logits[mask], label_names, threshold=threshold)
        row = {
            "model": "text_hgt",
            "target": args.target,
            "split_col": args.split_col,
            "eval_split": split_name,
            "n": int(mask.sum()),
            "best_epoch": int(best_epoch),
            "best_val_macro_f1": float(best_val),
            "seed": int(args.seed),
            **metrics,
        }
        metric_rows.append(row)
        true = y_cpu[mask]
        reports[split_name] = classification_report(true, y_pred, target_names=label_names, output_dict=True, zero_division=0)
        cms[split_name] = confusion_matrix(true, y_pred, labels=list(range(len(label_names)))).tolist()
        idxs = np.where(mask)[0]
        for local_i, node_idx in enumerate(idxs):
            d = {
                "award_node_idx": int(node_idx),
                "true_id": int(true[local_i]),
                "pred_id": int(y_pred[local_i]),
                "true_label": label_names[int(true[local_i])],
                "pred_label": label_names[int(y_pred[local_i])],
                "eval_split": split_name,
            }
            for ci, lab in enumerate(label_names):
                d[f"prob_{lab}"] = float(probs[local_i, ci])
            pred_rows.append(d)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(outdir / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pred_rows).to_csv(outdir / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(outdir / "training_history.csv", index=False, encoding="utf-8-sig")
    with open(outdir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)
    with open(outdir / "confusion_matrix.json", "w", encoding="utf-8") as f:
        json.dump({"labels": label_names, "matrices": cms}, f, indent=2)
    torch.save(best_state, outdir / "text_hgt_model.pt")
    summary = {
        "target": args.target,
        "split_col": args.split_col,
        "label_to_id": label_to_id,
        "device": str(device),
        "hidden_channels": args.hidden_channels,
        "num_layers": args.num_layers,
        "heads": args.heads,
        "dropout": args.dropout,
        "lr": args.lr,
        "epochs_ran": int(history[-1]["epoch"] if history else 0),
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val),
        "threshold": threshold,
        "n_train": int(train_mask.sum().item()),
        "n_validation": int(val_mask.sum().item()),
        "n_test": int(test_mask.sum().item()),
        "note": "TG4 text+HGT baseline. Award nodes use text embeddings with heterogeneous HGT message passing.",
    }
    with open(outdir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(metrics_df.to_string(index=False))
    print("Wrote", outdir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Train a structural-only GraphSAGE baseline on the MethodKG award graph.

This script expects the output of build_graphsage_award_graph.py. It trains on
benchmark award nodes only, but message passing can use the full unlabeled award
projection graph. It uses no title/abstract text and no candidate/annotation flags.
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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

try:
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.nn import SAGEConv
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "torch_geometric is required. Install PyTorch Geometric for your CUDA/PyTorch version. "
        "See README.md. Original import error: " + repr(exc)
    )


BINARY_TARGETS = {
    "target_integration_binary",
    "target_design_binary",
    "target_mmr_binary",
    "target_qual_binary",
    "target_quant_binary",
    "target_method_signal_binary",
}


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int, dropout: float):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(graph_dir: Path):
    data = torch.load(graph_dir / "graphsage_award_graph.pt", map_location="cpu")
    node_index = pd.read_csv(graph_dir / "graphsage_node_index.csv", low_memory=False, dtype=str, encoding="utf-8-sig")
    with open(graph_dir / "graphsage_label_maps.json", "r", encoding="utf-8") as f:
        label_maps = json.load(f)
    return data, node_index, label_maps


def encode_target(node_index: pd.DataFrame, target: str, label_maps: Dict[str, Dict[str, int]]) -> Tuple[np.ndarray, Dict[str, int]]:
    enc_col = target + "__encoded"
    if enc_col not in node_index.columns:
        raise ValueError(f"Encoded target column missing from node index: {enc_col}")
    y = pd.to_numeric(node_index[enc_col], errors="coerce").fillna(-1).astype(int).values
    label_map = label_maps.get(target, {"0": 0, "1": 1})
    return y, label_map


def make_masks(
    node_index: pd.DataFrame,
    y: np.ndarray,
    split_col: str,
    seed: int,
    internal_val_fraction: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if split_col not in node_index.columns:
        raise ValueError(f"Split column not found: {split_col}")

    split = node_index[split_col].fillna("unknown").astype(str).str.lower().values
    labeled = y >= 0
    train_idx = np.where(labeled & (split == "train"))[0]
    val_idx = np.where(labeled & ((split == "validation") | (split == "val")))[0]
    test_idx = np.where(labeled & (split == "test"))[0]

    if len(train_idx) == 0 or len(test_idx) == 0:
        counts = pd.Series(split[labeled]).value_counts().to_dict()
        raise ValueError(f"Split {split_col} must contain train and test labeled nodes. Counts: {counts}")

    if len(val_idx) == 0:
        # EDU->ENG transfer split has only train/test. Create an internal validation set from train.
        y_train = y[train_idx]
        stratify = y_train if len(np.unique(y_train)) > 1 and min(np.bincount(y_train.astype(int))) >= 2 else None
        tr, va = train_test_split(
            train_idx,
            test_size=internal_val_fraction,
            random_state=seed,
            stratify=stratify,
        )
        train_idx = np.array(tr, dtype=int)
        val_idx = np.array(va, dtype=int)

    return train_idx, val_idx, test_idx


def compute_class_weights(y_train: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y_train.astype(int), minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def full_forward(model: torch.nn.Module, data, device: torch.device, use_amp: bool = False) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                return model(x, edge_index).detach().cpu()
        return model(x, edge_index).detach().cpu()


def tune_binary_threshold(y_true: np.ndarray, prob: np.ndarray, metric: str = "positive_f1") -> float:
    best_t = 0.5
    best_score = -1.0
    for t in np.linspace(0.05, 0.95, 181):
        pred = (prob >= t).astype(int)
        if metric == "macro_f1":
            score = f1_score(y_true, pred, average="macro", zero_division=0)
        else:
            score = f1_score(y_true, pred, pos_label=1, zero_division=0)
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, prob_pos: np.ndarray) -> Dict[str, float]:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "positive_f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
    }
    if len(np.unique(y_true)) == 2:
        try:
            out["roc_auc"] = roc_auc_score(y_true, prob_pos)
        except Exception:
            out["roc_auc"] = np.nan
        try:
            out["pr_auc"] = average_precision_score(y_true, prob_pos)
        except Exception:
            out["pr_auc"] = np.nan
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan
    return out


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "positive_f1": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "roc_auc": np.nan,
        "pr_auc": np.nan,
    }


def choose_best_metric(target: str) -> str:
    if target in BINARY_TARGETS:
        return "val_positive_f1"
    return "val_macro_f1"


def evaluate_logits(
    logits: torch.Tensor,
    y: np.ndarray,
    idx: np.ndarray,
    target: str,
    threshold: float,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    y_true = y[idx].astype(int)
    sub = logits[idx]
    if target in BINARY_TARGETS:
        prob = torch.softmax(sub, dim=1).numpy()[:, 1]
        y_pred = (prob >= threshold).astype(int)
        metrics = binary_metrics(y_true, y_pred, prob)
        return metrics, y_pred, prob
    probs = torch.softmax(sub, dim=1).numpy()
    y_pred = probs.argmax(axis=1)
    metrics = multiclass_metrics(y_true, y_pred)
    return metrics, y_pred, probs.max(axis=1)


def prefix_metrics(prefix: str, d: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in d.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_dir", required=True, help="Directory created by build_graphsage_award_graph.py")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--split_col", required=True)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no_class_weights", action="store_true")
    parser.add_argument("--tune_threshold", action="store_true")
    parser.add_argument("--threshold_metric", default="positive_f1", choices=["positive_f1", "macro_f1"])
    parser.add_argument("--use_amp", action="store_true", help="Use mixed precision on CUDA.")
    args = parser.parse_args()

    set_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data, node_index, label_maps = load_data(Path(args.graph_dir))
    y, label_map = encode_target(node_index, args.target, label_maps)
    train_idx, val_idx, test_idx = make_masks(node_index, y, args.split_col, args.seed)

    num_classes = len(set(y[train_idx].astype(int)) | set(y[val_idx].astype(int)) | set(y[test_idx].astype(int)))
    if args.target in BINARY_TARGETS:
        num_classes = 2
    else:
        num_classes = max(int(y[y >= 0].max()) + 1, num_classes)

    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GraphSAGE(
        in_channels=int(data.x.shape[1]),
        hidden_channels=args.hidden_channels,
        out_channels=num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)
    train_t = torch.tensor(train_idx, dtype=torch.long, device=device)

    class_weights = None if args.no_class_weights else compute_class_weights(y[train_idx], num_classes, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.use_amp and device.type == "cuda"))

    best_metric_name = choose_best_metric(args.target)
    best_metric = -1.0
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(args.use_amp and device.type == "cuda")):
            logits = model(x, edge_index)
            loss = F.cross_entropy(logits[train_t], y_tensor[train_t], weight=class_weights)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            logits_cpu = logits.detach().cpu()
        val_threshold = 0.5
        if args.tune_threshold and args.target in BINARY_TARGETS:
            val_prob = torch.softmax(logits_cpu[val_idx], dim=1).numpy()[:, 1]
            val_threshold = tune_binary_threshold(y[val_idx].astype(int), val_prob, args.threshold_metric)
        train_metrics, _, _ = evaluate_logits(logits_cpu, y, train_idx, args.target, val_threshold)
        val_metrics, _, _ = evaluate_logits(logits_cpu, y, val_idx, args.target, val_threshold)

        row = {"epoch": epoch, "loss": float(loss.detach().cpu().item()), "threshold": val_threshold}
        row.update(prefix_metrics("train", train_metrics))
        row.update(prefix_metrics("val", val_metrics))
        history.append(row)

        current = row.get(best_metric_name, row.get("val_macro_f1", 0.0))
        if current > best_metric + 1e-8:
            best_metric = current
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 25 == 0 or epoch == 1:
            print(f"epoch={epoch} loss={row['loss']:.4f} val_macro_f1={row.get('val_macro_f1', np.nan):.4f} val_positive_f1={row.get('val_positive_f1', np.nan):.4f}")
        if epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch {best_epoch} {best_metric_name}={best_metric:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    logits = full_forward(model, data, device, use_amp=args.use_amp)

    threshold = 0.5
    if args.tune_threshold and args.target in BINARY_TARGETS:
        val_prob = torch.softmax(logits[val_idx], dim=1).numpy()[:, 1]
        threshold = tune_binary_threshold(y[val_idx].astype(int), val_prob, args.threshold_metric)

    train_metrics, _, _ = evaluate_logits(logits, y, train_idx, args.target, threshold)
    val_metrics, _, _ = evaluate_logits(logits, y, val_idx, args.target, threshold)
    test_metrics, test_pred, test_score = evaluate_logits(logits, y, test_idx, args.target, threshold)

    metrics = {
        "model": "graphsage_structural",
        "target": args.target,
        "split_col": args.split_col,
        "best_epoch": best_epoch,
        "threshold": threshold,
        "num_train": len(train_idx),
        "num_validation": len(val_idx),
        "num_test": len(test_idx),
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.shape[1]),
        "feature_dim": int(data.x.shape[1]),
        "hidden_channels": args.hidden_channels,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "device": str(device),
    }
    metrics.update(prefix_metrics("train", train_metrics))
    metrics.update(prefix_metrics("validation", val_metrics))
    metrics.update(prefix_metrics("test", test_metrics))

    pd.DataFrame([metrics]).to_csv(outdir / "metrics.csv", index=False)
    pd.DataFrame(history).to_csv(outdir / "training_history.csv", index=False)

    pred_df = node_index.iloc[test_idx].copy()
    pred_df["y_true_encoded"] = y[test_idx].astype(int)
    pred_df["y_pred_encoded"] = test_pred.astype(int)
    pred_df["score"] = test_score
    # Add decoded labels for multiclass.
    if args.target not in BINARY_TARGETS:
        inv = {v: k for k, v in label_map.items()}
        pred_df["y_true_label"] = pred_df["y_true_encoded"].map(inv)
        pred_df["y_pred_label"] = pred_df["y_pred_encoded"].map(inv)
    pred_df.to_csv(outdir / "predictions.csv", index=False, encoding="utf-8-sig")

    cm = confusion_matrix(y[test_idx].astype(int), test_pred.astype(int))
    pd.DataFrame(cm).to_csv(outdir / "confusion_matrix.csv", index=False)

    report = classification_report(y[test_idx].astype(int), test_pred.astype(int), output_dict=True, zero_division=0)
    with open(outdir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    torch.save(model.state_dict(), outdir / "graphsage_model.pt")
    with open(outdir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Wrote GraphSAGE results to", outdir.resolve())
    print(pd.DataFrame([metrics]).T.tail(20))


if __name__ == "__main__":
    main()

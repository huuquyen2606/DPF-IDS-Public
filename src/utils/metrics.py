"""Metric helpers for client and round evaluation."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

def metrics_from_confusion_matrix(cm: np.ndarray) -> Dict[str, float]:
    cm = np.asarray(cm, dtype=np.float64)
    total = cm.sum()
    if total <= 0:
        return {
            "accuracy": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_macro": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
        }

    tp = np.diag(cm)
    support = cm.sum(axis=1)
    pred_count = cm.sum(axis=0)

    precision_i = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    recall_i = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision_i + recall_i
    f1_i = np.divide(2 * precision_i * recall_i, denom, out=np.zeros_like(tp), where=denom > 0)

    weights = support / total
    return {
        "accuracy": float(tp.sum() / total),
        "precision": float(np.sum(precision_i * weights)),
        "recall": float(np.sum(recall_i * weights)),
        "f1": float(np.sum(f1_i * weights)),
        "precision_macro": float(np.mean(precision_i)),
        "recall_macro": float(np.mean(recall_i)),
        "f1_macro": float(np.mean(f1_i)),
    }


def save_confusion_matrix_csv(cm: np.ndarray, client_id: int, tag: str, cm_dir: str, num_classes: int):
    os.makedirs(cm_dir, exist_ok=True)
    labels = list(range(num_classes))
    path = os.path.join(cm_dir, f"confusion_matrix_{tag}_client_{int(client_id):04d}.csv")
    df = pd.DataFrame(cm, index=labels, columns=labels)
    df.index.name = "true_label"
    df.columns.name = "pred_label"
    df.to_csv(path)
    return path


def load_existing_client_metrics(csv_path: str, tag: str) -> Tuple[Dict[int, Dict[str, float]], List[Dict[str, Any]]]:
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return {}, []
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}, []
    if df.empty:
        return {}, []

    if "tag" in df.columns:
        df = df[df["tag"].astype(str) == str(tag)]
    if "scope" in df.columns:
        df = df[df["scope"].astype(str) == "client"]

    loaded = {}
    rows = []
    metric_cols = ["accuracy", "f1", "precision", "recall", "f1_macro", "precision_macro", "recall_macro"]
    for _, row in df.iterrows():
        try:
            cid = int(row["client_id"])
        except Exception:
            continue
        metrics = {c: float(row.get(c, 0.0)) for c in metric_cols}
        loaded[cid] = metrics
        rows.append({"tag": tag, "scope": "client", "client_id": cid, **metrics})
    return loaded, rows


def append_or_write_metrics(csv_path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def compute_average_metrics(per_client_metrics: Dict[int, Dict[str, float]]) -> Dict[str, float]:
    if not per_client_metrics:
        raise RuntimeError("No client metrics available to compute average.")
    metric_cols = ["accuracy", "f1", "precision", "recall", "f1_macro", "precision_macro", "recall_macro"]
    return {c: float(np.mean([m[c] for m in per_client_metrics.values()])) for c in metric_cols}

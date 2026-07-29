from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PAPER_METRICS = ("AUROC", "AUPRC", "Accuracy", "Precision", "Recall")
EXTENDED_METRICS = PAPER_METRICS + ("F1", "Specificity")


def _as_vectors(
    labels: Iterable[int | float],
    probabilities: Iterable[int | float],
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(list(labels), dtype=np.int64).reshape(-1)
    y_score = np.asarray(list(probabilities), dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        raise ValueError("labels cannot be empty")
    if y_true.shape != y_score.shape:
        raise ValueError("labels and probabilities must have the same shape")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("labels must be binary")
    if not np.isfinite(y_score).all():
        raise ValueError("probabilities contain non-finite values")
    if ((y_score < 0.0) | (y_score > 1.0)).any():
        raise ValueError("probabilities must be in [0, 1]")
    return y_true, y_score


def classification_metrics(
    labels: Iterable[int | float],
    probabilities: Iterable[int | float],
    threshold: float = 0.5,
) -> dict[str, float]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    y_true, y_score = _as_vectors(labels, probabilities)
    if np.unique(y_true).size < 2:
        raise ValueError("AUROC and AUPRC require both classes")
    y_pred = (y_score >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / max(tn + fp, 1)
    return {
        "AUROC": float(roc_auc_score(y_true, y_score)),
        "AUPRC": float(average_precision_score(y_true, y_score)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Specificity": float(specificity),
        "TP": float(tp),
        "TN": float(tn),
        "FP": float(fp),
        "FN": float(fn),
    }


def summarize_metrics(
    runs: Iterable[Mapping[str, float]],
    names: Iterable[str] = PAPER_METRICS,
) -> dict[str, dict[str, float]]:
    records = list(runs)
    if not records:
        raise ValueError("runs cannot be empty")
    summary: dict[str, dict[str, float]] = {}
    for name in names:
        values = np.asarray([float(record[name]) for record in records], dtype=float)
        summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary

"""Binary-classification metrics used by training and evaluation scripts."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
from sklearn import metrics


def classification_metrics(
    labels: Iterable[int],
    predictions: Iterable[int],
    valid_probabilities: Iterable[float],
) -> Dict[str, object]:
    """Calculate label-based metrics for invalid/valid classification."""
    y_true = np.asarray(list(labels), dtype=int)
    y_pred = np.asarray(list(predictions), dtype=int)
    y_score = np.asarray(list(valid_probabilities), dtype=float)
    matrix = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    result: Dict[str, object] = {
        "n": int(y_true.size),
        "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
        "precision": float(metrics.precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(metrics.recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(metrics.f1_score(y_true, y_pred, zero_division=0)),
        "cohen_kappa": float(metrics.cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": matrix.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if np.unique(y_true).size == 2:
        result["roc_auc"] = float(metrics.roc_auc_score(y_true, y_score))
        result["average_precision"] = float(
            metrics.average_precision_score(y_true, y_score)
        )
    else:
        result["roc_auc"] = None
        result["average_precision"] = None
    return result

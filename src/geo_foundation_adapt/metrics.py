"""Classification metrics that stay defined when a class is missing."""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[int(t), int(p)] += 1
    return cm


def accuracy_from_cm(cm: np.ndarray) -> float:
    total = int(cm.sum())
    if total == 0:
        return 0.0
    return float(np.trace(cm) / total)


def macro_f1_from_cm(cm: np.ndarray) -> float:
    """Unweighted mean of per-class F1. A class with no precision and no recall scores 0."""
    f1s: list[float] = []
    for c in range(cm.shape[0]):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)
        if tp == 0.0 and fp == 0.0 and fn == 0.0:
            f1s.append(0.0)
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall))
    return float(sum(f1s) / len(f1s))


def scores(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, float | list[list[int]]]:
    cm = confusion_matrix(y_true, y_pred, n_classes)
    return {
        "accuracy": accuracy_from_cm(cm),
        "macro_f1": macro_f1_from_cm(cm),
        "confusion_matrix": cm.tolist(),
    }


def mean_std(values: list[float]) -> tuple[float, float | None]:
    """Mean and sample standard deviation (divisor n-1). Std is None when n < 2."""
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean()) if len(arr) else 0.0
    if len(arr) < 2:
        return mean, None
    return mean, float(arr.std(ddof=1))


def format_mean_std(mean: float, std: float | None, digits: int = 3) -> str:
    if std is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"

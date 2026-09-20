"""Regression metrics for molecular property prediction."""

from __future__ import annotations

from typing import Dict

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination.

    Negative when the model is worse than predicting the mean -- which happens
    routinely on scaffold splits and is worth seeing rather than hiding behind
    a correlation coefficient.
    """
    residual = np.sum((y_true - y_pred) ** 2)
    total = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - residual / total) if total > 1e-12 else float("nan")


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Rank correlation.

    Often the metric that matters in practice: virtual screening cares about
    ranking candidates correctly, not about absolute predicted values.
    """
    from scipy import stats

    if len(y_true) < 3:
        return float("nan")
    return float(stats.spearmanr(y_true, y_pred).statistic)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if len(y_true) == 0:
        # Every metric below reduces over the array (mean, variance, ...),
        # so an empty batch would otherwise fall through to numpy and raise
        # a wall of "Mean of empty slice" / "divide by zero" RuntimeWarnings
        # instead of the same nan-filled report a degenerate non-empty batch
        # already produces.
        return {
            "rmse": float("nan"), "mae": float("nan"), "r2": float("nan"),
            "pearson": float("nan"), "spearman": float("nan"), "n": 0,
        }
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "pearson": pearson(y_true, y_pred),
        "spearman": spearman(y_true, y_pred),
        "n": int(len(y_true)),
    }


def format_report(results: Dict[str, float], name: str = "model") -> str:
    lines = [f"{name}:"]
    for key in ("rmse", "mae", "r2", "pearson", "spearman"):
        if key in results:
            lines.append(f"  {key.upper():<9} {results[key]:.4f}")
    return "\n".join(lines)

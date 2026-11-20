"""Cost-matrix-driven decision threshold optimization for DelhiCold Logistics."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from src.config import COST_MATRIX_INR


def theoretical_bayes_threshold(cost_matrix: Dict[str, float] | None = None) -> float:
    """Compute theoretical Bayes-optimal probability threshold from the cost matrix.

    Expected cost of flagging late (alert=1) at probability p:
        E[C | alert=1] = p * C_TP + (1 - p) * C_FP
    Expected cost of not flagging (alert=0) at probability p:
        E[C | alert=0] = p * C_FN + (1 - p) * C_TN

    Alerting is cheaper when:
        p * (C_FN - C_TP + C_FP - C_TN) >= C_FP - C_TN
        p* = (C_FP - C_TN) / ((C_FP - C_TN) + (C_FN - C_TP))
    """
    cm = cost_matrix or COST_MATRIX_INR
    num = cm["cost_fp"] - cm["cost_tn"]
    denom = num + (cm["cost_fn"] - cm["cost_tp"])
    if denom <= 0:
        return 0.5
    return float(np.clip(num / denom, 0.01, 0.99))


def evaluate_threshold_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    cost_matrix: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """Compute confusion counts, classification metrics, and INR business cost at a threshold."""
    cm = cost_matrix or COST_MATRIX_INR
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_prob, dtype=float) >= threshold).astype(int)

    tp = int(((y_true_arr == 1) & (y_pred == 1)).sum())
    fp = int(((y_true_arr == 0) & (y_pred == 1)).sum())
    fn = int(((y_true_arr == 1) & (y_pred == 0)).sum())
    tn = int(((y_true_arr == 0) & (y_pred == 0)).sum())

    total_cost = (
        tp * cm["cost_tp"]
        + fp * cm["cost_fp"]
        + fn * cm["cost_fn"]
        + tn * cm["cost_tn"]
    )
    n = max(len(y_true_arr), 1)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Baseline "do nothing" cost (flag 0 shipments -> all positives become FN)
    do_nothing_cost = int((y_true_arr == 1).sum()) * cm["cost_fn"]

    return {
        "threshold": round(float(threshold), 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "total_cost_inr": round(float(total_cost), 2),
        "cost_per_shipment_inr": round(float(total_cost / n), 2),
        "do_nothing_cost_inr": round(float(do_nothing_cost), 2),
        "savings_vs_do_nothing_inr": round(float(do_nothing_cost - total_cost), 2),
        "savings_pct_vs_do_nothing": round(
            float((do_nothing_cost - total_cost) / do_nothing_cost * 100.0) if do_nothing_cost > 0 else 0.0,
            2,
        ),
    }


def optimize_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_matrix: Dict[str, float] | None = None,
    grid_start: float = 0.05,
    grid_end: float = 0.95,
    grid_step: float = 0.01,
) -> Dict[str, Any]:
    """Grid-search probability thresholds on validation data to minimize expected INR cost."""
    cm = cost_matrix or COST_MATRIX_INR
    thresholds = np.arange(grid_start, grid_end + 1e-9, grid_step)
    curve: List[Dict[str, float]] = [
        evaluate_threshold_cost(y_true, y_prob, float(t), cm) for t in thresholds
    ]
    best = min(curve, key=lambda r: r["total_cost_inr"])
    default_05 = evaluate_threshold_cost(y_true, y_prob, 0.50, cm)

    return {
        "optimal_threshold": best["threshold"],
        "theoretical_bayes_threshold": round(theoretical_bayes_threshold(cm), 4),
        "optimal_metrics": best,
        "default_050_metrics": default_05,
        "savings_vs_default_050_inr": round(default_05["total_cost_inr"] - best["total_cost_inr"], 2),
        "cost_matrix_inr": cm,
        "threshold_curve": curve,
    }

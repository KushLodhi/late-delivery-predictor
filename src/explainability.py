"""Global and local SHAP explainability utilities for the DelhiCold pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _extract_shap_positive_class(shap_values: Any) -> np.ndarray:
    """Normalize SHAP output across tree/linear explainers and SHAP versions to 2D (n, f)."""
    if isinstance(shap_values, list):
        # Binary classifier returning [class0, class1]
        arr = np.asarray(shap_values[-1])
    else:
        arr = np.asarray(shap_values)
    if arr.ndim == 3:
        # Shape (n_samples, n_features, 2) -> positive class index 1
        arr = arr[:, :, 1]
    return arr


def _build_explainer(model_pipeline: Any, background_transformed: np.ndarray | None = None) -> Any:
    """Create the appropriate fast SHAP explainer for the fitted classifier."""
    import shap

    clf = model_pipeline.named_steps["classifier"]
    cls_name = type(clf).__name__.lower()

    if any(k in cls_name for k in ("lgbm", "xgb", "forest", "tree", "boost")):
        return shap.TreeExplainer(clf)
    if background_transformed is None:
        raise ValueError("LinearExplainer requires background_transformed data.")
    return shap.LinearExplainer(clf, background_transformed)


def compute_global_shap(
    model_pipeline: Any,
    X_sample: pd.DataFrame,
    save_json_path: Path | None = None,
    save_plot_path: Path | None = None,
    top_k: int = 15,
) -> Dict[str, Any]:
    """Compute global mean(|SHAP|) feature importance and optionally save bar plot & JSON."""
    preprocessor = model_pipeline.named_steps["preprocessor"]
    X_trans = preprocessor.transform(X_sample)
    feature_names = list(preprocessor.get_feature_names_out())

    explainer = _build_explainer(model_pipeline, background_transformed=X_trans[:200])
    raw_shap = explainer.shap_values(X_trans)
    shap_matrix = _extract_shap_positive_class(raw_shap)

    mean_abs = np.mean(np.abs(shap_matrix), axis=0)
    order = np.argsort(mean_abs)[::-1]

    ranking = [
        {
            "feature": feature_names[i],
            "mean_abs_shap": round(float(mean_abs[i]), 5),
        }
        for i in order[:top_k]
    ]

    result = {
        "n_explained_rows": int(len(X_sample)),
        "top_features": ranking,
    }

    if save_json_path is not None:
        save_json_path.parent.mkdir(parents=True, exist_ok=True)
        save_json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if save_plot_path is not None:
        save_plot_path.parent.mkdir(parents=True, exist_ok=True)
        names = [r["feature"] for r in ranking][::-1]
        vals = [r["mean_abs_shap"] for r in ranking][::-1]
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        ax.barh(names, vals, color="#0B5CAD")
        ax.set_xlabel("Mean |SHAP value| (impact on late-delivery log-odds)")
        ax.set_title("Global SHAP Feature Importance — DelhiCold Late-Delivery Model")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(save_plot_path, dpi=150)
        plt.close(fig)

    return result


def explain_single_shipment(
    model_pipeline: Any,
    X_row: pd.DataFrame,
    explainer: Any | None = None,
    top_k: int = 8,
) -> Dict[str, Any]:
    """Return individual SHAP feature contributions for a single shipment row."""
    preprocessor = model_pipeline.named_steps["preprocessor"]
    clf = model_pipeline.named_steps["classifier"]
    X_trans = preprocessor.transform(X_row)
    feature_names = list(preprocessor.get_feature_names_out())

    if explainer is None and hasattr(clf, "coef_"):
        # Exact linear SHAP relative to training mean (scaled numerics have mean 0)
        shap_vec = np.asarray(clf.coef_).ravel() * np.asarray(X_trans[0]).ravel()
        base_value = float(np.asarray(clf.intercept_).ravel()[-1])
    else:
        if explainer is None:
            explainer = _build_explainer(model_pipeline, background_transformed=np.zeros_like(X_trans))
        raw_shap = explainer.shap_values(X_trans)
        shap_vec = _extract_shap_positive_class(raw_shap)[0]
        expected_val = explainer.expected_value
        if isinstance(expected_val, (list, np.ndarray)):
            base_value = float(np.asarray(expected_val).ravel()[-1])
        else:
            base_value = float(expected_val)

    order = np.argsort(np.abs(shap_vec))[::-1][:top_k]
    contributions: List[Dict[str, Any]] = [
        {
            "feature": feature_names[i],
            "shap_value": round(float(shap_vec[i]), 5),
            "direction": "increases_late_risk" if shap_vec[i] > 0 else "decreases_late_risk",
        }
        for i in order
    ]

    return {
        "base_value": round(base_value, 5),
        "top_contributions": contributions,
    }

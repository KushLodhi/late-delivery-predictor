"""Feature drift monitoring using Population Stability Index (PSI) and KS tests.

Compares live incoming shipment features against reference distributions captured
during training (`models/reference_stats.json`) and logs structured warnings when
key operational features shift beyond thresholds.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES

logger = logging.getLogger("delhicold.drift")

# Industry-standard PSI thresholds
PSI_MODERATE_THRESHOLD = 0.10
PSI_SEVERE_THRESHOLD = 0.20
KS_PVALUE_THRESHOLD = 0.01

KEY_DRIFT_NUMERIC_FEATURES = [
    "distance_km",
    "planned_transit_hours",
    "warehouse_dwell_hours",
    "temp_deviation_c",
    "route_congestion_index",
    "vehicle_age_years",
    "payload_weight_kg",
    "dwell_to_transit_ratio",
]

KEY_DRIFT_CATEGORICAL_FEATURES = [
    "origin_hub_id",
    "dest_hub_id",
    "cargo_type",
    "weather_condition",
]


def compute_reference_stats(train_df: pd.DataFrame, n_bins: int = 10) -> Dict[str, Any]:
    """Capture quantile bin edges, proportions, and summary stats from training data."""
    numeric_stats: Dict[str, Any] = {}
    for col in KEY_DRIFT_NUMERIC_FEATURES:
        if col not in train_df.columns:
            continue
        vals = train_df[col].dropna().astype(float).values
        if len(vals) == 0:
            continue
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.quantile(vals, quantiles))
        if len(edges) < 3:
            edges = np.linspace(float(vals.min()), float(vals.max()) + 1e-6, 4)
        edges[0] = -np.inf
        edges[-1] = np.inf
        counts, _ = np.histogram(vals, bins=edges)
        props = np.clip(counts / max(len(vals), 1), 1e-6, 1.0)
        # Keep a deterministic subsample (up to 2,000 values) for two-sample KS test
        rng = np.random.default_rng(42)
        sample_vals = rng.choice(vals, size=min(len(vals), 2000), replace=False).round(4).tolist()
        numeric_stats[col] = {
            "bin_edges": [None if np.isinf(e) else round(float(e), 6) for e in edges],
            "expected_props": [round(float(p), 6) for p in props],
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "reference_sample": sample_vals,
        }

    categorical_stats: Dict[str, Any] = {}
    for col in KEY_DRIFT_CATEGORICAL_FEATURES:
        if col not in train_df.columns:
            continue
        vc = train_df[col].fillna("MISSING").astype(str).value_counts(normalize=True)
        categorical_stats[col] = {k: round(float(v), 6) for k, v in vc.items()}

    return {
        "n_reference_rows": int(len(train_df)),
        "psi_moderate_threshold": PSI_MODERATE_THRESHOLD,
        "psi_severe_threshold": PSI_SEVERE_THRESHOLD,
        "ks_pvalue_threshold": KS_PVALUE_THRESHOLD,
        "numeric": numeric_stats,
        "categorical": categorical_stats,
    }


def save_reference_stats(ref_stats: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ref_stats, indent=2), encoding="utf-8")


def load_reference_stats(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _compute_psi(expected_props: np.ndarray, actual_props: np.ndarray) -> float:
    eps = 1e-6
    exp = np.clip(expected_props, eps, 1.0)
    act = np.clip(actual_props, eps, 1.0)
    return float(np.sum((act - exp) * np.log(act / exp)))


def detect_drift(
    current_df: pd.DataFrame,
    ref_stats: Dict[str, Any],
    log_warnings: bool = True,
) -> Dict[str, Any]:
    """Compare incoming feature DataFrame against training reference stats.

    Returns per-feature PSI/KS metrics, drift severity status, and alert list.
    """
    n_current = len(current_df)
    feature_reports: Dict[str, Dict[str, Any]] = {}
    drifted_features: List[str] = []
    moderate_features: List[str] = []

    if n_current == 0:
        return {
            "n_current_rows": 0,
            "overall_status": "INSUFFICIENT_DATA",
            "drifted_features": [],
            "moderate_features": [],
            "features": {},
        }

    # Numeric feature drift (PSI + 2-sample KS test)
    for col, spec in ref_stats.get("numeric", {}).items():
        if col not in current_df.columns:
            continue
        vals = current_df[col].dropna().astype(float).values
        if len(vals) == 0:
            continue

        edges = np.array(
            [-np.inf if (i == 0 and e is None) else (np.inf if e is None else float(e)) for i, e in enumerate(spec["bin_edges"])]
        )
        counts, _ = np.histogram(vals, bins=edges)
        actual_props = counts / max(len(vals), 1)
        expected_props = np.array(spec["expected_props"], dtype=float)
        psi = _compute_psi(expected_props, actual_props)

        ks_stat, ks_pval = stats.ks_2samp(spec["reference_sample"], vals)

        if psi >= PSI_SEVERE_THRESHOLD or (ks_pval < KS_PVALUE_THRESHOLD and psi >= PSI_MODERATE_THRESHOLD):
            status = "DRIFT_DETECTED"
            drifted_features.append(col)
            if log_warnings:
                logger.warning(
                    "DATA DRIFT WARNING | feature=%s | PSI=%.4f | KS_stat=%.4f | KS_p=%.4g | "
                    "ref_mean=%.3f | cur_mean=%.3f",
                    col,
                    psi,
                    ks_stat,
                    ks_pval,
                    spec["mean"],
                    float(np.mean(vals)),
                )
        elif psi >= PSI_MODERATE_THRESHOLD:
            status = "MODERATE_SHIFT"
            moderate_features.append(col)
        else:
            status = "STABLE"

        feature_reports[col] = {
            "type": "numeric",
            "psi": round(float(psi), 4),
            "ks_stat": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_pval), 6),
            "ref_mean": spec["mean"],
            "current_mean": round(float(np.mean(vals)), 4),
            "status": status,
        }

    # Categorical feature drift (PSI over category proportions)
    for col, expected_map in ref_stats.get("categorical", {}).items():
        if col not in current_df.columns:
            continue
        actual_vc = current_df[col].fillna("MISSING").astype(str).value_counts(normalize=True).to_dict()
        all_cats = sorted(set(expected_map.keys()) | set(actual_vc.keys()))
        exp_arr = np.array([expected_map.get(c, 1e-6) for c in all_cats], dtype=float)
        act_arr = np.array([actual_vc.get(c, 1e-6) for c in all_cats], dtype=float)
        psi = _compute_psi(exp_arr, act_arr)

        if psi >= PSI_SEVERE_THRESHOLD:
            status = "DRIFT_DETECTED"
            drifted_features.append(col)
            if log_warnings:
                logger.warning("CATEGORICAL DRIFT WARNING | feature=%s | PSI=%.4f", col, psi)
        elif psi >= PSI_MODERATE_THRESHOLD:
            status = "MODERATE_SHIFT"
            moderate_features.append(col)
        else:
            status = "STABLE"

        feature_reports[col] = {
            "type": "categorical",
            "psi": round(float(psi), 4),
            "status": status,
        }

    if drifted_features:
        overall = "DRIFT_WARNING"
    elif moderate_features:
        overall = "MONITOR"
    else:
        overall = "HEALTHY"

    return {
        "n_current_rows": int(n_current),
        "overall_status": overall,
        "drifted_features": drifted_features,
        "moderate_features": moderate_features,
        "features": feature_reports,
    }

"""End-to-end training, model comparison, RandomizedSearchCV tuning, thresholding & SHAP script.

Usage:
    python -m src.train --rows 100000
    python -m src.train --rows 1500000   # Full 1.5M-row benchmark
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

from src.config import (
    COST_MATRIX_INR,
    DATA_DIR,
    METADATA_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    REFERENCE_STATS_PATH,
    SHAP_GLOBAL_PATH,
)
from src.data_generator import generate_coldchain_dataset
from src.drift import compute_reference_stats, save_reference_stats
from src.explainability import compute_global_shap
from src.pipeline import build_model_pipeline
from src.preprocessing import (
    chronological_split,
    clean_and_engineer_features,
    get_feature_columns,
)
from src.threshold import evaluate_threshold_cost, optimize_threshold


def _eval_probabilities(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "f1_at_050": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision_at_050": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall_at_050": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
    }


def _tune_best_model(
    best_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    scale_pos_weight: float,
    n_iter: int = 12,
) -> tuple[Any, Dict[str, Any]]:
    """Run RandomizedSearchCV with TimeSeriesSplit on the winning model family."""
    tscv = TimeSeriesSplit(n_splits=3)

    if best_name == "lightgbm":
        base = build_model_pipeline(
            LGBMClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            ),
            scale_numeric=False,
        )
        param_dist = {
            "classifier__n_estimators": [150, 250, 350, 450],
            "classifier__learning_rate": [0.03, 0.05, 0.08, 0.12],
            "classifier__num_leaves": [31, 47, 63, 95],
            "classifier__min_child_samples": [20, 40, 80],
            "classifier__subsample": [0.75, 0.85, 1.0],
            "classifier__colsample_bytree": [0.75, 0.85, 1.0],
        }
    elif best_name == "xgboost":
        base = build_model_pipeline(
            XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric="logloss",
                tree_method="hist",
            ),
            scale_numeric=False,
        )
        param_dist = {
            "classifier__n_estimators": [150, 250, 350],
            "classifier__learning_rate": [0.03, 0.06, 0.10],
            "classifier__max_depth": [4, 6, 8],
            "classifier__subsample": [0.75, 0.85, 1.0],
            "classifier__colsample_bytree": [0.75, 0.85, 1.0],
            "classifier__min_child_weight": [1, 5, 10],
        }
    elif best_name == "random_forest":
        base = build_model_pipeline(
            RandomForestClassifier(
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            scale_numeric=False,
        )
        param_dist = {
            "classifier__n_estimators": [120, 200, 300],
            "classifier__max_depth": [10, 14, 18, None],
            "classifier__min_samples_leaf": [5, 15, 30],
            "classifier__max_features": ["sqrt", "log2", 0.5],
        }
    else:
        base = build_model_pipeline(
            LogisticRegression(
                class_weight="balanced",
                max_iter=600,
                random_state=RANDOM_STATE,
            ),
            scale_numeric=True,
        )
        param_dist = {
            "classifier__C": [0.05, 0.2, 0.5, 1.0, 2.0, 5.0],
        }

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=min(n_iter, 12),
        scoring="average_precision",
        cv=tscv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    best_params = {k.replace("classifier__", ""): v for k, v in search.best_params_.items()}
    return search.best_estimator_, best_params


def run_training(n_rows: int = 100_000, tune_iter: int = 10, seed: int = RANDOM_STATE) -> Dict[str, Any]:
    """Execute full reproducible training pipeline and persist all artifacts."""
    t0 = time.perf_counter()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/7] Generating {n_rows:,} Delhi cold-chain shipment records (12 hubs)...")
    raw_df, scans_df = generate_coldchain_dataset(n_rows=n_rows, seed=seed)

    print("[2/7] Cleaning time zones, out-of-order scans, hub renames & engineering features...")
    clean_df = clean_and_engineer_features(raw_df, scans_df)
    feature_cols = get_feature_columns()

    print("[3/7] Performing strict chronological train (70%) / val (15%) / test (15%) split...")
    train_df, val_df, test_df = chronological_split(clean_df)
    X_train, y_train = train_df[feature_cols], train_df["is_late"].values
    X_val, y_val = val_df[feature_cols], val_df["is_late"].values
    X_test, y_test = test_df[feature_cols], test_df["is_late"].values

    neg_cnt = int((y_train == 0).sum())
    pos_cnt = max(int((y_train == 1).sum()), 1)
    scale_pos_weight = round(neg_cnt / pos_cnt, 3)

    # Baseline model (stratified / prior frequency baseline)
    baseline = build_model_pipeline(DummyClassifier(strategy="stratified", random_state=seed), scale_numeric=False)
    baseline.fit(X_train, y_train)
    baseline_val_prob = baseline.predict_proba(X_val)[:, 1]
    baseline_test_prob = baseline.predict_proba(X_test)[:, 1]
    baseline_val_metrics = _eval_probabilities(y_val, baseline_val_prob)
    baseline_test_metrics = _eval_probabilities(y_test, baseline_test_prob)

    print("[4/7] Training 4 candidate models with class-imbalance handling...")
    candidates = {
        "logistic_regression": build_model_pipeline(
            LogisticRegression(class_weight="balanced", max_iter=500, random_state=seed),
            scale_numeric=True,
        ),
        "random_forest": build_model_pipeline(
            RandomForestClassifier(
                n_estimators=150,
                max_depth=14,
                min_samples_leaf=15,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            ),
            scale_numeric=False,
        ),
        "xgboost": build_model_pipeline(
            XGBClassifier(
                n_estimators=220,
                learning_rate=0.06,
                max_depth=6,
                subsample=0.85,
                colsample_bytree=0.85,
                scale_pos_weight=scale_pos_weight,
                random_state=seed,
                n_jobs=-1,
                eval_metric="logloss",
                tree_method="hist",
            ),
            scale_numeric=False,
        ),
        "lightgbm": build_model_pipeline(
            LGBMClassifier(
                n_estimators=220,
                learning_rate=0.06,
                num_leaves=47,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
            ),
            scale_numeric=False,
        ),
    }

    leaderboard: Dict[str, Dict[str, Any]] = {"baseline_dummy": baseline_val_metrics}
    for name, pipe in candidates.items():
        m_start = time.perf_counter()
        pipe.fit(X_train, y_train)
        val_prob = pipe.predict_proba(X_val)[:, 1]
        metrics = _eval_probabilities(y_val, val_prob)
        metrics["train_seconds"] = round(time.perf_counter() - m_start, 2)
        leaderboard[name] = metrics
        print(f"   - {name:20s} | Val PR-AUC={metrics['pr_auc']:.4f} | ROC-AUC={metrics['roc_auc']:.4f} | {metrics['train_seconds']}s")

    best_model_name = max(candidates.keys(), key=lambda k: leaderboard[k]["pr_auc"])
    print(f"[5/7] Tuning best model ({best_model_name}) via RandomizedSearchCV (TimeSeriesSplit)...")
    tuned_pipeline, best_params = _tune_best_model(
        best_model_name, X_train, y_train, scale_pos_weight=scale_pos_weight, n_iter=tune_iter
    )

    val_prob_tuned = tuned_pipeline.predict_proba(X_val)[:, 1]
    test_prob_tuned = tuned_pipeline.predict_proba(X_test)[:, 1]

    print("[6/7] Optimizing decision threshold using cold-chain business cost matrix (INR)...")
    threshold_report = optimize_threshold(y_val, val_prob_tuned, cost_matrix=COST_MATRIX_INR)
    opt_threshold = threshold_report["optimal_threshold"]

    test_metrics_050 = _eval_probabilities(y_test, test_prob_tuned, threshold=0.50)
    test_cost_opt = evaluate_threshold_cost(y_test, test_prob_tuned, opt_threshold, COST_MATRIX_INR)
    test_cost_050 = evaluate_threshold_cost(y_test, test_prob_tuned, 0.50, COST_MATRIX_INR)

    pr_auc_lift_pct = round(
        (test_metrics_050["pr_auc"] - baseline_test_metrics["pr_auc"])
        / max(baseline_test_metrics["pr_auc"], 1e-6)
        * 100.0,
        2,
    )
    print(
        f"   - Optimal Threshold={opt_threshold:.2f} | Test PR-AUC={test_metrics_050['pr_auc']:.4f} "
        f"(+{pr_auc_lift_pct:.1f}% vs baseline {baseline_test_metrics['pr_auc']:.4f})"
    )
    print(
        f"   - Test Cost @ {opt_threshold:.2f}: INR {test_cost_opt['total_cost_inr']:,.0f} "
        f"(saves INR {test_cost_opt['savings_vs_do_nothing_inr']:,.0f} vs do-nothing, "
        f"INR {test_cost_050['total_cost_inr'] - test_cost_opt['total_cost_inr']:,.0f} vs 0.50 threshold)"
    )

    print("[7/7] Computing global SHAP explanations, drift reference stats & saving artifacts...")
    shap_sample = X_test.sample(n=min(len(X_test), 1000), random_state=seed)
    shap_global = compute_global_shap(
        tuned_pipeline,
        shap_sample,
        save_json_path=SHAP_GLOBAL_PATH,
        save_plot_path=MODELS_DIR / "shap_global_bar.png",
    )

    ref_stats = compute_reference_stats(train_df)
    save_reference_stats(ref_stats, REFERENCE_STATS_PATH)

    joblib.dump(tuned_pipeline, MODEL_PATH)

    metadata: Dict[str, Any] = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_rows": int(n_rows),
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "late_rate_train": round(float(y_train.mean()), 4),
        "late_rate_test": round(float(y_test.mean()), 4),
        "best_model_family": best_model_name,
        "best_params": best_params,
        "leaderboard_val": leaderboard,
        "baseline_test_metrics": baseline_test_metrics,
        "tuned_test_metrics": test_metrics_050,
        "pr_auc_lift_vs_baseline_pct": pr_auc_lift_pct,
        "meets_20pct_lift_target": bool(pr_auc_lift_pct >= 20.0),
        "optimal_threshold": opt_threshold,
        "theoretical_bayes_threshold": threshold_report["theoretical_bayes_threshold"],
        "cost_matrix_inr": COST_MATRIX_INR,
        "test_cost_at_optimal_threshold": test_cost_opt,
        "test_cost_at_default_050": test_cost_050,
        "top_global_shap_features": shap_global["top_features"][:10],
        "total_training_seconds": round(time.perf_counter() - t0, 2),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Done in {metadata['total_training_seconds']}s! Saved model -> {MODEL_PATH}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DelhiCold late-delivery ML model.")
    parser.add_argument("--rows", type=int, default=100_000, help="Dataset rows (use 1500000 for full scale).")
    parser.add_argument("--tune-iter", type=int, default=10, help="RandomizedSearchCV iterations.")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Random seed.")
    args = parser.parse_args()
    run_training(n_rows=args.rows, tune_iter=args.tune_iter, seed=args.seed)


if __name__ == "__main__":
    main()

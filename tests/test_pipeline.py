"""Unit tests for the reproducible scikit-learn pipeline."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.data_generator import generate_coldchain_dataset
from src.pipeline import build_model_pipeline
from src.preprocessing import (
    chronological_split,
    clean_and_engineer_features,
    get_feature_columns,
)


def test_sklearn_pipeline_fit_predict_handles_missing_and_unseen():
    raw_df, scans_df = generate_coldchain_dataset(n_rows=1200, seed=11)
    clean_df = clean_and_engineer_features(raw_df, scans_df)
    train_df, _, test_df = chronological_split(clean_df)

    cols = get_feature_columns()
    pipe = build_model_pipeline(
        LogisticRegression(class_weight="balanced", max_iter=300, random_state=42),
        scale_numeric=True,
    )
    pipe.fit(train_df[cols], train_df["is_late"].values)

    # Inject unseen category and NaN numeric value at inference time
    X_probe = test_df[cols].iloc[:10].copy()
    X_probe.loc[X_probe.index[0], "weather_condition"] = "unseen_sandstorm"
    X_probe.loc[X_probe.index[1], "warehouse_dwell_hours"] = np.nan

    probs = pipe.predict_proba(X_probe)[:, 1]
    assert probs.shape == (10,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert not np.isnan(probs).any()

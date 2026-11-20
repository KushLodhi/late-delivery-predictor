"""Reproducible scikit-learn preprocessing and modeling pipeline."""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Build reproducible scikit-learn ColumnTransformer for numeric & categorical features.

    - Numeric: median imputation + optional StandardScaler (critical for Logistic Regression).
    - Categorical: most-frequent imputation + OneHotEncoder(handle_unknown='ignore').
    """
    num_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    cat_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=num_steps), NUMERIC_FEATURES),
            ("cat", Pipeline(steps=cat_steps), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model_pipeline(estimator: Any, scale_numeric: bool = True) -> Pipeline:
    """Wrap preprocessor + classifier into a single serializable scikit-learn Pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=scale_numeric)),
            ("classifier", estimator),
        ]
    )

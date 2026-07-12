"""Feature preparation for the Isolation Forest detector.

Isolation Forest is scale-insensitive in theory, but standardising features
keeps the anomaly-score distribution well behaved and makes thresholds
comparable across features with wildly different magnitudes (bytes vs. ratios).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import LABEL_COLUMN, NUMERIC_FEATURES


def split_features_labels(df: pd.DataFrame):
    """Return the numeric feature matrix and (optional) label vector.

    The label column is optional: at inference time we usually do not have
    ground truth, so callers get ``None`` for the labels in that case.
    """
    missing = [c for c in NUMERIC_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required feature columns: {missing}")

    features = df[NUMERIC_FEATURES].copy()
    features = _sanitize(features)

    labels = df[LABEL_COLUMN].to_numpy() if LABEL_COLUMN in df.columns else None
    return features, labels


def _sanitize(features: pd.DataFrame) -> pd.DataFrame:
    """Replace infinities/NaNs that can arise from rate divisions."""
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True))
    # Any column that was entirely NaN falls back to zero.
    return features.fillna(0.0)


def fit_scaler(features: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on the training feature matrix."""
    scaler = StandardScaler()
    scaler.fit(features.to_numpy())
    return scaler


def transform(scaler: StandardScaler, features: pd.DataFrame) -> np.ndarray:
    """Apply a previously fitted scaler."""
    return scaler.transform(features.to_numpy())

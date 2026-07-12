"""The Isolation Forest intrusion detector.

Wraps scikit-learn's ``IsolationForest`` together with its feature scaler so
the two are always persisted and loaded as a single unit. The class exposes a
small, task-oriented API (`fit`, `predict`, `score`) rather than leaking the
sklearn estimator directly.

Why Isolation Forest for intrusion detection?
---------------------------------------------
Intrusions are, by definition, rare and varied — you cannot enumerate every
future attack to build a labelled classifier. Isolation Forest is an
*unsupervised* anomaly detector: it isolates points by random partitioning and
flags samples that are separated in few splits (i.e. that sit far from the
dense "normal" region). It trains on mostly-benign traffic and needs no attack
labels, which matches how real network defenders operate.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from . import preprocessing
from .config import ISOLATION_FOREST_PARAMS, MODEL_PATH, SCALER_PATH


@dataclass
class Detection:
    """Result of scoring a batch of flows."""

    predictions: np.ndarray   # 1 = attack/anomaly, 0 = normal
    scores: np.ndarray        # higher = more anomalous


class IntrusionDetector:
    """Isolation-Forest-based network intrusion detector."""

    def __init__(self, params: dict | None = None):
        self.params = params or dict(ISOLATION_FOREST_PARAMS)
        self.model: IsolationForest | None = None
        self.scaler: StandardScaler | None = None

    # -- Training ----------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "IntrusionDetector":
        """Fit the scaler and Isolation Forest on a DataFrame of flows.

        Labels, if present, are ignored during fitting — training is fully
        unsupervised. They are only used later for evaluation.
        """
        features, _ = preprocessing.split_features_labels(df)
        self.scaler = preprocessing.fit_scaler(features)
        X = preprocessing.transform(self.scaler, features)

        self.model = IsolationForest(**self.params)
        self.model.fit(X)
        return self

    # -- Inference ---------------------------------------------------------
    def score(self, df: pd.DataFrame) -> Detection:
        """Score flows and return per-sample predictions and anomaly scores.

        The raw sklearn ``decision_function`` returns *higher = more normal*.
        We negate it so that, intuitively, *higher = more anomalous*.
        """
        self._check_ready()
        features, _ = preprocessing.split_features_labels(df)
        X = preprocessing.transform(self.scaler, features)

        raw = self.model.predict(X)             # +1 normal, -1 anomaly
        predictions = (raw == -1).astype(int)   # 1 = attack, 0 = normal
        scores = -self.model.decision_function(X)
        return Detection(predictions=predictions, scores=scores)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Convenience wrapper returning only the 0/1 predictions."""
        return self.score(df).predictions

    # -- Persistence -------------------------------------------------------
    def save(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH) -> None:
        self._check_ready()
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

    @classmethod
    def load(cls, model_path=MODEL_PATH, scaler_path=SCALER_PATH) -> "IntrusionDetector":
        detector = cls()
        detector.model = joblib.load(model_path)
        detector.scaler = joblib.load(scaler_path)
        return detector

    # -- Internals ---------------------------------------------------------
    def _check_ready(self) -> None:
        if self.model is None or self.scaler is None:
            raise RuntimeError(
                "Detector is not trained. Call fit(...) or load(...) first."
            )

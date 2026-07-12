"""Traffic Anomaly Detection — Isolation Forest.

Flags days whose KPI profile is anomalous relative to normal operation: sudden
ranking-driven traffic drops, viral spikes, tracking outages, or bot floods.
Isolation Forest is unsupervised, so it needs no labelled incidents — exactly
right for catching novel, never-seen-before anomalies.

This is the evolution of the project's original CyberDetect intrusion-detection
core, repurposed here for growth-signal monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ..config import ANOMALY_MODEL, ISOLATION_FOREST_PARAMS

FEATURES = [
    "visitors",
    "organic_traffic",
    "paid_traffic",
    "bounce_rate",
    "avg_time_on_site",
    "ctr",
    "conversion_rate",
    "avg_position",
]


@dataclass
class AnomalyResult:
    dates: pd.Series
    is_anomaly: np.ndarray   # 1 = anomalous day, 0 = normal
    score: np.ndarray        # higher = more anomalous

    def anomalous_days(self) -> pd.DataFrame:
        mask = self.is_anomaly == 1
        return pd.DataFrame({
            "date": self.dates[mask].to_numpy(),
            "anomaly_score": self.score[mask],
        }).sort_values("anomaly_score", ascending=False).reset_index(drop=True)


class TrafficAnomalyDetector:
    """Isolation-Forest detector over daily KPI vectors."""

    def __init__(self, params: dict | None = None):
        self.params = params or dict(ISOLATION_FOREST_PARAMS)
        self.model: IsolationForest | None = None
        self.scaler: StandardScaler | None = None
        self.features = FEATURES

    def fit(self, daily: pd.DataFrame) -> "TrafficAnomalyDetector":
        X = daily[self.features].to_numpy()
        self.scaler = StandardScaler().fit(X)
        self.model = IsolationForest(**self.params)
        self.model.fit(self.scaler.transform(X))
        return self

    def detect(self, daily: pd.DataFrame) -> AnomalyResult:
        self._check_ready()
        X = self.scaler.transform(daily[self.features].to_numpy())
        raw = self.model.predict(X)                 # +1 normal, -1 anomaly
        is_anom = (raw == -1).astype(int)
        score = -self.model.decision_function(X)    # higher = more anomalous
        return AnomalyResult(
            dates=pd.to_datetime(daily["date"]).reset_index(drop=True),
            is_anomaly=is_anom,
            score=score,
        )

    def save(self, path=ANOMALY_MODEL) -> None:
        self._check_ready()
        joblib.dump(
            {"model": self.model, "scaler": self.scaler, "features": self.features},
            path,
        )

    @classmethod
    def load(cls, path=ANOMALY_MODEL) -> "TrafficAnomalyDetector":
        obj = cls()
        blob = joblib.load(path)
        obj.model = blob["model"]
        obj.scaler = blob["scaler"]
        obj.features = blob["features"]
        return obj

    def _check_ready(self) -> None:
        if self.model is None or self.scaler is None:
            raise RuntimeError("TrafficAnomalyDetector is not trained. Call fit() or load().")

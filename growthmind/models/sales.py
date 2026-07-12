"""Sales Prediction — XGBoost regressor.

Predicts daily revenue from traffic-quality signals (sessions, CTR, conversion
rate, average position, page speed, content score). Also exposes feature
importances so the recommendation engine can reason about *which* levers move
revenue the most.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from ..config import RANDOM_STATE, SALES_MODEL, XGB_REGRESSOR_PARAMS

FEATURES = [
    "sessions",
    "ctr",
    "conversion_rate",
    "avg_position",
    "page_speed",
    "content_score",
    "bounce_rate",
    "avg_time_on_site",
]
TARGET = "revenue"


@dataclass
class SalesMetrics:
    mae: float
    r2: float

    def pretty(self) -> str:
        return f"Sales predictor — MAE=${self.mae:,.0f}  R²={self.r2:.3f}"


class SalesPredictor:
    """XGBoost revenue predictor over traffic-quality features."""

    def __init__(self, params: dict | None = None):
        self.params = params or dict(XGB_REGRESSOR_PARAMS)
        self.model: XGBRegressor | None = None
        self.features = FEATURES

    def fit(self, daily: pd.DataFrame, test_size: float = 0.2) -> SalesMetrics:
        X = daily[self.features]
        y = daily[TARGET]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE
        )
        self.model = XGBRegressor(**self.params)
        self.model.fit(X_tr, y_tr)
        preds = self.model.predict(X_te)
        return SalesMetrics(
            mae=float(mean_absolute_error(y_te, preds)),
            r2=float(r2_score(y_te, preds)),
        )

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        self._check_ready()
        return np.clip(self.model.predict(features[self.features]), 0, None)

    def feature_importance(self) -> pd.Series:
        self._check_ready()
        return pd.Series(
            self.model.feature_importances_, index=self.features
        ).sort_values(ascending=False)

    def save(self, path=SALES_MODEL) -> None:
        self._check_ready()
        joblib.dump({"model": self.model, "features": self.features}, path)

    @classmethod
    def load(cls, path=SALES_MODEL) -> "SalesPredictor":
        obj = cls()
        blob = joblib.load(path)
        obj.model = blob["model"]
        obj.features = blob["features"]
        return obj

    def _check_ready(self) -> None:
        if self.model is None:
            raise RuntimeError("SalesPredictor is not trained. Call fit() or load().")

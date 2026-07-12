"""SEO Score Prediction — XGBoost regressor over on-page features.

Learns to predict a page's SEO score (0–100) from its on-page signals. Beyond
prediction, the model's feature importances tell the recommendation engine
which on-page factors drive rankings, and a per-page ``explain`` highlights the
weakest levers for a specific URL.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from ..config import PAGE_SEO_FEATURES, RANDOM_STATE, SEO_MODEL, XGB_REGRESSOR_PARAMS

TARGET = "seo_score"


@dataclass
class SEOMetrics:
    mae: float
    r2: float

    def pretty(self) -> str:
        return f"SEO scorer — MAE={self.mae:.2f} pts  R²={self.r2:.3f}"


class SEOScorer:
    """Predicts a 0–100 SEO score from on-page features."""

    def __init__(self, params: dict | None = None):
        self.params = params or dict(XGB_REGRESSOR_PARAMS)
        self.model: XGBRegressor | None = None
        self.features = PAGE_SEO_FEATURES

    def fit(self, pages: pd.DataFrame, test_size: float = 0.2) -> SEOMetrics:
        X = pages[self.features]
        y = pages[TARGET]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE
        )
        self.model = XGBRegressor(**self.params)
        self.model.fit(X_tr, y_tr)
        preds = self.model.predict(X_te)
        return SEOMetrics(
            mae=float(mean_absolute_error(y_te, preds)),
            r2=float(r2_score(y_te, preds)),
        )

    def predict(self, pages: pd.DataFrame) -> np.ndarray:
        self._check_ready()
        return np.clip(self.model.predict(pages[self.features]), 0, 100)

    def feature_importance(self) -> pd.Series:
        self._check_ready()
        return pd.Series(
            self.model.feature_importances_, index=self.features
        ).sort_values(ascending=False)

    def save(self, path=SEO_MODEL) -> None:
        self._check_ready()
        joblib.dump({"model": self.model, "features": self.features}, path)

    @classmethod
    def load(cls, path=SEO_MODEL) -> "SEOScorer":
        obj = cls()
        blob = joblib.load(path)
        obj.model = blob["model"]
        obj.features = blob["features"]
        return obj

    def _check_ready(self) -> None:
        if self.model is None:
            raise RuntimeError("SEOScorer is not trained. Call fit() or load().")

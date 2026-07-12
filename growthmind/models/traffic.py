"""Traffic Prediction Engine — XGBoost time-series forecaster.

Forecasts future daily visitors from lagged traffic plus calendar and rolling
signals. We frame the series as a supervised regression problem, but predict the
**day-over-day change** (``visitors - lag_1``) rather than the absolute level.
Tree ensembles cannot extrapolate beyond the range of values seen in training,
so predicting the level of a trending series fails on the future tail; modelling
the *delta* and adding it back to the last observation lets the forecast follow
the trend naturally. The model is then rolled forward one day at a time to
produce a multi-day forecast (recursive strategy).
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from xgboost import XGBRegressor

from ..config import (
    FORECAST_HORIZON,
    TRAFFIC_LAGS,
    TRAFFIC_MODEL,
    XGB_REGRESSOR_PARAMS,
)

TARGET = "visitors"
DELTA = "_delta"


@dataclass
class ForecastMetrics:
    mae: float
    mape: float
    r2: float

    def pretty(self) -> str:
        return (f"Traffic forecaster — MAE={self.mae:,.1f}  "
                f"MAPE={self.mape:.1%}  R²={self.r2:.3f}")


def _build_supervised(daily: pd.DataFrame) -> pd.DataFrame:
    """Turn the daily series into a lag/rolling feature matrix with a delta target."""
    df = daily.sort_values("date").reset_index(drop=True).copy()
    df["dow"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["date"]).dt.month

    for lag in TRAFFIC_LAGS:
        df[f"lag_{lag}"] = df[TARGET].shift(lag)
    df["roll_mean_7"] = df[TARGET].shift(1).rolling(7).mean()
    df["roll_std_7"] = df[TARGET].shift(1).rolling(7).std()
    # Recent momentum: change over the last week.
    df["momentum_7"] = df[f"lag_1"] - df[TARGET].shift(8)

    # Target: change vs. yesterday. Reconstructed level = lag_1 + delta.
    df[DELTA] = df[TARGET] - df["lag_1"]

    return df.dropna().reset_index(drop=True)


def _feature_columns() -> list[str]:
    cols = ["dow", "month", "roll_mean_7", "roll_std_7", "momentum_7"]
    cols += [f"lag_{lag}" for lag in TRAFFIC_LAGS]
    return cols


class TrafficForecaster:
    """XGBoost recursive multi-step traffic forecaster (predicts daily deltas)."""

    def __init__(self, params: dict | None = None):
        self.params = params or dict(XGB_REGRESSOR_PARAMS)
        self.model: XGBRegressor | None = None
        self.feature_cols = _feature_columns()

    def fit(self, daily: pd.DataFrame, eval_fraction: float = 0.15) -> ForecastMetrics:
        """Fit on all but the last ``eval_fraction`` of days; evaluate on the tail.

        Metrics are reported on the reconstructed visitor *level*
        (``lag_1 + predicted_delta``) so they are directly interpretable.
        """
        sup = _build_supervised(daily)
        split = int(len(sup) * (1 - eval_fraction))
        train, test = sup.iloc[:split], sup.iloc[split:]

        self.model = XGBRegressor(**self.params)
        self.model.fit(train[self.feature_cols], train[DELTA])

        pred_delta = self.model.predict(test[self.feature_cols])
        pred_level = np.clip(test["lag_1"].to_numpy() + pred_delta, 0, None)
        actual = test[TARGET].to_numpy()

        return ForecastMetrics(
            mae=float(mean_absolute_error(actual, pred_level)),
            mape=float(mean_absolute_percentage_error(actual, pred_level)),
            r2=float(r2_score(actual, pred_level)),
        )

    def forecast(self, daily: pd.DataFrame, horizon: int = FORECAST_HORIZON) -> pd.DataFrame:
        """Recursively forecast ``horizon`` days beyond the last observed date."""
        self._check_ready()
        history = daily.sort_values("date").reset_index(drop=True).copy()
        series = history[TARGET].astype(float).tolist()
        dates = pd.to_datetime(history["date"]).tolist()

        rows = []
        for _ in range(horizon):
            next_date = dates[-1] + pd.Timedelta(days=1)
            feat = {
                "dow": next_date.dayofweek,
                "month": next_date.month,
                "roll_mean_7": np.mean(series[-7:]),
                "roll_std_7": np.std(series[-7:]),
                "momentum_7": series[-1] - series[-8] if len(series) >= 8 else 0.0,
            }
            for lag in TRAFFIC_LAGS:
                feat[f"lag_{lag}"] = series[-lag]

            X = pd.DataFrame([feat])[self.feature_cols]
            delta = float(self.model.predict(X)[0])
            yhat = max(0.0, series[-1] + delta)

            rows.append({"date": next_date, "predicted_visitors": round(yhat)})
            series.append(yhat)
            dates.append(next_date)

        return pd.DataFrame(rows)

    def feature_importance(self) -> pd.Series:
        self._check_ready()
        return pd.Series(
            self.model.feature_importances_, index=self.feature_cols
        ).sort_values(ascending=False)

    def save(self, path=TRAFFIC_MODEL) -> None:
        self._check_ready()
        joblib.dump({"model": self.model, "feature_cols": self.feature_cols}, path)

    @classmethod
    def load(cls, path=TRAFFIC_MODEL) -> "TrafficForecaster":
        obj = cls()
        blob = joblib.load(path)
        obj.model = blob["model"]
        obj.feature_cols = blob["feature_cols"]
        return obj

    def _check_ready(self) -> None:
        if self.model is None:
            raise RuntimeError("TrafficForecaster is not trained. Call fit() or load().")

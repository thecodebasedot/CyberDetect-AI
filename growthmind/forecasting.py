"""Multi-model traffic forecasting + rolling-origin backtesting.

The v1 traffic engine used a single XGBoost model. Real forecasting practice is
to compare several model families and validate them with *walk-forward*
(rolling-origin) backtesting rather than a single train/test split — because a
lucky split can flatter a model that would fail in production.

This module wraps three forecaster families behind one interface:

* :class:`XGBoostForecaster`     — the v1 gradient-boosted delta model.
* :class:`HoltWintersForecaster` — classical triple exponential smoothing
  (trend + weekly seasonality), via statsmodels.
* :class:`ProphetForecaster`     — Facebook Prophet additive model (optional;
  gracefully skipped if the package is unavailable).

and evaluates them with :func:`rolling_backtest` / :func:`compare_forecasters`.
Each forecaster exposes ``fit(daily)`` and ``forecast(horizon)`` and returns a
``DataFrame[date, predicted_visitors]``, so they are interchangeable everywhere.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .models.traffic import TrafficForecaster

TARGET = "visitors"

# Prophet / cmdstanpy are extremely chatty; quiet them down.
for _noisy in ("prophet", "cmdstanpy", "prophet.models"):
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)


# --------------------------------------------------------------------------
# Forecaster interface + implementations
# --------------------------------------------------------------------------
class BaseForecaster:
    """Common interface: ``fit(daily)`` then ``forecast(horizon)``."""

    name = "base"

    def fit(self, daily: pd.DataFrame) -> "BaseForecaster":
        raise NotImplementedError

    def forecast(self, horizon: int) -> pd.DataFrame:
        raise NotImplementedError

    @staticmethod
    def _future_dates(last_date, horizon: int) -> list:
        last_date = pd.Timestamp(last_date)
        return [last_date + pd.Timedelta(days=i + 1) for i in range(horizon)]


class XGBoostForecaster(BaseForecaster):
    """Adapter over the v1 :class:`TrafficForecaster` (XGBoost on deltas)."""

    name = "XGBoost"

    def __init__(self):
        self._tf = TrafficForecaster()
        self._daily: pd.DataFrame | None = None

    def fit(self, daily: pd.DataFrame) -> "XGBoostForecaster":
        self._daily = daily.sort_values("date").reset_index(drop=True)
        self._tf.fit(self._daily)
        return self

    def forecast(self, horizon: int) -> pd.DataFrame:
        return self._tf.forecast(self._daily, horizon=horizon)


class HoltWintersForecaster(BaseForecaster):
    """Triple exponential smoothing (additive trend + weekly seasonality)."""

    name = "Holt-Winters"

    def __init__(self, seasonal_periods: int = 7):
        self.seasonal_periods = seasonal_periods
        self._res = None
        self._last_date = None

    def fit(self, daily: pd.DataFrame) -> "HoltWintersForecaster":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        daily = daily.sort_values("date").reset_index(drop=True)
        y = daily[TARGET].astype(float).to_numpy()
        self._last_date = daily["date"].iloc[-1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                y,
                trend="add",
                seasonal="add",
                seasonal_periods=self.seasonal_periods,
                initialization_method="estimated",
            )
            self._res = model.fit()
        return self

    def forecast(self, horizon: int) -> pd.DataFrame:
        preds = np.clip(np.asarray(self._res.forecast(horizon)), 0, None)
        return pd.DataFrame({
            "date": self._future_dates(self._last_date, horizon),
            "predicted_visitors": np.round(preds).astype(int),
        })


class ProphetForecaster(BaseForecaster):
    """Facebook Prophet additive model (optional dependency)."""

    name = "Prophet"

    def __init__(self):
        self._model = None
        self._last_date = None

    @staticmethod
    def is_available() -> bool:
        try:
            import prophet  # noqa: F401
            return True
        except Exception:
            return False

    def fit(self, daily: pd.DataFrame) -> "ProphetForecaster":
        from prophet import Prophet

        daily = daily.sort_values("date").reset_index(drop=True)
        self._last_date = daily["date"].iloc[-1]
        df = pd.DataFrame({
            "ds": pd.to_datetime(daily["date"]),
            "y": daily[TARGET].astype(float),
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = Prophet(
                weekly_seasonality=True,
                yearly_seasonality=True,
                daily_seasonality=False,
            )
            self._model.fit(df)
        return self

    def forecast(self, horizon: int) -> pd.DataFrame:
        future = self._model.make_future_dataframe(periods=horizon)
        pred = self._model.predict(future).tail(horizon)
        return pd.DataFrame({
            "date": pd.to_datetime(pred["ds"]).to_numpy(),
            "predicted_visitors": np.round(np.clip(pred["yhat"].to_numpy(), 0, None)).astype(int),
        })


def available_forecasters() -> list[type[BaseForecaster]]:
    """Return the forecaster classes usable in this environment."""
    classes: list[type[BaseForecaster]] = [XGBoostForecaster, HoltWintersForecaster]
    if ProphetForecaster.is_available():
        classes.append(ProphetForecaster)
    return classes


# --------------------------------------------------------------------------
# Metrics + rolling-origin backtest
# --------------------------------------------------------------------------
@dataclass
class BacktestResult:
    model: str
    mae: float
    mape: float
    rmse: float
    folds: int

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "MAE": round(self.mae, 1),
            "MAPE": round(self.mape, 4),
            "RMSE": round(self.rmse, 1),
            "folds": self.folds,
        }


def _fold_metrics(actual: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    err = actual - pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mask = actual != 0
    mape = float(np.mean(np.abs(err[mask] / actual[mask]))) if mask.any() else float("nan")
    return mae, mape, rmse


def rolling_backtest(
    daily: pd.DataFrame,
    factory,
    horizon: int = 14,
    folds: int = 3,
    min_train: int = 180,
) -> BacktestResult:
    """Walk-forward backtest: train on a growing window, forecast the next
    ``horizon`` days, and score against the held-out actuals — repeated over
    ``folds`` successive cut points near the end of the series.
    """
    daily = daily.sort_values("date").reset_index(drop=True)
    n = len(daily)
    maes, mapes, rmses = [], [], []
    used = 0

    for i in range(folds):
        cut = n - horizon * (folds - i)
        if cut < min_train:
            continue
        train = daily.iloc[:cut]
        actual = daily.iloc[cut:cut + horizon][TARGET].to_numpy()
        if len(actual) == 0:
            continue
        model = factory().fit(train)
        pred = model.forecast(horizon)["predicted_visitors"].to_numpy()[:len(actual)]
        mae, mape, rmse = _fold_metrics(actual.astype(float), pred.astype(float))
        maes.append(mae)
        mapes.append(mape)
        rmses.append(rmse)
        used += 1

    if used == 0:
        return BacktestResult(model=factory().name, mae=float("nan"),
                              mape=float("nan"), rmse=float("nan"), folds=0)

    return BacktestResult(
        model=factory().name,
        mae=float(np.mean(maes)),
        mape=float(np.nanmean(mapes)),
        rmse=float(np.mean(rmses)),
        folds=used,
    )


def compare_forecasters(
    daily: pd.DataFrame,
    horizon: int = 14,
    folds: int = 3,
) -> pd.DataFrame:
    """Backtest every available forecaster and return a metrics table (best first)."""
    rows = []
    for cls in available_forecasters():
        result = rolling_backtest(daily, cls, horizon=horizon, folds=folds)
        rows.append(result.as_dict())
    table = pd.DataFrame(rows)
    return table.sort_values("MAE").reset_index(drop=True)

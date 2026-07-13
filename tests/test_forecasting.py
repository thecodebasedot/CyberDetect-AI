"""Tests for the multi-model forecasting + backtesting module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from growthmind.data import generate_daily_metrics
from growthmind.forecasting import (
    BacktestResult,
    HoltWintersForecaster,
    ProphetForecaster,
    XGBoostForecaster,
    available_forecasters,
    compare_forecasters,
    rolling_backtest,
)


@pytest.fixture(scope="module")
def daily():
    return generate_daily_metrics(days=400, random_state=21)


def _assert_valid_forecast(fc: pd.DataFrame, horizon: int, last_date):
    assert list(fc.columns) == ["date", "predicted_visitors"]
    assert len(fc) == horizon
    assert (fc["predicted_visitors"] >= 0).all()
    assert pd.Timestamp(fc["date"].min()) > pd.Timestamp(last_date)


def test_available_forecasters_includes_core():
    names = {c.name for c in available_forecasters()}
    assert {"XGBoost", "Holt-Winters"}.issubset(names)


def test_holtwinters_forecast(daily):
    f = HoltWintersForecaster().fit(daily)
    _assert_valid_forecast(f.forecast(10), 10, daily["date"].iloc[-1])


def test_xgboost_forecaster_adapter(daily):
    f = XGBoostForecaster().fit(daily)
    _assert_valid_forecast(f.forecast(10), 10, daily["date"].iloc[-1])


@pytest.mark.skipif(not ProphetForecaster.is_available(), reason="prophet not installed")
def test_prophet_forecaster(daily):
    f = ProphetForecaster().fit(daily)
    _assert_valid_forecast(f.forecast(10), 10, daily["date"].iloc[-1])


def test_rolling_backtest_metrics_are_finite(daily):
    result = rolling_backtest(daily, HoltWintersForecaster, horizon=7, folds=2)
    assert isinstance(result, BacktestResult)
    assert result.folds == 2
    assert np.isfinite(result.mae)
    assert np.isfinite(result.mape)
    assert result.mae > 0


def test_backtest_respects_min_train():
    small = generate_daily_metrics(days=120, random_state=1)
    # min_train larger than the series -> no usable folds.
    result = rolling_backtest(small, HoltWintersForecaster, horizon=7, folds=3, min_train=200)
    assert result.folds == 0
    assert np.isnan(result.mae)


def test_compare_forecasters_table(daily):
    table = compare_forecasters(daily, horizon=7, folds=2)
    assert "model" in table.columns and "MAE" in table.columns
    # One row per available forecaster.
    assert len(table) == len(available_forecasters())
    # Sorted ascending by MAE (best first).
    maes = table["MAE"].to_numpy()
    assert np.all(np.diff(maes) >= 0)

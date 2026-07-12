"""Unit tests for GrowthMind AI.

Run with:  python -m pytest -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from growthmind.data import generate_daily_metrics, generate_pages, generate_users
from growthmind.health import compute_health
from growthmind.models import (
    SEOScorer,
    SalesPredictor,
    TrafficAnomalyDetector,
    TrafficForecaster,
    UserSegmenter,
)
from growthmind.recommend import page_recommendations, prioritize, site_recommendations
from growthmind.strategy import build_strategy


# --------------------------------------------------------------------------
# Data generation
# --------------------------------------------------------------------------
def test_datasets_shapes_and_reproducibility():
    a = generate_daily_metrics(days=365, random_state=5)
    b = generate_daily_metrics(days=365, random_state=5)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 365
    assert (a["visitors"] > 0).all()
    assert a["revenue"].sum() > 0


def test_pages_seo_score_is_bounded():
    pages = generate_pages(n=300, random_state=6)
    assert pages["seo_score"].between(0, 100).all()
    assert len(pages) == 300


def test_users_have_latent_segments():
    users = generate_users(n=1000, random_state=7)
    assert users["_latent_segment"].nunique() == 4
    # Buyers should account for the only non-zero spend.
    assert users.loc[users["num_orders"] > 0, "total_spent"].min() > 0


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def test_traffic_forecaster_learns_and_forecasts():
    daily = generate_daily_metrics(days=500, random_state=8)
    fc = TrafficForecaster()
    metrics = fc.fit(daily)
    # A trend-aware model should beat a naive ~10% error on this series.
    assert metrics.mape < 0.10
    out = fc.forecast(daily, horizon=14)
    assert len(out) == 14
    assert (out["predicted_visitors"] >= 0).all()
    # Forecast dates are strictly after the last observed date.
    assert out["date"].min() > daily["date"].max()


def test_sales_predictor_reasonable_r2():
    daily = generate_daily_metrics(days=500, random_state=9)
    model = SalesPredictor()
    metrics = model.fit(daily)
    assert metrics.r2 > 0.5
    imp = model.feature_importance()
    assert len(imp) == len(model.features)


def test_seo_scorer_recovers_scoring_function():
    pages = generate_pages(n=500, random_state=10)
    scorer = SEOScorer()
    metrics = scorer.fit(pages)
    # The SEO score is a deterministic-ish function of the features -> high R².
    assert metrics.r2 > 0.6
    preds = scorer.predict(pages.head(5))
    assert np.all((preds >= 0) & (preds <= 100))


def test_user_segmenter_recovers_structure():
    users = generate_users(n=1500, random_state=11)
    seg = UserSegmenter()
    metrics = seg.fit(users)
    assert metrics.silhouette > 0.2
    labelled = seg.predict(users)
    assert "segment_label" in labelled.columns
    assert "Buyer" in set(labelled["segment_label"])
    summary = seg.segment_summary(users)
    assert summary["est_clv"].max() > 0


def test_anomaly_detector_flags_injected_spike():
    daily = generate_daily_metrics(days=600, random_state=12)
    det = TrafficAnomalyDetector().fit(daily)
    result = det.detect(daily)
    assert result.is_anomaly.sum() > 0
    # The injected viral spike (index 500-504) should be flagged.
    flagged_idx = set(np.where(result.is_anomaly == 1)[0])
    assert any(500 <= i <= 505 for i in flagged_idx)


def test_model_unfit_raises():
    with pytest.raises(RuntimeError):
        SalesPredictor().predict(pd.DataFrame())


# --------------------------------------------------------------------------
# Recommendations, health, strategy
# --------------------------------------------------------------------------
def test_recommendations_and_strategy():
    daily = generate_daily_metrics(days=400, random_state=13)
    pages = generate_pages(n=200, random_state=13)

    recs = page_recommendations(pages, limit=5)
    recs += site_recommendations(daily)
    recs = prioritize(recs)
    assert recs, "expected at least one recommendation"
    # Priorities are sorted ascending (highest priority first).
    assert recs == sorted(recs, key=lambda r: r.priority)

    plan = build_strategy(recs)
    total = len(plan.today) + len(plan.this_week) + len(plan.this_month)
    assert total == len(recs)


def test_health_score_bounds():
    daily = generate_daily_metrics(days=400, random_state=14)
    pages = generate_pages(n=200, random_state=14)
    health = compute_health(daily, pages)
    assert 0 <= health.overall <= 100
    assert health.grade in {"A", "B", "C", "D", "F"}
    assert set(health.dimensions) == {"SEO", "Performance", "UX", "Security", "Content"}


def test_save_load_roundtrip(tmp_path):
    daily = generate_daily_metrics(days=400, random_state=15)
    model = SalesPredictor()
    model.fit(daily)
    p = tmp_path / "sales.joblib"
    model.save(p)
    restored = SalesPredictor.load(p)
    np.testing.assert_allclose(
        model.predict(daily.head(10)),
        restored.predict(daily.head(10)),
    )

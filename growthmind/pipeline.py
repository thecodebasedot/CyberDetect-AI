"""End-to-end orchestration for GrowthMind AI.

`train_all` fits every model on the (generated or loaded) datasets and persists
them. `analyze` loads the trained models and produces the full growth
intelligence bundle: forecasts, anomalies, segments, health score,
recommendations and a time-boxed strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import recommend, strategy
from .config import (
    DAILY_METRICS_CSV,
    FORECAST_HORIZON,
    PAGES_CSV,
    USERS_CSV,
)
from .data import generate_all, load_keywords
from .health import HealthScore, compute_health
from .models import (
    CustomerIntelligence,
    KeywordRankingModel,
    SEOScorer,
    SalesPredictor,
    TrafficAnomalyDetector,
    TrafficForecaster,
    UserSegmenter,
)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_datasets(regenerate: bool = False):
    """Load the three datasets, generating them on first run."""
    if regenerate or not (DAILY_METRICS_CSV.exists() and PAGES_CSV.exists() and USERS_CSV.exists()):
        return generate_all(save=True)
    daily = pd.read_csv(DAILY_METRICS_CSV, parse_dates=["date"])
    pages = pd.read_csv(PAGES_CSV)
    users = pd.read_csv(USERS_CSV)
    return daily, pages, users


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
@dataclass
class TrainingReport:
    metrics: dict = field(default_factory=dict)

    def pretty(self) -> str:
        lines = ["Model training report", "=" * 40]
        for line in self.metrics.values():
            lines.append("  " + line)
        return "\n".join(lines)


def train_all(regenerate: bool = False, save: bool = True) -> TrainingReport:
    """Fit and persist every model; return their evaluation metrics."""
    daily, pages, users = load_datasets(regenerate=regenerate)
    metrics: dict[str, str] = {}

    traffic = TrafficForecaster()
    metrics["traffic"] = traffic.fit(daily).pretty()

    sales = SalesPredictor()
    metrics["sales"] = sales.fit(daily).pretty()

    seo = SEOScorer()
    metrics["seo"] = seo.fit(pages).pretty()

    segmenter = UserSegmenter()
    metrics["segmentation"] = segmenter.fit(users).pretty()

    anomaly = TrafficAnomalyDetector().fit(daily)
    metrics["anomaly"] = "Traffic anomaly detector — fitted (Isolation Forest)"

    customer = CustomerIntelligence()
    metrics["customer"] = customer.fit(users).pretty()

    ranker = KeywordRankingModel()
    metrics["ranking"] = ranker.fit(load_keywords()).pretty()

    if save:
        traffic.save()
        sales.save()
        seo.save()
        segmenter.save()
        anomaly.save()
        customer.save()
        ranker.save()

    return TrainingReport(metrics=metrics)


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
@dataclass
class GrowthInsights:
    forecast: pd.DataFrame
    anomalies: pd.DataFrame
    segments: pd.DataFrame
    health: HealthScore
    recommendations: list
    strategy: object
    kpis: dict = field(default_factory=dict)
    customers: dict = field(default_factory=dict)


def analyze(horizon: int = FORECAST_HORIZON, regenerate: bool = False) -> GrowthInsights:
    """Load trained models and produce the full growth-intelligence bundle."""
    daily, pages, users = load_datasets(regenerate=regenerate)

    traffic = TrafficForecaster.load()
    sales = SalesPredictor.load()
    seo = SEOScorer.load()
    segmenter = UserSegmenter.load()
    anomaly = TrafficAnomalyDetector.load()
    customer = CustomerIntelligence.load()

    forecast = traffic.forecast(daily, horizon=horizon)
    anomaly_days = anomaly.detect(daily).anomalous_days()
    segment_summary = segmenter.segment_summary(users)
    customer_summary = customer.summary(users)
    health = compute_health(daily, pages)

    # Recommendations from pages + site trends + models.
    recs = recommend.page_recommendations(pages, limit=8)
    recs += recommend.site_recommendations(
        daily,
        sales_importance=sales.feature_importance(),
        anomalies=anomaly_days,
    )
    recs += recommend.customer_recommendations(customer_summary)
    recs = recommend.prioritize(recs)
    plan = strategy.build_strategy(recs)

    recent = daily.tail(30)
    kpis = {
        "avg_daily_visitors": int(recent["visitors"].mean()),
        "forecast_avg_visitors": int(forecast["predicted_visitors"].mean()),
        "forecast_horizon_days": horizon,
        "predicted_traffic_change_pct": round(
            100 * (forecast["predicted_visitors"].mean() / recent["visitors"].mean() - 1), 1
        ),
        "monthly_revenue": round(float(recent["revenue"].sum()), 2),
        "avg_conversion_rate": round(float(recent["conversion_rate"].mean()), 4),
        "health_score": health.overall,
        "health_grade": health.grade,
        "n_anomalies": int(len(anomaly_days)),
        "n_recommendations": len(recs),
        "revenue_at_risk": customer_summary["revenue_at_risk"],
        "high_churn_customers": customer_summary["high_churn_customers"],
        "conversion_opportunities": customer_summary["conversion_opportunities"],
    }

    return GrowthInsights(
        forecast=forecast,
        anomalies=anomaly_days,
        segments=segment_summary,
        customers=customer_summary,
        health=health,
        recommendations=recs,
        strategy=plan,
        kpis=kpis,
    )

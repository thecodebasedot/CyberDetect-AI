"""Central configuration for GrowthMind AI.

Paths, dataset schemas and model hyper-parameters live here so the rest of the
package stays declarative and every module reads from one source of truth.
"""

from __future__ import annotations

from pathlib import Path

# --- Project layout -------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
REPORTS_DIR = PROJECT_ROOT / "reports"

for _d in (DATASETS_DIR, MODELS_DIR, DASHBOARD_DIR, REPORTS_DIR):
    _d.mkdir(exist_ok=True)

# --- Dataset files --------------------------------------------------------
DAILY_METRICS_CSV = DATASETS_DIR / "daily_metrics.csv"   # time-series of site KPIs
PAGES_CSV = DATASETS_DIR / "pages.csv"                    # per-page SEO snapshot
USERS_CSV = DATASETS_DIR / "users.csv"                    # per-visitor behaviour

# --- Model artifacts ------------------------------------------------------
TRAFFIC_MODEL = MODELS_DIR / "traffic_forecaster.joblib"
SALES_MODEL = MODELS_DIR / "sales_predictor.joblib"
SEO_MODEL = MODELS_DIR / "seo_scorer.joblib"
SEGMENTER_MODEL = MODELS_DIR / "user_segmenter.joblib"
ANOMALY_MODEL = MODELS_DIR / "traffic_anomaly.joblib"

RANDOM_STATE = 42

# --- Schemas --------------------------------------------------------------
# Daily site-level KPIs (one row per calendar day).
DAILY_METRIC_COLUMNS = [
    "date",
    "visitors",
    "sessions",
    "organic_traffic",
    "paid_traffic",
    "bounce_rate",
    "avg_time_on_site",
    "ctr",
    "avg_position",
    "backlinks",
    "page_speed",
    "content_score",
    "conversion_rate",
    "sales",
    "revenue",
]

# Per-page SEO features used by the SEO scorer.
PAGE_SEO_FEATURES = [
    "word_count",
    "images",
    "alt_ratio",
    "internal_links",
    "external_links",
    "title_length",
    "meta_length",
    "h1_count",
    "page_speed",
    "has_schema",
    "mobile_friendly",
]

# Per-user behavioural features used by the segmenter.
USER_FEATURES = [
    "sessions",
    "pageviews",
    "avg_time_on_site",
    "recency_days",
    "num_orders",
    "total_spent",
]

# --- Model hyper-parameters ----------------------------------------------
XGB_REGRESSOR_PARAMS = {
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

KMEANS_PARAMS = {
    "n_clusters": 4,
    "n_init": 10,
    "random_state": RANDOM_STATE,
}

ISOLATION_FOREST_PARAMS = {
    "n_estimators": 200,
    "contamination": 0.06,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Number of lag days the traffic forecaster uses as features.
TRAFFIC_LAGS = [1, 2, 3, 7, 14]
FORECAST_HORIZON = 30  # default days to forecast

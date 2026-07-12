# GrowthMind AI — Architecture

This document explains how the pieces fit together and the reasoning behind the
main design decisions.

## Layers

GrowthMind is organized as four layers, each depending only on the one beneath
it:

1. **Data layer** (`growthmind/data/`) — generates or loads the three datasets.
2. **Model layer** (`growthmind/models/`) — five independent, self-persisting
   models, each with a small `fit / predict / save / load` API.
3. **Intelligence layer** (`recommend.py`, `health.py`, `strategy.py`) — turns
   raw model outputs into decisions.
4. **Delivery layer** (`main.py` CLI, `api/app.py` FastAPI, `report.py` HTML) —
   surfaces the intelligence to humans and machines.

`pipeline.py` is the orchestrator that wires layers 1–3 together
(`train_all` and `analyze`).

## Data model

Three tables mirror what a real business exports:

| Table | Grain | Purpose |
| ----- | ----- | ------- |
| `daily_metrics` | one row / day | time-series for forecasting & anomaly detection |
| `pages` | one row / URL | on-page SEO features for scoring & recommendations |
| `users` | one row / visitor | behavioural features for segmentation |

The generator builds **causal** relationships (revenue depends on
traffic × conversion; conversion depends on speed/content/bounce; `seo_score`
is a transparent function of on-page features) so the models learn real signal
rather than noise. Two anomalies are injected into the traffic series (a
ranking-drop and a viral spike) to give the anomaly detector something concrete
to catch — and the test-suite asserts it does.

## Model choices

- **Traffic forecaster — XGBoost on deltas.** Tree ensembles cannot extrapolate
  beyond values seen in training, which breaks naive level-forecasting of a
  trending series. We predict `visitors − lag_1` (the day-over-day change) and
  add it back to the last observation, so the forecast follows the trend. The
  model is rolled forward recursively for multi-step horizons.
- **Sales & SEO — XGBoost regressors.** Chosen for tabular accuracy and, just as
  importantly, **feature importances** that feed the recommendation engine.
- **Segmentation — K-Means** on standardized behavioural features, with
  automatic persona labelling from cluster centroids (buyer = highest spend,
  etc.) and a silhouette score for cluster quality.
- **Anomaly detection — Isolation Forest.** Unsupervised, so it flags *novel*
  anomalies with no labelled incidents. This is the original CyberDetect
  intrusion-detection core, repurposed for growth monitoring.

## Recommendation engine

Recommendations come from two sources and are merged, then priority-sorted:

- **Page rules** (`page_recommendations`) target pages with the highest
  *opportunity* = SEO gap × log(traffic) — i.e. pages that already get traffic
  but score poorly, where fixes pay off fastest.
- **Site signals** (`site_recommendations`) read recent trends (traffic drop,
  bounce, speed), the sales model's top lever, and detected anomalies.

Each recommendation carries a `problem`, `reason`, `action`, `expected_gain`
and `priority (1–5)`. `strategy.build_strategy` buckets them into
today / this week / this month.

## Persistence

Every model serializes itself (and its scaler / feature list) with `joblib`
into `models/`. The pipeline trains once and every downstream consumer
(`analyze`, the API, the dashboard) loads the artifacts — no retraining per
request.

## Delivery

- **CLI** (`main.py`) — the primary interface; `demo` runs the whole flow.
- **FastAPI** (`api/app.py`) — the same `analyze()` bundle exposed as JSON, plus
  a server-rendered `/dashboard`. Returns `503` until models are trained.
- **HTML report** (`report.py`) — a single dependency-free file (inline CSS +
  SVG sparkline). It is the shippable stand-in for the planned React frontend.

## Extending to real data

Replace `growthmind/data/generator.py` with loaders over your own exports. As
long as the column schemas in `growthmind/config.py` are preserved, every model,
recommendation and view keeps working unchanged.

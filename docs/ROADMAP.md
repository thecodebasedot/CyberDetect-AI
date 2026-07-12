# GrowthMind AI — Roadmap

## v1.0 — shipped ✅

A complete, working, offline growth-intelligence core:

- Synthetic GA/GSC/crawl datasets with causal structure
- 5 trained models: Traffic (XGBoost), Sales (XGBoost), SEO (XGBoost),
  Segmentation (K-Means), Anomaly (Isolation Forest)
- Recommendation engine, health score, growth strategy
- FastAPI backend + self-contained HTML dashboard
- 12-test pytest suite

## v1.1 — quality & breadth

- [ ] LightGBM ranking model for keyword-position prediction
- [ ] Prophet / statsmodels baseline alongside the XGBoost forecaster
- [ ] Backtesting harness (rolling-origin evaluation) for the forecaster
- [ ] Confidence intervals on forecasts
- [ ] Churn model and per-user CLV regression
- [ ] Competitor gap analysis (keyword / content / backlink deltas)

## v2.0 — from analytics to autonomous agent

The headline differentiator: GrowthMind stops merely *reporting* and starts
*acting*.

### Live data connectors
- [ ] Google Search Console API
- [ ] Google Analytics 4 (Data API)
- [ ] Bing Webmaster Tools
- [ ] Shopify / WordPress content APIs

### Autonomous Growth Agent
With explicit, scoped permission the agent will, on a daily schedule:
- [ ] Pull fresh GSC/GA data and detect abnormal changes
- [ ] Identify pages with SEO problems and rank fixes by expected impact
- [ ] Suggest new content topics and the highest-ROI pages to update
- [ ] Produce a daily Growth Report and notify the owner
- [ ] (Opt-in) apply low-risk fixes automatically and log every action

### Platform
- [x] React + Tailwind frontend consuming the existing FastAPI endpoints (`frontend/`)
- [ ] PostgreSQL persistence + historical trend storage
- [ ] Multi-site / multi-tenant support
- [ ] Auth, roles, and an audit log for agent actions

## Design principles carried forward

- **Human-in-the-loop by default** — the agent proposes; a human approves any
  outward-facing change until explicitly trusted.
- **Every recommendation is explainable** — problem, reason, action, expected
  gain. No black-box "do this."
- **Schema-stable** — swapping synthetic data for real exports must never
  require touching the models.

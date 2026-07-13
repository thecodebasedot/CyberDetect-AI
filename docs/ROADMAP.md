# GrowthMind AI — Roadmap

## v1.0 — shipped ✅

A complete, working, offline growth-intelligence core:

- Synthetic GA/GSC/crawl datasets with causal structure
- 5 trained models: Traffic (XGBoost), Sales (XGBoost), SEO (XGBoost),
  Segmentation (K-Means), Anomaly (Isolation Forest)
- Recommendation engine, health score, growth strategy
- FastAPI backend + self-contained HTML dashboard
- 29-test pytest suite

## v1.1 — quality & breadth

- [ ] LightGBM ranking model for keyword-position prediction
- [x] Prophet + statsmodels (Holt-Winters) baselines alongside XGBoost
      (`growthmind/forecasting.py`)
- [x] Backtesting harness (rolling-origin / walk-forward) comparing forecasters
      (`python main.py forecast-eval`)
- [ ] Confidence intervals on forecasts
- [ ] Churn model and per-user CLV regression
- [ ] Competitor gap analysis (keyword / content / backlink deltas)

## v2.0 — from analytics to autonomous agent

The headline differentiator: GrowthMind stops merely *reporting* and starts
*acting*.

### Live data connectors
- [x] Pluggable `DataSource` interface + registry (`growthmind/connectors/`)
- [x] Local / synthetic connector (default, fully working offline)
- [ ] Google Search Console API (interface + credential-gated stub shipped)
- [ ] Google Analytics 4 Data API (interface + credential-gated stub shipped)
- [ ] Bing Webmaster Tools
- [ ] Shopify / WordPress content APIs

### Autonomous Growth Agent
On a schedule, the agent:
- [x] Syncs data through a pluggable connector interface (`growthmind/connectors/`)
- [x] Detects abnormal changes by diffing each cycle's KPIs vs. the last snapshot
- [x] Ranks fixes by expected impact and builds a prioritized action plan
- [x] Produces a dated Growth Report (`reports/growth_report_NNNN.md`)
- [x] Human-in-the-loop autonomy: `propose` (default) vs `auto` (simulates
      low-risk actions only, logged to `reports/agent_actions.log`)
- [ ] Pull *live* GSC/GA data (needs the connectors below + credentials)
- [ ] Actually execute approved fixes once a real platform is connected
- [ ] Notify the owner (email / push) on each cycle

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

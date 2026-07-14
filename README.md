# 🧠 GrowthMind AI

> ### Predict. Optimize. Grow.
>
> An AI **growth-intelligence engine** that forecasts traffic & sales, scores
> SEO health, segments customers, detects traffic anomalies, and turns it all
> into a prioritized, one-click growth strategy.

GrowthMind AI answers the questions every online business struggles with:

- **Why is traffic falling?** → anomaly detection over daily KPIs
- **Which pages can rank?** → SEO scoring + on-page recommendations
- **What will next month look like?** → 7/30/90-day traffic & revenue forecasts
- **Which visitors will convert?** → behavioural segmentation + CLV
- **What should I do right now?** → an AI growth strategy (today / this week / this month)

Everything runs **fully offline** on synthetic-but-realistic data, so you can
clone, train, and see results in under a minute — no API keys, no accounts.

---

## Highlights

| Module | Algorithm | What it does |
| ------ | --------- | ------------ |
| **Traffic Forecasting** | XGBoost (delta modelling) | 7/30/90-day visitor forecast |
| **Sales Prediction** | XGBoost | revenue from traffic-quality signals |
| **SEO Scoring** | XGBoost | 0–100 page score + ranking drivers |
| **Customer Segmentation** | K-Means | New / Returning / Potential / Buyer + CLV |
| **Customer Intelligence** | XGBoost | purchase propensity · churn risk · lifetime value |
| **Keyword Ranking** | LightGBM (LambdaMART) | learning-to-rank + striking-distance SEO wins |
| **Anomaly Detection** | Isolation Forest | flags abnormal traffic days |
| **Recommendation Engine** | rules + model signals | prioritized fixes with impact estimates |
| **Health Score** | weighted composite | SEO · Performance · UX · Security · Content |
| **Growth Strategy** | priority bucketing | today / this week / this month plan |
| **Autonomous Agent** | snapshot diffing + planning | daily report, change alerts, action plan |

Plus a **FastAPI** backend, a **self-contained HTML dashboard**, and a
**React + Tailwind dashboard** (`frontend/`) that consumes the API.

---

## Quick start

```bash
git clone https://github.com/thecodebasedot/cyberdetect-ai.git
cd cyberdetect-ai
pip install -r requirements.txt

# Everything at once: generate data -> train 7 models -> analyze -> dashboard
python main.py demo
```

Then step through individual commands:

```bash
python main.py generate            # build the 4 synthetic datasets
python main.py train               # fit & persist all 7 models
python main.py analyze --horizon 30  # print the full growth report
python main.py customers           # purchase / churn / CLV predictions
python main.py keywords            # keyword ranking + striking-distance wins
python main.py dashboard           # write dashboard/index.html
python main.py agent               # run one Autonomous Growth Agent cycle
python main.py serve               # launch the FastAPI backend (docs at /docs)
```

### Example output

```
Model training report
========================================
  Traffic forecaster — MAE=124.4  MAPE=4.9%  R²=0.461
  Sales predictor — MAE=$208  R²=0.792
  SEO scorer — MAE=6.52 pts  R²=0.812
  User segmenter — 4 segments  silhouette=0.463
  Traffic anomaly detector — fitted (Isolation Forest)
  Customer intelligence — purchase AUC=0.840  churn AUC=0.812  CLV R²=0.379
  Keyword ranker — NDCG@10=0.973  NDCG@5=0.957  (60 test keywords)

Website Health Score: 68/100  (grade D)
  SEO             57  ███████████·········
  Performance     85  ████████████████····
  ...

AI Growth Strategy
==================
TODAY (do now)
  • [SEO] expand the article with useful sections, examples and FAQs
      ↳ /page/0359: thin content — +8–15% organic traffic
  • [Anomaly] investigate tracking / algorithm updates on the flagged dates
      ↳ 44 anomalous traffic day(s) detected — prevent silent traffic loss
```

---

## Architecture

```
                        ┌──────────────────────────┐
   synthetic data  ───► │  datasets/                │
   (GA + GSC + crawl)   │   daily_metrics · pages · users
                        └────────────┬─────────────┘
                                     ▼
        ┌────────────────────────────────────────────────────┐
        │  Models (growthmind/models/)                        │
        │  Traffic (XGB) · Sales (XGB) · SEO (XGB)            │
        │  Segmentation (KMeans) · Anomaly (IsolationForest) │
        └────────────┬───────────────────────────────────────┘
                     ▼
        ┌────────────────────────────────────────────────────┐
        │  Intelligence layer                                 │
        │  recommend.py · health.py · strategy.py             │
        └────────────┬───────────────────────────────────────┘
                     ▼
        ┌───────────────────┐   ┌───────────────────────────┐
        │  CLI (main.py)    │   │  FastAPI (api/app.py)     │
        │  report.py → HTML │   │  JSON + /dashboard        │
        └───────────────────┘   └───────────────────────────┘
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for a deeper walkthrough.

### Datasets

| File | Rows | Description |
| ---- | ---- | ----------- |
| `daily_metrics.csv` | 730 | daily time-series of site KPIs (trend + seasonality + injected anomalies) |
| `pages.csv` | 400 | per-page SEO snapshot with a learnable `seo_score` |
| `users.csv` | 5000 | per-visitor behaviour, churn label & 4 latent segments |
| `keywords.csv` | 3000 | keyword↔page ranking candidates (learning-to-rank) |

To use **real** data, export CSVs with the same columns
(`growthmind/config.py`) from Google Analytics / Search Console / a crawler and
drop them into `datasets/`.

---

## API

```bash
python main.py serve
```

| Endpoint | Returns |
| -------- | ------- |
| `GET /api/kpis` | headline KPIs (health, forecast, revenue, conversion) |
| `GET /api/forecast?horizon=30` | day-by-day traffic forecast |
| `GET /api/health-score` | composite score + per-dimension breakdown |
| `GET /api/segments` | user segments with estimated CLV |
| `GET /api/customers` | purchase propensity · churn · CLV summary |
| `GET /api/keywords` | keyword ranking factors + striking-distance opportunities |
| `GET /api/anomalies` | flagged anomalous days |
| `GET /api/recommendations` | prioritized recommendations |
| `GET /api/strategy` | today / this week / this month plan |
| `GET /dashboard` | the full HTML dashboard |

Interactive docs at `http://127.0.0.1:8000/docs`.

---

## Keyword Ranking (learning-to-rank)

A **LightGBM LambdaMART** ranker predicts how pages rank for a keyword and finds
the highest-ROI SEO wins (`growthmind/models/ranking.py`, `python main.py keywords`):

```
Keyword ranker — NDCG@10=0.973  NDCG@5=0.957  (60 test keywords)

Top ranking factors:  relevance · backlinks · domain_authority · word_count · page_speed

Striking-distance opportunities (our pages at positions 4–15), by search volume:
keyword  search_volume  keyword_difficulty  predicted_position  relevance  backlinks
kw_0187          23123                60.0                   5      0.539         12
kw_0012           7404                46.1                   5      0.605         54
...
```

Unlike a plain regressor, the ranker optimizes the *order* of candidate pages
per keyword (grouped learning-to-rank) and is evaluated with **NDCG** — the
standard ranking metric. The split is by keyword group so no keyword leaks
between train and test. `opportunities()` surfaces *striking-distance* keywords
— our pages sitting just off page one — ranked by search volume, the quick wins
an SEO team acts on first.

---

## Customer Intelligence

Segmentation tells you *who* your visitors are; this predicts what they'll *do*
(`growthmind/models/customer.py`, `python main.py customers`):

- **Purchase propensity** — probability a visitor converts, from behaviour only
  (purchase columns are excluded as inputs to avoid leakage). ROC-AUC ≈ 0.84.
- **Churn** — probability an existing customer lapses. ROC-AUC ≈ 0.81.
- **Customer Lifetime Value** — expected monetary value predicted from
  engagement behaviour. R² ≈ 0.38 (behaviour-only, no purchase-history leakage).

These roll up into headline retention KPIs — **revenue at risk** (expected CLV
weighted by churn probability) and **conversion opportunities** (high-propensity
non-buyers) — which the recommendation engine turns into win-back and
first-purchase actions.

---

## Forecasting model comparison + backtesting

The traffic forecaster is validated the way real forecasting is done — with
**walk-forward (rolling-origin) backtesting** across several model families, not
a single lucky train/test split:

```bash
python main.py forecast-eval --horizon 14 --folds 3
```

```
       model   MAE   MAPE  RMSE  folds
     Prophet 141.9 0.0565 159.3      3
Holt-Winters 238.7 0.0917 266.2      3
     XGBoost 459.0 0.1725 492.3      3

Best model by MAE: Prophet
```

Three families compete behind one interface (`growthmind/forecasting.py`):
**XGBoost** (v1 delta model), **Holt-Winters** triple exponential smoothing
(statsmodels), and **Prophet**. The honest result: over a 14-day *recursive*
horizon the tree model degrades (multi-step extrapolation is its weakness),
while Prophet's additive trend+seasonality wins — a finding a single-split
evaluation would have hidden. Prophet is optional; it's skipped automatically if
not installed.

---

## Autonomous Growth Agent (v2)

The agent runs the whole analysis on a schedule and reports **what changed**
since last time, not just the current state:

```bash
python main.py agent                      # propose mode (human approves actions)
python main.py agent --autonomy auto      # auto-simulate low-risk actions
python main.py agent --source local       # pluggable data source (local | gsc | ga4)
```

Each cycle:
1. **Syncs data** via a pluggable connector (`growthmind/connectors/`).
2. **Diffs** today's KPIs against the previous cycle's snapshot → change alerts
   (traffic drop, health regression, new anomalies, forecast shift).
3. **Plans** a prioritized action list from the recommendations.
4. **Writes** a dated Growth Report to `reports/growth_report_NNNN.md` and logs
   actions to `reports/agent_actions.log`.

**Autonomy is human-in-the-loop by default.** In `propose` mode the agent only
suggests. In `auto` mode it may act on **low-risk** actions only — and because
this build has no live platform connected, "acting" is *simulated* and logged,
never executed. Real execution arrives with the live connectors, behind the same
allow-list and an explicit opt-in.

### Data connectors

`growthmind/connectors/` defines a `DataSource` interface so the models never
change when the data source does:

| Source | Status |
| ------ | ------ |
| `local` | ✅ synthetic / local CSVs — the default, fully offline |
| `gsc` | 🔒 Google Search Console — interface shipped, needs OAuth credentials |
| `ga4` | 🔒 Google Analytics 4 — interface shipped, needs OAuth credentials |

Schedule a daily cycle with cron:

```cron
0 7 * * *  cd /path/to/repo && python main.py agent --autonomy propose
```

---

## React dashboard (v2 frontend)

A Vite + React + Tailwind dashboard lives in [`frontend/`](frontend/). It shows
the KPIs, forecast chart, health, segments, recommendations and strategy, and
falls back to demo data when the backend is offline.

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173 (proxies /api to the backend)
```

Run `python main.py serve` alongside it for live data. See
[`frontend/README.md`](frontend/README.md) for details.

---

## Project layout

```
GrowthMind-AI/
├── main.py                     # CLI (generate/train/analyze/dashboard/demo/serve)
├── requirements.txt
├── growthmind/
│   ├── config.py               # paths, schemas, hyper-parameters
│   ├── data/generator.py       # synthetic GA/GSC/crawl data
│   ├── models/
│   │   ├── traffic.py          # XGBoost traffic forecaster
│   │   ├── sales.py            # XGBoost sales predictor
│   │   ├── seo.py              # XGBoost SEO scorer
│   │   ├── segmentation.py     # K-Means user segmentation
│   │   └── anomaly.py          # Isolation Forest anomaly detector
│   ├── connectors/             # pluggable data sources (local, gsc, ga4)
│   ├── models/customer.py      # purchase / churn / CLV models
│   ├── models/ranking.py       # LightGBM keyword learning-to-rank
│   ├── forecasting.py          # XGBoost/Holt-Winters/Prophet + backtesting
│   ├── agent.py                # Autonomous Growth Agent
│   ├── recommend.py            # AI recommendation engine
│   ├── health.py               # website health score
│   ├── strategy.py             # growth strategy builder
│   ├── pipeline.py             # end-to-end orchestration
│   └── report.py               # self-contained HTML dashboard
├── api/app.py                  # FastAPI backend
├── frontend/                   # React + Tailwind dashboard (Vite)
├── tests/                      # pytest suite (44 tests)
│   ├── test_growthmind.py
│   ├── test_agent.py
│   └── test_forecasting.py
├── datasets/  models/  dashboard/  reports/   # generated artifacts
└── docs/                       # ARCHITECTURE.md · ROADMAP.md
```

---

## Tests

```bash
python -m pytest -q      # 44 tests
```

---

## Roadmap (v2)

v1 is a complete, working ML + API + dashboard core. v2 turns it into an
**autonomous growth agent** connected to real platforms. See
[`docs/ROADMAP.md`](docs/ROADMAP.md):

- Live connectors: Google Search Console, Google Analytics 4, Bing Webmaster,
  Shopify, WordPress
- LSTM / Prophet forecasting alongside XGBoost
- React + Tailwind frontend consuming the existing API
- An **AI Autonomous Growth Agent** that analyzes daily, prioritizes fixes, and
  (with permission) acts on them
- PostgreSQL persistence + scheduled daily growth reports

---

## Notes

> This project began as **CyberDetect AI**, an Isolation-Forest network
> intrusion detector. That anomaly-detection core lives on as GrowthMind's
> **traffic anomaly** module (`growthmind/models/anomaly.py`) — the same
> unsupervised technique, repurposed to catch abnormal growth signals.

## License

See [LICENSE](LICENSE).

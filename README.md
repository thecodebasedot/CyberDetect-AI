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
| **Anomaly Detection** | Isolation Forest | flags abnormal traffic days |
| **Recommendation Engine** | rules + model signals | prioritized fixes with impact estimates |
| **Health Score** | weighted composite | SEO · Performance · UX · Security · Content |
| **Growth Strategy** | priority bucketing | today / this week / this month plan |

Plus a **FastAPI** backend, a **self-contained HTML dashboard**, and a
**React + Tailwind dashboard** (`frontend/`) that consumes the API.

---

## Quick start

```bash
git clone https://github.com/thecodebasedot/cyberdetect-ai.git
cd cyberdetect-ai
pip install -r requirements.txt

# Everything at once: generate data -> train 5 models -> analyze -> dashboard
python main.py demo
```

Then step through individual commands:

```bash
python main.py generate            # build the 3 synthetic datasets
python main.py train               # fit & persist all 5 models
python main.py analyze --horizon 30  # print the full growth report
python main.py dashboard           # write dashboard/index.html
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
| `users.csv` | 5000 | per-visitor behaviour with 4 latent segments |

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
| `GET /api/anomalies` | flagged anomalous days |
| `GET /api/recommendations` | prioritized recommendations |
| `GET /api/strategy` | today / this week / this month plan |
| `GET /dashboard` | the full HTML dashboard |

Interactive docs at `http://127.0.0.1:8000/docs`.

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
│   ├── recommend.py            # AI recommendation engine
│   ├── health.py               # website health score
│   ├── strategy.py             # growth strategy builder
│   ├── pipeline.py             # end-to-end orchestration
│   └── report.py               # self-contained HTML dashboard
├── api/app.py                  # FastAPI backend
├── frontend/                   # React + Tailwind dashboard (Vite)
├── tests/test_growthmind.py    # pytest suite (12 tests)
├── datasets/  models/  dashboard/  reports/   # generated artifacts
└── docs/                       # ARCHITECTURE.md · ROADMAP.md
```

---

## Tests

```bash
python -m pytest -q      # 12 tests
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

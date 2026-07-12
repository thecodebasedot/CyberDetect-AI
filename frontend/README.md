# GrowthMind AI — React Dashboard (v2 frontend)

A Vite + React + Tailwind dashboard for the GrowthMind AI FastAPI backend. It
visualizes KPIs, the traffic forecast, website health, user segments,
AI recommendations, and the growth strategy.

The dashboard falls back to bundled demo data when the backend is unreachable,
so it renders as a static preview *and* live against the API.

## Develop

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173  (proxies /api to :8000)
```

In another terminal, start the backend so the dashboard shows live data:

```bash
python main.py serve  # FastAPI on http://127.0.0.1:8000
```

The Vite dev server proxies `/api` and `/dashboard` to the backend (see
`vite.config.js`), so no CORS setup is needed.

## Build

```bash
npm run build        # outputs static assets to dist/
npm run preview      # serve the production build locally
```

To point a static build at a remote API, set `VITE_API_BASE`:

```bash
VITE_API_BASE=https://your-api.example.com npm run build
```

## Structure

```
frontend/
├── index.html
├── vite.config.js          # dev proxy to the FastAPI backend
├── tailwind.config.js
└── src/
    ├── App.jsx             # layout + data loading
    ├── api.js              # fetch client with demo-data fallback
    └── components/
        ├── KpiCard.jsx
        ├── ForecastChart.jsx   # Recharts area chart
        ├── HealthPanel.jsx
        ├── SegmentsTable.jsx
        ├── Recommendations.jsx
        └── StrategyBoard.jsx
```

// API client for the GrowthMind AI FastAPI backend.
//
// Each call falls back to bundled mock data when the backend is unreachable, so
// the dashboard renders in a static preview (e.g. GitHub Pages) as well as when
// wired to `python main.py serve`.

const BASE = import.meta.env.VITE_API_BASE || ''

async function get(path, fallback) {
  try {
    const res = await fetch(`${BASE}${path}`)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    return { data: await res.json(), live: true }
  } catch (e) {
    return { data: fallback, live: false }
  }
}

export const api = {
  kpis: (horizon = 30) => get(`/api/kpis?horizon=${horizon}`, mock.kpis),
  forecast: (horizon = 30) => get(`/api/forecast?horizon=${horizon}`, mock.forecast),
  healthScore: () => get('/api/health-score', mock.health),
  segments: () => get('/api/segments', mock.segments),
  anomalies: () => get('/api/anomalies', mock.anomalies),
  recommendations: () => get('/api/recommendations', mock.recommendations),
  strategy: () => get('/api/strategy', mock.strategy),
}

// --- Mock data (mirrors the shape returned by the backend) -----------------
const mockForecast = Array.from({ length: 30 }, (_, i) => ({
  date: `2025-01-${String((i % 28) + 1).padStart(2, '0')}`,
  predicted_visitors: Math.round(2300 + 120 * Math.sin(i / 3) + i * 4),
}))

export const mock = {
  kpis: {
    health_score: 68,
    health_grade: 'D',
    avg_daily_visitors: 2588,
    forecast_avg_visitors: 2360,
    forecast_horizon_days: 30,
    predicted_traffic_change_pct: -8.8,
    monthly_revenue: 64693,
    avg_conversion_rate: 0.0137,
    n_anomalies: 44,
    n_recommendations: 10,
  },
  forecast: { horizon: 30, forecast: mockForecast },
  health: {
    overall: 68,
    grade: 'D',
    dimensions: { SEO: 57, Performance: 85, UX: 43, Security: 90, Content: 79 },
  },
  segments: [
    { segment_label: 'Buyer', users: 530, avg_orders: 5.1, avg_spent: 268.5, avg_pageviews: 30.3, est_clv: 1641.08 },
    { segment_label: 'Returning Visitor', users: 1252, avg_orders: 0, avg_spent: 0, avg_pageviews: 13.4, est_clv: 0 },
    { segment_label: 'Potential Buyer', users: 1013, avg_orders: 0, avg_spent: 0, avg_pageviews: 14.7, est_clv: 0 },
    { segment_label: 'New Visitor', users: 2205, avg_orders: 0, avg_spent: 0, avg_pageviews: 2.2, est_clv: 0 },
  ],
  anomalies: [
    { date: '2024-05-15', anomaly_score: 0.21 },
    { date: '2024-10-27', anomaly_score: 0.18 },
    { date: '2024-11-02', anomaly_score: 0.16 },
  ],
  recommendations: [
    { area: 'SEO', problem: '/page/0359: thin content', reason: 'only 180 words', action: 'expand the article with useful sections and FAQs', expected_gain: '+8–15% organic traffic', priority: 1 },
    { area: 'Anomaly', problem: '44 anomalous traffic day(s) detected', reason: 'unusual KPI patterns', action: 'investigate tracking / algorithm updates on the flagged dates', expected_gain: 'prevent silent traffic loss', priority: 1 },
    { area: 'Performance', problem: 'slow average page speed', reason: 'site-wide speed 66/100', action: 'optimize images, adopt a CDN', expected_gain: '+5–12% traffic & conversions', priority: 2 },
    { area: 'Conversion', problem: 'revenue sensitive to one lever', reason: "sales model ranks 'conversion_rate' highest", action: "run experiments improving 'conversion_rate'", expected_gain: 'highest revenue lift per effort', priority: 2 },
    { area: 'SEO', problem: '/page/0002: no structured data', reason: 'no schema markup', action: 'add JSON-LD schema', expected_gain: '+4–9% CTR', priority: 2 },
  ],
  strategy: {
    today: [
      { area: 'SEO', problem: '/page/0359: thin content', action: 'expand the article', expected_gain: '+8–15% organic traffic', priority: 1 },
      { area: 'Anomaly', problem: '44 anomalous days', action: 'investigate flagged dates', expected_gain: 'prevent silent traffic loss', priority: 1 },
    ],
    this_week: [
      { area: 'Performance', problem: 'slow page speed', action: 'optimize images & CDN', expected_gain: '+5–12%', priority: 2 },
      { area: 'Conversion', problem: 'top revenue lever', action: "improve conversion_rate", expected_gain: 'highest lift', priority: 2 },
    ],
    this_month: [
      { area: 'SEO', problem: '/page/0111: title length', action: 'rewrite title to 50–60 chars', expected_gain: '+2–5% CTR', priority: 4 },
    ],
  },
}

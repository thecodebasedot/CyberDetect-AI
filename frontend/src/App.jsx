import { useEffect, useState } from 'react'
import { api } from './api'
import KpiCard from './components/KpiCard'
import ForecastChart from './components/ForecastChart'
import HealthPanel from './components/HealthPanel'
import SegmentsTable from './components/SegmentsTable'
import Recommendations from './components/Recommendations'
import StrategyBoard from './components/StrategyBoard'

export default function App() {
  const [state, setState] = useState({ loading: true, live: false })

  useEffect(() => {
    let cancelled = false
    async function load() {
      const [kpis, forecast, health, segments, recs, strategy, anomalies] = await Promise.all([
        api.kpis(),
        api.forecast(30),
        api.healthScore(),
        api.segments(),
        api.recommendations(),
        api.strategy(),
        api.anomalies(),
      ])
      if (cancelled) return
      setState({
        loading: false,
        live: kpis.live, // if the KPI call reached the backend, we're live
        kpis: kpis.data,
        forecast: forecast.data,
        health: health.data,
        segments: segments.data,
        recs: recs.data,
        strategy: strategy.data,
        anomalies: anomalies.data,
      })
    }
    load()
    return () => {
      cancelled = true
    }
  }, [])

  if (state.loading) {
    return <div className="p-10 text-gray-400">Loading GrowthMind AI…</div>
  }

  const k = state.kpis
  const change = k.predicted_traffic_change_pct
  const changeColor = change >= 0 ? '#2ecc71' : '#e74c3c'

  return (
    <div className="min-h-full">
      <header className="px-8 py-5 border-b border-line flex items-baseline gap-4 flex-wrap">
        <h1 className="text-xl font-bold m-0">🧠 GrowthMind AI</h1>
        <span className="text-accent font-semibold">Predict. Optimize. Grow.</span>
        <span className="ml-auto text-xs text-gray-400">
          {state.live ? '● live API' : '○ demo data (start the backend for live data)'}
        </span>
      </header>

      <main className="max-w-6xl mx-auto px-8 py-6 space-y-6">
        <div className="grid gap-3.5 grid-cols-2 md:grid-cols-3 lg:grid-cols-6">
          <KpiCard label="Health Score" value={`${Math.round(k.health_score)}/100`} sub={`Grade ${k.health_grade}`} />
          <KpiCard label="Avg Daily Visitors" value={k.avg_daily_visitors.toLocaleString()} sub="last 30 days" />
          <KpiCard
            label="30-Day Forecast"
            value={k.forecast_avg_visitors.toLocaleString()}
            sub={`${change >= 0 ? '+' : ''}${change}% vs now`}
            accent={changeColor}
          />
          <KpiCard label="Monthly Revenue" value={`$${Math.round(k.monthly_revenue).toLocaleString()}`} sub="last 30 days" />
          <KpiCard label="Conversion" value={`${(k.avg_conversion_rate * 100).toFixed(2)}%`} sub="last 30 days" />
          <KpiCard label="Anomalies" value={k.n_anomalies} sub="flagged days" />
        </div>

        <ForecastChart data={state.forecast} />

        <div className="grid gap-6 md:grid-cols-2">
          <HealthPanel data={state.health} />
          <SegmentsTable data={state.segments} />
        </div>

        <Recommendations data={state.recs} />
        <StrategyBoard data={state.strategy} />

        <footer className="text-center text-gray-500 text-xs py-6">
          GrowthMind AI · v1 core (5 ML models) + v2 React dashboard
        </footer>
      </main>
    </div>
  )
}

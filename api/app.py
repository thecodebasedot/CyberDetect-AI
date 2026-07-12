"""GrowthMind AI — FastAPI backend.

Exposes the trained models and the growth-intelligence pipeline over HTTP so a
frontend (or another service) can consume forecasts, health scores,
recommendations and the growth strategy as JSON. Also serves the generated HTML
dashboard at ``/dashboard``.

Run with:  python main.py serve        (or)   uvicorn api.app:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from growthmind import __tagline__, __version__
from growthmind.config import TRAFFIC_MODEL
from growthmind.pipeline import analyze

app = FastAPI(
    title="GrowthMind AI",
    description="Predict. Optimize. Grow. — AI growth-intelligence API.",
    version=__version__,
)


def _require_trained() -> None:
    if not TRAFFIC_MODEL.exists():
        raise HTTPException(
            status_code=503,
            detail="Models are not trained yet. Run `python main.py train` first.",
        )


@app.get("/")
def root():
    return {"name": "GrowthMind AI", "tagline": __tagline__, "version": __version__,
            "docs": "/docs", "dashboard": "/dashboard"}


@app.get("/health")
def health():
    return {"status": "ok", "models_trained": TRAFFIC_MODEL.exists()}


@app.get("/api/kpis")
def kpis(horizon: int = Query(30, ge=1, le=180)):
    _require_trained()
    return analyze(horizon=horizon).kpis


@app.get("/api/forecast")
def forecast(horizon: int = Query(30, ge=1, le=180)):
    _require_trained()
    fc = analyze(horizon=horizon).forecast
    return {"horizon": horizon, "forecast": fc.assign(
        date=fc["date"].astype(str)).to_dict(orient="records")}


@app.get("/api/health-score")
def health_score():
    _require_trained()
    h = analyze().health
    return {"overall": h.overall, "grade": h.grade, "dimensions": h.dimensions}


@app.get("/api/segments")
def segments():
    _require_trained()
    return analyze().segments.to_dict(orient="records")


@app.get("/api/anomalies")
def anomalies():
    _require_trained()
    a = analyze().anomalies
    return a.assign(date=a["date"].astype(str)).to_dict(orient="records")


@app.get("/api/recommendations")
def recommendations():
    _require_trained()
    return [r.as_dict() for r in analyze().recommendations]


@app.get("/api/strategy")
def strategy():
    _require_trained()
    plan = analyze().strategy
    return {
        "today": [r.as_dict() for r in plan.today],
        "this_week": [r.as_dict() for r in plan.this_week],
        "this_month": [r.as_dict() for r in plan.this_month],
    }


@app.post("/api/agent/run")
def agent_run(
    autonomy: str = Query("propose", pattern="^(propose|auto)$"),
    source: str = Query("local"),
    horizon: int = Query(30, ge=1, le=180),
):
    """Run one Autonomous Growth Agent cycle and return its report."""
    _require_trained()
    from growthmind.agent import AutonomousGrowthAgent
    from growthmind.connectors import get_connector

    connector = get_connector(source)
    if not connector.is_available():
        connector = get_connector("local")

    report = AutonomousGrowthAgent(connector=connector, autonomy=autonomy).run_cycle(horizon=horizon)
    return {
        "cycle": report.cycle,
        "timestamp": report.timestamp,
        "source": report.source_summary,
        "kpis": report.kpis,
        "alerts": [a.as_dict() for a in report.alerts],
        "actions": [a.as_dict() for a in report.actions],
        "report_path": report.report_path,
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(horizon: int = Query(30, ge=1, le=180)):
    _require_trained()
    from growthmind.report import render_html
    return render_html(analyze(horizon=horizon))

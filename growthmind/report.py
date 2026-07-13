"""Self-contained HTML dashboard generator.

Renders a GrowthInsights bundle into a single, dependency-free HTML file (inline
CSS, inline SVG sparkline) that opens in any browser. This stands in for the
planned React dashboard while remaining fully functional and shippable today.
"""

from __future__ import annotations

import html
from datetime import datetime

import pandas as pd

from . import __tagline__, __version__
from .config import DASHBOARD_DIR
from .pipeline import GrowthInsights, load_datasets


def _sparkline(values, width=520, height=90, color="#4f8cff") -> str:
    """Return an inline SVG sparkline for a list of numbers."""
    values = list(values)
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    rng = (hi - lo) or 1
    step = width / (len(values) - 1)
    pts = " ".join(
        f"{i * step:.1f},{height - (v - lo) / rng * (height - 10) - 5:.1f}"
        for i, v in enumerate(values)
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'preserveAspectRatio="none">'
        f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{pts}"/>'
        f'</svg>'
    )


def _kpi_card(label: str, value: str, sub: str = "") -> str:
    return (
        '<div class="card kpi">'
        f'<div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="kpi-value">{html.escape(value)}</div>'
        f'<div class="kpi-sub">{html.escape(sub)}</div>'
        '</div>'
    )


def _table(df: pd.DataFrame, max_rows: int = 10) -> str:
    df = df.head(max_rows)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = ""
    for _, row in df.iterrows():
        cells = "".join(f"<td>{html.escape(str(v))}</td>" for v in row)
        body += f"<tr>{cells}</tr>"
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def render_html(insights: GrowthInsights) -> str:
    daily, pages, users = load_datasets()
    k = insights.kpis

    change = k["predicted_traffic_change_pct"]
    change_color = "#2ecc71" if change >= 0 else "#e74c3c"
    change_sign = "+" if change >= 0 else ""

    # KPI row.
    kpis_html = "".join([
        _kpi_card("Health Score", f"{k['health_score']:.0f}/100", f"Grade {k['health_grade']}"),
        _kpi_card("Avg Daily Visitors", f"{k['avg_daily_visitors']:,}", "last 30 days"),
        _kpi_card(f"{k['forecast_horizon_days']}-Day Forecast",
                  f"{k['forecast_avg_visitors']:,}",
                  f"{change_sign}{change}% vs now"),
        _kpi_card("Monthly Revenue", f"${k['monthly_revenue']:,.0f}", "last 30 days"),
        _kpi_card("Conversion Rate", f"{k['avg_conversion_rate']:.2%}", "last 30 days"),
        _kpi_card("Anomalies", f"{k['n_anomalies']}", "flagged days"),
    ] + ([
        _kpi_card("Revenue at Risk", f"${k['revenue_at_risk']:,.0f}",
                  f"{k['high_churn_customers']:,} high-churn customers"),
        _kpi_card("Conversion Opportunities", f"{k['conversion_opportunities']:,}",
                  "high-intent non-buyers"),
    ] if "revenue_at_risk" in k else []))

    # Health dimensions.
    health_bars = ""
    for name, val in insights.health.dimensions.items():
        health_bars += (
            f'<div class="bar-row"><span>{name}</span>'
            f'<div class="bar"><div class="fill" style="width:{val}%"></div></div>'
            f'<b>{val:.0f}</b></div>'
        )

    # Forecast sparkline (recent history + forecast).
    hist = daily["visitors"].tail(60).tolist()
    fc = insights.forecast["predicted_visitors"].tolist()
    spark = _sparkline(hist + fc)

    # Recommendations.
    rec_html = ""
    for r in insights.recommendations[:12]:
        rec_html += (
            '<div class="rec">'
            f'<span class="pill p{min(r.priority,5)}">P{r.priority}</span>'
            f'<div><b>{html.escape(r.area)} — {html.escape(r.problem)}</b>'
            f'<div class="muted">{html.escape(r.reason)}</div>'
            f'<div>➜ {html.escape(r.action)} '
            f'<span class="gain">{html.escape(r.expected_gain)}</span></div></div>'
            '</div>'
        )

    generated = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GrowthMind AI — Dashboard</title>
<style>
  :root {{ --bg:#0e1117; --card:#161b22; --line:#232a34; --text:#e6edf3; --muted:#8b949e; --accent:#4f8cff; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
         background:var(--bg); color:var(--text); }}
  header {{ padding:24px 32px; border-bottom:1px solid var(--line);
            display:flex; align-items:baseline; gap:16px; flex-wrap:wrap; }}
  header h1 {{ margin:0; font-size:22px; }}
  header .tag {{ color:var(--accent); font-weight:600; }}
  header .meta {{ margin-left:auto; color:var(--muted); font-size:13px; }}
  main {{ padding:24px 32px; max-width:1200px; margin:0 auto; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:14px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px; }}
  .kpi-label {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
  .kpi-value {{ font-size:26px; font-weight:700; margin:6px 0 2px; }}
  .kpi-sub {{ color:var(--muted); font-size:12px; }}
  section {{ margin-top:26px; }}
  section h2 {{ font-size:16px; border-left:3px solid var(--accent); padding-left:10px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th,td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); }}
  th {{ color:var(--muted); font-weight:600; }}
  .bar-row {{ display:flex; align-items:center; gap:12px; margin:8px 0; font-size:13px; }}
  .bar-row span {{ width:100px; color:var(--muted); }}
  .bar {{ flex:1; height:10px; background:#0d1117; border-radius:6px; overflow:hidden; }}
  .fill {{ height:100%; background:linear-gradient(90deg,#4f8cff,#2ecc71); }}
  .rec {{ display:flex; gap:12px; padding:12px 0; border-bottom:1px solid var(--line); }}
  .muted {{ color:var(--muted); font-size:13px; margin:2px 0; }}
  .gain {{ color:#2ecc71; font-size:12px; }}
  .pill {{ height:22px; min-width:30px; padding:0 8px; border-radius:11px; font-size:12px;
           display:inline-flex; align-items:center; justify-content:center; font-weight:700; }}
  .p1 {{ background:#e74c3c; }} .p2 {{ background:#e67e22; }} .p3 {{ background:#f1c40f; color:#111; }}
  .p4 {{ background:#3498db; }} .p5 {{ background:#7f8c8d; }}
  .two {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
  @media (max-width:800px) {{ .two {{ grid-template-columns:1fr; }} }}
</style></head>
<body>
<header>
  <h1>🧠 GrowthMind AI</h1><span class="tag">{html.escape(__tagline__)}</span>
  <span class="meta">v{__version__} · generated {generated}</span>
</header>
<main>
  <section><div class="grid">{kpis_html}</div></section>

  <section>
    <h2>Traffic — last 60 days + {k['forecast_horizon_days']}-day forecast</h2>
    <div class="card">{spark}
      <div class="muted">Blue line: observed visitors followed by the model forecast
      (<b style="color:{change_color}">{change_sign}{change}%</b> projected change).</div>
    </div>
  </section>

  <div class="two">
    <section>
      <h2>Website Health</h2>
      <div class="card">{health_bars}
        <div class="muted" style="margin-top:10px">Overall
        <b>{insights.health.overall:.0f}/100</b> (grade {insights.health.grade})</div>
      </div>
    </section>
    <section>
      <h2>User Segments</h2>
      <div class="card">{_table(insights.segments)}</div>
    </section>
  </div>

  <section>
    <h2>AI Recommendations</h2>
    <div class="card">{rec_html or '<div class="muted">No issues found.</div>'}</div>
  </section>

  <section>
    <h2>Detected Traffic Anomalies</h2>
    <div class="card">{_table(insights.anomalies) if len(insights.anomalies) else '<div class="muted">No anomalies detected.</div>'}</div>
  </section>
</main>
</body></html>"""


def write_dashboard(insights: GrowthInsights, path=None) -> str:
    """Render and write the dashboard HTML; return the file path."""
    path = path or (DASHBOARD_DIR / "index.html")
    html_str = render_html(insights)
    path.write_text(html_str, encoding="utf-8")
    return str(path)

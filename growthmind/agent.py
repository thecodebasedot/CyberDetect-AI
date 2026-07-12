"""Autonomous Growth Agent (v2).

The agent turns GrowthMind from a report you *read* into a system that *runs on
a schedule*. Each cycle it:

  1. Syncs data from a connector (local synthetic, or a live source in v2).
  2. Ensures the models are trained.
  3. Runs the full analysis.
  4. Diffs today's KPIs against the previous cycle's snapshot to surface
     *changes* (traffic drop, health regression, new anomalies, forecast shift).
  5. Builds a prioritized action plan from the recommendations + change alerts.
  6. Writes a dated Growth Report and persists the snapshot for next time.

Autonomy is deliberately conservative and human-in-the-loop by default:

  * ``propose`` (default) — the agent only *proposes* actions; a human acts.
  * ``auto``             — the agent may act on **low-risk** actions. Because
    this build has no connected platform to change, "acting" is *simulated*:
    every intended action is recorded in the action log with status
    ``simulated``. Real execution arrives with the live connectors (v2), gated
    behind the same low-risk allow-list and an explicit opt-in.

Nothing here ever calls an external system, so running the agent is always safe.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime

from .config import REPORTS_DIR, TRAFFIC_MODEL
from .connectors import DataSource, get_connector
from .pipeline import analyze, train_all

SNAPSHOT_PATH = REPORTS_DIR / "agent_snapshot.json"
ACTION_LOG_PATH = REPORTS_DIR / "agent_actions.log"

# Actions the agent is allowed to auto-apply in `auto` mode. Deliberately narrow
# and reversible; everything else always waits for a human.
LOW_RISK_AREAS = {"SEO", "Content"}


@dataclass
class Alert:
    level: str        # info | warning | critical
    metric: str
    message: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class PlannedAction:
    area: str
    action: str
    expected_gain: str
    priority: int
    status: str       # proposed | simulated

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentReport:
    cycle: int
    timestamp: str
    source_summary: dict
    kpis: dict
    alerts: list = field(default_factory=list)          # list[Alert]
    actions: list = field(default_factory=list)         # list[PlannedAction]
    report_path: str = ""

    def markdown(self) -> str:
        return _render_markdown(self)


# --------------------------------------------------------------------------
# Snapshot diffing (pure, unit-testable)
# --------------------------------------------------------------------------
def diff_snapshots(prev: dict | None, curr: dict) -> list[Alert]:
    """Compare two KPI snapshots and emit change alerts."""
    alerts: list[Alert] = []
    if not prev:
        alerts.append(Alert("info", "baseline",
                            "First run — no previous snapshot to compare against. "
                            "Baseline saved."))
        return alerts

    # Traffic.
    p_v, c_v = prev.get("avg_daily_visitors", 0), curr.get("avg_daily_visitors", 0)
    if p_v:
        change = (c_v - p_v) / p_v
        if change <= -0.05:
            alerts.append(Alert("critical", "traffic",
                                f"Average daily visitors fell {abs(change):.0%} "
                                f"({p_v:,} → {c_v:,})."))
        elif change >= 0.10:
            alerts.append(Alert("info", "traffic",
                                f"Traffic up {change:.0%} ({p_v:,} → {c_v:,})."))

    # Health score.
    p_h, c_h = prev.get("health_score", 0), curr.get("health_score", 0)
    if c_h - p_h <= -3:
        alerts.append(Alert("warning", "health",
                            f"Health score dropped {p_h:.0f} → {c_h:.0f}."))
    elif c_h - p_h >= 3:
        alerts.append(Alert("info", "health",
                            f"Health score improved {p_h:.0f} → {c_h:.0f}."))

    # Anomalies.
    p_a, c_a = prev.get("n_anomalies", 0), curr.get("n_anomalies", 0)
    if c_a > p_a:
        alerts.append(Alert("warning", "anomaly",
                            f"{c_a - p_a} new anomalous day(s) detected "
                            f"({p_a} → {c_a})."))

    # Forecast direction.
    fc = curr.get("predicted_traffic_change_pct", 0)
    if fc <= -5:
        alerts.append(Alert("warning", "forecast",
                            f"Model forecasts a {fc:.1f}% traffic decline ahead."))

    if not alerts:
        alerts.append(Alert("info", "stable", "No material changes since last cycle."))
    return alerts


# --------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------
class AutonomousGrowthAgent:
    """Runs scheduled growth-analysis cycles with human-in-the-loop autonomy."""

    def __init__(self, connector: DataSource | None = None, autonomy: str = "propose"):
        if autonomy not in {"propose", "auto"}:
            raise ValueError("autonomy must be 'propose' or 'auto'")
        self.connector = connector or get_connector("local")
        self.autonomy = autonomy

    def run_cycle(self, horizon: int = 30, now: datetime | None = None) -> AgentReport:
        """Execute one full agent cycle and return the report."""
        now = now or datetime.utcnow()

        # 1. Sync data.
        source_summary = self.connector.sync()

        # 2. Ensure models exist.
        if not TRAFFIC_MODEL.exists():
            train_all(save=True)

        # 3. Analyse.
        insights = analyze(horizon=horizon)

        # 4. Diff against previous snapshot.
        prev = self._load_snapshot()
        alerts = diff_snapshots(prev, insights.kpis)

        # 5. Plan actions.
        actions = self._plan_actions(insights)

        # 6. Persist + report.
        state = self._load_state()
        cycle = int(state.get("cycle", 0)) + 1
        report = AgentReport(
            cycle=cycle,
            timestamp=now.strftime("%Y-%m-%d %H:%M UTC"),
            source_summary=source_summary,
            kpis=insights.kpis,
            alerts=alerts,
            actions=actions,
        )
        report.report_path = self._write_report(report)
        self._save_state(cycle, insights.kpis, now)
        self._append_action_log(report)
        return report

    # -- planning ----------------------------------------------------------
    def _plan_actions(self, insights) -> list[PlannedAction]:
        actions: list[PlannedAction] = []
        for rec in insights.recommendations[:10]:
            auto_ok = self.autonomy == "auto" and rec.area in LOW_RISK_AREAS and rec.priority <= 2
            actions.append(PlannedAction(
                area=rec.area,
                action=rec.action,
                expected_gain=rec.expected_gain,
                priority=rec.priority,
                # 'simulated' = would have been auto-applied if a live platform
                # were connected; logged, never actually executed here.
                status="simulated" if auto_ok else "proposed",
            ))
        return actions

    # -- state persistence -------------------------------------------------
    def _load_state(self) -> dict:
        if SNAPSHOT_PATH.exists():
            return json.loads(SNAPSHOT_PATH.read_text())
        return {}

    def _load_snapshot(self) -> dict | None:
        return self._load_state().get("kpis")

    def _save_state(self, cycle: int, kpis: dict, now: datetime) -> None:
        SNAPSHOT_PATH.write_text(json.dumps(
            {"cycle": cycle, "updated": now.isoformat(), "kpis": kpis}, indent=2
        ))

    def _write_report(self, report: AgentReport) -> str:
        path = REPORTS_DIR / f"growth_report_{report.cycle:04d}.md"
        path.write_text(report.markdown(), encoding="utf-8")
        return str(path)

    def _append_action_log(self, report: AgentReport) -> None:
        with ACTION_LOG_PATH.open("a", encoding="utf-8") as fh:
            for a in report.actions:
                fh.write(f"{report.timestamp}\tcycle={report.cycle}\t"
                         f"{a.status}\t[{a.area}] {a.action}\n")


# --------------------------------------------------------------------------
# Report rendering
# --------------------------------------------------------------------------
def _render_markdown(report: AgentReport) -> str:
    k = report.kpis
    lines = [
        f"# GrowthMind AI — Daily Growth Report #{report.cycle}",
        f"_{report.timestamp} · source: {report.source_summary.get('source', 'local')}_",
        "",
        "## Key metrics",
        f"- **Health score:** {k['health_score']:.0f}/100 (grade {k['health_grade']})",
        f"- **Avg daily visitors:** {k['avg_daily_visitors']:,}",
        f"- **{k['forecast_horizon_days']}-day forecast:** {k['forecast_avg_visitors']:,} "
        f"({k['predicted_traffic_change_pct']:+.1f}%)",
        f"- **Monthly revenue:** ${k['monthly_revenue']:,.0f}",
        f"- **Conversion rate:** {k['avg_conversion_rate']:.2%}",
        f"- **Anomalous days:** {k['n_anomalies']}",
        "",
        "## What changed since last cycle",
    ]
    for a in report.alerts:
        icon = {"critical": "🔴", "warning": "🟠", "info": "🟢"}.get(a.level, "•")
        lines.append(f"- {icon} **{a.metric}** — {a.message}")

    lines += ["", "## Action plan"]
    if not report.actions:
        lines.append("- (no actions)")
    for a in report.actions:
        tag = "🤖 auto (simulated)" if a.status == "simulated" else "🙋 proposed"
        lines.append(f"- **[P{a.priority}] {a.area}** — {a.action} "
                     f"_({a.expected_gain})_ · {tag}")

    lines += [
        "",
        "---",
        "_Generated by the GrowthMind Autonomous Growth Agent. "
        "Proposed actions require human approval; 'simulated' actions are logged, "
        "not executed (no live platform is connected in this build)._",
    ]
    return "\n".join(lines)

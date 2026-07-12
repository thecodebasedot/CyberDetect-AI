"""Tests for the connectors and the Autonomous Growth Agent."""

from __future__ import annotations

import pytest

from growthmind.agent import AutonomousGrowthAgent, diff_snapshots
from growthmind.connectors import (
    LocalConnector,
    available_sources,
    get_connector,
)


# --------------------------------------------------------------------------
# Connectors
# --------------------------------------------------------------------------
def test_registry_and_local_connector():
    assert "local" in available_sources()
    assert isinstance(get_connector("local"), LocalConnector)
    with pytest.raises(ValueError):
        get_connector("does-not-exist")


def test_local_connector_sync_summary():
    summary = LocalConnector().sync()
    assert summary["source"] == "local"
    assert summary["daily_rows"] > 0
    assert len(summary["date_range"]) == 2


def test_google_connectors_are_gated_stubs():
    gsc = get_connector("gsc")
    # No credentials in the test environment -> unavailable and not implemented.
    assert gsc.is_available() is False
    with pytest.raises(NotImplementedError):
        gsc.sync()


# --------------------------------------------------------------------------
# Snapshot diffing (pure logic)
# --------------------------------------------------------------------------
def test_diff_first_run_is_baseline():
    alerts = diff_snapshots(None, {"avg_daily_visitors": 1000, "health_score": 70})
    assert len(alerts) == 1
    assert alerts[0].metric == "baseline"


def test_diff_detects_traffic_drop():
    prev = {"avg_daily_visitors": 1000, "health_score": 70, "n_anomalies": 5,
            "predicted_traffic_change_pct": 0}
    curr = {"avg_daily_visitors": 900, "health_score": 70, "n_anomalies": 5,
            "predicted_traffic_change_pct": 0}
    alerts = diff_snapshots(prev, curr)
    metrics = {a.metric for a in alerts}
    assert "traffic" in metrics
    assert any(a.level == "critical" for a in alerts)


def test_diff_detects_health_and_anomaly_changes():
    prev = {"avg_daily_visitors": 1000, "health_score": 80, "n_anomalies": 3,
            "predicted_traffic_change_pct": 0}
    curr = {"avg_daily_visitors": 1000, "health_score": 74, "n_anomalies": 7,
            "predicted_traffic_change_pct": 0}
    alerts = diff_snapshots(prev, curr)
    metrics = {a.metric for a in alerts}
    assert "health" in metrics
    assert "anomaly" in metrics


def test_diff_stable_when_no_change():
    snap = {"avg_daily_visitors": 1000, "health_score": 80, "n_anomalies": 3,
            "predicted_traffic_change_pct": 0}
    alerts = diff_snapshots(dict(snap), dict(snap))
    assert any(a.metric == "stable" for a in alerts)


# --------------------------------------------------------------------------
# Agent cycle (integration)
# --------------------------------------------------------------------------
def test_agent_cycle_produces_report(tmp_path, monkeypatch):
    # Redirect agent state/report files into a temp dir for isolation.
    import growthmind.agent as agent_mod
    monkeypatch.setattr(agent_mod, "SNAPSHOT_PATH", tmp_path / "snap.json")
    monkeypatch.setattr(agent_mod, "ACTION_LOG_PATH", tmp_path / "actions.log")
    monkeypatch.setattr(agent_mod, "REPORTS_DIR", tmp_path)

    agent = AutonomousGrowthAgent(autonomy="propose")

    r1 = agent.run_cycle(horizon=14)
    assert r1.cycle == 1
    assert r1.kpis["health_score"] > 0
    assert any(a.metric == "baseline" for a in r1.alerts)
    assert (tmp_path / "snap.json").exists()
    assert "GrowthMind" in r1.markdown()

    # Second cycle increments and compares against the baseline snapshot.
    r2 = agent.run_cycle(horizon=14)
    assert r2.cycle == 2
    assert not any(a.metric == "baseline" for a in r2.alerts)


def test_agent_auto_mode_only_simulates_low_risk(tmp_path, monkeypatch):
    import growthmind.agent as agent_mod
    monkeypatch.setattr(agent_mod, "SNAPSHOT_PATH", tmp_path / "snap.json")
    monkeypatch.setattr(agent_mod, "ACTION_LOG_PATH", tmp_path / "actions.log")
    monkeypatch.setattr(agent_mod, "REPORTS_DIR", tmp_path)

    report = AutonomousGrowthAgent(autonomy="auto").run_cycle(horizon=14)
    # Any simulated action must be a low-risk area; risky areas stay proposed.
    for a in report.actions:
        if a.status == "simulated":
            assert a.area in agent_mod.LOW_RISK_AREAS
            assert a.priority <= 2
    assert any(a.status == "proposed" for a in report.actions)


def test_agent_rejects_bad_autonomy():
    with pytest.raises(ValueError):
        AutonomousGrowthAgent(autonomy="yolo")

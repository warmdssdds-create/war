"""Tests for src/monitoring/health.py."""

from __future__ import annotations

import json
import urllib.request

import pytest

from src.monitoring.health import HealthServer, HealthState, update_state, get_state


def test_health_state_to_dict_contains_checked_at() -> None:
    state = HealthState()
    d = state.to_dict()
    assert "checked_at" in d
    assert "status" in d


def test_health_state_update() -> None:
    state = HealthState()
    state.update(status="running", daily_pnl=-5.0, consecutive_losses=2)
    assert state.status == "running"
    assert state.daily_pnl == -5.0
    assert state.consecutive_losses == 2


def test_health_state_update_ignores_unknown_keys() -> None:
    state = HealthState()
    # Should not raise
    state.update(nonexistent_key="value")
    assert not hasattr(state, "nonexistent_key")


def test_health_server_serves_json() -> None:
    server = HealthServer(host="127.0.0.1", port=18080)
    server.start()
    try:
        update_state(status="running", last_signal="Buy")
        with urllib.request.urlopen("http://127.0.0.1:18080/health", timeout=3) as resp:
            data = json.loads(resp.read())
        assert data["status"] == "running"
        assert "checked_at" in data
    finally:
        server.stop()


def test_health_server_404_for_unknown_path() -> None:
    server = HealthServer(host="127.0.0.1", port=18081)
    server.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen("http://127.0.0.1:18081/unknown", timeout=3)
        assert exc_info.value.code == 404
    finally:
        server.stop()

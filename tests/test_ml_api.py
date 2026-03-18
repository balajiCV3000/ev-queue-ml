"""Tests for RL policy API endpoints."""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


@pytest.fixture
def client():
    import os

    os.environ["BOOTSTRAP_SIMULATION"] = "false"
    import importlib

    import app as app_module
    import config

    importlib.reload(config)
    importlib.reload(app_module)
    return app_module.app.test_client()


def test_get_policy_status(client):
    resp = client.get("/api/policy")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "active" in data
    assert "available" in data
    assert "greedy" in data["available"]


def test_set_policy_greedy(client):
    resp = client.post("/api/policy", json={"policy": "greedy"})
    assert resp.status_code == 200
    assert resp.get_json()["success"] is True


def test_set_policy_unknown_returns_400(client):
    resp = client.post("/api/policy", json={"policy": "nonexistent"})
    assert resp.status_code == 400


def test_state_includes_policy_field(client):
    resp = client.get("/api/simulation/state")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "policy" in data

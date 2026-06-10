import importlib
import os
import pathlib
import sys

import pytest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


@pytest.fixture
def app_module():
    os.environ["BOOTSTRAP_SIMULATION"] = "false"
    import app as app_module
    import config

    importlib.reload(config)
    importlib.reload(app_module)
    return app_module


@pytest.fixture
def client(app_module):
    return app_module.app.test_client()


def test_results_page_returns_200_and_distinguishes_rl(client):
    resp = client.get("/results")

    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "RL vs. Baseline Results" in body
    assert "<strong>rl</strong>" in body
    assert "greedy" in body


def test_results_page_shows_placeholder_when_files_missing(app_module, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "RESULTS_DIR", str(tmp_path))
    client = app_module.app.test_client()

    resp = client.get("/results")

    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "python -m ml.evaluate" in body


def test_how_it_works_page_returns_200_with_both_diagrams(client):
    resp = client.get("/how-it-works")

    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "Full Pipeline" in body
    assert "RL Decision" in body

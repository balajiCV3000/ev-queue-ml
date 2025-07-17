import importlib
import pathlib
import sys


def test_health_endpoint_returns_healthy(monkeypatch):
    monkeypatch.setenv("BOOTSTRAP_SIMULATION", "false")
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

    app_module = importlib.import_module("app")
    client = app_module.app.test_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "healthy"}

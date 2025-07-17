import importlib
import pathlib
import sys


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def test_reset_endpoint_returns_json_on_reset_failure(monkeypatch):
    monkeypatch.setenv("BOOTSTRAP_SIMULATION", "false")
    app_module = importlib.import_module("app")

    class BrokenSimulation:
        def reset(self):
            raise RuntimeError("reset exploded")

    monkeypatch.setattr(app_module, "simulation", BrokenSimulation())
    client = app_module.app.test_client()

    response = client.post("/api/simulation/reset")

    assert response.status_code == 500
    assert response.is_json
    assert response.get_json() == {"success": False, "error": "reset exploded"}

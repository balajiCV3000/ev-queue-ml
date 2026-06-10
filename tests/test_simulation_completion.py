import pathlib
import sys
import time


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from models.ev import EV
from models.simulation import Simulation


def _make_single_step_ev():
    """An EV whose single-point route completes on its very first move()."""
    return EV(
        id="ev-1",
        origin=(12.9716, 77.5946),
        destination=(12.9716, 77.5946),
        battery_capacity=60,
        initial_soc=0.9,
        consumption_rate=0.2,
        route=[(12.9716, 77.5946)],
    )


def test_run_headless_reports_done_and_completion_metrics():
    ev = _make_single_step_ev()
    simulation = Simulation([ev], [], [])

    result = simulation.run_headless(max_steps=10)

    assert result["done"] is True
    assert ev.trip_completed is True


def test_thread_driven_simulation_stops_itself_and_reports_completed_status():
    ev = _make_single_step_ev()
    simulation = Simulation([ev], [], [])

    assert simulation.start() is True

    deadline = time.time() + 5
    while time.time() < deadline and not simulation.completed:
        time.sleep(0.05)

    assert simulation.completed is True
    assert simulation.running is False
    assert simulation.get_current_state()["status"] == "completed"

    simulation.stop()


def test_start_refuses_when_already_completed():
    ev = _make_single_step_ev()
    simulation = Simulation([ev], [], [])
    simulation.completed = True

    assert simulation.start() is False

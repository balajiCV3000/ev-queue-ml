import pathlib
import sys
from datetime import datetime


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from models.ev import EV
from models.simulation import Simulation
from models.station import ChargingStation


def test_current_state_is_available_before_first_recorded_step():
    simulation = Simulation([], [], [])

    state = simulation.get_current_state()

    assert state["step"] == 0
    assert state["running"] is False
    assert state["evs"] == []
    assert state["stations"] == []
    assert state["metrics"]["max_queue_length"] == 0


def test_step_handles_empty_station_list():
    simulation = Simulation([], [], [])

    simulation.step()

    assert simulation.current_step == 1
    assert simulation.metrics["max_queue_length"] == 0


def test_reset_restores_ev_and_station_runtime_state():
    ev = EV(
        id="ev-1",
        origin=(12.9716, 77.5946),
        destination=(12.9816, 77.6046),
        battery_capacity=60,
        initial_soc=0.42,
        consumption_rate=0.2,
        route=[(12.9716, 77.5946), (12.9816, 77.6046)],
    )
    station = ChargingStation(id="station-1", location=(12.975, 77.6), num_chargers=1)
    simulation = Simulation([ev], [station], [])

    ev.current_position = ev.destination
    ev.route_index = 1
    ev.soc = 0.12
    ev.charging = True
    ev.in_queue = True
    ev.assigned_station = station
    ev.queue_arrival_time = datetime.now()
    ev.charging_start_time = datetime.now()
    ev.waiting_time = 300
    ev.target_soc = 1.0
    ev.trip_completed = True
    ev.abandoned = True
    ev.trip_end_time = datetime.now()
    station.charging_evs = [ev]
    station.queue.append(ev)
    station.total_served = 3
    station.total_wait_time = 900
    station.max_queue_length = 2

    assert simulation.reset() is True

    assert ev.current_position == ev.origin
    assert ev.route_index == 0
    assert ev.soc == ev.initial_soc
    assert ev.charging is False
    assert ev.in_queue is False
    assert ev.assigned_station is None
    assert ev.queue_arrival_time is None
    assert ev.charging_start_time is None
    assert ev.waiting_time == 0
    assert ev.target_soc == 0.8
    assert ev.trip_completed is False
    assert ev.abandoned is False
    assert ev.trip_end_time is None
    assert len(ev.journey_log) == 1
    assert station.charging_evs == []
    assert len(station.queue) == 0
    assert station.total_served == 0
    assert station.total_wait_time == 0
    assert station.max_queue_length == 0

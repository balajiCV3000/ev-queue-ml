"""Tests for ML policy parity, env determinism, masking, RewardTracker, fallback."""

import copy
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml.features import NUM_FEATURES, extract_features, build_feature_matrix
from ml.policies.greedy import GreedyPolicy
from ml.policies.nearest import NearestPolicy
from ml.policies.registry import get_policy
from ml.policies.rl_policy import RLPolicy
from ml.tracking import RewardTracker
from models.ev import EV
from models.optimization import optimize_charging
from models.simulation import Simulation
from models.station import ChargingStation


def _make_ev(ev_id="ev-1", soc=0.15):
    route = [(12.9716, 77.5946), (12.975, 77.598), (12.9816, 77.6046)]
    return EV(
        id=ev_id,
        origin=route[0],
        destination=route[-1],
        battery_capacity=50,
        initial_soc=soc,
        consumption_rate=0.2,
        route=route,
    )


def _make_station(station_id="station-1", chargers=2):
    return ChargingStation(
        id=station_id,
        location=(12.973, 77.596),
        num_chargers=chargers,
        charging_rate=11.0,
    )


class TestGreedyPolicyParity:
    def test_greedy_matches_optimize_charging(self):
        evs = [_make_ev("ev-1"), _make_ev("ev-2", soc=0.18)]
        stations = [_make_station("s1"), _make_station("s2", chargers=3)]
        ctx = {"current_step": 0}

        policy_assignments, policy_abandoned = GreedyPolicy().assign(
            copy.deepcopy(evs), copy.deepcopy(stations), ctx
        )
        direct_assignments, direct_abandoned = optimize_charging(
            copy.deepcopy(evs), copy.deepcopy(stations)
        )

        assert policy_assignments == direct_assignments
        assert policy_abandoned == direct_abandoned


class TestFeatureExtraction:
    def test_feature_count(self):
        ev = _make_ev()
        station = _make_station()
        features = extract_features(ev, station)
        assert features.shape == (NUM_FEATURES,)
        assert features.dtype == np.float32

    def test_masking_unreachable_station(self):
        ev = _make_ev()
        ev.soc = 0.001
        stations = [_make_station("far", chargers=1)]
        stations[0].location = (13.5, 78.5)
        features, _, _, ev_ids, _ = build_feature_matrix([ev], stations)
        assert len(features) == 0
        assert len(ev_ids) == 0


class TestRewardTracker:
    def test_wait_time_accumulation(self):
        tracker = RewardTracker()
        ev = _make_ev()
        ev.waiting_time = 0
        stations = [_make_station()]

        tracker.observe([ev], stations, 60)
        ev.waiting_time = 120
        tracker.observe([ev], stations, 60)

        summary = tracker.summary()
        assert summary["total_wait"] == 120
        assert summary["reward"] == -summary["total_system"]

    def test_travel_time_accrues_at_default_weight(self):
        tracker = RewardTracker()
        ev = _make_ev()
        ev.waiting_time = 0
        ev.en_route_to_charger = True
        stations = [_make_station()]

        tracker.observe([ev], stations, 60)
        tracker.observe([ev], stations, 60)

        summary = tracker.summary()
        assert summary["total_travel"] == 120
        assert summary["total_system"] == 120
        assert summary["reward"] == -120

    def test_travel_weight_scales_system_time_not_total_travel(self):
        tracker = RewardTracker(travel_weight=1.5)
        ev = _make_ev()
        ev.waiting_time = 0
        ev.en_route_to_charger = True
        stations = [_make_station()]

        tracker.observe([ev], stations, 60)
        tracker.observe([ev], stations, 60)

        summary = tracker.summary()
        assert summary["total_travel"] == 120
        assert summary["total_system"] == 180
        assert summary["reward"] == -180


class TestEnvDeterminism:
    def test_same_seed_same_reward(self):
        def run(seed):
            policy = GreedyPolicy()
            evs = [_make_ev(f"ev-{i}") for i in range(3)]
            stations = [_make_station(f"s-{i}") for i in range(2)]
            sim = Simulation(copy.deepcopy(evs), copy.deepcopy(stations), [], policy)
            result = sim.run_headless(max_steps=50)
            return result["reward_summary"]["reward"]

        assert run(42) == run(42)


class TestRLFallback:
    def test_missing_model_falls_back_to_greedy(self):
        policy = RLPolicy(model_path="artifacts/nonexistent.npz")
        assert policy.fallback_active is True
        evs = [_make_ev()]
        stations = [_make_station()]
        assignments, _ = policy.assign(evs, stations, {})
        direct, _ = optimize_charging(copy.deepcopy(evs), copy.deepcopy(stations))
        assert assignments == direct

    def test_smoke_model_loads(self):
        smoke_path = pathlib.Path("artifacts/policy_smoke.npz")
        if not smoke_path.exists():
            pytest.skip("smoke model not generated yet")
        policy = RLPolicy(model_path=str(smoke_path))
        assert policy.model_loaded is True
        assert policy.fallback_active is False

        evs = [_make_ev()]
        stations = [_make_station()]
        assignments, abandoned = policy.assign(evs, stations, {})
        assert isinstance(assignments, dict)
        assert isinstance(abandoned, list)


def _make_congested_fixture():
    """Hand-crafted EV/station pair where nearest-distance and total-time
    (travel + wait + charge) heuristics disagree.

    Station A is very close to the EV but has a full charger plus a long
    queue (large wait estimate); Station B is a few km away but has no
    queue and free chargers. A pure-distance policy (nearest) should pick A;
    a total-time-aware policy (greedy/RL) should pick B instead.

    B sits at a realistic detour distance (~3.4km, well within the range of
    assignments seen in training) rather than tens of km out: since travel
    time was priced into the reward, the RL policy deliberately no longer
    detours to far-off stations, and Q-value extrapolation on
    far-off-distribution (distance, wait) pairs is unreliable -- the old
    19km placement only "worked" because the pre-fix model treated travel
    as free.
    """
    ev = _make_ev("target", soc=0.3)

    station_a = _make_station("A", chargers=1)
    station_a.location = (12.9720, 77.5950)  # very close to ev's origin
    occupant = _make_ev("occupant", soc=0.05)
    occupant.charging = True
    station_a.charging_evs = [occupant]
    for i in range(5):
        station_a.queue.append(_make_ev(f"queued-{i}", soc=0.1))

    station_b = _make_station("B", chargers=2)
    station_b.location = (12.99, 77.62)  # ~3.4km away, but empty and free

    assert station_a.get_current_wait_time_estimate() > station_b.get_current_wait_time_estimate()
    return ev, station_a, station_b


class TestContentionDivergence:
    """Verify greedy/RL don't collapse to nearest-distance behavior under
    real contention (queued/occupied close station vs. free distant one)."""

    def test_greedy_differs_from_nearest_under_contention(self):
        ev, station_a, station_b = _make_congested_fixture()

        nearest_assignments, _ = NearestPolicy().assign([ev], [station_a, station_b], {})
        greedy_assignments, _ = GreedyPolicy().assign([ev], [station_a, station_b], {})

        assert nearest_assignments["target"] == "A"
        assert greedy_assignments["target"] == "B"
        assert nearest_assignments["target"] != greedy_assignments["target"]

    def test_rl_uses_loaded_weights_and_differs_from_nearest(self):
        policy = RLPolicy()
        assert policy.model_loaded is True
        assert policy.fallback_active is False

        ev, station_a, station_b = _make_congested_fixture()

        nearest_assignments, _ = NearestPolicy().assign([ev], [station_a, station_b], {})
        rl_assignments, _ = policy.assign([ev], [station_a, station_b], {})

        assert nearest_assignments["target"] == "A"
        assert rl_assignments["target"] == "B"
        assert rl_assignments["target"] != nearest_assignments["target"]


class TestSimulationStrategy:
    def test_policy_in_state(self):
        sim = Simulation([], [], [], GreedyPolicy())
        state = sim.get_current_state()
        assert state["policy"] == "greedy"
        assert "fallback_active" in state

    def test_run_headless_completes(self):
        evs = [_make_ev(f"ev-{i}") for i in range(2)]
        stations = [_make_station(f"s-{i}") for i in range(2)]
        sim = Simulation(evs, stations, [], GreedyPolicy())
        result = sim.run_headless(max_steps=100)
        assert "reward_summary" in result
        assert "reward" in result["reward_summary"]

    def test_set_policy(self):
        sim = Simulation([], [], [], GreedyPolicy())
        sim.set_policy("nearest")
        assert sim.get_policy_name() == "nearest"

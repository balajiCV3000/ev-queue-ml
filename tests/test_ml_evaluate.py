"""Integration tests for ml.evaluate.run_single under a congested scenario."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml.evaluate import run_single


class TestRunSingleCongested:
    """Verify the metrics pipeline actually produces non-zero, populated
    values under real contention, instead of collapsing to the old broken
    baseline (average_wait_time=0, max_queue_length=0, etc.)."""

    def test_congested_scenario_produces_nonzero_metrics(self):
        row = run_single(
            "greedy",
            seed=1,
            num_evs=40,
            num_stations=5,
            max_steps=150,
            scenario="congested",
        )

        assert row["scenario"] == "congested"
        assert row["average_wait_time"] > 0
        assert row["max_queue_length"] > 0
        assert row["avg_queue_length"] > 0
        assert row["total_travel_distance_km"] > 0
        assert row["avg_station_utilization"] > 0
        assert row["vehicles_served"] > 0
        assert row["model_loaded"] is True
        assert row["rl_fallback"] is False

    def test_congested_scenario_rl_uses_real_model(self):
        row = run_single(
            "rl",
            seed=1,
            num_evs=40,
            num_stations=5,
            max_steps=150,
            scenario="congested",
        )

        assert row["model_loaded"] is True
        assert row["rl_fallback"] is False
        assert row["total_travel_distance_km"] > 0

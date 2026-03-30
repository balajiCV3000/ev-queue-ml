"""Tests for experiment scenarios."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experiments.scenarios import apply_scenario, _apply_outage
from models.ev import EV
from models.station import ChargingStation


def _station(n_chargers):
    return ChargingStation(
        id=f"s-{n_chargers}",
        location=(12.97, 77.59),
        num_chargers=n_chargers,
    )


class TestOutageScenario:
    def test_outage_degrades_to_one_charger_not_zero(self):
        stations = [_station(4), _station(3), _station(2), _station(2)]
        _apply_outage(stations)
        zero = [s for s in stations if s.num_chargers == 0]
        assert len(zero) == 0
        assert stations[0].num_chargers == 1
        assert stations[1].num_chargers == 1
        assert stations[2].num_chargers == 1
        assert stations[3].num_chargers == 2


class TestRushScenario:
    def test_rush_sets_low_soc(self):
        evs = [
            EV(
                id="ev-1",
                origin=(12.97, 77.59),
                destination=(12.98, 77.60),
                battery_capacity=50,
                initial_soc=0.8,
                consumption_rate=0.2,
                route=[(12.97, 77.59), (12.98, 77.60)],
            )
        ]
        apply_scenario("rush", evs, [], step=0)
        assert 0.15 <= evs[0].soc <= 0.30

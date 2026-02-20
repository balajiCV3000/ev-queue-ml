"""Experiment scenario definitions."""

import copy
import random

from utils.data_generator import generate_synthetic_data, interpolate_route
from models.ev import EV


def scenario_density(num_evs, seed, num_stations=20):
    """Standard density scenario."""
    def fn(s):
        return generate_synthetic_data(
            num_evs=num_evs, num_stations=num_stations,
            num_nodes=40, num_routes=80, use_cache=False, seed=s,
        )
    return fn


def scenario_station_outage(num_evs=50, num_stations=20, outage_step=300, seed=42):
    """Degrade top-capacity stations to 1 charger at outage_step."""
    base_fn = scenario_density(num_evs, seed, num_stations)

    class OutageWrapper:
        def __init__(self):
            self._outage_applied = False
            self._outage_step = outage_step

        def __call__(self, s):
            return base_fn(s)

        def check_outage(self, sim):
            if self._outage_applied or sim.current_step < self._outage_step:
                return
            ranked = sorted(sim.stations, key=lambda s: s.num_chargers, reverse=True)
            for station in ranked[:3]:
                station.num_chargers = 1
            self._outage_applied = True

    return OutageWrapper()


def scenario_rush_burst(num_evs=50, num_stations=20, burst_size=20, seed=42):
    """Rush scenario with burst of low-SoC EVs."""
    def fn(s):
        evs, stations, routes = generate_synthetic_data(
            num_evs=num_evs - burst_size, num_stations=num_stations,
            num_nodes=40, num_routes=80, use_cache=False, seed=s,
        )
        rng = random.Random(s)
        for i in range(burst_size):
            route = rng.choice(routes)
            ev = EV(
                id=f"rush-ev-{i+1}",
                origin=route["origin"],
                destination=route["destination"],
                battery_capacity=rng.uniform(30, 50),
                initial_soc=rng.uniform(0.15, 0.30),
                consumption_rate=rng.uniform(0.15, 0.25),
                route=route["points"],
            )
            evs.append(ev)
        return evs, stations, routes
    return fn


def scenario_congested(num_evs=180, num_stations=12, seed=42, soc_range=(0.10, 0.25)):
    """High-traffic / congested scenario data generator.

    Deliberately skews the EV:charger ratio high and initial SoC low so a
    large fraction of the fleet needs charging concurrently and queues
    actually form (see plan doc: this is the scenario used to make
    nearest/greedy/RL policies diverge).
    """
    def fn(s):
        evs, stations, routes = generate_synthetic_data(
            num_evs=num_evs, num_stations=num_stations,
            num_nodes=40, num_routes=80, use_cache=False, seed=s,
        )
        rng = random.Random(s)
        lo, hi = soc_range
        for ev in evs:
            ev.initial_soc = rng.uniform(lo, hi)
            ev.soc = ev.initial_soc
        return evs, stations, routes
    return fn


# Named traffic-density configs, exposed as a plain dict so the grid-running
# phase can iterate over "low"/"medium"/"high"/"congested" conditions without
# needing to re-derive EV:charger ratios or SoC ranges. `num_evs`/`num_stations`
# control contention via the EV:charger ratio; `soc_range` controls how many
# EVs need to charge early in the episode.
TRAFFIC_DENSITY_SCENARIOS = {
    "low": {"num_evs": 40, "num_stations": 25, "soc_range": (0.35, 0.80)},
    "medium": {"num_evs": 100, "num_stations": 20, "soc_range": (0.20, 0.60)},
    "high": {"num_evs": 180, "num_stations": 15, "soc_range": (0.15, 0.40)},
    "congested": {"num_evs": 220, "num_stations": 10, "soc_range": (0.10, 0.25)},
}


def scenario_traffic(density, seed=42):
    """Return a data-generator fn for a named traffic-density scenario.

    Usable as `EnvWrapper(scenario_fn=scenario_traffic("high", seed=1))`.
    `density` must be a key of TRAFFIC_DENSITY_SCENARIOS
    ("low", "medium", "high", "congested").
    """
    cfg = TRAFFIC_DENSITY_SCENARIOS[density]
    return scenario_congested(
        num_evs=cfg["num_evs"],
        num_stations=cfg["num_stations"],
        seed=seed,
        soc_range=cfg["soc_range"],
    )


SCENARIOS = {
    "density_50": lambda: scenario_density(50, seed=42),
    "density_100": lambda: scenario_density(100, seed=42),
    "density_200": lambda: scenario_density(200, seed=42),
    "station_outage": lambda: scenario_station_outage(),
    "rush_burst": lambda: scenario_rush_burst(),
    "traffic_low": lambda: scenario_traffic("low", seed=42),
    "traffic_medium": lambda: scenario_traffic("medium", seed=42),
    "traffic_high": lambda: scenario_traffic("high", seed=42),
    "traffic_congested": lambda: scenario_traffic("congested", seed=42),
}


def _apply_outage(stations, top_n=3):
    """Degrade top-capacity stations to 1 charger (not zero)."""
    ranked = sorted(stations, key=lambda s: s.num_chargers, reverse=True)
    for station in ranked[:top_n]:
        station.num_chargers = 1


def _apply_congestion(evs, soc_range=(0.10, 0.25), step=0):
    """Force a large fraction of an already-generated fleet to need charging.

    Complements `scenario_traffic`/`scenario_congested` (which control the
    EV:charger ratio at data-generation time) for callers that only have
    access to `apply_scenario(name, evs, stations, step)`, e.g. EnvWrapper's
    string `scenario=` path.
    """
    rng = random.Random(step)
    lo, hi = soc_range
    for ev in evs:
        ev.soc = rng.uniform(lo, hi)
        ev.initial_soc = ev.soc


def apply_scenario(name, evs, stations, step=0):
    """Apply in-simulation scenario mutations."""
    if name == "rush":
        rng = random.Random(step)
        for ev in evs:
            ev.soc = rng.uniform(0.15, 0.30)
            ev.initial_soc = ev.soc
    elif name == "outage" and step >= 300:
        _apply_outage(stations)
    elif name == "congested" and step == 0:
        _apply_congestion(evs)

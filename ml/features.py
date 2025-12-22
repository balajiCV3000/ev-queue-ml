"""Feature extraction for EV-station pair scoring (23 features).

Features 19-22 give the network system-wide contention context (fleet
load relative to total charger capacity, mean utilization/queue across
all stations, and this station's queue relative to that mean). Without
them, a pair-scoring network cannot tell "medium load, this station is
relatively busy" from "high load, this station is actually one of the
better options" -- the optimal distance-vs-queue trade-off shifts with
overall system load, and the earlier 19-feature network had no way to
condition on which regime it was in. This was diagnosed as a likely
cause of RL systematically over-preferring far stations (avg travel
distance ~7-8km vs nearest's ~2-2.5km, essentially independent of
density) while leaving nearby stations under-utilized at higher density.
"""

import numpy as np
from models.maps_service import calculate_distance

NUM_FEATURES = 23
FEATURE_NAMES = [
    "ev_soc",
    "ev_battery_capacity_norm",
    "ev_consumption_rate",
    "remaining_distance_km",
    "energy_needed_kwh",
    "energy_needed_soc",
    "critical_soc",
    "travel_distance_km",
    "travel_time_s",
    "wait_time_s",
    "charge_time_s",
    "total_time_s",
    "on_route",
    "station_num_chargers",
    "station_charging_rate_norm",
    "station_queue_length",
    "station_utilization",
    "route_progress",
    "shadow_pending",
    "fleet_load_ratio",
    "mean_station_utilization",
    "mean_queue_length_norm",
    "station_queue_relative",
]


def _travel_time_s(ev, station):
    """Same haversine/30 km/h formula as greedy scorer."""
    travel_km = calculate_distance(ev.current_position, station.location) / 1000.0
    return travel_km / 30.0 * 3600.0


def _charge_time_s(ev, station):
    energy_needed = ev.calculate_energy_needed_for_destination()
    current_energy = ev.soc * ev.battery_capacity
    energy_to_charge = max(0.0, energy_needed - current_energy)
    min_charge = 0.1 * ev.battery_capacity
    energy_to_charge = max(energy_to_charge, min_charge)

    if station.get_queue_length() > 0 or len(station.charging_evs) >= station.num_chargers:
        energy_80pct = 0.8 * ev.battery_capacity - current_energy
        if energy_needed > 0.8 * ev.battery_capacity:
            energy_to_charge = max(energy_to_charge, min_charge)
        else:
            energy_to_charge = min(energy_to_charge, max(energy_80pct, min_charge))

    return (energy_to_charge / station.charging_rate) * 3600.0


def compute_global_context(evs_needing_charge, stations):
    """Fleet/system-wide contention summary, computed once per optimization round.

    Cheap to compute (O(num_stations)) so callers should build this once per
    round and pass it to every `extract_pair_features` call that round rather
    than recomputing per (EV, station) pair.
    """
    total_chargers = sum(s.num_chargers for s in stations) or 1
    fleet_load_ratio = len(evs_needing_charge) / total_chargers

    if stations:
        mean_utilization = sum(
            (len(s.charging_evs) / s.num_chargers) if s.num_chargers > 0 else 0.0
            for s in stations
        ) / len(stations)
        mean_queue_length = sum(s.get_queue_length() for s in stations) / len(stations)
    else:
        mean_utilization = 0.0
        mean_queue_length = 0.0

    return {
        "fleet_load_ratio": fleet_load_ratio,
        "mean_utilization": mean_utilization,
        "mean_queue_length": mean_queue_length,
    }


def extract_pair_features(ev, station, shadow_pending=0, global_ctx=None):
    """Extract 23-dim feature vector for one (EV, station) candidate.

    `global_ctx` should be the dict returned by `compute_global_context` for
    the current optimization round; if omitted, the four contention features
    default to 0 (acceptable for one-off/legacy callers, but training and
    serving should always pass it -- see module docstring).
    """
    global_ctx = global_ctx or {}
    fleet_load_ratio = global_ctx.get("fleet_load_ratio", 0.0)
    mean_utilization = global_ctx.get("mean_utilization", 0.0)
    mean_queue_length = global_ctx.get("mean_queue_length", 0.0)
    station_queue_relative = (station.get_queue_length() - mean_queue_length) / 5.0

    travel_km = calculate_distance(ev.current_position, station.location) / 1000.0
    travel_time = _travel_time_s(ev, station)
    wait_time = station.get_current_wait_time_estimate()
    if shadow_pending > 0:
        wait_time += shadow_pending * 1800.0
    charge_time = _charge_time_s(ev, station)
    total_time = travel_time + wait_time + charge_time

    on_route = ev.is_station_on_route(station.location, max_detour=1000)
    if on_route:
        total_time *= 0.9
    if ev.soc < 0.1:
        total_time += travel_time * 5

    energy_needed = ev.calculate_energy_needed_for_destination()
    remaining_km = ev.calculate_remaining_distance()
    route_len = max(len(ev.route) - 1, 1)
    route_progress = ev.route_index / route_len
    utilization = (
        len(station.charging_evs) / station.num_chargers if station.num_chargers > 0 else 0.0
    )

    return np.array(
        [
            ev.soc,
            ev.battery_capacity / 60.0,
            ev.consumption_rate,
            remaining_km,
            energy_needed,
            energy_needed / max(ev.battery_capacity, 1e-6),
            1.0 if ev.soc < 0.1 else 0.0,
            travel_km,
            travel_time,
            wait_time,
            charge_time,
            total_time,
            1.0 if on_route else 0.0,
            station.num_chargers / 4.0,
            station.charging_rate / 22.0,
            station.get_queue_length() / 10.0,
            utilization,
            route_progress,
            shadow_pending / 5.0,
            fleet_load_ratio,
            mean_utilization,
            mean_queue_length / 10.0,
            station_queue_relative,
        ],
        dtype=np.float32,
    )


def get_reachable_candidates(ev, stations):
    """Return reachable (station, on_route) pairs for an EV."""
    candidates = []
    for station in stations:
        if ev.can_reach_station(station.location):
            on_route = ev.is_station_on_route(station.location, max_detour=1000)
            candidates.append((station, on_route))
    candidates.sort(key=lambda x: 0 if x[1] else 1)
    return candidates


def build_feature_matrix(evs, stations, shadow_pending=None):
    """
    Build feature matrix and index maps for batch scoring.

    Returns:
        features (N, 23), ev_indices, station_indices, ev_ids, station_ids
    """
    rows = []
    ev_indices = []
    station_indices = []
    ev_ids = []
    station_ids = []
    pending = shadow_pending or {}
    global_ctx = compute_global_context(evs, stations)

    for ei, ev in enumerate(evs):
        for si, station in enumerate(stations):
            if not ev.can_reach_station(station.location):
                continue
            rows.append(
                extract_pair_features(ev, station, pending.get(station.id, 0), global_ctx)
            )
            ev_indices.append(ei)
            station_indices.append(si)
            ev_ids.append(ev.id)
            station_ids.append(station.id)

    if not rows:
        return (
            np.zeros((0, NUM_FEATURES), dtype=np.float32),
            ev_indices,
            station_indices,
            ev_ids,
            station_ids,
        )
    return np.stack(rows), ev_indices, station_indices, ev_ids, station_ids


def extract_features(ev, station, ctx=None):
    """Alias used by train/rl_policy; supports pending_counts + global_ctx in ctx."""
    shadow = 0
    global_ctx = None
    if ctx:
        pending = ctx.get("pending_counts") or ctx.get("pending") or {}
        shadow = pending.get(station.id, 0)
        global_ctx = ctx.get("global_ctx")
    return extract_pair_features(ev, station, shadow, global_ctx)


build_candidate_matrix = build_feature_matrix


def compute_norm_stats(feature_matrix):
    """Compute mean/std normalization stats from feature matrix."""
    if len(feature_matrix) == 0:
        return {"mean": [0.0] * NUM_FEATURES, "std": [1.0] * NUM_FEATURES}
    mean = feature_matrix.mean(axis=0)
    std = feature_matrix.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return {"mean": mean.tolist(), "std": std.tolist()}


def normalize_features(features, norm_stats):
    """Normalize a feature vector using stored mean/std."""
    mean = np.array(norm_stats["mean"], dtype=np.float32)
    std = np.array(norm_stats["std"], dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return ((features - mean) / std).astype(np.float32)

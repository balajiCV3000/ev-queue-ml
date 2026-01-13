"""Greedy policy with shadow pending counter to reduce herding."""

from collections import Counter

from models.optimization import optimize_charging
from ml.policies.base import AssignmentPolicy


class GreedySequentialPolicy(AssignmentPolicy):
    """
    Greedy assignment with a shadow pending counter within each round.
    Inflates wait estimates for stations already chosen this round.
    """

    name = "greedy_sequential"

    def assign(self, evs, stations, ctx):
        pending = Counter()
        assignments = {}
        abandoned = []

        for ev in evs:
            shadow_stations = []
            for station in stations:
                shadow = _ShadowStation(station, pending[station.id])
                shadow_stations.append(shadow)

            ev_assignments, ev_abandoned = optimize_charging([ev], shadow_stations)
            if ev.id in ev_assignments:
                sid = ev_assignments[ev.id]
                assignments[ev.id] = sid
                pending[sid] += 1
            elif ev.id in ev_abandoned:
                abandoned.append(ev.id)

        return assignments, abandoned


class _ShadowStation:
    """Proxy that inflates queue length for shadow pending assignments."""

    def __init__(self, station, extra_pending):
        self._station = station
        self._extra_pending = extra_pending

    def __getattr__(self, name):
        return getattr(self._station, name)

    def get_queue_length(self):
        return self._station.get_queue_length() + self._extra_pending

    def get_current_wait_time_estimate(self):
        base = self._station.get_current_wait_time_estimate()
        if self._extra_pending <= 0:
            return base
        avg_charge = 1800
        return base + self._extra_pending * avg_charge

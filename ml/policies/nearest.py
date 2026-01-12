"""Nearest reachable station policy."""

from models.maps_service import calculate_distance
from ml.policies.base import AssignmentPolicy


class NearestPolicy(AssignmentPolicy):
    """Assign each EV to the nearest reachable station."""

    name = "nearest"

    def assign(self, evs, stations, ctx):
        assignments = {}
        abandoned = []

        for ev in evs:
            best_id = None
            best_dist = float("inf")
            for station in stations:
                if ev.can_reach_station(station.location):
                    dist = calculate_distance(ev.current_position, station.location)
                    if dist < best_dist:
                        best_dist = dist
                        best_id = station.id
            if best_id is not None:
                assignments[ev.id] = best_id
            else:
                abandoned.append(ev.id)

        return assignments, abandoned

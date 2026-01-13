"""Random assignment policy for baseline comparison."""

import random

from ml.policies.base import AssignmentPolicy


class RandomPolicy(AssignmentPolicy):
    """Assign each EV to a random reachable station."""

    name = "random"

    def __init__(self, seed=None):
        self._rng = random.Random(seed)

    def assign(self, evs, stations, ctx):
        seed = ctx.get("seed")
        rng = random.Random(seed) if seed is not None else self._rng

        assignments = {}
        abandoned = []

        for ev in evs:
            reachable = [s for s in stations if ev.can_reach_station(s.location)]
            if reachable:
                assignments[ev.id] = rng.choice(reachable).id
            else:
                abandoned.append(ev.id)

        return assignments, abandoned

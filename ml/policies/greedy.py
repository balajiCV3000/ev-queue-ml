"""Greedy policy wrapping the existing optimize_charging heuristic."""

from models.optimization import optimize_charging
from ml.policies.base import AssignmentPolicy


class GreedyPolicy(AssignmentPolicy):
    """Wraps optimize_charging for parity with the original simulation."""

    name = "greedy"

    def assign(self, evs, stations, ctx):
        return optimize_charging(evs, stations)

"""RL policy with numpy forward pass and greedy fallback."""

import json
import os
from collections import Counter

import numpy as np

import config
from ml.features import (
    build_candidate_matrix,
    compute_global_context,
    extract_features,
    normalize_features,
    NUM_FEATURES,
)
from ml.networks import load_npz, numpy_forward
from ml.policies.base import AssignmentPolicy
from ml.policies.greedy import GreedyPolicy
from models.optimization import optimize_charging


class RLPolicy(AssignmentPolicy):
    """Score EV-station pairs with a trained MLP; fall back to greedy on failure."""

    name = "rl"

    def __init__(self, model_path=None, norm_stats_path=None):
        self._greedy = GreedyPolicy()
        self._weights = None
        self._norm_stats = None
        self.fallback_active = False
        self.model_loaded = False

        model_path = model_path or config.RL_MODEL_PATH
        norm_stats_path = norm_stats_path or config.RL_NORM_STATS_PATH

        try:
            if os.path.exists(model_path) and os.path.exists(norm_stats_path):
                self._weights = load_npz(model_path)
                with open(norm_stats_path) as f:
                    self._norm_stats = json.load(f)
                self.model_loaded = True
            else:
                self.fallback_active = True
        except Exception:
            self.fallback_active = True
            self._weights = None
            self._norm_stats = None

    def assign(self, evs, stations, ctx):
        if self.fallback_active or self._weights is None:
            return self._greedy.assign(evs, stations, ctx)

        try:
            return self._rl_assign(evs, stations, ctx)
        except Exception:
            return self._greedy.assign(evs, stations, ctx)

    def _rl_assign(self, evs, stations, ctx):
        pending = Counter()
        assignments = {}
        abandoned = []
        # Computed once per round: as EVs are assigned during the round the
        # queues/utilization it reflects (pre-round state) go slightly stale,
        # but recomputing station-by-station wouldn't reflect the round's own
        # pending assignments either (they aren't real charging_evs/queue
        # entries yet) -- pending is separately tracked via shadow_pending.
        global_ctx = compute_global_context(evs, stations)

        for ev in evs:
            ctx_with_pending = dict(ctx or {})
            ctx_with_pending["pending_counts"] = dict(pending)
            ctx_with_pending["global_ctx"] = global_ctx

            best_station = None
            best_q = float("-inf")
            has_reachable = False

            for station in stations:
                if not ev.can_reach_station(station.location):
                    continue
                has_reachable = True
                feats = extract_features(ev, station, ctx_with_pending)
                normed = normalize_features(feats, self._norm_stats)
                q = float(numpy_forward(self._weights, normed.reshape(1, -1))[0])

                if q > best_q:
                    best_q = q
                    best_station = station.id

            if best_station is not None:
                assignments[ev.id] = best_station
                pending[best_station] += 1
            elif not has_reachable:
                abandoned.append(ev.id)

        return assignments, abandoned

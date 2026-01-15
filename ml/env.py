"""Gym-style environment wrapper around headless simulation."""

import copy

import config
from ml.features import (
    NUM_FEATURES,
    build_feature_matrix,
    compute_global_context,
    extract_pair_features,
)
from ml.policies.registry import get_policy
from models.simulation import Simulation
from utils.data_generator import generate_synthetic_data


class EnvWrapper:
    """Wrap Simulation for RL training and evaluation."""

    def __init__(
        self,
        num_evs=50,
        num_stations=20,
        max_steps=1500,
        policy_name="greedy",
        seed=None,
        scenario=None,
        scenario_fn=None,
    ):
        self.num_evs = num_evs
        self.num_stations = num_stations
        self.max_steps = max_steps
        self.policy_name = policy_name
        self.seed = seed if seed is not None else config.SIM_SEED
        # `scenario` is a string name applied post-generation via
        # `experiments.scenarios.apply_scenario` (e.g. "rush", "outage",
        # "congested"). `scenario_fn` is a callable data generator (e.g.
        # from `experiments.scenarios.scenario_traffic(...)`) that replaces
        # the default `generate_synthetic_data` call entirely, so density
        # presets (num_evs/num_stations/soc_range) actually take effect.
        # These are kept separate (previously collapsed into one field,
        # which silently no-op'd scenario_fn since apply_scenario only
        # matches string names).
        self.scenario = scenario
        self.scenario_fn = scenario_fn
        self.simulation = None
        self.sim = None  # alias used by train.py
        self._shadow_pending = {}
        self._step_count = 0

    def reset(self):
        """Reset environment and return initial observation."""
        if self.scenario_fn:
            evs, stations, routes = self.scenario_fn(self.seed)
        else:
            evs, stations, routes = generate_synthetic_data(
                self.num_evs,
                self.num_stations,
                80,
                240,
                use_cache=True,
                seed=self.seed,
            )
        if self.scenario:
            from experiments.scenarios import apply_scenario

            apply_scenario(self.scenario, evs, stations, step=0)

        policy = get_policy(self.policy_name)
        self.simulation = Simulation(copy.deepcopy(evs), copy.deepcopy(stations), routes, policy)
        self.sim = self.simulation
        self.simulation.reset()
        self._shadow_pending = {}
        self._step_count = 0
        return self._get_obs()

    def _apply_scenario(self):
        if self.scenario:
            from experiments.scenarios import apply_scenario

            apply_scenario(
                self.scenario,
                self.simulation.evs,
                self.simulation.stations,
                step=self.simulation.current_step,
            )

    def _get_obs(self):
        """Current observation: features for EVs needing charge."""
        evs = self.simulation.get_evs_needing_charge()
        if not evs:
            return {
                "evs": [],
                "features": [],
                "stations": [s.id for s in self.simulation.stations],
                "step": self.simulation.current_step,
            }

        global_ctx = compute_global_context(evs, self.simulation.stations)
        features = []
        for ev in evs:
            for station in self.simulation.stations:
                if ev.can_reach_station(station.location):
                    features.append(
                        extract_pair_features(
                            ev,
                            station,
                            self._shadow_pending.get(station.id, 0),
                            global_ctx,
                        )
                    )
        return {
            "evs": [ev.id for ev in evs],
            "features": features,
            "stations": [s.id for s in self.simulation.stations],
            "step": self.simulation.current_step,
        }

    def step(self, assignments=None):
        """
        Advance simulation one step.

        If assignments provided, apply them before stepping (training mode).
        Otherwise use built-in policy via auto_optimize (evaluation mode).
        """
        reward = 0.0
        if assignments is not None:
            evs_needing = self.simulation.get_evs_needing_charge()
            station_map = {s.id: s for s in self.simulation.stations}
            for ev in evs_needing:
                sid = assignments.get(ev.id)
                if sid and sid in station_map:
                    station = station_map[sid]
                    self.simulation.assign_ev_to_station(ev, station)
                    self._shadow_pending[sid] = self._shadow_pending.get(sid, 0) + 1
                elif sid is None:
                    ev.abandon("No assignment")
            self.simulation.last_optimization_step = self.simulation.current_step
            reward = self.simulation.step(auto_optimize=False) or 0.0
        else:
            reward = self.simulation.step(auto_optimize=True) or 0.0

        self._step_count += 1
        self._apply_scenario()
        done = self.simulation.is_done() or self._step_count >= self.max_steps
        info = {
            "metrics": self.simulation.metrics.copy(),
            "reward_summary": self.simulation.reward_tracker.summary(),
        }
        return self._get_obs(), reward, done, info

    def step_with_assignments(self, assignments):
        """Apply explicit assignments then advance simulation."""
        obs, reward, done, info = self.step(assignments)
        return obs, reward, done, info

    def run_episode(self, action_fn=None):
        """Run full episode; uses policy_name unless action_fn provided."""
        self.reset()
        total_reward = 0.0

        while self._step_count < self.max_steps and not self.simulation.is_done():
            if self.simulation.optimization_due():
                evs = self.simulation.get_evs_needing_charge()
                if evs:
                    if action_fn:
                        ctx = {
                            "current_step": self.simulation.current_step,
                            "seed": self.seed + self._step_count,
                            "pending_counts": dict(self._shadow_pending),
                        }
                        assignments = action_fn(evs, self.simulation.stations, ctx)
                        if isinstance(assignments, tuple):
                            assignments = assignments[0]
                    else:
                        assignments, _ = self.simulation.policy.assign(
                            evs, self.simulation.stations,
                            {
                                "current_step": self.simulation.current_step,
                                "seed": self.seed + self._step_count,
                            },
                        )
                    _, reward, done, info = self.step(assignments)
                    total_reward += reward
                    if done:
                        break
                    continue

            reward = self.simulation.step(auto_optimize=False) or 0.0
            total_reward += reward
            self._step_count += 1
            self._apply_scenario()

            if self.simulation.is_done():
                break

        return {
            "metrics": self.simulation.metrics.copy(),
            "reward_summary": self.simulation.reward_tracker.summary(),
            "total_reward": total_reward,
            "step": self._step_count,
        }

    def get_pair_features(self, evs=None):
        """Build feature matrix for current candidates."""
        evs = evs or self.simulation.get_evs_needing_charge()
        return build_feature_matrix(evs, self.simulation.stations, self._shadow_pending)

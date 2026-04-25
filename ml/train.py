"""Training script for the station-assignment Q-network.

Algorithm: episodic Monte-Carlo return regression (see `train_dqn` docstring
for why this replaced the previous "Double DQN" label, which was never
actually implemented -- `gamma`/`target_net` existed but were unused, and
the loss regressed Q(s, a) directly onto the immediate per-assignment
reward with no notion of a return at all).
"""

import argparse
import contextlib
import copy
import csv
import io
import json
import os
import random
from collections import Counter, deque
from pathlib import Path

import numpy as np

import config
from experiments.scenarios import TRAFFIC_DENSITY_SCENARIOS, scenario_traffic
from ml.env import EnvWrapper
from ml.features import (
    extract_features,
    compute_global_context,
    compute_norm_stats,
    normalize_features,
    FEATURE_NAMES,
    NUM_FEATURES,
)
from ml.networks import build_mlp, export_weights, save_npz, TORCH_AVAILABLE, init_numpy_weights
from ml.policies.greedy_sequential import GreedySequentialPolicy
from ml.tracking import RewardTracker
from models.simulation import Simulation

_TRAVEL_TIME_FEATURE_IDX = FEATURE_NAMES.index("travel_time_s")
_WAIT_TIME_FEATURE_IDX = FEATURE_NAMES.index("wait_time_s")
_CHARGE_TIME_FEATURE_IDX = FEATURE_NAMES.index("charge_time_s")

# Travel must cost MORE per second than the wait it can displace: while an EV
# is en route it accrues no waiting_time, so at weight 1.0 the policy trades
# travel for wait on any positive margin (greedy's exact 1:1 total-time
# formula), which is what produced the ~9.4km average detours. A detour is
# only worth taking when it saves at least this multiple of its travel time
# in realized wait+charge -- a clear margin. Training-only: the evaluation
# RewardTracker keeps travel_weight=1.0 so the reported reward stays true
# (unshaped) system time for every policy. Validated on the full grid:
# 2.0 cut distance further but degraded true system time, utilization,
# and completions at high/congested; 1.5 dominated it on the overall
# objective, so 1.5 is the shipped value.
TRAVEL_TIME_WEIGHT = 1.5

# Realized per-EV cost targets (see `_PendingDecision`) are in seconds and
# can run into the thousands (charge/wait time) to ~3600 (abandon penalty).
# Divide by this before regression so a freshly-initialized MLP's near-zero
# output isn't many orders of magnitude away from the target, which is what
# made the whole-episode system-wide return (~1e8, see git history/PR
# description) impossible to usefully regress onto.
TARGET_SCALE = 1000.0

# Densities to sample training episodes from. Deliberately skips "low" --
# training must see real contention (the acceptance criterion is RL
# differing from/beating nearest under congestion), and mixes medium/high/
# congested so the policy also sees moderate load, not only extreme queues.
TRAIN_DENSITIES = ["medium", "high", "congested"]


def create_smoke_artifacts(output_dir="artifacts"):
    """Create minimal smoke model for CI tests without torch."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    weights = init_numpy_weights()
    save_npz(str(output_path / "policy_smoke.npz"), weights)
    save_npz(str(output_path / "policy_dqn.npz"), weights)

    stats = {"mean": [0.0] * NUM_FEATURES, "std": [1.0] * NUM_FEATURES}
    with open(output_path / "norm_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    curve_path = output_path / "learning_curve.csv"
    with open(curve_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "reward", "moving_avg"])
        writer.writeheader()
        writer.writerow({"episode": 0, "reward": 0.0, "moving_avg": 0.0})

    model_card = output_path / "model_card.md"
    if not model_card.exists():
        model_card.write_text(
            "# EV Queue RL Policy (Smoke Model)\n\n"
            "Minimal randomly-initialized weights for CI testing.\n"
            "Train with `python -m ml.train` for production weights.\n",
            encoding="utf-8",
        )
    return {"smoke": str(output_path / "policy_smoke.npz")}


def collect_norm_stats(num_samples=5, seed=42):
    """Collect feature statistics from episodes across all traffic densities.

    Sampling only from the small default-density env (as the old version
    did) under-represents the queue-length/wait-time feature range the
    policy actually sees once trained under `TRAIN_DENSITIES` contention, so
    this now draws pairs from a world generated at each named density.
    """
    all_features = []
    densities = list(TRAFFIC_DENSITY_SCENARIOS.keys())
    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(num_samples):
            density = densities[i % len(densities)]
            gen = scenario_traffic(density, seed=seed + i)
            evs, stations, _ = gen(seed + i)
            global_ctx = compute_global_context(evs, stations)
            for ev in evs[:10]:
                for station in stations[:10]:
                    if ev.can_reach_station(station.location):
                        all_features.append(
                            extract_features(ev, station, {"global_ctx": global_ctx})
                        )
    matrix = np.stack(all_features) if all_features else np.zeros((1, NUM_FEATURES))
    return compute_norm_stats(matrix)


def _build_world_pool(densities, seed, pool_size=6):
    """Pre-generate a small pool of distinct worlds per density.

    `experiments.scenarios.scenario_congested` regenerates an entire city
    (nodes + 80 routes) with `use_cache=False` on every call, which is the
    right choice for one-off evaluation runs but far too slow to call fresh
    every training episode (hundreds of full city/route generations). This
    generates `pool_size` distinct worlds per density once up front; each
    episode then draws a randomized shallow variant (deep-copied so EV/
    station mutable state doesn't leak between episodes) from the matching
    density's pool, keeping cross-episode world diversity while making
    training tractable.
    """
    pool = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for density in densities:
            worlds = []
            for i in range(pool_size):
                gen = scenario_traffic(density, seed=seed)
                world_seed = seed + i * 4177
                evs, stations, routes = gen(world_seed)
                worlds.append((evs, stations, routes))
            pool[density] = worlds
    return pool


def _make_pool_scenario_fn(world):
    """Wrap a pre-generated (evs, stations, routes) world as a scenario_fn."""
    evs, stations, routes = world

    def fn(_seed):
        return copy.deepcopy(evs), copy.deepcopy(stations), copy.deepcopy(routes)

    return fn


def _select_round_assignments(evs, stations, q_net, norm_stats, epsilon, rng, torch):
    """Assign every EV needing charge this round to its best reachable station.

    Mirrors the sequential shadow-pending pattern used in
    `ml/policies/rl_policy.py::_rl_assign` and
    `ml/policies/greedy_sequential.py` -- each EV is scored/assigned in turn
    within the round, and a `pending` counter (not yet reflected in the
    simulated queue) discourages herding multiple EVs onto the same
    station in one optimization round. Using the same pattern in training
    as in serving keeps the learned Q-values consistent with how they'll be
    used at inference time.

    Returns:
        assignments: {ev.id: station.id}
        chosen: list of (ev, normalized_feats) for every EV assigned this
            round, for later pairing with its realized-outcome target.
    """
    pending = Counter()
    assignments = {}
    chosen = []
    global_ctx = compute_global_context(evs, stations)

    for ev in evs:
        reachable_stations = []
        reachable_feats = []
        for station in stations:
            if not ev.can_reach_station(station.location):
                continue
            feats = extract_features(
                ev, station, {"pending_counts": dict(pending), "global_ctx": global_ctx}
            )
            reachable_stations.append(station)
            reachable_feats.append(feats)

        if not reachable_stations:
            continue

        if rng.random() < epsilon:
            idx = rng.randrange(len(reachable_stations))
        else:
            normed_batch = np.stack(
                [normalize_features(f, norm_stats) for f in reachable_feats]
            )
            with torch.no_grad():
                q_values = q_net(torch.from_numpy(normed_batch)).squeeze(-1).numpy()
            idx = int(np.argmax(q_values))

        station = reachable_stations[idx]
        assignments[ev.id] = station.id
        pending[station.id] += 1
        chosen.append((ev, normalize_features(reachable_feats[idx], norm_stats)))

    return assignments, chosen


def _collect_imitation_dataset(
    world_pool, norm_stats, densities, worlds_per_density=2, steps_per_world=40, seed=0
):
    """Collect (normalized_features, target) pairs that imitate a
    travel-weighted version of the greedy heuristic's time estimate, for
    warm-starting the Q-network.

    Rationale: at episode 0 the network's Q-values are near-random, so
    early epsilon-greedy/argmax exploration wastes rounds on arbitrary
    (often bad, e.g. far-station) choices before enough Monte-Carlo return
    signal accumulates to correct them -- and diagnosis showed the fully
    from-scratch network converged to a *systematic* bias toward far
    stations (avg travel distance ~7-8km vs nearest's ~2-2.5km almost
    independent of density), consistent with extrapolation error on
    under-sampled far-station states. Regressing onto
    `-(TRAVEL_TIME_WEIGHT*travel + wait + charge)` gives the network a
    sane, distance-aware starting point that optimizes the *same* weighted
    objective as the Monte-Carlo fine-tuning phase, so fine-tuning
    specializes the heuristic instead of pulling toward a different one.

    Runs `GreedySequentialPolicy` (not epsilon-random) over a few worlds
    per density so the recorded candidate pairs reflect realistic queue
    states (including contention greedy itself creates), not just the
    idle-station states seen at t=0.
    """
    rng = random.Random(seed)
    pairs = []
    for density in densities:
        for world in world_pool[density][:worlds_per_density]:
            evs, stations, routes = world
            sim = Simulation(
                copy.deepcopy(evs), copy.deepcopy(stations), routes, GreedySequentialPolicy()
            )
            sim.reset()
            for _ in range(steps_per_world):
                needing = sim.get_evs_needing_charge()
                if needing and sim.optimization_due():
                    global_ctx = compute_global_context(needing, sim.stations)
                    for ev in needing:
                        for station in sim.stations:
                            if not ev.can_reach_station(station.location):
                                continue
                            feats = extract_features(ev, station, {"global_ctx": global_ctx})
                            # Weighted travel + wait + charge, matching the
                            # Monte-Carlo fine-tuning objective exactly (the
                            # total_time_s feature's on-route x0.9 / low-SoC
                            # x5 heuristic adjustments are deliberately NOT
                            # part of the target -- the MC return contains
                            # neither, and the realized 3600s abandonment
                            # penalty already covers the low-SoC safety case).
                            target = (
                                -(
                                    TRAVEL_TIME_WEIGHT * float(feats[_TRAVEL_TIME_FEATURE_IDX])
                                    + float(feats[_WAIT_TIME_FEATURE_IDX])
                                    + float(feats[_CHARGE_TIME_FEATURE_IDX])
                                )
                                / TARGET_SCALE
                            )
                            pairs.append((normalize_features(feats, norm_stats), target))
                sim.step(auto_optimize=True)
                if sim.is_done():
                    break

    max_pairs = 20000
    if len(pairs) > max_pairs:
        pairs = rng.sample(pairs, max_pairs)
    return pairs


def _pretrain_imitation(q_net, optimizer, loss_fn, pairs, rng, torch, batch_size=128, steps=800):
    """Supervised warm-start: regress Q(s, a) onto the greedy heuristic target."""
    if not pairs:
        return
    for _ in range(steps):
        batch = [pairs[rng.randrange(len(pairs))] for _ in range(batch_size)]
        states = torch.FloatTensor(np.array([b[0] for b in batch]))
        targets = torch.FloatTensor(np.array([b[1] for b in batch]))
        pred = q_net(states).squeeze(1)
        loss = loss_fn(pred, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class _PendingDecision:
    """Tracks the realized outcome of one (EV, station) assignment.

    Each decision is scored by the *same* EV's own realized weighted-travel
    + wait + charge (+ abandonment penalty) time from the moment it's
    assigned until its trip actually ends (completes or is abandoned),
    discounted step-by-step
    by `gamma` -- i.e. a proper n-step/Monte-Carlo return, but attributed to
    the single EV the decision was actually made for. Crucially, if the
    same EV needs to charge again later in the episode (e.g. because a
    busy station only partially charged it), *both* the earlier and the
    later decision keep accumulating cost all the way to trip end -- an
    earlier version stopped/finalized decision N as soon as decision N+1
    was made, which let the network learn to prefer stations that minimize
    only the *immediate* wait+charge (e.g. a busy station capping target
    SoC low so it charges quickly) while completely externalizing the cost
    of the resulting extra charging stop onto a separate, differently-
    scored decision; that measurably made the trained policy worse than
    nearest/greedy in practice. Tracking all of a trip's decisions to the
    same terminal outcome fixes that externality.

    This also replaces an even earlier attempt at using the whole-episode,
    whole-fleet `-total_system` return as the target for every simultaneous
    decision in a round: with 50-200 EVs assigned per round, sharing one
    system-wide number as the target for every one of them gives almost no
    signal about whether any individual station choice was actually good
    (verified empirically: the resulting Q-ranking didn't even correlate
    with the `total_time` heuristic feature it was supposed to learn to
    improve on). Using each EV's own downstream cost is directly
    attributable to its station choice and is exactly the per-EV
    decomposition of the same `RewardTracker`/`-total_system` metric used
    for evaluation, so training pressure still matches the eval metric.
    """

    __slots__ = ("ev", "feats", "tracker", "prev_total", "discount", "target")

    def __init__(self, ev, feats):
        self.ev = ev
        self.feats = feats
        # Weighted tracker: en-route travel costs TRAVEL_TIME_WEIGHT x its
        # duration in the training target (see constant's comment), while
        # the fleet-wide evaluation tracker stays at weight 1.0.
        self.tracker = RewardTracker(travel_weight=TRAVEL_TIME_WEIGHT)
        self.prev_total = 0.0
        self.discount = 1.0
        self.target = 0.0

    def advance(self, time_step, gamma):
        self.tracker.observe([self.ev], [], time_step)
        new_total = self.tracker.total_system
        delta = new_total - self.prev_total
        self.target += self.discount * delta
        self.discount *= gamma
        self.prev_total = new_total

    def finalize(self):
        """Return the (feats, scaled_target) training pair for this decision."""
        return self.feats, -self.target / TARGET_SCALE


def _run_training_episode(env, q_net, norm_stats, epsilon, gamma, rng, torch):
    """Run one episode, assigning all EVs needing charge each round.

    Returns:
        finalized: list of (feats, target) pairs -- see `_PendingDecision`.
        total_reward: sum of the fleet-wide `-total_system` step rewards
            (same quantity `ml.evaluate` scores policies on), for learning
            curve / logging purposes only (not used as a training target).
    """
    env.reset()
    # ev.id -> list of every not-yet-finalized decision made for that EV
    # this episode; all of them keep accumulating cost until the EV's trip
    # actually ends (see `_PendingDecision` docstring on why decisions
    # aren't finalized early just because a new one was made for the same
    # EV).
    pending = {}
    finalized = []
    total_reward = 0.0

    def _sweep_pending():
        time_step = env.simulation.time_step
        resolved_ids = []
        for ev_id, decisions in pending.items():
            ev = decisions[0].ev
            for decision in decisions:
                decision.advance(time_step, gamma)
            if ev.trip_completed or ev.abandoned:
                resolved_ids.append(ev_id)
        for ev_id in resolved_ids:
            for decision in pending.pop(ev_id):
                finalized.append(decision.finalize())

    while env._step_count < env.max_steps and not env.simulation.is_done():
        if env.simulation.optimization_due():
            evs = env.simulation.get_evs_needing_charge()
            if evs:
                assignments, chosen = _select_round_assignments(
                    evs, env.simulation.stations, q_net, norm_stats, epsilon, rng, torch
                )
                for ev, feats in chosen:
                    pending.setdefault(ev.id, []).append(_PendingDecision(ev, feats))

                _, reward, done, _ = env.step(assignments)
                total_reward += reward
                _sweep_pending()
                if done:
                    break
                continue

        reward = env.simulation.step(auto_optimize=False) or 0.0
        env._step_count += 1
        env._apply_scenario()
        total_reward += reward
        _sweep_pending()
        if env.simulation.is_done():
            break

    # Decisions still unresolved when the episode ends (truncated by
    # max_steps) are finalized with whatever cost has accrued so far --
    # a valid (if incomplete) partial return, better than discarding them.
    for decisions in pending.values():
        for decision in decisions:
            finalized.append(decision.finalize())

    return finalized, total_reward


def train_dqn(
    episodes=50,
    gamma=0.99,
    lr=1e-3,
    batch_size=128,
    replay_size=100000,
    target_sync=2000,
    epsilon=0.2,
    seed=42,
    output_dir="artifacts",
    bandit=False,
    densities=None,
    max_steps=350,
    simulate_travel_time=True,
    world_pool_size=9,
):
    """Train the Q-network with per-EV Monte-Carlo return regression.

    Why not Double DQN: assignment here is a per-round, multi-EV
    combinatorial decision (every EV needing charge is assigned a station
    in the same optimization round via `_select_round_assignments`), so
    there isn't a clean single-agent "next state" to bootstrap a one-step
    TD target from the way classic (Double) DQN assumes. Rather than fake a
    next-state and call it Double DQN (as the previous implementation's
    docstring did, while actually never using `gamma` or `target_net` at
    all), each (EV, station) decision is regressed onto that specific EV's
    own realized downstream cost (see `_PendingDecision`): the discounted
    sum of its wait/charge/abandonment-penalty time from the moment it's
    assigned until it resolves,

        target(s_t, a_t) = -sum_{k=0}^{K-1} gamma^k * cost_increment_k

    which is the same per-EV decomposition of the `-total_system`
    (`RewardTracker`) reward used to score policies in `ml.evaluate`, so
    training pressure matches the evaluation metric, while being properly
    attributable to the individual decision (an earlier version of this
    function shared one whole-episode, whole-fleet return across every
    simultaneous decision in a round, which gave almost no signal -- see
    module docstring on `_PendingDecision`). `gamma` genuinely affects the
    target (verified: bandit/gamma=0 training, which only credits the
    first step's cost, produces different weights than gamma>0).
    `target_sync`/a separate target network are not needed since there is
    no bootstrapping (the target for each decision is a fully realized
    Monte-Carlo return, not another Q-value estimate); the parameter is
    kept only for CLI/API backward compatibility and is unused.

    Args:
        densities: list of `experiments.scenarios.TRAFFIC_DENSITY_SCENARIOS`
            keys to sample episodes from (default `TRAIN_DENSITIES`, i.e.
            medium/high/congested -- real contention, not "low").
        simulate_travel_time: sets `config.SIMULATE_TRAVEL_TIME` for the
            duration of training so the learned policy experiences the same
            travel-then-queue dynamics it's evaluated under (restored to
            its prior value afterwards).
    """
    if not TORCH_AVAILABLE:
        print("torch not installed; creating smoke artifacts only")
        return create_smoke_artifacts(output_dir)

    import torch
    import torch.nn as nn
    import torch.optim as optim

    densities = densities or TRAIN_DENSITIES

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)

    if bandit:
        gamma = 0.0

    os.makedirs(output_dir, exist_ok=True)
    norm_stats = collect_norm_stats(seed=seed)
    with open(os.path.join(output_dir, "norm_stats.json"), "w", encoding="utf-8") as f:
        json.dump(norm_stats, f, indent=2)

    q_net = build_mlp()
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()
    replay = deque(maxlen=replay_size)

    curve_path = os.path.join(output_dir, "learning_curve.csv")
    curve_rows = []
    global_step = 0

    # Weight episodes toward the denser scenarios. A "medium" episode's
    # stations rarely fill up, so nearly all of its (state, target) pairs
    # are drawn from empty-queue, zero-wait states; if densities were
    # sampled uniformly, those low-signal examples would dominate the
    # replay buffer (medium/high/congested episodes differ hugely in how
    # many optimization rounds actually see contention) and dilute the
    # higher-wait/queued examples the network most needs to learn to
    # penalize station herding under load. "high" is weighted heaviest:
    # diagnosis showed RL was specifically worst (relative to greedy and
    # even nearest) at "high" density -- likely under-represented relative
    # to "congested" in earlier runs despite being the regime that most
    # needs the distance-vs-queue trade-off to be learned correctly.
    density_weights = {"medium": 1, "high": 3, "congested": 2}
    weighted_pool = [d for d in densities for _ in range(density_weights.get(d, 1))]
    density_sequence = [weighted_pool[i % len(weighted_pool)] for i in range(episodes)]
    rng.shuffle(density_sequence)

    print(f"Pre-generating world pool ({world_pool_size} worlds x {len(densities)} densities)...")
    world_pool = _build_world_pool(densities, seed, pool_size=world_pool_size)

    print("Collecting greedy-imitation warm-start dataset...")
    imitation_pairs = _collect_imitation_dataset(world_pool, norm_stats, densities, seed=seed)
    print(f"  {len(imitation_pairs)} imitation (feature, target) pairs collected")
    _pretrain_imitation(q_net, optimizer, loss_fn, imitation_pairs, rng, torch, batch_size=batch_size)
    print("Warm-start pretraining complete; starting Monte-Carlo fine-tuning.")

    # Held-out worlds used only to *measure* the learning curve (greedy/
    # epsilon=0 policy, deterministic given fixed weights), kept separate
    # from the training pool above. Rather than a single fixed eval world
    # (which risks reading a curve that reflects overfitting to that one
    # world, and previously only ever measured "congested" -- exactly the
    # density RL was weakest at relative to itself, not "high" where it
    # was weakest relative to the other policies), rotate evaluation
    # round-robin across one held-out world per density in `densities` so
    # every regime gets tracked; each density keeps its own moving average
    # since raw reward magnitude differs by orders of magnitude across
    # densities (congested >> medium) and would otherwise dominate a
    # combined average even while genuinely improving elsewhere.
    eval_densities = [d for d in ("medium", "high", "congested") if d in densities] or [densities[-1]]
    eval_cfgs = {d: TRAFFIC_DENSITY_SCENARIOS[d] for d in eval_densities}
    eval_seed = seed + 999983
    eval_worlds = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for d in eval_densities:
            eval_worlds[d] = scenario_traffic(d, seed=eval_seed)(eval_seed)
    moving_rewards_by_density = {d: deque(maxlen=20) for d in eval_densities}

    prior_travel_flag = config.SIMULATE_TRAVEL_TIME
    config.SIMULATE_TRAVEL_TIME = simulate_travel_time
    try:
        for episode in range(episodes):
            density = density_sequence[episode]
            cfg = TRAFFIC_DENSITY_SCENARIOS[density]
            episode_seed = seed + episode
            world = rng.choice(world_pool[density])
            env = EnvWrapper(
                seed=episode_seed,
                num_evs=cfg["num_evs"],
                num_stations=cfg["num_stations"],
                max_steps=max_steps,
                scenario_fn=_make_pool_scenario_fn(world),
            )

            eps = max(0.02, epsilon * (1.0 - episode / max(episodes, 1)))
            finalized, ep_reward = _run_training_episode(
                env, q_net, norm_stats, eps, gamma, rng, torch
            )
            for feats, target in finalized:
                replay.append((feats, target))

            # Fixed number of gradient steps per episode (rather than scaled
            # by that episode's own decision count) so low-density episodes
            # -- which produce few new (state, target) pairs -- still train
            # from the accumulated replay buffer instead of contributing
            # almost no learning signal. Raised from 8 to 32: with only ~50
            # episodes' worth of updates previously, the network barely
            # moved past its imitation warm-start; more updates per episode
            # lets Monte-Carlo fine-tuning actually specialize beyond it
            # within a comparable episode budget.
            updates_this_episode = 32
            for _ in range(updates_this_episode):
                if len(replay) < batch_size:
                    break
                batch = rng.sample(replay, batch_size)
                states = torch.FloatTensor(np.array([b[0] for b in batch]))
                targets = torch.FloatTensor(np.array([b[1] for b in batch]))
                pred = q_net(states).squeeze(1)
                loss = loss_fn(pred, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                global_step += 1

            eval_density_this = eval_densities[episode % len(eval_densities)]
            eval_cfg = eval_cfgs[eval_density_this]
            eval_env = EnvWrapper(
                seed=eval_seed,
                num_evs=eval_cfg["num_evs"],
                num_stations=eval_cfg["num_stations"],
                max_steps=max_steps,
                scenario_fn=_make_pool_scenario_fn(eval_worlds[eval_density_this]),
            )
            _, eval_reward = _run_training_episode(
                eval_env, q_net, norm_stats, 0.0, gamma, rng, torch
            )

            moving_rewards_by_density[eval_density_this].append(eval_reward)
            moving_avg = sum(moving_rewards_by_density[eval_density_this]) / len(
                moving_rewards_by_density[eval_density_this]
            )
            curve_rows.append({
                "episode": episode,
                "eval_density": eval_density_this,
                "reward": eval_reward,
                "moving_avg": moving_avg,
            })
            if episode % 10 == 0 or episode == episodes - 1:
                per_density = ", ".join(
                    f"{d}={sum(r) / len(r):.0f}" if (r := moving_rewards_by_density[d]) else f"{d}=n/a"
                    for d in eval_densities
                )
                print(
                    f"Episode {episode} [{density}]: train_reward={ep_reward:.1f}, "
                    f"eval[{eval_density_this}]_reward={eval_reward:.1f}, "
                    f"moving_avg[{eval_density_this}]={moving_avg:.1f}, "
                    f"moving_avg_by_density=({per_density}), n_decisions={len(finalized)}"
                )
    finally:
        config.SIMULATE_TRAVEL_TIME = prior_travel_flag

    with open(curve_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "eval_density", "reward", "moving_avg"])
        writer.writeheader()
        writer.writerows(curve_rows)

    weights = export_weights(q_net)
    model_name = "policy_bandit.npz" if bandit else "policy_dqn.npz"
    save_npz(os.path.join(output_dir, model_name), weights)
    print(f"Model saved to {output_dir}/{model_name}")
    return {"model": os.path.join(output_dir, model_name)}


def main():
    parser = argparse.ArgumentParser(description="Train station assignment Q-network")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=350)
    parser.add_argument("--bandit", action="store_true", help="Bandit ablation (gamma=0)")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Initial epsilon (decays to 0.02)")
    parser.add_argument("--world-pool-size", type=int, default=9)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--densities", default="medium,high,congested",
        help="Comma-separated TRAFFIC_DENSITY_SCENARIOS keys to train on",
    )
    parser.add_argument("--smoke-only", action="store_true", help="Create smoke artifacts only")
    args = parser.parse_args()

    if args.smoke_only:
        result = create_smoke_artifacts(args.output_dir)
        print(f"Smoke artifacts: {result}")
        return

    train_dqn(
        episodes=args.episodes,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        seed=args.seed,
        output_dir=args.output_dir,
        bandit=args.bandit,
        epsilon=args.epsilon,
        world_pool_size=args.world_pool_size,
        densities=[d.strip() for d in args.densities.split(",") if d.strip()],
    )


if __name__ == "__main__":
    main()

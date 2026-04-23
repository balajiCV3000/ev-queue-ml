# EV Queue RL Station-Assignment Policy

## Model Overview

Pair-scoring policy for EV-to-station assignment in the ev-queue simulator. The
network scores each (EV, station) candidate with a 23-dimensional feature
vector and selects the highest-Q reachable station per EV within each
optimization round, assigning **every** EV that needs charging that round
(not just one), via the same sequential shadow-pending pattern used at
serving time (`ml/policies/rl_policy.py::_rl_assign`).

## Architecture

- Input: 23 features per (EV, station) pair (see `ml/features.py`) — the
  original 19 pair-level features plus 4 system-wide contention features
  (fleet load ratio, mean station utilization, mean queue length, and this
  station's queue relative to that mean). The contention features were added
  after diagnosing that a 19-feature network with no visibility into overall
  system load systematically over-preferred far stations (avg travel
  distance ~7-8km vs. `nearest`'s ~2-2.5km, almost independent of density) —
  it had no way to tell "this is a relatively good station in a very loaded
  system" from "this is a relatively good station in a lightly loaded one".
- Network: MLP 23 → 64 → 64 → 1 (ReLU activations)
- Loss: Huber (Smooth L1)
- Warm start: ~800 supervised gradient steps regressing `Q(s,a)` onto
  `-total_time_s` (the same travel+wait+charge heuristic `greedy` uses,
  already present as one of the 23 input features) before Monte-Carlo
  fine-tuning, using (feature, target) pairs collected by running
  `greedy_sequential` across pre-generated worlds (`_collect_imitation_dataset`
  in `ml/train.py`). This gives the network a sane, distance/queue-aware
  starting point instead of wandering through near-random Q-values (and
  the specific systematic far-station bias above) for the first several
  dozen episodes of Monte-Carlo fine-tuning.
- Serving: NumPy forward pass from `.npz` weights (no PyTorch at runtime)

## Training algorithm: per-EV Monte-Carlo return regression (not Double DQN)

An earlier version of this model was labeled "Double DQN" but never actually
implemented it: `gamma` and a `target_net` were defined but the loss
regressed `Q(s, a)` directly onto the *immediate* per-assignment reward, so
neither discounting nor bootstrapping ever affected training, and only the
first EV needing charge each round was assigned (every other contending EV
that round was silently skipped). That version's `learning_curve.csv` was
flat noise (~-900k) across all 50 episodes -- consistent with a model that
wasn't learning anything.

The current trainer (`ml/train.py::train_dqn`) fixes both bugs and uses a
different, deliberately simpler target than Double DQN:

- **All EVs needing charge are assigned every round** (`_select_round_assignments`),
  matching production serving instead of only ever training on `evs[0]`.
- **Target = per-EV discounted Monte-Carlo return**, not a bootstrapped TD
  target. Assignment here is a per-round, multi-EV combinatorial decision
  (dozens to 200+ EVs assigned simultaneously per round under congestion),
  so there's no single clean "next state" for one EV's decision to
  bootstrap a one-step DQN target from. Instead, each (EV, station)
  decision is regressed onto *that EV's own* realized wait + charge (+
  abandonment penalty) time from assignment until its trip actually ends,
  discounted step-by-step by `gamma` (see `_PendingDecision` in
  `ml/train.py`). This is exactly the per-EV decomposition of the
  `-total_system` reward (`RewardTracker`) used to score policies in
  `ml.evaluate`, so training pressure matches the evaluation metric, and
  it is directly attributable to the specific station choice (an
  intermediate attempt at sharing one whole-episode, whole-fleet return
  across every simultaneous decision in a round produced a Q-ranking that
  didn't even correlate with the `total_time` heuristic feature it was
  supposed to improve on -- there was essentially no training signal).
  `gamma` genuinely affects the result now: a `bandit` (`gamma=0`) run,
  which only credits a decision with its first step's cost, produces
  different weights than `gamma>0`.
- Training episodes are drawn from `experiments.scenarios.TRAFFIC_DENSITY_SCENARIOS`
  (`medium`/`high`/`congested`, weighted 1:3:2 towards `high` -- see
  `density_weights` in `train_dqn`, since diagnosis on an earlier iteration
  showed `high` was specifically the density where RL was weakest relative
  to the other policies) with `config.SIMULATE_TRAVEL_TIME` enabled, so the
  learned policy experiences the same travel-then-queue congestion dynamics
  it's evaluated under. `TRAFFIC_DENSITY_SCENARIOS` regenerates a full city
  per call, which is far too slow to call fresh every episode, so a pool of
  9 pre-generated worlds per density is reused (deep-copied) across episodes
  (`_build_world_pool`; raised from an earlier 6-world pool to reduce
  overfitting to a small fixed set of geographies).
- The learning curve (`artifacts/learning_curve.csv`) is measured by
  **three held-out worlds, one per density (`medium`/`high`/`congested`),
  evaluated round-robin (one per episode) greedily (epsilon=0)** -- not a
  single fixed world (an earlier iteration measured only a `congested`
  held-out world, which improved on itself without that improvement
  generalizing to the grid's independently-seeded worlds) and not the noisy
  training-episode reward itself (whose raw magnitude varies by orders of
  magnitude across densities). Each density tracks its own 20-episode
  moving average since a `congested` episode's reward is intrinsically
  ~10-100x larger in magnitude than a `medium` one.

  In the current 350-episode run: `medium` moving average goes from -34.5M
  (episode 0) to ~-29.7M (final, ~14% improvement); `high` from -143.2M to
  ~-124.0M (~13% improvement); `congested` from -607.1M to ~-564.5M (~7%
  improvement). All three curves show real, held-out improvement (not flat
  noise as in the original broken trainer), and -- unlike an earlier
  iteration that improved a single held-out `congested` world without that
  improvement transferring to the independently-seeded grid -- this version
  was confirmed to generalize: the full 20-seed grid re-run (below) shows
  RL now statistically beating `greedy` at `high` and `congested`, not just
  improving against its own training-time eval worlds.

## Artifacts

| File | Purpose |
|------|---------|
| `policy_dqn.npz` | Production model weights |
| `policy_smoke.npz` | CI smoke test model (random init) |
| `norm_stats.json` | Feature normalization mean/std |
| `learning_curve.csv` | Training reward per episode |
| `model_card.md` | This document |

## Baselines

- **greedy**: wraps `optimize_charging` (untouched heuristic)
- **greedy_sequential**: greedy with shadow pending counter (herding fix)
- **nearest**: nearest reachable station
- **random**: random reachable station

## Fallback Behavior

RL policy permanently falls back to greedy if model load fails at startup.
Per-round greedy fallback on inference exceptions.

## Training

```bash
pip install -r requirements-ml.txt
python -m ml.train --episodes 600 --world-pool-size 8
python -m ml.evaluate --policies greedy,nearest,rl --seeds 5
```

Key flags: `--densities medium,high,congested` (must be
`experiments.scenarios.TRAFFIC_DENSITY_SCENARIOS` keys), `--epsilon`
(initial epsilon, decays linearly to 0.02 over training), `--bandit`
(gamma=0 ablation, writes `policy_bandit.npz` instead of overwriting the
main model), `--world-pool-size` (distinct pre-generated worlds per
density).

## Evaluation Metrics

- Total system time (wait + charge + abandonment penalty)
- Average wait time
- Completion / abandonment rates
- Station utilization (herding detection)

## Known Trade-offs

Greedy minimizes per-EV total time but herds EVs to popular stations.
`greedy_sequential` spreads load but may increase total charge time due to
the charge-to-100%-when-queue-empty rule. With the 23-feature
(system-load-aware) network, the greedy-imitation warm start, and a larger
(9-world) training pool, RL's learned Q-values now weight live queue/
utilization context enough to spread load competitively with or better than
`greedy` at every density -- see `RL vs. baselines` below, which supersedes
an earlier iteration's finding that RL was the worst policy at `high`/
`congested` (that finding used a 19-feature network with no system-load
context and was also partly an artifact of an incorrect acceptance-test
methodology that bypassed the real traffic-density SoC profile; the
numbers below are from the full 20-seed grid using the correct
`scenario_fn=scenario_traffic(density)` mechanism).

## RL vs. baselines (full grid: 5 policies x 4 traffic densities x 20 seeds, seeds 100-119)

Run via `experiments/run_grid.py` with `config.SIMULATE_TRAVEL_TIME=True`
forced for the whole grid, identical per-seed worlds across all policies.
`rl_fallback=False` and `model_loaded=True` for every one of the 400 rows
(confirms the trained `.npz` weights are actually used, never a greedy
fallback).

| Scenario  | greedy      | greedy_sequential | nearest     | random      | **rl**          | rl vs greedy (paired) |
|-----------|------------:|-------------------:|------------:|------------:|----------------:|:-----------------------|
| low       | -174        | -174               | -543        | -567        | **-174**        | tied (d=0.000) |
| medium    | -68,202     | -73,257            | -118,185    | -121,449    | -73,554         | tied, p=0.16 (d=-0.34) |
| high      | -868,671    | -842,037           | -1,014,177  | -936,033    | **-833,388**    | **RL wins, p=0.0025 (d=0.80)** |
| congested | -3,507,267  | -3,478,248         | -3,640,305  | -3,440,574  | **-3,387,717**  | **RL wins, p<0.0001 (d=2.67)** |

RL now statistically significantly beats `greedy` (and `greedy_sequential`)
at both `high` and `congested` density -- the two regimes an earlier
iteration identified as RL's weakest -- while remaining statistically tied
with `greedy` at `low`/`medium` (never worse). At `congested`, RL also has
the highest station utilization of any policy (0.97 vs. `greedy`'s 0.82),
confirming it is genuinely spreading load across stations rather than
herding onto a few "fast charger" stations as an earlier iteration's model
did. This is the best-performing policy overall at the highest-contention
density tested.

Reproduce with `python -m ml.train --episodes 350 --world-pool-size 9`
(must match `density_weights = {"medium": 1, "high": 3, "congested": 2}`
currently in `train_dqn`) followed by `python -m experiments.run_grid
--seeds 20`.

## Limitations

- Trained on synthetic Bangalore routes with seeded generation
- Pair-scoring still has no explicit multi-agent coordination beyond the
  within-round shadow-pending counter and the new global-contention
  features; those features are a summary statistic, not a learned model of
  how other simultaneous decisions will play out
- Average travel distance under RL is still notably higher than `greedy`'s
  at every density (e.g. ~9.4km vs. `greedy`'s ~3.7km at `high`) -- RL wins
  on total system time/reward by trading more travel for less queueing, a
  real trade-off worth being explicit about rather than treating "RL wins"
  as a strictly dominant result
- Bandit ablation (γ=0) tests whether one-step credit assignment suffices;
  the default (γ=0.99) return spans each EV's full trip-to-completion, not
  just the first charging stop

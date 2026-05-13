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
  `-(TRAVEL_TIME_WEIGHT*travel + wait + charge)` (a travel-weighted version
  of the total-time heuristic `greedy` uses, matching the Monte-Carlo
  fine-tuning objective) before Monte-Carlo
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
  decision is regressed onto *that EV's own* realized weighted-travel +
  wait + charge (+ abandonment penalty) time from assignment until its
  trip actually ends, discounted step-by-step by `gamma` (see
  `_PendingDecision` in `ml/train.py`). This is the per-EV decomposition
  of the `-total_system` reward (`RewardTracker`) used to score policies
  in `ml.evaluate` -- with travel weighted 1.5x during training (see the
  reward section below) -- so training pressure matches the evaluation
  metric while pricing detours at a premium, and
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

  In the current 400-episode run (travel-priced reward, w=1.5): `medium`
  moving average improves from -31.5M (first 5 episodes) to ~-19.3M
  (final); `high` from -160.6M to ~-148.2M; `congested` from -635.6M to
  ~-631.8M. All three curves show real, held-out improvement (not flat
  noise as in the original broken trainer), and the full 20-seed grid
  re-run (below) confirms the behavior generalizes to
  independently-seeded worlds, not just the training-time eval worlds.

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

- Total system time (travel + wait + charge + abandonment penalty)
- Average wait time
- Completion / abandonment rates
- Station utilization (herding detection)
- Total / per-assignment travel distance

## Reward: travel time is priced (2026-07 change)

The reward previously omitted en-route travel entirely; because
`waiting_time` does not accrue while an EV is `en_route_to_charger`,
detouring to a far empty station *displaced* penalized queue time with
unpriced travel time, which is why earlier models averaged ~9.4km per
assignment vs `greedy`'s ~3.7km. `RewardTracker` now accrues travel time:
at weight 1.0 in evaluation (so `total_reward` is true system time for
every policy) and at `TRAVEL_TIME_WEIGHT = 1.5` in the training target
(`ml/train.py`), so a detour must save clearly more wait than the travel
it adds. Because evaluation now counts travel, `total_reward` values in
this section are NOT comparable to grids run before this change
(`results_pre_travel_fix/` preserves the last pre-change grid).

Weight ablations (full 20-seed grids each): `TRAVEL_TIME_WEIGHT=2.0` cut
distance further (e.g. 5.8km vs 7.7km at `high`) but degraded true system
time, utilization, and completions at `high`/`congested`; an undiscounted
(`--gamma 1.0`) variant at w=1.5 behaved similarly. w=1.5 with the default
gamma=0.99 dominated both on the overall objective and is the shipped
model.

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

Total reward = negative true system time (travel now included, see the
reward section above; numbers below are NOT comparable to the table this
replaces, which was measured on the old travel-free metric):

| Scenario  | greedy      | greedy_sequential | nearest     | random      | **rl**          | rl vs greedy (paired) | rl vs nearest (paired) |
|-----------|------------:|-------------------:|------------:|------------:|----------------:|:----------------------|:-----------------------|
| low       | -192        | -192               | -561        | -612        | **-192**        | tied (d=0.000)        | tied, p=0.33 |
| medium    | -72,051     | -77,997            | -120,486    | -132,360    | -76,704         | tied, p=0.17 (d=-0.33)| **RL wins, p<0.0001** |
| high      | -897,834    | -880,164           | -1,029,720  | -998,580    | -965,673        | greedy wins, p<0.0001 (d=-1.52) | **RL wins, p=0.0015** |
| congested | -3,600,270  | -3,604,656         | -3,695,412  | -3,611,007  | -3,617,085      | tied, p=0.12 (d=-0.37)| **RL wins, p=0.0007** |

Per-metric picture (20-seed means): RL has the lowest average wait time of
all five policies at every contended density (medium 118s vs greedy 527s /
nearest 832s; high 3,766s vs 5,084s / 4,871s; congested 6,907s vs 7,610s /
7,740s; all paired p<=0.03), the lowest avg/max queue lengths of
greedy/nearest/rl at every density, and higher utilization than greedy at
high (0.74 vs 0.52) and congested (0.94 vs 0.82) while serving more
vehicles (high 64.5 vs 61.4; congested 75.2 vs 73.4). Completion rate is
equal at medium (1.00), slightly below greedy at high (0.868 vs 0.899) and
congested (0.342 vs 0.347), and above nearest everywhere. On true system
time RL beats `nearest` at every contended density and is statistically
tied with `greedy` at medium/congested but loses at high (-7.6%): greedy's
per-EV total-time heuristic is genuinely strong once travel is priced.

Reproduce with `python -m ml.train` (400 episodes, defaults; must match
`density_weights = {"medium": 1, "high": 3, "congested": 2}` in
`train_dqn`) followed by `python -m experiments.run_grid --seeds 20`.

## Limitations

- Trained on synthetic Bangalore routes with seeded generation
- Pair-scoring still has no explicit multi-agent coordination beyond the
  within-round shadow-pending counter and the new global-contention
  features; those features are a summary statistic, not a learned model of
  how other simultaneous decisions will play out
- Average travel distance under RL improved substantially once travel was
  priced into the reward (medium 9.0 -> 4.4km, high 9.4 -> 7.7km, congested
  9.7 -> 8.5km per assignment) but remains above `greedy` (3.0/3.7/4.2km)
  and `nearest` (1.9/2.0/2.5km) at every contended density: the policy
  still buys its wait-time/queue-balance wins with extra travel, now at a
  bounded exchange rate rather than for free. Raising the travel weight
  further (2.0) reduced distance more but made overall system time worse
  (see reward section) -- the residual gap is a real trade-off, not a
  tuning artifact left unexamined
- Bandit ablation (γ=0) tests whether one-step credit assignment suffices;
  the default (γ=0.99) return spans each EV's full trip-to-completion, not
  just the first charging stop

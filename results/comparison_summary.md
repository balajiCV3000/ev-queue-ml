# RL vs. Baselines — travel-priced reward (July 2026)

Full paired grid: 5 policies x 4 traffic densities x 20 seeds (100-119),
`experiments/run_grid.py`, `SIMULATE_TRAVEL_TIME=True`, identical per-seed
worlds across policies. Every one of the 80 `rl` rows has
`model_loaded=True` and `rl_fallback=False` (the trained `.npz` weights
were exercised in every run, never the greedy fallback).

**Metric change note:** `total_reward` is now negative *true system time*
(travel + wait + charge + abandonment). Earlier grids
(`results_pre_travel_fix/`) omitted travel from the reward, so their
`total_reward` values — including the previous version of this document,
which reported "RL matches or beats greedy at every density" — are not
comparable to these. All other metrics are computed identically.

**Shipped model:** `TRAVEL_TIME_WEIGHT=1.5`, gamma=0.99, 400 episodes.
Ablations run on the full grid and rejected on the data: w=2.0 (less
distance, but worse system time, utilization, and completions at
high/congested) and w=1.5 with gamma=1.0 (same pathology).

## Statistical summary (paired t-tests, n=20 seeds)

| Scenario | RL vs Greedy (total reward) | RL vs Nearest (total reward) |
|---|---|---|
| low | tied (identical assignments) | tied, p=0.33 |
| medium | tied, p=0.17 | **RL wins, p<0.0001** |
| high | greedy wins, p<0.0001 (−7.6%) | **RL wins, p=0.0015** |
| congested | tied, p=0.12 | **RL wins, p=0.0007** |

RL has the lowest average wait time of all five policies at every
contended density (paired p≤0.03 vs both baselines), the lowest avg/max
queue lengths among greedy/nearest/rl, and the highest charger
utilization at high/congested — while traveling 1.45–2.08x greedy's
distance (down from 2.3–3.0x before travel was priced into the reward:
9.0/9.4/9.7 km per assignment → 4.4/7.7/8.5 km at medium/high/congested).

## Per-density comparison (20-seed means)

### low

| Metric | Greedy | Nearest | RL | RL − Greedy | RL − Nearest |
|---|---:|---:|---:|---:|---:|
| Avg wait time (s) | 0.0 | 0.0 | 0.0 | +0.0 | +0.0 |
| Total travel distance (km) | 0.2 | 0.2 | 0.2 | +0.0 (+0.0%) | +0.0 (+0.4%) |
| Per-assignment travel distance (km) | 0.17 | 0.16 | 0.17 | +0.00 (+0.0%) | +0.00 (+0.4%) |
| Avg queue length | 0.00 | 0.00 | 0.00 | +0.00 | +0.00 |
| Max queue length | 0.0 | 0.0 | 0.0 | +0.0 | +0.0 |
| Avg charger utilization | 0.001 | 0.001 | 0.001 | +0.000 | −0.000 |
| Vehicles served (completed sessions) | 0.1 | 0.1 | 0.1 | +0.0 | +0.0 |
| Trip completion rate | 1.000 | 1.000 | 1.000 | +0.000 | +0.000 |
| Avg charging completion time (s) | 174 | 543 | 174 | +0 (+0.0%) | −369 (−68.0%) |
| Total reward (= −true system time) | −192 | −561 | −192 | +0 (+0.0%) | +369 (+65.8%) |

(At `low` density there is almost no charging demand; all policies behave
near-identically.)

### medium

| Metric | Greedy | Nearest | RL | RL − Greedy | RL − Nearest |
|---|---:|---:|---:|---:|---:|
| Avg wait time (s) | 527.5 | 831.7 | 118.2 | −409.2 (−77.6%) | −713.5 (−85.8%) |
| Total travel distance (km) | 34.7 | 21.9 | 50.1 | +15.4 (+44.4%) | +28.2 (+129.1%) |
| Per-assignment travel distance (km) | 3.03 | 1.90 | 4.39 | +1.36 (+45.0%) | +2.49 (+131.1%) |
| Avg queue length | 0.44 | 0.69 | 0.08 | −0.36 (−81.0%) | −0.61 (−87.8%) |
| Max queue length | 1.6 | 1.6 | 0.3 | −1.2 (−80.6%) | −1.2 (−80.6%) |
| Avg charger utilization | 0.107 | 0.138 | 0.127 | +0.020 (+18.6%) | −0.011 (−8.2%) |
| Vehicles served (completed sessions) | 11.4 | 11.3 | 11.4 | +0.0 (+0.0%) | +0.1 (+1.3%) |
| Trip completion rate | 0.999 | 0.989 | 0.999 | +0.000 | +0.010 (+1.0%) |
| Avg charging completion time (s) | 5,402 | 9,068 | 5,905 | +503 (+9.3%) | −3,163 (−34.9%) |
| Total reward (= −true system time) | −72,051 | −120,486 | −76,704 | −4,653 (−6.5%) | +43,782 (+36.3%) |

### high

| Metric | Greedy | Nearest | RL | RL − Greedy | RL − Nearest |
|---|---:|---:|---:|---:|---:|
| Avg wait time (s) | 5,083.8 | 4,870.8 | 3,765.6 | −1,318.3 (−25.9%) | −1,105.3 (−22.7%) |
| Total travel distance (km) | 259.4 | 143.4 | 540.9 | +281.5 (+108.5%) | +397.5 (+277.2%) |
| Per-assignment travel distance (km) | 3.69 | 2.04 | 7.67 | +3.98 (+108.1%) | +5.63 (+276.1%) |
| Avg queue length | 23.38 | 28.47 | 16.98 | −6.39 (−27.3%) | −11.48 (−40.3%) |
| Max queue length | 44.9 | 43.9 | 33.0 | −11.9 (−26.4%) | −10.9 (−24.8%) |
| Avg charger utilization | 0.524 | 0.591 | 0.740 | +0.216 (+41.2%) | +0.149 (+25.2%) |
| Vehicles served (completed sessions) | 61.4 | 53.5 | 64.5 | +3.1 (+5.1%) | +10.9 (+20.4%) |
| Trip completion rate | 0.899 | 0.843 | 0.868 | −0.031 (−3.4%) | +0.026 (+3.0%) |
| Avg charging completion time (s) | 6,190 | 7,867 | 8,421 | +2,231 (+36.0%) | +555 (+7.0%) |
| Total reward (= −true system time) | −897,834 | −1,029,720 | −965,673 | −67,839 (−7.6%) | +64,047 (+6.2%) |

### congested

| Metric | Greedy | Nearest | RL | RL − Greedy | RL − Nearest |
|---|---:|---:|---:|---:|---:|
| Avg wait time (s) | 7,609.6 | 7,739.9 | 6,906.7 | −702.9 (−9.2%) | −833.2 (−10.8%) |
| Total travel distance (km) | 820.7 | 500.0 | 1,670.8 | +850.1 (+103.6%) | +1,170.8 (+234.2%) |
| Per-assignment travel distance (km) | 4.16 | 2.53 | 8.47 | +4.30 (+103.4%) | +5.93 (+234.0%) |
| Avg queue length | 146.14 | 153.18 | 139.40 | −6.74 (−4.6%) | −13.78 (−9.0%) |
| Max queue length | 173.8 | 175.5 | 171.0 | −2.8 (−1.6%) | −4.5 (−2.6%) |
| Avg charger utilization | 0.823 | 0.820 | 0.943 | +0.121 (+14.7%) | +0.123 (+15.0%) |
| Vehicles served (completed sessions) | 73.4 | 63.9 | 75.2 | +1.8 (+2.4%) | +11.3 (+17.6%) |
| Trip completion rate | 0.347 | 0.300 | 0.342 | −0.006 (−1.6%) | +0.041 (+13.7%) |
| Avg charging completion time (s) | 6,327 | 7,400 | 7,176 | +848 (+13.4%) | −224 (−3.0%) |
| Total reward (= −true system time) | −3,600,270 | −3,695,412 | −3,617,085 | −16,815 (−0.5%) | +78,327 (+2.1%) |

## Honest assessment against the success bar

**Where RL wins** (both baselines, contended densities): waiting time
(−9% to −86%), queue lengths (avg and max), charger utilization at
high/congested, vehicles served, and — against `nearest` — total system
time at every contended density (p≤0.0015). Trip completion beats
`nearest` everywhere and is within 0.6pp/3.1pp of greedy at
congested/high.

**Where RL does not win:** per-assignment travel distance improved
2x-relative to the pre-fix model at medium (9.0→4.4 km) and materially at
high/congested (9.4→7.7, 9.7→8.5 km), but remains 1.45–2.08x greedy and
2.3–3.8x nearest — the distance bar ("competitive with or better than
both baselines") is **not met** at high/congested. On true total system
time RL is statistically tied with greedy at medium/congested and loses
at high (−7.6%): once travel is actually counted, greedy's
total-time heuristic is a genuinely strong optimum, and the pre-fix "RL
beats greedy everywhere" result was partly an artifact of the old metric
not charging for travel.

**Why we did not push distance further:** raising the travel weight to
2.0 did cut distance (7.67→5.78 km at high) but degraded true system
time, utilization, and completions at high/congested — over-optimizing
the single distance metric at the expense of the others, which the
success criteria explicitly forbid. The shipped w=1.5 model is the best
overall-objective policy of the three trained candidates.

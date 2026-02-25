"""Run paired experiment grid: N seeds x 5 policies x 4 traffic-density scenarios.

Policies (greedy, greedy_sequential, nearest, random, rl) are run on
*identical* seeds within each traffic-density condition (low / medium /
high / congested, from `experiments.scenarios.TRAFFIC_DENSITY_SCENARIOS`),
so per-seed comparisons are apples-to-apples: the same generated world is
handed to every policy for a given (scenario, seed) pair.

`config.SIMULATE_TRAVEL_TIME` is forced on for the whole run (matching how
the RL policy was trained/evaluated: assigned EVs incur a real travel delay
before joining a station queue instead of being teleported in instantly).
"""

import argparse
import csv
from pathlib import Path

import config
from experiments.scenarios import TRAFFIC_DENSITY_SCENARIOS, scenario_traffic
from ml.evaluate import run_single

POLICIES = ["greedy", "greedy_sequential", "nearest", "random", "rl"]
DENSITIES = ["low", "medium", "high", "congested"]
NUM_SEEDS = 20
SEED_START = 100
MAX_STEPS = 350


def run_grid(output_dir="results", num_seeds=NUM_SEEDS, seed_start=SEED_START,
             max_steps=MAX_STEPS, densities=DENSITIES, policies=POLICIES,
             simulate_travel_time=True):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "results.csv"
    rows = []

    prior_travel_flag = config.SIMULATE_TRAVEL_TIME
    config.SIMULATE_TRAVEL_TIME = simulate_travel_time
    try:
        for density in densities:
            cfg = TRAFFIC_DENSITY_SCENARIOS[density]
            for seed in range(seed_start, seed_start + num_seeds):
                # Same seed -> same scenario_traffic(density) world handed to
                # every policy below, so the comparison is paired per-seed.
                for policy in policies:
                    try:
                        scenario_fn = scenario_traffic(density)
                        row = run_single(
                            policy,
                            seed,
                            num_evs=cfg["num_evs"],
                            num_stations=cfg["num_stations"],
                            max_steps=max_steps,
                            scenario_fn=scenario_fn,
                            scenario_label=density,
                        )
                        rows.append(row)
                    except Exception as exc:
                        print(f"Failed {policy}/{density}/seed={seed}: {exc}")
    finally:
        config.SIMULATE_TRAVEL_TIME = prior_travel_flag

    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    _write_stats_summary(rows, output_path / "stats_summary.md", densities, policies)
    return rows


def _write_stats_summary(rows, path, densities=DENSITIES, policies=POLICIES):
    """Write paired stats summary (t-test/Wilcoxon vs greedy when scipy is
    available, else mean +/- std per policy per scenario)."""
    lines = ["# Experiment Stats Summary\n"]
    if not rows:
        lines.append("No results collected.\n")
        path.write_text("".join(lines), encoding="utf-8")
        return

    try:
        from scipy import stats
        import numpy as np

        for density in densities:
            scen_rows = [r for r in rows if r.get("scenario") == density]
            greedy_rewards = [
                float(r["total_reward"]) for r in scen_rows if r["policy"] == "greedy"
            ]
            lines.append(f"## Scenario: {density}\n\n")
            for policy in policies:
                policy_rows = [r for r in scen_rows if r["policy"] == policy]
                policy_rewards = [float(r["total_reward"]) for r in policy_rows]
                if not policy_rewards:
                    continue
                mean_r = float(np.mean(policy_rewards))
                std_r = float(np.std(policy_rewards))
                lines.append(
                    f"- **{policy}**: mean total_reward={mean_r:.1f} "
                    f"(std={std_r:.1f}, n={len(policy_rewards)})\n"
                )
                if policy == "greedy":
                    continue
                if len(greedy_rewards) == len(policy_rewards) and len(greedy_rewards) >= 5:
                    t_stat, p_val = stats.ttest_rel(policy_rewards, greedy_rewards)
                    try:
                        w_stat, w_p = stats.wilcoxon(policy_rewards, greedy_rewards)
                    except ValueError:
                        w_stat, w_p = float("nan"), float("nan")
                    diff = np.array(policy_rewards) - np.array(greedy_rewards)
                    cohens_d = diff.mean() / (diff.std() + 1e-8)
                    lines.append(
                        f"  - vs greedy: t-test p={p_val:.4f}, Wilcoxon p={w_p:.4f}, "
                        f"Cohen's d={cohens_d:.3f}\n"
                    )
            lines.append("\n")
    except ImportError:
        lines.append("scipy not installed; showing mean +/- std only.\n\n")
        for density in densities:
            scen_rows = [r for r in rows if r.get("scenario") == density]
            lines.append(f"## Scenario: {density}\n\n")
            by_policy = {}
            for r in scen_rows:
                by_policy.setdefault(r["policy"], []).append(float(r["total_reward"]))
            for policy, rewards in sorted(by_policy.items()):
                if not rewards:
                    continue
                mean_r = sum(rewards) / len(rewards)
                var_r = sum((x - mean_r) ** 2 for x in rewards) / len(rewards)
                lines.append(
                    f"- {policy}: mean total_reward={mean_r:.1f} "
                    f"(std={var_r ** 0.5:.1f}, n={len(rewards)})\n"
                )
            lines.append("\n")

    path.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results")
    parser.add_argument("--seeds", type=int, default=NUM_SEEDS)
    parser.add_argument("--seed-start", type=int, default=SEED_START)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument(
        "--densities", default=",".join(DENSITIES),
        help="Comma-separated TRAFFIC_DENSITY_SCENARIOS keys to evaluate",
    )
    parser.add_argument(
        "--policies", default=",".join(POLICIES),
        help="Comma-separated policy names to evaluate",
    )
    args = parser.parse_args()
    run_grid(
        args.output,
        args.seeds,
        args.seed_start,
        max_steps=args.max_steps,
        densities=[d.strip() for d in args.densities.split(",") if d.strip()],
        policies=[p.strip() for p in args.policies.split(",") if p.strip()],
    )

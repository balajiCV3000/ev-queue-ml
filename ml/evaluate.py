"""CLI for evaluating assignment policies."""

import argparse
import csv
import os

from ml.env import EnvWrapper
from ml.policies.registry import list_policies


def run_single(policy_name, seed, num_evs=50, num_stations=20, max_steps=500, scenario=None,
               scenario_fn=None, scenario_label=None):
    """Run one policy/seed/scenario episode and return a single row-dict.

    This is the shared building block for both the CLI (`run_evaluation`)
    and the experiment grid (`experiments/run_grid.py`): one call = one
    fully-run episode = one comparable row for a results.csv table.

    Args:
        policy_name: name registered in `ml.policies.registry` (or "rl").
        seed: seed used for both world generation and policy RNG (passed
            through to EnvWrapper, which derives per-step seeds for
            stochastic policies so evaluation runs are reproducible).
        num_evs, num_stations, max_steps: episode sizing (ignored for
            num_evs/num_stations when `scenario_fn` is provided, since the
            scenario_fn fully controls world generation; the actual counts
            used are read back from the simulation after the episode runs).
        scenario: scenario name string (e.g. "rush", "outage", "congested")
            applied via `experiments.scenarios.apply_scenario`, or None for
            the baseline (no scenario mutation). Mutually usable alongside
            `scenario_fn`.
        scenario_fn: optional callable data generator (e.g.
            `experiments.scenarios.scenario_traffic("high", seed)`) that
            replaces the default world generation entirely, so traffic
            density presets (num_evs/num_stations/soc_range) actually take
            effect. Passed straight through to `EnvWrapper`.
        scenario_label: label recorded in the output row's "scenario" field
            when `scenario_fn` is used (since `scenario_fn` itself carries
            no name). Falls back to `scenario` or "baseline" if omitted.

    Returns:
        dict with keys:
            policy, seed, scenario, num_evs, num_stations, steps,
            completion_rate, abandoned_rate,
            average_wait_time, max_queue_length, avg_queue_length,
            total_travel_distance_km, average_travel_distance_km,
            average_charging_completion_time,
            avg_station_utilization, vehicles_served, throughput_per_hour,
            total_reward, reward, total_system_time, total_wait, total_charge,
            completed, abandoned, rl_fallback, model_loaded
    """
    env = EnvWrapper(
        policy_name=policy_name,
        seed=seed,
        num_evs=num_evs,
        num_stations=num_stations,
        max_steps=max_steps,
        scenario=scenario,
        scenario_fn=scenario_fn,
    )
    info = env.run_episode()
    metrics = info["metrics"]
    reward_summary = info.get("reward_summary", {})

    policy = env.simulation.policy
    is_rl = policy_name == "rl" or getattr(policy, "name", None) == "rl"
    rl_fallback = bool(getattr(policy, "fallback_active", False)) if is_rl else False
    model_loaded = bool(getattr(policy, "model_loaded", False)) if is_rl else True

    reward = reward_summary.get("reward", info.get("total_reward", 0))

    actual_num_evs = len(env.simulation.evs) if scenario_fn else num_evs
    actual_num_stations = len(env.simulation.stations) if scenario_fn else num_stations
    scenario_name = scenario_label or scenario or "baseline"

    return {
        "policy": policy_name,
        "seed": seed,
        "scenario": scenario_name,
        "num_evs": actual_num_evs,
        "num_stations": actual_num_stations,
        "steps": info.get("step", max_steps),
        "completion_rate": metrics.get("completion_rate", 0),
        "abandoned_rate": metrics.get("abandoned_rate", 0),
        "average_wait_time": metrics.get("average_wait_time", 0),
        "max_queue_length": metrics.get("max_queue_length", 0),
        "avg_queue_length": metrics.get("avg_queue_length", 0),
        "total_travel_distance_km": metrics.get("total_travel_distance_km", 0),
        "average_travel_distance_km": metrics.get("average_travel_distance_km", 0),
        "average_charging_completion_time": metrics.get("average_charging_completion_time", 0),
        "avg_station_utilization": metrics.get("avg_station_utilization", 0),
        "vehicles_served": metrics.get("vehicles_served", 0),
        "throughput_per_hour": metrics.get("throughput_per_hour", 0),
        "total_reward": reward,
        "reward": reward,
        "total_system_time": reward_summary.get("total_system", 0),
        "total_wait": reward_summary.get("total_wait", 0),
        "total_charge": reward_summary.get("total_charge", 0),
        "completed": reward_summary.get("completed", 0),
        "abandoned": reward_summary.get("abandoned", 0),
        "rl_fallback": rl_fallback,
        "model_loaded": model_loaded,
    }


def run_evaluation(policies, seeds, num_evs=50, num_stations=20, max_steps=500,
                   output_dir="results", scenario=None):
    """Run policies over seeds and write results CSV. Built on `run_single`."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "results.csv")

    rows = []
    for policy_name in policies:
        for seed in seeds:
            row = run_single(
                policy_name, seed,
                num_evs=num_evs,
                num_stations=num_stations,
                max_steps=max_steps,
                scenario=scenario,
            )
            rows.append(row)
            print(f"  {policy_name} seed={seed}: wait={row['average_wait_time']:.1f}s "
                  f"completion={row['completion_rate']:.2%}")

    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results written to {csv_path}")
    return csv_path, rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate station assignment policies")
    parser.add_argument("--policies", default="greedy,nearest,random",
                        help="Comma-separated policy names")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of seeds to evaluate")
    parser.add_argument("--num-evs", type=int, default=30)
    parser.add_argument("--num-stations", type=int, default=15)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    available = list_policies()
    for p in policies:
        if p not in available and p != "rl":
            print(f"Warning: unknown policy '{p}', will attempt anyway")

    seeds = list(range(args.seeds))
    print(f"Evaluating {policies} over {len(seeds)} seeds...")
    csv_path, rows = run_evaluation(
        policies, seeds,
        num_evs=args.num_evs,
        num_stations=args.num_stations,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
    )

    try:
        from ml.plots import plot_evaluation_results
        plot_evaluation_results(csv_path, args.output_dir)
    except ImportError:
        pass

    return rows


if __name__ == "__main__":
    main()

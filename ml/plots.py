"""Plotting utilities for experiment results."""

import os


def plot_evaluation_results(csv_path, output_dir="results"):
    """Generate box plots from evaluation CSV."""
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    policies = df["policy"].unique()
    wait_data = [df[df["policy"] == p]["average_wait_time"].values for p in policies]
    axes[0].boxplot(wait_data, labels=policies)
    axes[0].set_title("Average Wait Time by Policy")
    axes[0].set_ylabel("Wait Time (s)")

    reward_data = [df[df["policy"] == p]["total_reward"].values for p in policies]
    axes[1].boxplot(reward_data, labels=policies)
    axes[1].set_title("Total Reward by Policy")
    axes[1].set_ylabel("Reward")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "policy_comparison.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")


def plot_learning_curve(csv_path, output_dir="results"):
    """Plot training learning curve."""
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["episode"], df["reward"], alpha=0.3, label="episode reward")
    if "moving_avg" in df.columns:
        ax.plot(df["episode"], df["moving_avg"], label="moving avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Learning Curve")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "learning_curve.png")
    plt.savefig(path)
    plt.close()


def plot_utilization_heatmap(utilization_data, output_dir="results"):
    """Plot station utilization heatmap (herding visualization)."""
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not utilization_data:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    data = np.array(utilization_data)
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    ax.set_xlabel("Station")
    ax.set_ylabel("Time Step")
    ax.set_title("Station Utilization Over Time")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    path = os.path.join(output_dir, "utilization_heatmap.png")
    plt.savefig(path)
    plt.close()

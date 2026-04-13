"""
Plot learning curves averaged across seeds with confidence bands.
Following Henderson et al. (2018) best practices for RL reporting.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def load_training_logs(model_dirs: list[Path]) -> list[list[dict]]:
    """Load training_log.json from each directory, return list of histories."""
    histories = []
    for d in model_dirs:
        log_path = d / "training_log.json"
        if log_path.exists():
            with open(log_path) as f:
                data = json.load(f)
            histories.append(data["history"])
    return histories


def align_histories(histories: list[list[dict]], key: str, x_key: str = "step"):
    """
    Align histories to common x-axis. Pad shorter runs with NaN.
    Returns (x_values, array of shape [n_seeds, n_epochs]).
    """
    max_len = max(len(h) for h in histories)
    x_values = None
    for h in histories:
        if len(h) == max_len:
            x_values = np.array([entry[x_key] for entry in h])
            break
    if x_values is None:
        x_values = np.array([histories[0][i][x_key] for i in range(len(histories[0]))])
        x_values = np.concatenate([x_values, np.arange(len(x_values) + 1, max_len + 1) * x_values[0]])

    data = np.full((len(histories), max_len), np.nan)
    for i, h in enumerate(histories):
        for j, entry in enumerate(h):
            data[i, j] = entry[key]

    # Use common x_values up to max_len
    if len(x_values) < max_len:
        step_size = x_values[1] - x_values[0] if len(x_values) > 1 else 2000
        x_values = np.arange(1, max_len + 1) * step_size

    return x_values[:max_len], data


def plot_metric(ax, x, data, label, color, alpha_fill=0.2):
    """Plot mean line with shaded std region."""
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    n_valid = np.sum(~np.isnan(data), axis=0)
    se = std / np.sqrt(np.maximum(n_valid, 1))

    ax.plot(x, mean, color=color, linewidth=2, label=label)
    ax.fill_between(x, mean - se, mean + se, color=color, alpha=alpha_fill)


def main():
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Define model directories for each algorithm ──────────────────
    cql_dirs = [
        Path("models/cql"),
        Path("models/cql_seed1"),
        Path("models/cql_seed2"),
        Path("models/cql_seed3"),
        Path("models/cql_seed4"),
    ]

    dist_cql_dirs = [
        Path("models/dist_cql_alpha05"),
        Path("models/dist_cql_alpha05_seed1"),
        Path("models/dist_cql_alpha05_seed2"),
        Path("models/dist_cql_alpha05_seed3"),
        Path("models/dist_cql_alpha05_seed4"),
    ]

    # ── Load histories ───────────────────────────────────────────────
    cql_histories = load_training_logs(cql_dirs)
    dist_histories = load_training_logs(dist_cql_dirs)

    print(f"Loaded {len(cql_histories)} CQL training logs")
    print(f"Loaded {len(dist_histories)} DistCQL training logs")

    colors = {
        "CQL": "#1f77b4",
        "DistCQL": "#ff7f0e",
        "DQN": "#2ca02c",
    }

    # ── Figure 1: Validation Sharpe across training ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # CQL
    x_cql, sharpe_cql = align_histories(cql_histories, "val_sharpe")
    plot_metric(axes[0], x_cql, sharpe_cql, "CQL", colors["CQL"])

    # DistCQL
    x_dist, sharpe_dist = align_histories(dist_histories, "val_sharpe")
    plot_metric(axes[0], x_dist, sharpe_dist, "DistCQL (a=0.5)", colors["DistCQL"])

    axes[0].set_xlabel("Training Steps", fontsize=12)
    axes[0].set_ylabel("Validation Sharpe Ratio", fontsize=12)
    axes[0].set_title("Validation Sharpe During Training\n(mean +/- SE across 5 seeds)", fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[0].grid(True, alpha=0.3)

    # ── Validation Return ────────────────────────────────────────────
    _, ret_cql = align_histories(cql_histories, "val_total_return")
    plot_metric(axes[1], x_cql, ret_cql, "CQL", colors["CQL"])

    _, ret_dist = align_histories(dist_histories, "val_total_return")
    plot_metric(axes[1], x_dist, ret_dist, "DistCQL (a=0.5)", colors["DistCQL"])

    axes[1].set_xlabel("Training Steps", fontsize=12)
    axes[1].set_ylabel("Validation Total Return", fontsize=12)
    axes[1].set_title("Validation Return During Training\n(mean +/- SE across 5 seeds)", fontsize=13)
    axes[1].legend(fontsize=11)
    axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "learning_curves_val.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 'learning_curves_val.png'}")

    # ── Figure 2: Q-value diagnostics ────────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

    # Q-mean
    _, qmean_cql = align_histories(cql_histories, "q_mean")
    plot_metric(axes2[0], x_cql, qmean_cql, "CQL", colors["CQL"])
    _, qmean_dist = align_histories(dist_histories, "q_mean")
    plot_metric(axes2[0], x_dist, qmean_dist, "DistCQL (a=0.5)", colors["DistCQL"])
    axes2[0].set_xlabel("Training Steps", fontsize=12)
    axes2[0].set_ylabel("Mean Q-value", fontsize=12)
    axes2[0].set_title("Q-value Scale During Training", fontsize=13)
    axes2[0].legend(fontsize=11)
    axes2[0].grid(True, alpha=0.3)

    # Q-gap
    _, qgap_cql = align_histories(cql_histories, "q_gap_mean")
    plot_metric(axes2[1], x_cql, qgap_cql, "CQL", colors["CQL"])
    _, qgap_dist = align_histories(dist_histories, "q_gap_mean")
    plot_metric(axes2[1], x_dist, qgap_dist, "DistCQL (a=0.5)", colors["DistCQL"])
    axes2[1].set_xlabel("Training Steps", fontsize=12)
    axes2[1].set_ylabel("Q-gap (max - min)", fontsize=12)
    axes2[1].set_title("Action Preference Strength", fontsize=13)
    axes2[1].legend(fontsize=11)
    axes2[1].grid(True, alpha=0.3)

    # Conservatism proxy
    _, cons_cql = align_histories(cql_histories, "conservatism_proxy")
    plot_metric(axes2[2], x_cql, cons_cql, "CQL", colors["CQL"])
    _, cons_dist = align_histories(dist_histories, "conservatism_proxy")
    plot_metric(axes2[2], x_dist, cons_dist, "DistCQL (a=0.5)", colors["DistCQL"])
    axes2[2].set_xlabel("Training Steps", fontsize=12)
    axes2[2].set_ylabel("Conservatism Proxy", fontsize=12)
    axes2[2].set_title("CQL Conservatism Proxy\n(logsumexp Q - max Q)", fontsize=13)
    axes2[2].legend(fontsize=11)
    axes2[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig2.savefig(output_dir / "learning_curves_qdiag.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 'learning_curves_qdiag.png'}")

    # ── Figure 3: Action distributions during training ───────────────
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

    # CQL short fraction
    _, short_cql = align_histories(cql_histories, "val_action_frac_short")
    _, long_cql = align_histories(cql_histories, "val_action_frac_long")
    plot_metric(axes3[0], x_cql, short_cql, "Short %", "red")
    plot_metric(axes3[0], x_cql, long_cql, "Long %", "green")
    axes3[0].set_xlabel("Training Steps", fontsize=12)
    axes3[0].set_ylabel("Action Fraction", fontsize=12)
    axes3[0].set_title("CQL: Action Distribution During Training\n(mean +/- SE across 5 seeds)", fontsize=13)
    axes3[0].legend(fontsize=11)
    axes3[0].set_ylim(-0.05, 1.05)
    axes3[0].grid(True, alpha=0.3)

    # DistCQL short fraction
    _, short_dist = align_histories(dist_histories, "val_action_frac_short")
    _, long_dist = align_histories(dist_histories, "val_action_frac_long")
    plot_metric(axes3[1], x_dist, short_dist, "Short %", "red")
    plot_metric(axes3[1], x_dist, long_dist, "Long %", "green")
    axes3[1].set_xlabel("Training Steps", fontsize=12)
    axes3[1].set_ylabel("Action Fraction", fontsize=12)
    axes3[1].set_title("DistCQL (a=0.5): Action Distribution During Training\n(mean +/- SE across 5 seeds)", fontsize=13)
    axes3[1].legend(fontsize=11)
    axes3[1].set_ylim(-0.05, 1.05)
    axes3[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig3.savefig(output_dir / "learning_curves_actions.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 'learning_curves_actions.png'}")

    # ── Figure 4: Individual seed learning curves ────────────────────
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))

    seed_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    seed_labels = ["s42", "s1", "s2", "s3", "s4"]

    # CQL individual seeds
    for i, h in enumerate(cql_histories):
        steps = [e["step"] for e in h]
        sharpes = [e["val_sharpe"] for e in h]
        axes4[0].plot(steps, sharpes, color=seed_colors[i], linewidth=1.5,
                      label=seed_labels[i], alpha=0.8)
    axes4[0].set_xlabel("Training Steps", fontsize=12)
    axes4[0].set_ylabel("Validation Sharpe", fontsize=12)
    axes4[0].set_title("CQL: Individual Seed Learning Curves", fontsize=13)
    axes4[0].legend(fontsize=10, ncol=3)
    axes4[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes4[0].grid(True, alpha=0.3)

    # DistCQL individual seeds
    for i, h in enumerate(dist_histories):
        steps = [e["step"] for e in h]
        sharpes = [e["val_sharpe"] for e in h]
        axes4[1].plot(steps, sharpes, color=seed_colors[i], linewidth=1.5,
                      label=seed_labels[i], alpha=0.8)
    axes4[1].set_xlabel("Training Steps", fontsize=12)
    axes4[1].set_ylabel("Validation Sharpe", fontsize=12)
    axes4[1].set_title("DistCQL (a=0.5): Individual Seed Learning Curves", fontsize=13)
    axes4[1].legend(fontsize=10, ncol=3)
    axes4[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes4[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig4.savefig(output_dir / "learning_curves_individual.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 'learning_curves_individual.png'}")

    print("\nAll figures saved to results/figures/")


if __name__ == "__main__":
    main()

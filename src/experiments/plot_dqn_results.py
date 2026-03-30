"""Generate report figures for the DQN baseline (Step 6)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from src.agents.dqn_baseline import load_model
from src.env.portfolio import PortfolioConfig
from src.experiments.eval_policies import rollout_policy

LOG = logging.getLogger(__name__)

OUT_DIR = Path("reports/figures")
SPLITS = {
    "val": "data/processed/btc_daily_val.parquet",
    "test": "data/processed/btc_daily_test.parquet",
}

# Epoch-by-epoch val Sharpe from training logs (manually extracted)
EPOCH_STEPS = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]
EPOCH_VAL_SHARPE = [-0.5374, -0.0771, 1.4669, -0.1936, 0.2231, 0.4470, -0.5420, -0.5751, -0.2462, -0.2461]

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
})


def _run_buy_and_hold(data_path: str, split: str) -> np.ndarray:
    """Simple buy-and-hold equity curve."""
    from src.experiments.baselines import run_buy_and_hold_on_split
    bh = run_buy_and_hold_on_split(data_path, split_name=split)
    return bh.equity


def plot_equity_curves(algo, out_dir: Path) -> None:
    """Fig 1: Equity curves — DQN vs Buy-and-Hold on val and test."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    for ax, (split, path) in zip(axes, SPLITS.items()):
        # DQN rollout
        result = rollout_policy(algo, data_path=path, split_name=split, d3rlpy_actions=True)
        dqn_eq = result.equity

        # Buy-and-hold
        bh_eq = _run_buy_and_hold(path, split)

        n = min(len(dqn_eq), len(bh_eq))
        days = np.arange(n)

        ax.plot(days, bh_eq[:n], color="#2196F3", linewidth=1.8, label="Buy & Hold")
        ax.plot(days, dqn_eq[:n], color="#E53935", linewidth=1.8, label="DoubleDQN (best)")
        ax.axhline(1.0, color="grey", linewidth=0.7, linestyle="--", alpha=0.5)

        ax.set_title(f"{split.capitalize()} Split")
        ax.set_xlabel("Trading Days")
        ax.set_ylabel("Portfolio Value (start = 1.0)")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        # Annotate final returns
        dqn_ret = (dqn_eq[-1] / dqn_eq[0] - 1) * 100
        bh_ret = (bh_eq[-1] / bh_eq[0] - 1) * 100
        ax.annotate(
            f"{dqn_ret:+.0f}%", xy=(n - 1, dqn_eq[n - 1]),
            fontsize=9, color="#E53935", fontweight="bold",
            xytext=(5, 0), textcoords="offset points", va="center",
        )
        ax.annotate(
            f"{bh_ret:+.0f}%", xy=(n - 1, bh_eq[n - 1]),
            fontsize=9, color="#2196F3", fontweight="bold",
            xytext=(5, 0), textcoords="offset points", va="center",
        )

    fig.suptitle("DQN Baseline: Equity Curves vs Buy & Hold", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "dqn_equity_curves.png", bbox_inches="tight")
    plt.close(fig)
    LOG.info("Saved → %s", out_dir / "dqn_equity_curves.png")


def plot_overfitting_trajectory(out_dir: Path) -> None:
    """Fig 2: Val Sharpe across training epochs — shows overfitting peak."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    epochs = list(range(1, len(EPOCH_VAL_SHARPE) + 1))

    ax.plot(epochs, EPOCH_VAL_SHARPE, "o-", color="#7B1FA2", linewidth=2, markersize=7)

    # Highlight best epoch
    best_idx = int(np.argmax(EPOCH_VAL_SHARPE))
    ax.plot(
        epochs[best_idx], EPOCH_VAL_SHARPE[best_idx],
        "o", color="#E53935", markersize=12, zorder=5, markeredgecolor="white", markeredgewidth=2,
    )
    ax.annotate(
        f"Best: Sharpe = {EPOCH_VAL_SHARPE[best_idx]:.2f}\n(epoch {epochs[best_idx]}, step {EPOCH_STEPS[best_idx]})",
        xy=(epochs[best_idx], EPOCH_VAL_SHARPE[best_idx]),
        xytext=(epochs[best_idx] + 1.5, EPOCH_VAL_SHARPE[best_idx] - 0.3),
        fontsize=10, color="#E53935", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#E53935", lw=1.5),
    )

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Sharpe Ratio")
    ax.set_title("Overfitting Trajectory: Val Sharpe Across Training", fontweight="bold")
    ax.set_xticks(epochs)
    ax.set_xticklabels([f"{e}\n({s // 1000}k)" for e, s in zip(epochs, EPOCH_STEPS)], fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "dqn_overfitting_trajectory.png", bbox_inches="tight")
    plt.close(fig)
    LOG.info("Saved → %s", out_dir / "dqn_overfitting_trajectory.png")


def plot_action_distribution(algo, out_dir: Path) -> None:
    """Fig 3: Action distribution on val vs test — shows regime mismatch."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    colors = {"Short": "#E53935", "Flat": "#9E9E9E", "Long": "#43A047"}
    labels = ["Short", "Flat", "Long"]

    for ax, (split, path) in zip(axes, SPLITS.items()):
        result = rollout_policy(algo, data_path=path, split_name=split, d3rlpy_actions=True)
        fracs = [
            result.metrics["action_frac_short"],
            result.metrics["action_frac_flat"],
            result.metrics["action_frac_long"],
        ]

        bars = ax.bar(labels, [f * 100 for f in fracs], color=[colors[l] for l in labels], edgecolor="white", width=0.6)
        for bar, frac in zip(bars, fracs):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{frac:.0%}", ha="center", va="bottom", fontweight="bold", fontsize=11,
            )

        ret = result.metrics["total_return"]
        sharpe = result.metrics["sharpe"]
        ax.set_title(f"{split.capitalize()} Split\n(return {ret:+.0%}, Sharpe {sharpe:.2f})", fontsize=11)
        ax.set_ylabel("% of Trading Days")
        ax.set_ylim(0, 75)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("DQN Action Distribution: Val vs Test", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "dqn_action_distribution.png", bbox_inches="tight")
    plt.close(fig)
    LOG.info("Saved → %s", out_dir / "dqn_action_distribution.png")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    LOG.info("Loading best DQN model...")
    algo = load_model("models/dqn_baseline/model_best.d3")

    plot_equity_curves(algo, OUT_DIR)
    plot_overfitting_trajectory(OUT_DIR)
    plot_action_distribution(algo, OUT_DIR)

    LOG.info("All figures saved to %s", OUT_DIR)


if __name__ == "__main__":
    main()

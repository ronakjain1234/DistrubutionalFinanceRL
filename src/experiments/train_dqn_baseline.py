"""
Train offline DQN / DoubleDQN baseline and evaluate vs buy-and-hold (Step 6).

Usage
-----
::

    python -m src.experiments.train_dqn_baseline          # defaults
    python -m src.experiments.train_dqn_baseline --algo dqn --n_steps 100000
    python -m src.experiments.train_dqn_baseline --lr 1e-4 --hidden 128 128

The script:

1. Loads the offline dataset built in Step 5.
2. Trains a DQN (or DoubleDQN) on it via d3rlpy.
3. Evaluates the learned policy on validation and test splits.
4. Prints a comparison table vs buy-and-hold.
5. Saves the trained model to ``models/dqn_baseline/``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from src.data.build_offline_dataset import load_offline_dataset, to_d3rlpy_dataset
from src.agents.dqn_baseline import DQNBaselineConfig, create_dqn, save_model
from src.experiments.eval_policies import evaluate_on_splits

LOG = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train offline DQN baseline for BTC trading (Step 6).",
    )

    # Algorithm
    p.add_argument(
        "--algo", choices=["dqn", "double_dqn"], default="double_dqn",
        help="DQN variant (default: double_dqn).",
    )

    # Network
    p.add_argument(
        "--hidden", type=int, nargs="+", default=[256, 256],
        help="Hidden layer sizes (default: 256 256).",
    )
    p.add_argument("--activation", default="relu", help="Activation function.")

    # Optimisation
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    p.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    p.add_argument(
        "--target_update", type=int, default=1000,
        help="Target network update interval (steps).",
    )

    # Training budget
    p.add_argument(
        "--n_steps", type=int, default=50_000,
        help="Total gradient steps.",
    )
    p.add_argument(
        "--n_steps_per_epoch", type=int, default=5_000,
        help="Steps per epoch (logging frequency).",
    )

    # Data
    p.add_argument(
        "--dataset", default="data/processed/offline_dataset_train.npz",
        help="Path to offline dataset.",
    )

    # Device
    p.add_argument("--device", default=None, help="PyTorch device (auto if omitted).")

    # Output
    p.add_argument(
        "--save_dir", default="models/dqn_baseline",
        help="Directory for saved model.",
    )

    return p.parse_args(argv)


# ── Main ──────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-36s  %(levelname)-7s  %(message)s",
    )

    args = parse_args(argv)

    # ── 1. Load offline dataset ───────────────────────────────────────
    LOG.info("Loading offline dataset from %s ...", args.dataset)
    raw = load_offline_dataset(args.dataset)
    dataset = to_d3rlpy_dataset(raw)
    LOG.info(
        "Dataset: %d transitions, obs_dim=%d, action_space=%s",
        dataset.transition_count,
        raw["observations"].shape[1],
        "Discrete(3)",
    )

    # ── 2. Create DQN agent ───────────────────────────────────────────
    cfg = DQNBaselineConfig(
        algo=args.algo,
        hidden_units=args.hidden,
        activation=args.activation,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        gamma=args.gamma,
        target_update_interval=args.target_update,
        n_steps=args.n_steps,
        n_steps_per_epoch=args.n_steps_per_epoch,
        device=args.device,
        save_dir=Path(args.save_dir),
    )

    algo = create_dqn(cfg)

    # ── 3. Train ──────────────────────────────────────────────────────
    LOG.info("Starting training: %d steps ...", cfg.n_steps)
    algo.fit(
        dataset,
        n_steps=cfg.n_steps,
        n_steps_per_epoch=cfg.n_steps_per_epoch,
        experiment_name="dqn_baseline",
        show_progress=True,
        logging_steps=500,
    )
    LOG.info("Training complete.")

    # ── 4. Save model ─────────────────────────────────────────────────
    save_path = cfg.save_dir / "model.d3"
    save_model(algo, save_path)

    # ── 5. Evaluate on val / test vs buy-and-hold ─────────────────────
    results = evaluate_on_splits(
        algo,
        policy_name=cfg.algo.upper(),
        d3rlpy_actions=True,
        verbose=True,
    )

    # ── 6. Quick summary ──────────────────────────────────────────────
    for split_name, result in results.items():
        m = result.metrics
        LOG.info(
            "%s %s: return=%.4f  sharpe=%.4f  maxDD=%.4f  actions=[S:%.0f%% F:%.0f%% L:%.0f%%]",
            cfg.algo.upper(),
            split_name,
            m["total_return"],
            m["sharpe"],
            m["max_drawdown"],
            m.get("action_frac_short", 0) * 100,
            m.get("action_frac_flat", 0) * 100,
            m.get("action_frac_long", 0) * 100,
        )


if __name__ == "__main__":
    main()

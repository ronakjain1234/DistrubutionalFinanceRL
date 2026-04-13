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
import copy
import logging
import sys
from pathlib import Path

import numpy as np

from src.data.build_offline_dataset import (
    load_offline_dataset,
    to_d3rlpy_dataset,
    get_n_actions,
    get_position_levels,
)
from src.agents.dqn_baseline import DQNBaselineConfig, create_dqn, save_model
from src.experiments.eval_policies import (
    evaluate_on_splits,
    rollout_policy,
)

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
        "--hidden", type=int, nargs="+", default=[128, 128],
        help="Hidden layer sizes (default: 128 128).",
    )
    p.add_argument("--activation", default="relu", help="Activation function.")
    p.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout rate (0 = disabled, default: 0.1).",
    )
    p.add_argument(
        "--no_layer_norm", action="store_true",
        help="Disable layer normalisation.",
    )

    # Optimisation
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    p.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    p.add_argument(
        "--gamma", type=float, default=0.95,
        help="Discount factor (default: 0.95).",
    )
    p.add_argument(
        "--weight_decay", type=float, default=1e-4,
        help="L2 weight decay (default: 1e-4).",
    )
    p.add_argument(
        "--clip_grad_norm", type=float, default=1.0,
        help="Max gradient norm (0 = disabled, default: 1.0).",
    )
    p.add_argument(
        "--n_critics", type=int, default=3,
        help="Number of Q-network ensemble members (default: 3).",
    )
    p.add_argument(
        "--target_update", type=int, default=1000,
        help="Target network update interval (steps).",
    )

    # Training budget
    p.add_argument(
        "--n_steps", type=int, default=20_000,
        help="Total gradient steps (default: 20000).",
    )
    p.add_argument(
        "--n_steps_per_epoch", type=int, default=2_000,
        help="Steps per epoch (default: 2000).",
    )

    # Early stopping
    p.add_argument(
        "--patience", type=int, default=3,
        help="Early-stop patience in epochs (0 = disabled, default: 3).",
    )

    # Data
    p.add_argument(
        "--dataset", default="data/processed/offline_dataset_train.npz",
        help="Path to offline dataset.",
    )
    p.add_argument(
        "--val_path", default="data/processed/btc_daily_val.parquet",
        help="Validation split parquet (for early stopping).",
    )
    p.add_argument(
        "--log_return_column", default="log_return_next_1d",
        help="Forward return column name (log_return_next_1d or log_return_next_1h).",
    )
    p.add_argument(
        "--periods_per_year", type=int, default=252,
        help="Periods per year for annualization (252=daily, 8760=hourly).",
    )

    # Device
    p.add_argument("--device", default=None, help="PyTorch device (auto if omitted).")

    # Output
    p.add_argument(
        "--save_dir", default="models/dqn_baseline",
        help="Directory for saved model.",
    )

    return p.parse_args(argv)


# ── Early-stopping callback ──────────────────────────────────────────────

class _EarlyStopState:
    """Tracks validation Sharpe across epochs for early stopping."""

    def __init__(
        self,
        patience: int,
        save_dir: Path,
        val_path: str | Path = "data/processed/btc_daily_val.parquet",
        log_return_column: str = "log_return_next_1d",
        periods_per_year: int = 252,
        position_levels: tuple[float, ...] = (-1.0, 0.0, 1.0),
    ):
        self.patience = patience
        self.save_dir = save_dir
        self.val_path = Path(val_path)
        self.log_return_column = log_return_column
        self.periods_per_year = periods_per_year
        self.position_levels = position_levels
        self.best_sharpe: float = -float("inf")
        self.epochs_without_improvement: int = 0
        self.best_epoch: int = 0
        self.should_stop: bool = False

    def check(self, algo: object, epoch: int, total_step: int) -> None:
        """epoch_callback compatible with d3rlpy .fit()."""
        from src.experiments.eval_policies import rollout_policy

        if not self.val_path.is_file():
            return

        result = rollout_policy(
            algo,  # type: ignore[arg-type]
            data_path=self.val_path,
            split_name="val",
            d3rlpy_actions=True,
            log_return_column=self.log_return_column,
            periods_per_year=self.periods_per_year,
            position_levels=self.position_levels,
        )
        sharpe = result.metrics["sharpe"]
        ret = result.metrics["total_return"]

        LOG.info(
            "Epoch %d (step %d) — val Sharpe=%.4f  return=%.4f  (best=%.4f @ epoch %d)",
            epoch, total_step, sharpe, ret, self.best_sharpe, self.best_epoch,
        )

        if sharpe > self.best_sharpe:
            self.best_sharpe = sharpe
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            # Save best checkpoint
            best_path = self.save_dir / "model_best.d3"
            best_path.parent.mkdir(parents=True, exist_ok=True)
            algo.save(str(best_path))  # type: ignore[union-attr]
            LOG.info("  New best model saved → %s", best_path)
        else:
            self.epochs_without_improvement += 1
            if self.patience > 0 and self.epochs_without_improvement >= self.patience:
                LOG.info(
                    "  Early stopping triggered: %d epochs without improvement.",
                    self.epochs_without_improvement,
                )
                self.should_stop = True


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
    n_actions = get_n_actions(raw)
    position_levels = get_position_levels(raw)
    LOG.info(
        "Dataset: %d transitions, obs_dim=%d, action_space=Discrete(%d)",
        dataset.transition_count,
        raw["observations"].shape[1],
        n_actions,
    )

    # ── 2. Create DQN agent ───────────────────────────────────────────
    cfg = DQNBaselineConfig(
        algo=args.algo,
        hidden_units=args.hidden,
        activation=args.activation,
        use_layer_norm=not args.no_layer_norm,
        dropout_rate=args.dropout if args.dropout > 0 else None,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        clip_grad_norm=args.clip_grad_norm if args.clip_grad_norm > 0 else None,
        batch_size=args.batch_size,
        gamma=args.gamma,
        n_critics=args.n_critics,
        target_update_interval=args.target_update,
        n_steps=args.n_steps,
        n_steps_per_epoch=args.n_steps_per_epoch,
        early_stopping_patience=args.patience,
        device=args.device,
        save_dir=Path(args.save_dir),
    )

    algo = create_dqn(cfg)

    # ── 3. Train (with best-checkpoint tracking) ────────────────────────
    LOG.info("Starting training: %d steps ...", cfg.n_steps)

    es = _EarlyStopState(
        cfg.early_stopping_patience,
        cfg.save_dir,
        val_path=args.val_path,
        log_return_column=args.log_return_column,
        periods_per_year=args.periods_per_year,
        position_levels=position_levels,
    )

    algo.fit(
        dataset,
        n_steps=cfg.n_steps,
        n_steps_per_epoch=cfg.n_steps_per_epoch,
        experiment_name="dqn_baseline",
        show_progress=True,
        logging_steps=500,
        epoch_callback=es.check,
    )
    LOG.info("Training complete.")

    # ── 4. Save final model ───────────────────────────────────────────
    save_path = cfg.save_dir / "model.d3"
    save_model(algo, save_path)

    # ── 5. Load best model for evaluation (if early stopping saved one)
    best_path = cfg.save_dir / "model_best.d3"
    if best_path.exists():
        LOG.info("Loading best checkpoint (epoch %d) for evaluation.", es.best_epoch)
        from src.agents.dqn_baseline import load_model
        eval_algo = load_model(best_path, cfg.device)
        label = f"{cfg.algo.upper()} (best@ep{es.best_epoch})"
    else:
        eval_algo = algo
        label = cfg.algo.upper()

    # ── 6. Evaluate on val / test vs buy-and-hold ─────────────────────
    # Derive split paths from val_path (e.g. btc_hourly_val -> btc_hourly_test)
    val_p = str(args.val_path)
    test_p = val_p.replace("_val.", "_test.")
    eval_splits = {"val": val_p, "test": test_p}

    results = evaluate_on_splits(
        eval_algo,
        policy_name=label,
        splits=eval_splits,
        d3rlpy_actions=True,
        verbose=True,
        log_return_column=args.log_return_column,
        periods_per_year=args.periods_per_year,
        position_levels=position_levels,
    )

    # ── 7. Quick summary ──────────────────────────────────────────────
    for split_name, result in results.items():
        m = result.metrics
        LOG.info(
            "%s %s: return=%.4f  sharpe=%.4f  maxDD=%.4f  actions=[S:%.0f%% F:%.0f%% L:%.0f%%]",
            label,
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

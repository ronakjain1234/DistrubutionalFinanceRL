"""
Train offline Discrete CQL agent and evaluate vs DQN baseline + buy-and-hold (Step 7).

Usage
-----
::

    python -m src.experiments.train_cql                  # defaults
    python -m src.experiments.train_cql --alpha 2.0      # stronger conservatism
    python -m src.experiments.train_cql --alpha 0.5 --lr 3e-4 --n_steps 50000
    python -m src.experiments.train_cql --hidden 256 256 128 --dropout 0.15

The script:

1. Loads the offline dataset built in Step 5.
2. Trains DiscreteCQL on it via d3rlpy with the specified conservatism α.
3. Each epoch, computes on the validation split:

   * rollout Sharpe & return  (policy-level metric)
   * mean / std of Q(s,·)     (Q-value sanity check)
   * mean Q-gap (max_a Q - min_a Q) (action preference strength)
   * **conservatism proxy**: logsumexp_a Q(s,a) - max_a Q(s,a)
     — a direct analogue of the CQL penalty on unseen states.
     Small values = Q is peaky on the chosen action.
     Large values = Q is flat across actions (heavily regularised).
4. Early-stops on val Sharpe and saves the best checkpoint.
5. Evaluates the best model on val + test and prints a comparison table
   against the DQN baseline (if available) and buy-and-hold.
6. Dumps per-epoch diagnostics to ``models/cql/training_log.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.data.build_offline_dataset import (
    load_offline_dataset,
    to_d3rlpy_dataset,
    get_n_actions,
    get_position_levels,
)
from src.agents.cql import CQLConfig, create_cql, save_model, load_model
from src.experiments.eval_policies import (
    evaluate_on_splits,
    rollout_policy,
)

LOG = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train offline Discrete CQL for BTC trading (Step 7).",
    )

    # Conservatism
    p.add_argument(
        "--alpha", type=float, default=1.0,
        help="CQL conservatism weight α (default: 1.0). "
             "0 = DoubleDQN, higher = stronger anchoring to behaviour data.",
    )

    # Network
    p.add_argument(
        "--hidden", type=int, nargs="+", default=[256, 256],
        help="Hidden layer sizes (default: 256 256).",
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
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
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
        "--target_update", type=int, default=2_000,
        help="Target network update interval (steps, default: 2000).",
    )

    # Training budget
    p.add_argument(
        "--n_steps", type=int, default=30_000,
        help="Total gradient steps (default: 30000).",
    )
    p.add_argument(
        "--n_steps_per_epoch", type=int, default=2_000,
        help="Steps per epoch (default: 2000).",
    )

    # Early stopping
    p.add_argument(
        "--patience", type=int, default=4,
        help="Early-stop patience in epochs (0 = disabled, default: 4).",
    )

    # Data
    p.add_argument(
        "--dataset", default="data/processed/offline_dataset_train.npz",
        help="Path to offline dataset.",
    )
    p.add_argument(
        "--val_path", default="data/processed/btc_daily_val.parquet",
        help="Validation split parquet (for early stopping and Q-stats).",
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
        "--save_dir", default="models/cql",
        help="Directory for saved model and training log.",
    )

    # Optional comparison with DQN baseline
    p.add_argument(
        "--dqn_model",
        default="models/dqn_baseline/model_best.d3",
        help="Path to DQN baseline checkpoint for side-by-side comparison "
             "(skipped if the file does not exist).",
    )

    return p.parse_args(argv)


# ── Q-value diagnostics ──────────────────────────────────────────────────

def compute_q_stats(
    algo, val_observations: np.ndarray, n_actions: int = 3,
) -> dict[str, float]:
    """
    Compute Q-value statistics on a batch of validation observations.

    Produces the "conservatism behavior" diagnostics required by Step 7:

    * ``q_mean`` / ``q_std``    — marginal value scale
    * ``q_gap_mean``            — average (max_a Q - min_a Q); bigger gap
                                  = sharper action preferences
    * ``conservatism_proxy``    — average logsumexp_a Q(s,a) - max_a Q(s,a);
                                  the logsumexp term is *exactly* what CQL's
                                  penalty minimises.  As α grows, the learned
                                  Q should push this quantity **down** toward
                                  log(n_actions) (= fully flat Q).
    * ``greedy_action_frac_*``  — fraction of val states where each discrete
                                  action is argmax (sanity check: does the
                                  policy actually use all actions?)
    """
    # Query Q(s, a) for each of the three discrete actions.
    # d3rlpy expects action arrays shaped (N,).
    n = val_observations.shape[0]
    q_values = np.empty((n, n_actions), dtype=np.float64)
    for a in range(n_actions):
        acts = np.full((n,), a, dtype=np.int64)
        q_values[:, a] = algo.predict_value(val_observations, acts).ravel()

    # Core stats
    q_mean = float(q_values.mean())
    q_std = float(q_values.std())

    # Action-spread metrics
    q_max = q_values.max(axis=1)
    q_min = q_values.min(axis=1)
    q_gap_mean = float((q_max - q_min).mean())

    # Conservatism proxy = logsumexp(Q) - max(Q)
    # (a = max-row-value → stable implementation)
    logsumexp = q_max + np.log(np.exp(q_values - q_max[:, None]).sum(axis=1))
    conservatism = float((logsumexp - q_max).mean())

    # Greedy action distribution on val
    greedy = q_values.argmax(axis=1)
    action_frac = {
        f"greedy_action_frac_{a}": float((greedy == a).mean())
        for a in range(n_actions)
    }

    return {
        "q_mean": q_mean,
        "q_std": q_std,
        "q_gap_mean": q_gap_mean,
        "conservatism_proxy": conservatism,
        **action_frac,
    }


# ── Early-stopping + Q-stats callback ────────────────────────────────────

@dataclass
class _TrainingDiagnostics:
    """Tracks validation metrics + Q-stats across epochs."""

    patience: int
    save_dir: Path
    val_observations: np.ndarray
    val_path: Path
    log_return_column: str = "log_return_next_1d"
    periods_per_year: int = 252
    n_actions: int = 3
    position_levels: tuple[float, ...] = (-1.0, 0.0, 1.0)

    best_sharpe: float = field(default=-float("inf"))
    best_epoch: int = 0
    epochs_without_improvement: int = 0
    should_stop: bool = False

    history: list[dict] = field(default_factory=list)

    def check(self, algo: object, epoch: int, total_step: int) -> None:
        """epoch_callback compatible with d3rlpy .fit()."""
        if not self.val_path.is_file():
            return

        # ── 1. Rollout on validation split ────────────────────────────
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
        max_dd = result.metrics["max_drawdown"]

        # ── 2. Q-value diagnostics on held-out states ─────────────────
        try:
            q_stats = compute_q_stats(
                algo, self.val_observations, n_actions=self.n_actions,
            )
        except Exception as e:  # noqa: BLE001
            LOG.warning("Q-stats computation failed: %s", e)
            q_stats = {}

        LOG.info(
            "Epoch %d (step %d) — val Sharpe=%.4f  ret=%+.4f  maxDD=%.4f"
            "  | Q mean=%.4f gap=%.4f cons=%.4f"
            "  | best=%.4f @ ep%d",
            epoch, total_step, sharpe, ret, max_dd,
            q_stats.get("q_mean", float("nan")),
            q_stats.get("q_gap_mean", float("nan")),
            q_stats.get("conservatism_proxy", float("nan")),
            self.best_sharpe, self.best_epoch,
        )

        # ── 3. Record history ─────────────────────────────────────────
        self.history.append({
            "epoch": int(epoch),
            "step": int(total_step),
            "val_sharpe": float(sharpe),
            "val_total_return": float(ret),
            "val_max_drawdown": float(max_dd),
            "val_action_frac_short": float(
                result.metrics.get("action_frac_short", 0.0)
            ),
            "val_action_frac_flat": float(
                result.metrics.get("action_frac_flat", 0.0)
            ),
            "val_action_frac_long": float(
                result.metrics.get("action_frac_long", 0.0)
            ),
            **q_stats,
        })

        # ── 4. Best-checkpoint tracking ──────────────────────────────
        if sharpe > self.best_sharpe:
            self.best_sharpe = sharpe
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
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

    def dump(self, path: Path) -> None:
        """Persist diagnostics to JSON for later analysis / plotting."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "best_sharpe": float(self.best_sharpe),
            "best_epoch": int(self.best_epoch),
            "history": self.history,
        }
        path.write_text(json.dumps(payload, indent=2))
        LOG.info("Training log saved → %s", path)


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

    # ── 2. Create CQL agent ───────────────────────────────────────────
    cfg = CQLConfig(
        alpha=args.alpha,
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

    algo = create_cql(cfg)

    # ── 3. Pre-load a batch of val observations for Q-stats ───────────
    #      (We use a fixed sample so diagnostics are comparable across epochs.)
    val_path = Path(args.val_path)
    if val_path.is_file():
        import pandas as pd
        from src.env.offline_trading_env import _infer_feature_columns

        val_df = pd.read_parquet(val_path)
        feat_cols = _infer_feature_columns(val_df)
        val_feats = val_df[feat_cols].to_numpy(dtype=np.float32)
        # Env appends current position; use 0 (flat) since Q-stats are a
        # snapshot and we don't want position bias.
        val_obs = np.concatenate(
            [val_feats, np.zeros((len(val_feats), 1), dtype=np.float32)],
            axis=1,
        )
        LOG.info("Loaded %d val observations for Q-stats.", len(val_obs))
    else:
        LOG.warning("Val split not found at %s — Q-stats disabled.", val_path)
        val_obs = np.zeros((0, raw["observations"].shape[1]), dtype=np.float32)

    diagnostics = _TrainingDiagnostics(
        patience=cfg.early_stopping_patience,
        save_dir=cfg.save_dir,
        val_observations=val_obs,
        val_path=val_path,
        log_return_column=args.log_return_column,
        periods_per_year=args.periods_per_year,
        n_actions=n_actions,
        position_levels=position_levels,
    )

    # ── 4. Train ──────────────────────────────────────────────────────
    LOG.info(
        "Starting training: %d steps, %d per epoch, alpha=%.3f",
        cfg.n_steps, cfg.n_steps_per_epoch, cfg.alpha,
    )

    algo.fit(
        dataset,
        n_steps=cfg.n_steps,
        n_steps_per_epoch=cfg.n_steps_per_epoch,
        experiment_name="cql",
        show_progress=True,
        logging_steps=500,
        epoch_callback=diagnostics.check,
    )
    LOG.info("Training complete.")

    # ── 5. Save final model + training log ───────────────────────────
    final_path = cfg.save_dir / "model.d3"
    save_model(algo, final_path)
    diagnostics.dump(cfg.save_dir / "training_log.json")

    # ── 6. Load best model for evaluation ────────────────────────────
    best_path = cfg.save_dir / "model_best.d3"
    if best_path.exists():
        LOG.info("Loading best checkpoint (epoch %d) for evaluation.",
                 diagnostics.best_epoch)
        eval_algo = load_model(best_path, cfg.device)
        label = f"CQL(a={cfg.alpha:g},best@ep{diagnostics.best_epoch})"
    else:
        eval_algo = algo
        label = f"CQL(a={cfg.alpha:g})"

    # ── 7. Evaluate on val / test vs buy-and-hold ────────────────────
    val_p = str(args.val_path)
    test_p = val_p.replace("_val.", "_test.")
    eval_splits = {"val": val_p, "test": test_p}

    cql_results = evaluate_on_splits(
        eval_algo,
        policy_name=label,
        splits=eval_splits,
        d3rlpy_actions=True,
        verbose=True,
        log_return_column=args.log_return_column,
        periods_per_year=args.periods_per_year,
        position_levels=position_levels,
    )

    # ── 8. Optional side-by-side comparison with DQN baseline ────────
    dqn_path = Path(args.dqn_model)
    if dqn_path.is_file():
        LOG.info("Found DQN baseline at %s — running comparison.", dqn_path)
        try:
            dqn_algo = load_model(dqn_path, cfg.device)
            evaluate_on_splits(
                dqn_algo,
                policy_name="DQN (baseline)",
                splits=eval_splits,
                d3rlpy_actions=True,
                verbose=True,
                log_return_column=args.log_return_column,
                periods_per_year=args.periods_per_year,
                position_levels=position_levels,
            )
        except Exception as e:  # noqa: BLE001
            LOG.warning("Could not evaluate DQN baseline: %s", e)

    # ── 9. Summary line ──────────────────────────────────────────────
    for split_name, result in cql_results.items():
        m = result.metrics
        LOG.info(
            "%s %s: return=%+.4f  sharpe=%.4f  maxDD=%.4f  actions=[S:%.0f%% F:%.0f%% L:%.0f%%]",
            label, split_name,
            m["total_return"], m["sharpe"], m["max_drawdown"],
            m.get("action_frac_short", 0) * 100,
            m.get("action_frac_flat", 0) * 100,
            m.get("action_frac_long", 0) * 100,
        )


if __name__ == "__main__":
    main()

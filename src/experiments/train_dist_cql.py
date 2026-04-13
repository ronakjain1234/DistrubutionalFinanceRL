"""
Train Distributional CQL (QR-CQL) and evaluate vs CQL / DQN / buy-and-hold (Step 8).

Usage
-----
::

    python -m src.experiments.train_dist_cql                         # defaults
    python -m src.experiments.train_dist_cql --alpha 2.0             # stronger conservatism
    python -m src.experiments.train_dist_cql --n_quantiles 31        # fewer quantiles
    python -m src.experiments.train_dist_cql --action_selection cvar_10  # risk-sensitive

The script:

1. Loads the offline dataset from Step 5.
2. Trains a Distributional CQL agent (quantile-regression Q-networks
   with conservative and tail-aware regularisation).
3. Each epoch, evaluates on the validation split:
   - Rollout Sharpe, return, max drawdown, action distribution.
   - Q-value diagnostics: mean, gap, conservatism proxy.
   - Distributional diagnostics: quantile spread, CVaR estimates.
4. Early-stops on val Sharpe and saves the best checkpoint.
5. Evaluates the best model on val + test vs buy-and-hold, and
   optionally compares against CQL and DQN baselines.
6. Dumps per-epoch diagnostics to ``models/dist_cql/training_log.json``.
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
    get_n_actions,
    get_position_levels,
)
from src.agents.distributional_qnet import (
    DistCQLConfig,
    DistributionalCQLAgent,
)
from src.experiments.eval_policies import (
    evaluate_on_splits,
    rollout_policy,
)

LOG = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Distributional CQL (QR-CQL) for BTC trading (Step 8).",
    )

    # Distributional
    p.add_argument("--n_quantiles", type=int, default=51,
                   help="Number of quantile levels (default: 51).")
    p.add_argument("--no_tail_focus", action="store_true",
                   help="Disable tail-focused quantile spacing.")
    p.add_argument("--kappa", type=float, default=1.0,
                   help="Huber loss threshold (default: 1.0).")

    # CQL conservatism
    p.add_argument("--alpha", type=float, default=1.0,
                   help="CQL conservatism weight on mean Q (default: 1.0).")
    p.add_argument("--alpha_tail", type=float, default=0.5,
                   help="CQL penalty on upper-tail quantiles (default: 0.5, 0=off).")

    # Network
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    p.add_argument("--activation", default="relu")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no_layer_norm", action="store_true")

    # Ensemble
    p.add_argument("--n_ensemble", type=int, default=3)
    p.add_argument("--ensemble_penalty", type=float, default=0.0,
                   help="Ensemble disagreement penalty (0=min-of-means).")

    # Optimisation
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)
    p.add_argument("--target_update", type=int, default=2_000)

    # Training budget
    p.add_argument("--n_steps", type=int, default=30_000)
    p.add_argument("--n_steps_per_epoch", type=int, default=2_000)

    # Early stopping
    p.add_argument("--patience", type=int, default=4)

    # Action selection
    p.add_argument("--action_selection", default="mean",
                   choices=["mean", "cvar_10", "cvar_25", "quantile_10"],
                   help="Action selection strategy (default: mean).")

    # Data
    p.add_argument("--dataset", default="data/processed/offline_dataset_train.npz")
    p.add_argument("--val_path", default="data/processed/btc_daily_val.parquet")
    p.add_argument("--log_return_column", default="log_return_next_1d")
    p.add_argument("--periods_per_year", type=int, default=252)

    # Device
    p.add_argument("--device", default=None)

    # Output
    p.add_argument("--save_dir", default="models/dist_cql")

    # Baselines for comparison
    p.add_argument("--cql_model", default="models/cql/model_best.d3",
                   help="CQL checkpoint for comparison (skipped if missing).")
    p.add_argument("--dqn_model", default="models/dqn_baseline/model_best.d3",
                   help="DQN checkpoint for comparison (skipped if missing).")

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


# ── Q-value & distributional diagnostics ─────────────────────────────────

def compute_q_stats(
    agent: DistributionalCQLAgent,
    val_observations: np.ndarray,
    n_actions: int = 3,
) -> dict[str, float]:
    """
    Q-value and distributional diagnostics on validation observations.

    Extends the CQL Q-stats with quantile-specific metrics:
    - quantile_spread: mean range of quantile values per action
    - cvar_10 / cvar_25: conditional value-at-risk estimates
    """
    n = val_observations.shape[0]

    # Mean Q-values per action (via predict_value)
    q_values = np.empty((n, n_actions), dtype=np.float64)
    for a in range(n_actions):
        acts = np.full((n,), a, dtype=np.int64)
        q_values[:, a] = agent.predict_value(val_observations, acts).ravel()

    q_mean = float(q_values.mean())
    q_std = float(q_values.std())
    q_max = q_values.max(axis=1)
    q_min = q_values.min(axis=1)
    q_gap_mean = float((q_max - q_min).mean())

    # Conservatism proxy
    logsumexp = q_max + np.log(np.exp(q_values - q_max[:, None]).sum(axis=1))
    conservatism = float((logsumexp - q_max).mean())

    # Greedy action distribution
    greedy = q_values.argmax(axis=1)
    action_frac = {
        f"greedy_action_frac_{a}": float((greedy == a).mean())
        for a in range(n_actions)
    }

    # Distributional metrics: quantile spread and CVaR
    # Sample a subset for efficiency
    n_sample = min(500, n)
    idx = np.random.choice(n, n_sample, replace=False)
    sample_obs = val_observations[idx]
    quantiles = agent.predict_quantiles(sample_obs)  # (n_sample, A, N)
    taus = agent.get_taus()

    # Spread: mean range of quantile values across all actions and samples
    q_range = quantiles.max(axis=-1) - quantiles.min(axis=-1)  # (n_sample, A)
    quantile_spread = float(q_range.mean())

    # CVaR at 10th and 25th percentile (averaged across actions, for greedy action)
    greedy_sample = q_values[idx].argmax(axis=1)  # (n_sample,)
    greedy_quantiles = quantiles[np.arange(n_sample), greedy_sample, :]  # (n_sample, N)

    cvar_10 = cvar_25 = float("nan")
    mask_10 = taus <= 0.10
    mask_25 = taus <= 0.25
    if mask_10.any():
        cvar_10 = float(greedy_quantiles[:, mask_10].mean())
    if mask_25.any():
        cvar_25 = float(greedy_quantiles[:, mask_25].mean())

    return {
        "q_mean": q_mean,
        "q_std": q_std,
        "q_gap_mean": q_gap_mean,
        "conservatism_proxy": conservatism,
        "quantile_spread": quantile_spread,
        "cvar_10": cvar_10,
        "cvar_25": cvar_25,
        **action_frac,
    }


# ── Training diagnostics callback ────────────────────────────────────────

@dataclass
class _TrainingDiagnostics:
    """Validation metrics + Q-stats + distributional diagnostics."""

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

    history: list[dict] = field(default_factory=list)

    def check(
        self,
        agent: DistributionalCQLAgent,
        epoch: int,
        total_step: int,
    ) -> bool:
        """
        Epoch callback.  Returns True to signal early stop.
        """
        if not self.val_path.is_file():
            return False

        # 1. Rollout on validation split
        result = rollout_policy(
            agent,
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

        # 2. Q-value + distributional diagnostics
        try:
            q_stats = compute_q_stats(
                agent, self.val_observations, n_actions=self.n_actions,
            )
        except Exception as e:  # noqa: BLE001
            LOG.warning("Q-stats failed: %s", e)
            q_stats = {}

        LOG.info(
            "Epoch %d (step %d) — val Sharpe=%.4f  ret=%+.4f  maxDD=%.4f"
            "  | Q mean=%.4f gap=%.4f cons=%.4f spread=%.4f"
            "  | best=%.4f @ ep%d",
            epoch, total_step, sharpe, ret, max_dd,
            q_stats.get("q_mean", float("nan")),
            q_stats.get("q_gap_mean", float("nan")),
            q_stats.get("conservatism_proxy", float("nan")),
            q_stats.get("quantile_spread", float("nan")),
            self.best_sharpe, self.best_epoch,
        )

        # 3. Record
        self.history.append({
            "epoch": int(epoch),
            "step": int(total_step),
            "val_sharpe": float(sharpe),
            "val_total_return": float(ret),
            "val_max_drawdown": float(max_dd),
            "val_action_frac_short": float(result.metrics.get("action_frac_short", 0)),
            "val_action_frac_flat": float(result.metrics.get("action_frac_flat", 0)),
            "val_action_frac_long": float(result.metrics.get("action_frac_long", 0)),
            **{k: float(v) for k, v in q_stats.items()},
        })

        # 4. Best-checkpoint tracking
        if sharpe > self.best_sharpe:
            self.best_sharpe = sharpe
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            best_path = self.save_dir / "model_best"
            agent.save(best_path)
            LOG.info("  New best model saved -> %s", best_path)
        else:
            self.epochs_without_improvement += 1
            if self.patience > 0 and self.epochs_without_improvement >= self.patience:
                LOG.info(
                    "  Early stopping: %d epochs without improvement.",
                    self.epochs_without_improvement,
                )
                return True

        return False

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "best_sharpe": float(self.best_sharpe),
            "best_epoch": int(self.best_epoch),
            "history": self.history,
        }
        path.write_text(json.dumps(payload, indent=2))
        LOG.info("Training log saved -> %s", path)


# ── Main ──────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-36s  %(levelname)-7s  %(message)s",
    )

    args = parse_args(argv)

    # ── 1. Load offline dataset ──────────────────────────────────────
    LOG.info("Loading offline dataset from %s ...", args.dataset)
    raw = load_offline_dataset(args.dataset)
    n_actions = get_n_actions(raw)
    position_levels = get_position_levels(raw)

    # Convert actions: new datasets already {0,...,n-1}, old ones need offset
    offset = int(raw.get("action_offset", np.array([0]))[0])
    observations = raw["observations"]
    actions = raw["actions"].astype(np.int64) + offset
    rewards = raw["rewards"].astype(np.float32)
    next_observations = raw["next_observations"]
    terminals = raw["terminals"].astype(np.float32)

    obs_dim = observations.shape[1]
    LOG.info(
        "Dataset: %d transitions, obs_dim=%d, n_actions=%d",
        len(observations), obs_dim, n_actions,
    )

    # ── 2. Create agent ──────────────────────────────────────────────
    cfg = DistCQLConfig(
        n_quantiles=args.n_quantiles,
        kappa=args.kappa,
        tail_quantile_focus=not args.no_tail_focus,
        alpha=args.alpha,
        alpha_tail=args.alpha_tail,
        hidden_units=args.hidden,
        activation=args.activation,
        use_layer_norm=not args.no_layer_norm,
        dropout_rate=args.dropout if args.dropout > 0 else 0.0,
        n_ensemble=args.n_ensemble,
        ensemble_penalty=args.ensemble_penalty,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        clip_grad_norm=args.clip_grad_norm if args.clip_grad_norm > 0 else 0.0,
        batch_size=args.batch_size,
        gamma=args.gamma,
        target_update_interval=args.target_update,
        n_steps=args.n_steps,
        n_steps_per_epoch=args.n_steps_per_epoch,
        early_stopping_patience=args.patience,
        device=args.device,
        save_dir=Path(args.save_dir),
        action_selection=args.action_selection,
        seed=args.seed,
    )

    agent = DistributionalCQLAgent(cfg, obs_dim=obs_dim, n_actions=n_actions)

    # ── 3. Pre-load val observations for diagnostics ─────────────────
    val_path = Path(args.val_path)
    if val_path.is_file():
        import pandas as pd
        from src.env.offline_trading_env import _infer_feature_columns

        val_df = pd.read_parquet(val_path)
        feat_cols = _infer_feature_columns(val_df)
        val_feats = val_df[feat_cols].to_numpy(dtype=np.float32)
        val_obs = np.concatenate(
            [val_feats, np.zeros((len(val_feats), 1), dtype=np.float32)],
            axis=1,
        )
        LOG.info("Loaded %d val observations for Q-stats.", len(val_obs))
    else:
        LOG.warning("Val split not found at %s — Q-stats disabled.", val_path)
        val_obs = np.zeros((0, obs_dim), dtype=np.float32)

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

    # ── 4. Train ─────────────────────────────────────────────────────
    LOG.info(
        "Starting training: %d steps, %d per epoch, alpha=%.2f, "
        "alpha_tail=%.2f, n_quantiles=%d, tail_focus=%s",
        cfg.n_steps, cfg.n_steps_per_epoch,
        cfg.alpha, cfg.alpha_tail, cfg.n_quantiles,
        cfg.tail_quantile_focus,
    )

    agent.fit(
        observations, actions, rewards, next_observations, terminals,
        n_steps=cfg.n_steps,
        n_steps_per_epoch=cfg.n_steps_per_epoch,
        epoch_callback=diagnostics.check,
        show_progress=True,
    )
    LOG.info("Training complete.")

    # ── 5. Save final model + training log ───────────────────────────
    final_path = cfg.save_dir / "model_final"
    agent.save(final_path)
    diagnostics.dump(cfg.save_dir / "training_log.json")

    # ── 6. Load best model for evaluation ────────────────────────────
    best_path = cfg.save_dir / "model_best"
    if best_path.exists():
        LOG.info(
            "Loading best checkpoint (epoch %d) for evaluation.",
            diagnostics.best_epoch,
        )
        eval_agent = DistributionalCQLAgent.load(best_path, cfg.device)
    else:
        eval_agent = agent

    label = (
        f"DistCQL(a={cfg.alpha:g},at={cfg.alpha_tail:g},"
        f"q={cfg.n_quantiles},best@ep{diagnostics.best_epoch})"
    )

    # ── 7. Evaluate on val + test vs buy-and-hold ────────────────────
    val_p = str(args.val_path)
    test_p = val_p.replace("_val.", "_test.")
    eval_splits = {"val": val_p, "test": test_p}

    dist_results = evaluate_on_splits(
        eval_agent,
        policy_name=label,
        splits=eval_splits,
        d3rlpy_actions=True,
        verbose=True,
        log_return_column=args.log_return_column,
        periods_per_year=args.periods_per_year,
        position_levels=position_levels,
    )

    # ── 8. Compare with CQL baseline ────────────────────────────────
    cql_path = Path(args.cql_model)
    if cql_path.is_file():
        LOG.info("Found CQL baseline at %s -- running comparison.", cql_path)
        try:
            import d3rlpy
            cql_algo = d3rlpy.load_learnable(str(cql_path), device="cpu:0")
            evaluate_on_splits(
                cql_algo,
                policy_name="CQL (baseline)",
                splits=eval_splits,
                d3rlpy_actions=True,
                verbose=True,
                log_return_column=args.log_return_column,
                periods_per_year=args.periods_per_year,
                position_levels=position_levels,
            )
        except Exception as e:  # noqa: BLE001
            LOG.warning("Could not evaluate CQL baseline: %s", e)

    # ── 9. Compare with DQN baseline ─────────────────────────────────
    dqn_path = Path(args.dqn_model)
    if dqn_path.is_file():
        LOG.info("Found DQN baseline at %s -- running comparison.", dqn_path)
        try:
            import d3rlpy
            dqn_algo = d3rlpy.load_learnable(str(dqn_path), device="cpu:0")
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

    # ── 10. Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  DISTRIBUTIONAL CQL — TRAINING SUMMARY")
    print("=" * 72)
    print(f"  Quantiles       : {cfg.n_quantiles} ({'tail-focused' if cfg.tail_quantile_focus else 'uniform'})")
    print(f"  CQL alpha       : {cfg.alpha}  (tail: {cfg.alpha_tail})")
    print(f"  Ensemble        : {cfg.n_ensemble} networks")
    print(f"  Action selection: {cfg.action_selection}")
    print(f"  Best epoch      : {diagnostics.best_epoch}  (val Sharpe = {diagnostics.best_sharpe:.4f})")
    print()

    for split_name, result in dist_results.items():
        m = result.metrics
        print(
            f"  {split_name:>5s}: return={m['total_return']:+.4f}  "
            f"sharpe={m['sharpe']:.4f}  maxDD={m['max_drawdown']:.4f}  "
            f"actions=[S:{m.get('action_frac_short', 0):.0%} "
            f"F:{m.get('action_frac_flat', 0):.0%} "
            f"L:{m.get('action_frac_long', 0):.0%}]"
        )

    print("=" * 72)


if __name__ == "__main__":
    main()

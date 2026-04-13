"""
Ensemble evaluation: combine multiple model predictions.

Strategies:
1. average_position: Average all models' position levels, snap to nearest valid action
2. majority_vote: Each model votes for an action; most common wins (ties → long)
3. veto_short: Only short if ALL models agree; otherwise use average of non-short models
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from collections import Counter

import numpy as np

from src.env.offline_trading_env import OfflineTradingEnv, EnvConfig, POSITION_LEVELS_3
from src.env.portfolio import PortfolioConfig
from src.experiments.baselines import equity_metrics, run_buy_and_hold_on_split
from src.agents.distributional_qnet import DistributionalCQLAgent
from src.agents.cql import load_model as load_cql
from src.agents.dqn_baseline import load_model as load_dqn

LOG = logging.getLogger(__name__)

POSITION_LEVELS = POSITION_LEVELS_3  # (-1.0, 0.0, 1.0)


class EnsemblePolicy:
    """Combine N models into one policy."""

    def __init__(self, models: list, model_names: list[str],
                 d3rlpy_flags: list[bool], strategy: str = "average_position"):
        self.models = models
        self.model_names = model_names
        self.d3rlpy_flags = d3rlpy_flags
        self.strategy = strategy

    def _get_all_actions(self, x: np.ndarray) -> list[int]:
        """Get action index from each model."""
        actions = []
        for model in self.models:
            actions.append(int(model.predict(x)[0]))
        return actions

    def predict(self, x: np.ndarray) -> np.ndarray:
        actions = self._get_all_actions(x)
        positions = [POSITION_LEVELS[a] for a in actions]

        if self.strategy == "average_position":
            avg_pos = sum(positions) / len(positions)
            final_action = int(np.argmin([abs(avg_pos - p) for p in POSITION_LEVELS]))

        elif self.strategy == "majority_vote":
            counts = Counter(actions)
            # Most common action; ties broken by highest action index (long bias)
            final_action = max(counts.keys(), key=lambda a: (counts[a], a))

        elif self.strategy == "veto_short":
            # Only short if ALL models agree
            if all(p == -1.0 for p in positions):
                final_action = 0  # short
            else:
                # Average the non-short positions (or all if none are short)
                non_short = [p for p in positions if p > -1.0]
                if not non_short:
                    non_short = positions
                avg = sum(non_short) / len(non_short)
                final_action = int(np.argmin([abs(avg - p) for p in POSITION_LEVELS]))

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return np.array([final_action])


def run_rollout(policy, data_path, split_name, log_return_column, periods_per_year):
    """Run a single rollout and return metrics."""
    env_cfg = EnvConfig(
        data_path=Path(data_path),
        portfolio_cfg=PortfolioConfig(),
        log_return_column=log_return_column,
        position_levels=POSITION_LEVELS,
    )
    env = OfflineTradingEnv(env_cfg)
    obs, info = env.reset()

    equities = [info["equity"]]
    log_rets = []
    actions_list = []

    while True:
        obs_batch = obs.reshape(1, -1)
        action = int(policy.predict(obs_batch)[0])
        obs, reward, terminated, truncated, info = env.step(action)
        equities.append(info["equity"])
        log_rets.append(reward)
        actions_list.append(action)
        if terminated or truncated:
            break

    equity_arr = np.array(equities, dtype=np.float64)
    log_ret_arr = np.array(log_rets, dtype=np.float32)

    m = equity_metrics(equity_arr, log_ret_arr, periods_per_year=periods_per_year)

    act_arr = np.array(actions_list)
    n = len(act_arr)
    m["action_frac_short"] = float(np.sum(act_arr == 0)) / n
    m["action_frac_flat"] = float(np.sum(act_arr == 1)) / n
    m["action_frac_long"] = float(np.sum(act_arr == 2)) / n

    return m, act_arr


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-36s  %(levelname)-7s  %(message)s",
    )

    p = argparse.ArgumentParser(description="Ensemble evaluation")
    p.add_argument("--dist_cql_path", default="models/dist_cql/model_best")
    p.add_argument("--cql_path", default="models/cql/model_best.d3")
    p.add_argument("--dqn_path", default="models/dqn_baseline/model_best.d3")
    p.add_argument("--strategy", nargs="+",
                   default=["average_position", "majority_vote", "veto_short"])
    p.add_argument("--split", default="both", choices=["val", "test", "both"])
    p.add_argument("--models", nargs="+", default=["dist_cql", "cql", "dqn"],
                   help="Which models to include (default: all three)")
    args = p.parse_args()

    # Load requested models
    all_models = []
    all_names = []
    all_d3rlpy = []

    if "dist_cql" in args.models:
        LOG.info("Loading DistCQL from %s", args.dist_cql_path)
        dist_cql = DistributionalCQLAgent.load(args.dist_cql_path)
        all_models.append(dist_cql)
        all_names.append("DistCQL")
        all_d3rlpy.append(False)

    if "cql" in args.models:
        LOG.info("Loading CQL from %s", args.cql_path)
        cql = load_cql(args.cql_path)
        all_models.append(cql)
        all_names.append("CQL")
        all_d3rlpy.append(True)

    if "dqn" in args.models:
        LOG.info("Loading DQN from %s", args.dqn_path)
        dqn = load_dqn(args.dqn_path)
        all_models.append(dqn)
        all_names.append("DQN")
        all_d3rlpy.append(True)

    LOG.info("Ensemble members: %s", all_names)

    # Determine splits
    splits = {}
    if args.split in ("val", "both"):
        splits["val"] = "data/processed/btc_daily_val.parquet"
    if args.split in ("test", "both"):
        splits["test"] = "data/processed/btc_daily_test.parquet"

    log_return_column = "log_return_next_1d"
    periods_per_year = 252

    # Run all strategies
    ensemble_label = "+".join(all_names)
    print("\n" + "=" * 90)
    print(f"  ENSEMBLE EVALUATION — {ensemble_label}")
    print("=" * 90)

    for strategy in args.strategy:
        print(f"\n--- Strategy: {strategy} ---")
        ensemble = EnsemblePolicy(all_models, all_names, all_d3rlpy, strategy=strategy)

        for split_name, path in splits.items():
            m, acts = run_rollout(ensemble, path, split_name, log_return_column, periods_per_year)

            bh = run_buy_and_hold_on_split(
                path, split_name=split_name,
                log_return_column=log_return_column,
                periods_per_year=periods_per_year,
            )

            print(f"  {split_name:>5s}: return={m['total_return']:+.1%}  "
                  f"sharpe={m['sharpe']:.2f}  maxDD={m['max_drawdown']:.1%}  "
                  f"actions=[S:{m['action_frac_short']:.0%} F:{m['action_frac_flat']:.0%} L:{m['action_frac_long']:.0%}]  "
                  f"(B&H: {bh.metrics['total_return']:+.1%})")

    # Also run 2-model subsets for comparison
    if len(all_models) == 3:
        print(f"\n--- 2-Model Subsets (average_position) ---")
        pairs = [
            ([0, 1], "DistCQL+CQL"),
            ([0, 2], "DistCQL+DQN"),
            ([1, 2], "CQL+DQN"),
        ]
        for idxs, pair_name in pairs:
            pair_models = [all_models[i] for i in idxs]
            pair_names = [all_names[i] for i in idxs]
            pair_flags = [all_d3rlpy[i] for i in idxs]
            ensemble = EnsemblePolicy(pair_models, pair_names, pair_flags, strategy="average_position")

            for split_name, path in splits.items():
                m, acts = run_rollout(ensemble, path, split_name, log_return_column, periods_per_year)
                print(f"  {pair_name:>15s} {split_name:>5s}: return={m['total_return']:+.1%}  "
                      f"sharpe={m['sharpe']:.2f}  maxDD={m['max_drawdown']:.1%}  "
                      f"actions=[S:{m['action_frac_short']:.0%} F:{m['action_frac_flat']:.0%} L:{m['action_frac_long']:.0%}]")

    # Individual models for reference
    print(f"\n--- Individual Models (reference) ---")
    for model, name, is_d3rlpy in zip(all_models, all_names, all_d3rlpy):
        for split_name, path in splits.items():
            from src.experiments.eval_policies import rollout_policy
            result = rollout_policy(
                model, data_path=path, split_name=split_name,
                d3rlpy_actions=is_d3rlpy,
                log_return_column=log_return_column,
                periods_per_year=periods_per_year,
                position_levels=POSITION_LEVELS,
            )
            m = result.metrics
            print(f"  {name:>10s} {split_name:>5s}: return={m['total_return']:+.1%}  "
                  f"sharpe={m['sharpe']:.2f}  maxDD={m['max_drawdown']:.1%}  "
                  f"actions=[S:{m.get('action_frac_short',0):.0%} F:{m.get('action_frac_flat',0):.0%} L:{m.get('action_frac_long',0):.0%}]")

    print()


if __name__ == "__main__":
    main()

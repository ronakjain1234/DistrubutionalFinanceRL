"""
Ensemble evaluation: combine daily DistCQL + CQL predictions.

Strategies:
1. majority_vote: Both models vote; ties broken by DistCQL (the better model)
2. average_position: Average the position levels, snap to nearest valid action
3. veto_short: Use DistCQL's action unless both agree to short
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from src.env.offline_trading_env import OfflineTradingEnv, EnvConfig, POSITION_LEVELS_3
from src.env.portfolio import PortfolioConfig
from src.experiments.baselines import equity_metrics, run_buy_and_hold_on_split
from src.agents.distributional_qnet import DistributionalCQLAgent
from src.agents.cql import load_model as load_cql

LOG = logging.getLogger(__name__)

POSITION_LEVELS = POSITION_LEVELS_3  # (-1.0, 0.0, 1.0)


class EnsemblePolicy:
    """Combine two models into one policy."""

    def __init__(self, dist_cql, cql_model, strategy: str = "majority_vote"):
        self.dist_cql = dist_cql
        self.cql = cql_model
        self.strategy = strategy

    def predict(self, x: np.ndarray) -> np.ndarray:
        # Get actions from both models
        # DistCQL: predict returns action indices directly
        dist_action = int(self.dist_cql.predict(x)[0])
        cql_action = int(self.cql.predict(x)[0])

        # Map to position levels: 0=-1.0, 1=0.0, 2=+1.0
        dist_pos = POSITION_LEVELS[dist_action]
        cql_pos = POSITION_LEVELS[cql_action]

        if self.strategy == "majority_vote":
            # If they agree, use that action
            if dist_action == cql_action:
                final_action = dist_action
            else:
                # Tie: use DistCQL (the better model)
                final_action = dist_action

        elif self.strategy == "average_position":
            avg_pos = (dist_pos + cql_pos) / 2.0
            # Snap to nearest valid position level
            final_action = int(np.argmin([abs(avg_pos - p) for p in POSITION_LEVELS]))

        elif self.strategy == "veto_short":
            # Only go short if BOTH models agree
            if dist_pos == -1.0 and cql_pos == -1.0:
                final_action = 0  # short
            elif dist_pos >= 0 and cql_pos >= 0:
                # Both non-short: take the more aggressive (higher) position
                final_action = max(dist_action, cql_action)
            else:
                # One wants short, other doesn't: go flat or long
                # Use the non-short model's action
                if dist_pos >= 0:
                    final_action = dist_action
                else:
                    final_action = cql_action

        elif self.strategy == "cql_unless_short":
            # Use CQL (the active trader) but override shorts with DistCQL
            if cql_pos == -1.0 and dist_pos >= 0:
                final_action = dist_action  # DistCQL overrides the short
            else:
                final_action = cql_action

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
    p.add_argument("--strategy", nargs="+",
                   default=["majority_vote", "average_position", "veto_short", "cql_unless_short"])
    p.add_argument("--split", default="both", choices=["val", "test", "both"])
    args = p.parse_args()

    # Load models
    LOG.info("Loading DistCQL from %s", args.dist_cql_path)
    dist_cql = DistributionalCQLAgent.load(args.dist_cql_path)

    LOG.info("Loading CQL from %s", args.cql_path)
    cql = load_cql(args.cql_path)

    # Determine splits
    splits = {}
    if args.split in ("val", "both"):
        splits["val"] = "data/processed/btc_daily_val.parquet"
    if args.split in ("test", "both"):
        splits["test"] = "data/processed/btc_daily_test.parquet"

    log_return_column = "log_return_next_1d"
    periods_per_year = 252

    # Run all strategies
    print("\n" + "=" * 90)
    print("  ENSEMBLE EVALUATION")
    print("=" * 90)

    for strategy in args.strategy:
        print(f"\n--- Strategy: {strategy} ---")
        ensemble = EnsemblePolicy(dist_cql, cql, strategy=strategy)

        for split_name, path in splits.items():
            m, acts = run_rollout(ensemble, path, split_name, log_return_column, periods_per_year)

            # Also get B&H for comparison
            bh = run_buy_and_hold_on_split(
                path, split_name=split_name,
                log_return_column=log_return_column,
                periods_per_year=periods_per_year,
            )

            print(f"  {split_name:>5s}: return={m['total_return']:+.1%}  "
                  f"sharpe={m['sharpe']:.2f}  maxDD={m['max_drawdown']:.1%}  "
                  f"actions=[S:{m['action_frac_short']:.0%} F:{m['action_frac_flat']:.0%} L:{m['action_frac_long']:.0%}]  "
                  f"(B&H: {bh.metrics['total_return']:+.1%})")

    # Also show individual models for comparison
    print(f"\n--- Individual Models (reference) ---")
    for model_name, model, d3rlpy_actions in [
        ("DistCQL", dist_cql, False),
        ("CQL", cql, True),
    ]:
        for split_name, path in splits.items():
            from src.experiments.eval_policies import rollout_policy
            result = rollout_policy(
                model, data_path=path, split_name=split_name,
                d3rlpy_actions=d3rlpy_actions,
                log_return_column=log_return_column,
                periods_per_year=periods_per_year,
                position_levels=POSITION_LEVELS,
            )
            m = result.metrics
            print(f"  {model_name:>10s} {split_name:>5s}: return={m['total_return']:+.1%}  "
                  f"sharpe={m['sharpe']:.2f}  maxDD={m['max_drawdown']:.1%}  "
                  f"actions=[S:{m.get('action_frac_short',0):.0%} F:{m.get('action_frac_flat',0):.0%} L:{m.get('action_frac_long',0):.0%}]")

    print()


if __name__ == "__main__":
    main()

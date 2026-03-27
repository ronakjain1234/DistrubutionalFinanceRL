"""
Common evaluation harness for all trained agents (Step 6+).

Given a policy (any callable  obs → action), this module:

1. Rolls the policy through ``OfflineTradingEnv`` on a chosen split.
2. Collects the equity curve, log-returns, and actions.
3. Computes portfolio metrics (via ``baselines.equity_metrics``).
4. Optionally compares against buy-and-hold on the same split.

Design: policy-agnostic.  Works with d3rlpy models, custom PyTorch nets,
or any object that implements ``predict(obs_array) → action_array``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from src.env.offline_trading_env import OfflineTradingEnv, EnvConfig
from src.env.portfolio import PortfolioConfig
from src.experiments.baselines import (
    equity_metrics,
    run_buy_and_hold_on_split,
    BuyAndHoldReport,
)

LOG = logging.getLogger(__name__)

# ── Action mapping ────────────────────────────────────────────────────────
# Environment uses {-1, 0, +1};  d3rlpy predicts {0, 1, 2}.
D3RLPY_TO_ENV_OFFSET = -1


# ── Policy protocol ──────────────────────────────────────────────────────

class PolicyProtocol(Protocol):
    """Minimal interface a policy must satisfy for evaluation."""

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return action(s) given observation(s).  May be batched."""
        ...


# ── Rollout result ────────────────────────────────────────────────────────

@dataclass
class RolloutResult:
    """Full record of one policy rollout on a data split."""

    split_name: str
    source_path: Path
    n_steps: int

    # Time series (lengths: equity = n_steps+1, rest = n_steps)
    equity: np.ndarray
    step_log_returns: np.ndarray
    actions: np.ndarray          # env actions in {-1, 0, +1}
    positions: np.ndarray        # same as actions (no slippage model changes them)
    turnovers: np.ndarray
    drawdowns: np.ndarray

    metrics: dict[str, float]


# ── Core rollout ──────────────────────────────────────────────────────────

def rollout_policy(
    policy: PolicyProtocol,
    *,
    data_path: str | Path,
    split_name: str = "split",
    portfolio_cfg: PortfolioConfig | None = None,
    d3rlpy_actions: bool = True,
) -> RolloutResult:
    """
    Roll out *policy* through the offline trading environment on one split.

    Parameters
    ----------
    policy
        Anything with a ``predict(obs_array) → action_array`` method.
        For d3rlpy models this is the native interface.
    data_path
        Path to the processed parquet for the split.
    split_name
        Human label (e.g. "val", "test").
    portfolio_cfg
        Portfolio simulation config (fees, slippage, etc.).
    d3rlpy_actions
        If True, assumes ``policy.predict`` returns actions in {0,1,2}
        and remaps to {-1,0,+1} for the environment.  Set False if
        the policy already returns env-native actions.
    """
    env_cfg = EnvConfig(
        data_path=Path(data_path),
        portfolio_cfg=portfolio_cfg or PortfolioConfig(),
    )
    env = OfflineTradingEnv(env_cfg)

    obs, info = env.reset()

    equities = [info["equity"]]
    log_rets: list[float] = []
    actions_list: list[int] = []
    turnovers: list[float] = []
    drawdowns: list[float] = [info["drawdown"]]

    while True:
        # d3rlpy expects (batch, obs_dim) — reshape single obs
        obs_batch = obs.reshape(1, -1)
        raw_action = int(policy.predict(obs_batch)[0])

        if d3rlpy_actions:
            env_action = raw_action + D3RLPY_TO_ENV_OFFSET
        else:
            env_action = raw_action

        obs, reward, terminated, truncated, info = env.step(env_action)

        equities.append(info["equity"])
        log_rets.append(reward)
        actions_list.append(env_action)
        turnovers.append(info["turnover"])
        drawdowns.append(info["drawdown"])

        if terminated or truncated:
            break

    equity_arr = np.array(equities, dtype=np.float64)
    log_ret_arr = np.array(log_rets, dtype=np.float32)
    n_steps = len(log_rets)

    m = equity_metrics(equity_arr, log_ret_arr)

    # Action distribution for logging
    act_arr = np.array(actions_list, dtype=np.int64)
    for label, val in [("short", -1), ("flat", 0), ("long", 1)]:
        m[f"action_frac_{label}"] = float(np.mean(act_arr == val))
    m["total_turnover"] = float(np.sum(turnovers))
    m["mean_turnover"] = float(np.mean(turnovers))

    return RolloutResult(
        split_name=split_name,
        source_path=Path(data_path).resolve(),
        n_steps=n_steps,
        equity=equity_arr,
        step_log_returns=log_ret_arr,
        actions=act_arr,
        positions=act_arr.copy(),
        turnovers=np.array(turnovers, dtype=np.float64),
        drawdowns=np.array(drawdowns[1:], dtype=np.float64),  # align with steps
        metrics=m,
    )


# ── Comparison table ──────────────────────────────────────────────────────

@dataclass
class ComparisonRow:
    """One row in the comparison table: policy name + split metrics."""

    policy_name: str
    split_name: str
    metrics: dict[str, float]


def compare_to_buy_and_hold(
    rollout: RolloutResult,
    policy_name: str = "DQN",
    portfolio_cfg: PortfolioConfig | None = None,
) -> list[ComparisonRow]:
    """
    Return a two-row comparison: [policy, buy-and-hold] on the same split.
    """
    bh = run_buy_and_hold_on_split(
        rollout.source_path,
        split_name=rollout.split_name,
        portfolio_cfg=portfolio_cfg,
    )
    return [
        ComparisonRow(policy_name, rollout.split_name, rollout.metrics),
        ComparisonRow("BuyAndHold", rollout.split_name, bh.metrics),
    ]


def print_comparison(rows: list[ComparisonRow]) -> None:
    """Pretty-print a comparison table to stdout."""
    # Collect all metric keys
    all_keys = list(rows[0].metrics.keys()) if rows else []
    # Core metrics to show first
    core = [
        "n_steps", "total_return", "annualized_return",
        "annualized_vol", "sharpe", "max_drawdown",
    ]
    extra = [k for k in all_keys if k not in core]

    print()
    hdr = f"  {'Policy':<16s} {'Split':<6s}"
    for k in core:
        hdr += f" {k:>16s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for row in rows:
        line = f"  {row.policy_name:<16s} {row.split_name:<6s}"
        for k in core:
            v = row.metrics.get(k, float("nan"))
            if k == "n_steps":
                line += f" {int(v):>16d}"
            else:
                line += f" {v:>16.4f}"
        print(line)

    # Extra metrics (agent only — buy-and-hold won't have them)
    if extra:
        print()
        for row in rows:
            has_extra = any(k in row.metrics for k in extra)
            if has_extra:
                parts = [f"{k}={row.metrics[k]:.4f}" for k in extra if k in row.metrics]
                print(f"  {row.policy_name} extras: {', '.join(parts)}")
    print()


# ── Multi-split evaluation ────────────────────────────────────────────────

def evaluate_on_splits(
    policy: PolicyProtocol,
    policy_name: str = "Agent",
    *,
    splits: dict[str, str | Path] | None = None,
    portfolio_cfg: PortfolioConfig | None = None,
    d3rlpy_actions: bool = True,
    verbose: bool = True,
) -> dict[str, RolloutResult]:
    """
    Evaluate *policy* on multiple data splits and optionally print comparisons.

    Parameters
    ----------
    splits
        Mapping ``{split_name: parquet_path}``.  Defaults to val + test.
    """
    if splits is None:
        splits = {
            "val": "data/processed/btc_daily_val.parquet",
            "test": "data/processed/btc_daily_test.parquet",
        }

    results: dict[str, RolloutResult] = {}
    all_rows: list[ComparisonRow] = []

    for split_name, path in splits.items():
        if not Path(path).is_file():
            LOG.warning("Split %s not found at %s — skipping.", split_name, path)
            continue

        result = rollout_policy(
            policy,
            data_path=path,
            split_name=split_name,
            portfolio_cfg=portfolio_cfg,
            d3rlpy_actions=d3rlpy_actions,
        )
        results[split_name] = result
        all_rows.extend(
            compare_to_buy_and_hold(result, policy_name, portfolio_cfg)
        )

    if verbose and all_rows:
        print("\n" + "=" * 72)
        print(f"  EVALUATION: {policy_name}")
        print("=" * 72)
        print_comparison(all_rows)

    return results

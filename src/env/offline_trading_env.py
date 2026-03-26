"""
Offline trading environment (Gym-style) for BTC.

Step 4 in the project roadmap.  Wraps a preprocessed BTC dataset into a
gymnasium-compatible environment that an RL agent can interact with.

The environment is **deterministic and sequential**: it walks through the
dataset one row at a time.  There is no random sampling or on-the-fly data
fetching --- the agent simply replays the historical tape.

Timing
------
At each step *t* the agent:

1. Observes normalised market features (and optionally its current position).
2. Chooses an action  →  new signed exposure  w_t  in {-1, 0, +1}.
3. Pays turnover cost  (fee + slippage) * |w_t - w_{t-1}|  on the current NAV.
4. Receives one-period return  1 + w_t * (P_{t+1}/P_t - 1)  on the post-cost NAV.
5. The reward is  log(NAV_{t+1} / NAV_t) , matching ``portfolio.simulate_portfolio``.

The episode terminates after every row has been acted upon.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

from .portfolio import (
    ACTION_FLAT,
    ACTION_LONG,
    ACTION_SHORT,
    PortfolioConfig,
)

# ---------------------------------------------------------------------------
# Action mapping: Discrete(3) integers  →  signed positions
# ---------------------------------------------------------------------------
_ACTION_TO_POSITION: dict[int, int] = {
    0: ACTION_SHORT,  # -1
    1: ACTION_FLAT,   #  0
    2: ACTION_LONG,   # +1
}
N_ACTIONS = len(_ACTION_TO_POSITION)

# ---------------------------------------------------------------------------
# Columns that are metadata / targets, not features for the agent
# ---------------------------------------------------------------------------
_NON_FEATURE_COLS = frozenset({
    "timestamp", "open", "high", "low", "close",
    "next_close", "volume", "log_return_next_1d",
})


def _infer_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return sorted list of columns that are actual market features."""
    return sorted(c for c in df.columns if c not in _NON_FEATURE_COLS)


# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnvConfig:
    """All knobs for :class:`OfflineTradingEnv` in one place."""

    data_path: Path = Path("data/processed/btc_daily_train.parquet")
    """Path to the processed (normalised) parquet for one split."""

    portfolio_cfg: PortfolioConfig = field(default_factory=PortfolioConfig)
    """Portfolio simulation parameters (costs, initial equity, etc.)."""

    include_position_in_obs: bool = True
    """If True, append the agent's current signed position to the feature vector."""

    feature_columns: list[str] | None = None
    """
    Explicit list of feature column names.  If ``None``, all columns that
    are not in ``_NON_FEATURE_COLS`` are used (auto-detected).
    """

    log_return_column: str = "log_return_next_1d"
    """Column holding the forward 1-day log-return (target / price info)."""

    timestamp_column: str = "timestamp"
    """Column with row timestamps (used for info dicts, not observations)."""


# ---------------------------------------------------------------------------
# The environment
# ---------------------------------------------------------------------------

class OfflineTradingEnv(gym.Env):
    """
    Deterministic, sequential offline trading environment for BTC.

    Parameters
    ----------
    cfg : EnvConfig, optional
        Full configuration object.  You can also pass individual keyword
        arguments (they are forwarded to ``EnvConfig``).

    Examples
    --------
    >>> from src.env.offline_trading_env import OfflineTradingEnv, EnvConfig
    >>> env = OfflineTradingEnv(EnvConfig(data_path="data/processed/btc_daily_train.parquet"))
    >>> obs, info = env.reset()
    >>> obs.shape
    (15,)
    >>> for _ in range(5):
    ...     action = env.action_space.sample()
    ...     obs, reward, terminated, truncated, info = env.step(action)
    ...     if terminated:
    ...         break
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self, cfg: EnvConfig | None = None, **kwargs: Any) -> None:
        super().__init__()

        if cfg is None:
            cfg = EnvConfig(**kwargs)
        self._cfg = cfg
        self._pcfg = cfg.portfolio_cfg

        # ── Load and validate dataset ────────────────────────────────
        df = pd.read_parquet(cfg.data_path)

        if cfg.log_return_column not in df.columns:
            raise ValueError(
                f"Column '{cfg.log_return_column}' not found in {cfg.data_path}. "
                f"Available: {list(df.columns)}"
            )

        # ── Determine feature columns ────────────────────────────────
        if cfg.feature_columns is not None:
            self._feature_cols: list[str] = list(cfg.feature_columns)
        else:
            self._feature_cols = _infer_feature_columns(df)

        missing = set(self._feature_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Feature columns not in data: {sorted(missing)}")

        # ── Pre-extract numpy arrays (fast row access during rollout) ─
        self._features: np.ndarray = df[self._feature_cols].to_numpy(dtype=np.float32)
        self._log_returns: np.ndarray = df[cfg.log_return_column].to_numpy(dtype=np.float64)
        self._timestamps: np.ndarray | None = (
            df[cfg.timestamp_column].values
            if cfg.timestamp_column in df.columns
            else None
        )
        self._n_rows: int = len(df)

        if self._n_rows == 0:
            raise ValueError(f"Dataset at {cfg.data_path} is empty.")

        # ── Define observation and action spaces ─────────────────────
        n_feat = len(self._feature_cols)
        obs_dim = n_feat + 1 if cfg.include_position_in_obs else n_feat

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(N_ACTIONS)

        # ── Episode state (set properly in reset()) ──────────────────
        self._t: int = 0
        self._position: float = 0.0
        self._equity: float = 1.0
        self._peak_equity: float = 1.0
        self._done: bool = True  # force user to call reset() first

    # ------------------------------------------------------------------ #
    #  Read-only properties                                               #
    # ------------------------------------------------------------------ #

    @property
    def feature_columns(self) -> list[str]:
        """Names of the market-feature columns used as observations."""
        return list(self._feature_cols)

    @property
    def n_rows(self) -> int:
        """Total rows in the underlying dataset."""
        return self._n_rows

    @property
    def current_step(self) -> int:
        """Index of the row the agent will act on next (0-based)."""
        return self._t

    # ------------------------------------------------------------------ #
    #  gymnasium API                                                      #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment to the beginning of the dataset.

        Returns the first observation and an info dict.
        """
        super().reset(seed=seed)

        self._t = 0
        self._position = self._pcfg.initial_position
        self._equity = self._pcfg.initial_equity
        self._peak_equity = self._equity
        self._done = False

        obs = self._build_obs(self._t)
        info = self._build_info(step_reward=0.0, turnover=0.0, step_cost=0.0)
        return obs, info

    def step(
        self, action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Take one trading action and advance the clock by one period.

        Parameters
        ----------
        action : int
            0 = short, 1 = flat, 2 = long.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        if self._done:
            raise RuntimeError(
                "Episode is over.  Call reset() before stepping again."
            )

        # ── Map discrete action to signed position ───────────────────
        new_position = float(_ACTION_TO_POSITION[int(action)])

        # ── Transaction cost (same accounting as simulate_portfolio) ──
        turnover = abs(new_position - self._position)
        cost_frac = (self._pcfg.fee_rate + self._pcfg.slippage_rate) * turnover

        # ── One-period portfolio return ──────────────────────────────
        price_rel = np.exp(self._log_returns[self._t])  # P_{t+1} / P_t
        r_asset = price_rel - 1.0                        # simple return of BTC

        equity_before = self._equity
        v_after_cost = equity_before * (1.0 - cost_frac)
        v_next = v_after_cost * (1.0 + new_position * r_asset)

        # Reward = log(NAV_{t+1} / NAV_t), matching simulate_portfolio
        if equity_before <= 0.0 or v_next <= 0.0:
            reward = float("nan")
        else:
            reward = float(np.log(v_next / equity_before))

        step_cost = equity_before * cost_frac

        # ── Update state ─────────────────────────────────────────────
        self._equity = v_next
        self._position = new_position
        self._peak_equity = max(self._peak_equity, self._equity)
        self._t += 1

        # ── Termination ──────────────────────────────────────────────
        terminated = self._t >= self._n_rows
        truncated = False
        self._done = terminated

        # ── Next observation ─────────────────────────────────────────
        if terminated:
            # Terminal obs — zeros (won't be used for learning)
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._build_obs(self._t)

        info = self._build_info(
            step_reward=reward, turnover=turnover, step_cost=step_cost,
        )
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _build_obs(self, t: int) -> np.ndarray:
        """Assemble the observation vector for row *t*."""
        feat = self._features[t]  # (n_features,), float32
        if self._cfg.include_position_in_obs:
            # Append current position as an extra feature
            return np.append(feat, np.float32(self._position))
        return feat.copy()

    def _build_info(
        self,
        *,
        step_reward: float,
        turnover: float,
        step_cost: float,
    ) -> dict[str, Any]:
        """Build the info dict returned by reset() and step()."""
        drawdown = (
            (self._equity - self._peak_equity) / self._peak_equity
            if self._peak_equity > 0 else 0.0
        )
        info: dict[str, Any] = {
            "equity": self._equity,
            "position": self._position,
            "drawdown": drawdown,
            "turnover": turnover,
            "step_cost": step_cost,
            "step_reward": step_reward,
            "step": self._t,
        }
        if self._timestamps is not None and self._t < self._n_rows:
            info["timestamp"] = self._timestamps[self._t]
        return info

    # ------------------------------------------------------------------ #
    #  Convenience                                                        #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"OfflineTradingEnv(data={self._cfg.data_path.name!r}, "
            f"rows={self._n_rows}, obs_dim={self.observation_space.shape[0]})"
        )


# ---------------------------------------------------------------------------
# Quick smoke-test when run as a script
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    """
    Run a random-policy rollout through the training split and print
    summary statistics.  Useful for verifying the env works end-to-end.

    Usage:  python -m src.env.offline_trading_env
    """
    import sys

    train_path = Path("data/processed/btc_daily_train.parquet")
    if not train_path.exists():
        print(f"[smoke-test] {train_path} not found — run the data pipeline first.")
        sys.exit(1)

    cfg = EnvConfig(data_path=train_path)
    env = OfflineTradingEnv(cfg)
    print(f"Created: {env}")
    print(f"  Observation space : {env.observation_space}")
    print(f"  Action space      : {env.action_space}")
    print(f"  Feature columns   : {env.feature_columns}")
    print()

    # Roll through the entire episode with random actions
    obs, info = env.reset()
    total_reward = 0.0
    n_steps = 0

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        n_steps += 1
        if terminated or truncated:
            break

    print(f"  Episode length    : {n_steps} steps")
    print(f"  Cumulative reward : {total_reward:+.6f}")
    print(f"  Final equity      : {info['equity']:.6f}")
    print(f"  Max drawdown      : {info['drawdown']:.4%}")
    print()

    # Also do a buy-and-hold rollout to cross-check with baselines.py
    obs, _ = env.reset()
    bnh_reward = 0.0
    while True:
        obs, reward, terminated, _, info = env.step(2)  # always long
        bnh_reward += reward
        if terminated:
            break

    print(f"  Buy-and-hold (env): cumulative log-return = {bnh_reward:+.6f}")
    print(f"  Buy-and-hold (env): final equity          = {info['equity']:.6f}")
    print()
    print("[smoke-test] Done.")


if __name__ == "__main__":
    _smoke_test()

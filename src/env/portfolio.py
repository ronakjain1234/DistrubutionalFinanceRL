"""
Portfolio mechanics for BTC offline RL: positions, PnL, costs, and rewards.

Action space (Step 3 roadmap)
-----------------------------
* **Signed exposure**: -1 (short), 0 (flat), +1 (long).

At each step *t*, the chosen position applies to the interval from *t* to *t+1* using the
price relative :math:`P_{t+1}/P_t`. Turnover-based fees are charged when the position changes
before that return is applied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Action constants (signed exposure)
# ---------------------------------------------------------------------------

ACTION_SHORT = -1
ACTION_FLAT = 0
ACTION_LONG = 1


# ---------------------------------------------------------------------------
# Config & results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortfolioConfig:
    """Simulation parameters for one run."""

    initial_equity: float = 1.0
    """Starting portfolio value (NAV)."""

    initial_position: float = 0.0
    """Exposure *before* the first step, in {-1, 0, +1}."""

    fee_rate: float = 0.0005
    """
    Proportional cost per unit of position change, applied to NAV at rebalance.
    Example: 0.0005 ≈ 5 bps of NAV when |Δw| = 1 (e.g. flat -> long).
    """

    slippage_rate: float = 0.0
    """Additional turnover cost (same units as ``fee_rate``), optional."""


@dataclass
class PortfolioResult:
    """Outputs of :func:`simulate_portfolio`."""

    equity: np.ndarray
    """NAV at the start of step 0, then after each step; shape ``(n_steps + 1,)``."""

    step_log_returns: np.ndarray
    """Natural log of growth factor each step (after costs); shape ``(n_steps,)``."""

    turnover: np.ndarray
    """|Δw| each step; shape ``(n_steps,)``."""

    positions: np.ndarray
    """Signed position applied each step; shape ``(n_steps,)``."""


def simulate_portfolio(
    price_relatives: np.ndarray,
    positions: np.ndarray,
    cfg: PortfolioConfig | None = None,
) -> PortfolioResult:
    """
    Simulate NAV through time from per-step price relatives and target positions.

    Parameters
    ----------
    price_relatives
        :math:`P_{t+1}/P_t` for each step *t*; shape ``(n_steps,)``.
    positions
        Signed exposure in {-1, 0, +1} for each step; same length as ``price_relatives``.

    Mechanics
    ---------
    1. Pay turnover cost on NAV: ``(fee_rate + slippage_rate) * |w_t - w_{t-1}|``.
    2. Apply one-period linear return: ``1 + w_t * (P_{t+1}/P_t - 1)``.

    The step reward is ``log(V_{t+1}/V_t)`` (after costs), unless ``reward_kind`` is extended
    later in :class:`PortfolioConfig`.
    """
    cfg = cfg if cfg is not None else PortfolioConfig()

    pr = np.asarray(price_relatives, dtype=np.float64).ravel()
    w = np.asarray(positions, dtype=np.float64).ravel()
    if pr.shape != w.shape:
        raise ValueError("price_relatives and positions must have the same shape.")
    if pr.size == 0:
        raise ValueError("Need at least one step to simulate.")

    n = pr.size
    equity = np.empty(n + 1, dtype=np.float64)
    step_log_returns = np.empty(n, dtype=np.float64)
    turnover = np.empty(n, dtype=np.float64)

    v = float(cfg.initial_equity)
    w_prev = float(cfg.initial_position)
    equity[0] = v

    cost_per_unit_turnover = cfg.fee_rate + cfg.slippage_rate

    for t in range(n):
        w_t = float(w[t])
        turnover[t] = abs(w_t - w_prev)
        cost_frac = cost_per_unit_turnover * turnover[t]
        v_after_cost = v * (1.0 - cost_frac)
        r_asset = pr[t] - 1.0
        v_next = v_after_cost * (1.0 + w_t * r_asset)
        if v <= 0.0 or v_next <= 0.0:
            step_log_returns[t] = np.nan
        else:
            step_log_returns[t] = np.log(v_next / v)
        v = v_next
        w_prev = w_t
        equity[t + 1] = v

    return PortfolioResult(
        equity=equity,
        step_log_returns=step_log_returns,
        turnover=turnover,
        positions=w.copy(),
    )


def price_relative_from_log_return(log_r: float | np.ndarray) -> float | np.ndarray:
    """Convert one-step log return ``log(P_{t+1}/P_t)`` to price relative ``P_{t+1}/P_t``."""
    return np.exp(np.asarray(log_r, dtype=np.float64))

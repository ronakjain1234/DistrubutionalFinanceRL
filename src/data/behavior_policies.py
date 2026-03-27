"""
Behavior policies for generating offline RL training data.

In offline RL the agent can only learn about state-action regions that the
behavior policy visited.  A single deterministic policy (e.g. buy-and-hold)
would produce a dataset with zero coverage of short or flat actions, making
it impossible for the learner to evaluate alternatives.

We therefore implement several policies with *distinct trading philosophies*
and combine them via a mixture, so the resulting dataset covers:

  * all three actions  {short, flat, long}
  * diverse market regimes  (trending, mean-reverting, volatile)
  * both good and bad decisions  (essential for CQL to learn what NOT to do)

Each policy operates on the z-scored observation vector produced by
:class:`OfflineTradingEnv`.  Because features are standardised to
training-period mean=0, std=1, a threshold of 0 corresponds to the
historical average signal strength — a natural decision boundary.

Policy catalogue
----------------
* **BuyAndHold** — passive beta, always long
* **Random** — uniform over {-1, 0, +1}, maximal action coverage
* **TrendFollowing** — momentum + moving-average confirmation
* **MeanReversion** — contrarian RSI / MA signals
* **MACDCrossover** — MACD histogram sign
* **VolatilityRegime** — risk-off in high-vol, trend-follow in low-vol
* **EpsilonGreedy** — wraps any policy with ε-random exploration
* **Mixture** — per-step random delegation to sub-policies
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


# ── helpers ────────────────────────────────────────────────────────────────

VALID_ACTIONS = (-1, 0, 1)


def _feature_index(feature_columns: list[str], name: str) -> int:
    """Look up *name* in the feature list; raise a clear error if missing."""
    try:
        return feature_columns.index(name)
    except ValueError:
        raise ValueError(
            f"Feature '{name}' not found. Available: {feature_columns}"
        ) from None


# ── abstract base ──────────────────────────────────────────────────────────


class BehaviorPolicy(ABC):
    """Interface every behavior policy must satisfy."""

    @abstractmethod
    def select_action(self, obs: np.ndarray) -> int:
        """Return an action in {-1, 0, +1} given the current observation."""
        ...

    def reset(self) -> None:
        """Reset internal state at the start of an episode (default: no-op)."""

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ── deterministic core policies ────────────────────────────────────────────


class BuyAndHoldPolicy(BehaviorPolicy):
    """Always long.  Captures the long-run beta premium of BTC."""

    def select_action(self, obs: np.ndarray) -> int:
        return 1


class TrendFollowingPolicy(BehaviorPolicy):
    """
    Go long when recent momentum *and* price-vs-MA are both positive;
    short when both are negative; flat when the signals disagree.

    Primary signals (z-scored):
      * ``log_ret_5``   — 5-day cumulative log-return (short-term momentum)
      * ``ma_ratio_20`` — close / SMA(20) - 1  (trend confirmation)

    A threshold of 0 means "above/below the training-period average".
    """

    def __init__(
        self,
        feature_columns: list[str],
        threshold: float = 0.0,
    ) -> None:
        self._mom_idx = _feature_index(feature_columns, "log_ret_5")
        self._ma_idx = _feature_index(feature_columns, "ma_ratio_20")
        self._thr = threshold

    def select_action(self, obs: np.ndarray) -> int:
        mom = obs[self._mom_idx]
        ma = obs[self._ma_idx]
        if mom > self._thr and ma > self._thr:
            return 1
        if mom < -self._thr and ma < -self._thr:
            return -1
        return 0


class MeanReversionPolicy(BehaviorPolicy):
    """
    Contrarian: buy oversold dips, sell overbought rallies.

    Uses the z-scored RSI-14 as an oscillator and the MA-ratio for
    confirmation.  After standardisation, RSI < −0.5 roughly maps to the
    oversold territory of the raw RSI scale, and > +0.5 to overbought.
    """

    def __init__(
        self,
        feature_columns: list[str],
        rsi_threshold: float = 0.5,
    ) -> None:
        self._rsi_idx = _feature_index(feature_columns, "rsi_14")
        self._ma_idx = _feature_index(feature_columns, "ma_ratio_20")
        self._thr = rsi_threshold

    def select_action(self, obs: np.ndarray) -> int:
        rsi = obs[self._rsi_idx]
        ma = obs[self._ma_idx]
        if rsi < -self._thr and ma < 0:
            return 1          # oversold → buy the dip
        if rsi > self._thr and ma > 0:
            return -1         # overbought → fade the rally
        return 0


class MACDCrossoverPolicy(BehaviorPolicy):
    """
    Classic momentum via the MACD histogram sign.

    Long when the histogram (MACD − signal line) is positive (bullish
    crossover regime), short when negative, flat near zero.
    """

    def __init__(
        self,
        feature_columns: list[str],
        hist_threshold: float = 0.0,
    ) -> None:
        self._hist_idx = _feature_index(feature_columns, "macd_hist")
        self._thr = hist_threshold

    def select_action(self, obs: np.ndarray) -> int:
        h = obs[self._hist_idx]
        if h > self._thr:
            return 1
        if h < -self._thr:
            return -1
        return 0


class SupplyDemandPolicy(BehaviorPolicy):
    """
    Trade based on proximity to supply and demand zones.

    Supply and demand zones encode **price memory** — levels where
    institutional buying or selling previously caused sharp reversals.
    When price revisits those levels, the same imbalance is expected to
    recur.

    After z-scoring:
      * ``sd_dist_demand`` < −threshold → price is *closer than usual* to
        a demand zone → go long (expect bounce).
      * ``sd_dist_supply`` < −threshold → price is *closer than usual* to
        a supply zone → go short (expect rejection).
      * Otherwise → flat.

    The ``sd_zone_signal`` feature is used as a tiebreaker when both
    zones are near: its sign already resolves the ambiguity.
    """

    def __init__(
        self,
        feature_columns: list[str],
        proximity_threshold: float = -0.3,
    ) -> None:
        self._demand_idx = _feature_index(feature_columns, "sd_dist_demand")
        self._supply_idx = _feature_index(feature_columns, "sd_dist_supply")
        self._signal_idx = _feature_index(feature_columns, "sd_zone_signal")
        self._thr = proximity_threshold

    def select_action(self, obs: np.ndarray) -> int:
        demand = obs[self._demand_idx]   # z-scored distance to demand
        supply = obs[self._supply_idx]   # z-scored distance to supply
        signal = obs[self._signal_idx]   # z-scored zone signal

        near_demand = demand < self._thr
        near_supply = supply < self._thr

        if near_demand and near_supply:
            # Both zones close — let the composite signal decide
            return 1 if signal > 0 else (-1 if signal < 0 else 0)
        if near_demand:
            return 1    # approaching demand → long
        if near_supply:
            return -1   # approaching supply → short
        return 0


class VolatilityRegimePolicy(BehaviorPolicy):
    """
    Risk-adaptive: reduce exposure during high-volatility regimes.

    When 20-day realised vol is above its training-period average (z > thr),
    the policy goes flat — mimicking a disciplined trader who steps aside
    during turbulence.  In calmer markets it follows the 20-day momentum.

    This is philosophically aligned with the project's focus on risk-sensitive
    decision-making: a good offline dataset should contain examples of
    *choosing not to trade* when uncertainty is high.
    """

    def __init__(
        self,
        feature_columns: list[str],
        vol_threshold: float = 0.5,
    ) -> None:
        self._vol_idx = _feature_index(feature_columns, "vol_20")
        self._mom_idx = _feature_index(feature_columns, "log_ret_20")
        self._thr = vol_threshold

    def select_action(self, obs: np.ndarray) -> int:
        vol = obs[self._vol_idx]
        if vol > self._thr:
            return 0              # high vol → risk-off
        mom = obs[self._mom_idx]
        if mom > 0:
            return 1
        if mom < 0:
            return -1
        return 0


# ── stochastic policies ───────────────────────────────────────────────────


class RandomPolicy(BehaviorPolicy):
    """Uniform random over {-1, 0, +1}.  Maximises action-space coverage."""

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self._rng = rng or np.random.default_rng()

    def select_action(self, obs: np.ndarray) -> int:
        return int(self._rng.choice(VALID_ACTIONS))


class EpsilonGreedyPolicy(BehaviorPolicy):
    """
    Wraps a deterministic policy: with probability ε choose a uniform
    random action, otherwise delegate to the base policy.

    This is the standard way to inject exploration into an otherwise
    deterministic strategy, ensuring that the offline dataset contains
    *some* coverage of every action in every state the base policy visits.
    """

    def __init__(
        self,
        base: BehaviorPolicy,
        epsilon: float = 0.15,
        rng: np.random.Generator | None = None,
    ) -> None:
        self._base = base
        self._eps = epsilon
        self._rng = rng or np.random.default_rng()

    @property
    def name(self) -> str:
        return f"EpsGreedy({self._base.name}, eps={self._eps})"

    def reset(self) -> None:
        self._base.reset()

    def select_action(self, obs: np.ndarray) -> int:
        if self._rng.random() < self._eps:
            return int(self._rng.choice(VALID_ACTIONS))
        return self._base.select_action(obs)


class MixturePolicy(BehaviorPolicy):
    """
    Per-step random delegation: at each step, sample one sub-policy
    according to the given weights and return its action.

    This is the workhorse for building diverse offline datasets.  By
    combining policies with *opposed* philosophies (trend-following vs
    mean-reversion, always-on vs volatility-gated), the mixture produces
    a dataset where the learner can contrast outcomes of conflicting
    decisions in similar market states — exactly the signal CQL needs to
    form conservative but accurate Q-estimates.
    """

    def __init__(
        self,
        policies: list[BehaviorPolicy],
        weights: list[float] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        if not policies:
            raise ValueError("MixturePolicy requires at least one sub-policy.")
        self._policies = policies
        w = np.asarray(weights if weights else [1.0] * len(policies), dtype=np.float64)
        self._weights = w / w.sum()
        self._rng = rng or np.random.default_rng()

    @property
    def name(self) -> str:
        inner = ", ".join(p.name for p in self._policies)
        return f"Mixture[{inner}]"

    def reset(self) -> None:
        for p in self._policies:
            p.reset()

    def select_action(self, obs: np.ndarray) -> int:
        idx = self._rng.choice(len(self._policies), p=self._weights)
        return self._policies[idx].select_action(obs)

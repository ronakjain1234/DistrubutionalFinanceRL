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
* **Random** — uniform over position levels, maximal action coverage
* **TrendFollowing** — momentum + moving-average confirmation
* **MeanReversion** — contrarian RSI / MA signals
* **MACDCrossover** — MACD histogram sign
* **VolatilityRegime** — risk-off in high-vol, trend-follow in low-vol
* **BollingerBreakout** — momentum breakout via Bollinger %B
* **CandlePattern** — intrabar conviction candles
* **ParkinsonVolRegime** — Parkinson-vol-based risk-off
* **AutocorrRegime** — adaptive trend/mean-reversion
* **VolSizedTrend** — trend-following with vol-scaled fractional sizing
* **BollingerSized** — continuous Bollinger %B-based sizing
* **GradualPosition** — ramps into positions over consecutive bars
* **EpsilonGreedy** — wraps any policy with ε-random exploration
* **Mixture** — per-step random delegation to sub-policies

All policies return **float positions** in [-1, +1].  The data collection
layer discretises these to the nearest valid action index.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


# ── helpers ────────────────────────────────────────────────────────────────

# Legacy 3-action positions (backward compat)
VALID_POSITIONS_3 = (-1.0, 0.0, 1.0)
# 7-action fractional positions
VALID_POSITIONS_7 = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0)


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
    def select_action(self, obs: np.ndarray) -> float:
        """Return a desired position in [-1, +1].

        The data-collection layer snaps this to the nearest valid
        action index for the configured position levels.
        """
        ...

    def reset(self) -> None:
        """Reset internal state at the start of an episode (default: no-op)."""

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ── deterministic core policies ────────────────────────────────────────────


class BuyAndHoldPolicy(BehaviorPolicy):
    """Always long.  Captures the long-run beta premium of BTC."""

    def select_action(self, obs: np.ndarray) -> float:
        return 1.0


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
        *,
        momentum_feature: str = "log_ret_5",
        ma_feature: str = "ma_ratio_20",
    ) -> None:
        self._mom_idx = _feature_index(feature_columns, momentum_feature)
        self._ma_idx = _feature_index(feature_columns, ma_feature)
        self._thr = threshold

    def select_action(self, obs: np.ndarray) -> float:
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
        *,
        rsi_feature: str = "rsi_14",
        ma_feature: str = "ma_ratio_20",
    ) -> None:
        self._rsi_idx = _feature_index(feature_columns, rsi_feature)
        self._ma_idx = _feature_index(feature_columns, ma_feature)
        self._thr = rsi_threshold

    def select_action(self, obs: np.ndarray) -> float:
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

    def select_action(self, obs: np.ndarray) -> float:
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

    def select_action(self, obs: np.ndarray) -> float:
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
        *,
        vol_feature: str = "vol_20",
        momentum_feature: str = "log_ret_20",
    ) -> None:
        self._vol_idx = _feature_index(feature_columns, vol_feature)
        self._mom_idx = _feature_index(feature_columns, momentum_feature)
        self._thr = vol_threshold

    def select_action(self, obs: np.ndarray) -> float:
        vol = obs[self._vol_idx]
        if vol > self._thr:
            return 0              # high vol → risk-off
        mom = obs[self._mom_idx]
        if mom > 0:
            return 1
        if mom < 0:
            return -1
        return 0


# ── OHLC-aware policies (hourly) ─────────────────────────────────────────


class BollingerBreakoutPolicy(BehaviorPolicy):
    """
    Momentum breakout via Bollinger %B.

    When price punches above the upper Bollinger band (%B > 1), it signals
    a volatility breakout — institutional buying overwhelmed the recent
    range.  Go long to ride the continuation.  When price drops below the
    lower band (%B < 0), the reverse.

    A thin neutral zone around the bands avoids whipsaws in ranging markets.
    After z-scoring, %B typically fluctuates around 0.5 (mid-band); values
    beyond ±1 std are the signals.
    """

    def __init__(
        self,
        feature_columns: list[str],
        upper_threshold: float = 0.5,
        lower_threshold: float = -0.5,
    ) -> None:
        self._bb_idx = _feature_index(feature_columns, "bollinger_pctb")
        self._upper = upper_threshold
        self._lower = lower_threshold

    def select_action(self, obs: np.ndarray) -> float:
        bb = obs[self._bb_idx]
        if bb > self._upper:
            return 1   # breakout above band -> long
        if bb < self._lower:
            return -1  # breakdown below band -> short
        return 0


class CandlePatternPolicy(BehaviorPolicy):
    """
    Intrabar price action: trade conviction candles, avoid indecision.

    A strong bullish candle has high close_in_range (buyers pushed close
    to the top of the bar) AND high bar_body_ratio (most of the range
    was directional, not wicks).  This combination — a full-bodied bar
    closing near its high — is the classic institutional footprint.

    Conversely, a low bar_body_ratio (doji/spinning top) indicates
    indecision regardless of where the close falls.

    After z-scoring, thresholds of 0 correspond to "above/below the
    training-period average candle shape".
    """

    def __init__(
        self,
        feature_columns: list[str],
        close_threshold: float = 0.3,
        body_threshold: float = 0.0,
    ) -> None:
        self._cir_idx = _feature_index(feature_columns, "close_in_range")
        self._bbr_idx = _feature_index(feature_columns, "bar_body_ratio")
        self._close_thr = close_threshold
        self._body_thr = body_threshold

    def select_action(self, obs: np.ndarray) -> float:
        cir = obs[self._cir_idx]   # z-scored close-in-range
        bbr = obs[self._bbr_idx]   # z-scored bar body ratio

        if bbr < self._body_thr:
            return 0               # indecision candle -> flat

        if cir > self._close_thr:
            return 1               # bullish conviction -> long
        if cir < -self._close_thr:
            return -1              # bearish conviction -> short
        return 0


class ParkinsonVolRegimePolicy(BehaviorPolicy):
    """
    Volatility regime detection using Parkinson (high-low) volatility.

    Parkinson vol is ~5x more statistically efficient than close-to-close
    vol because it uses the full intrabar range.  This means it detects
    regime changes faster.

    When the short-term (24h) Parkinson vol spikes above its average
    (z > threshold), the market is in a turbulent regime — go flat.
    When vol is compressing (short-term << average), conditions favor
    breakouts — follow the 4h momentum direction.

    This is philosophically the same as VolatilityRegimePolicy but uses
    a superior volatility estimator and shorter reaction time.
    """

    def __init__(
        self,
        feature_columns: list[str],
        vol_threshold: float = 0.5,
        *,
        vol_feature: str = "parkinson_vol_24",
        momentum_feature: str = "log_ret_4",
    ) -> None:
        self._vol_idx = _feature_index(feature_columns, vol_feature)
        self._mom_idx = _feature_index(feature_columns, momentum_feature)
        self._thr = vol_threshold

    def select_action(self, obs: np.ndarray) -> float:
        vol = obs[self._vol_idx]
        if vol > self._thr:
            return 0              # high vol regime -> risk-off
        mom = obs[self._mom_idx]
        if mom > 0:
            return 1
        if mom < 0:
            return -1
        return 0


class AutocorrRegimePolicy(BehaviorPolicy):
    """
    Adaptive trend/mean-reversion based on return autocorrelation.

    Markets oscillate between trending (positive autocorrelation) and
    mean-reverting (negative autocorrelation) regimes.  This policy
    estimates the current regime from rolling return autocorrelation
    and adapts its strategy accordingly:

    * Positive autocorrelation at lag 4: recent returns predict future
      returns in the same direction -> trend-follow (go with log_ret_4).
    * Negative autocorrelation: recent returns predict reversal ->
      mean-revert (go against log_ret_4).
    * Near-zero autocorrelation: no predictability -> flat.

    This is the only policy in the suite that explicitly adapts its
    *philosophy* based on market microstructure rather than just price
    level or momentum.
    """

    def __init__(
        self,
        feature_columns: list[str],
        autocorr_threshold: float = 0.3,
        *,
        autocorr_feature: str = "ret_autocorr_4",
        momentum_feature: str = "log_ret_4",
    ) -> None:
        self._ac_idx = _feature_index(feature_columns, autocorr_feature)
        self._mom_idx = _feature_index(feature_columns, momentum_feature)
        self._thr = autocorr_threshold

    def select_action(self, obs: np.ndarray) -> float:
        ac = obs[self._ac_idx]
        mom = obs[self._mom_idx]

        if ac > self._thr:
            # Trending regime -> follow momentum
            if mom > 0:
                return 1
            if mom < 0:
                return -1
            return 0
        elif ac < -self._thr:
            # Mean-reverting regime -> fade momentum
            if mom > 0:
                return -1
            if mom < 0:
                return 1
            return 0
        else:
            return 0  # no clear regime -> flat


# ── fractional-position policies (require 7-action space) ────────────────


class VolSizedTrendPolicy(BehaviorPolicy):
    """
    Trend-following with volatility-scaled position sizing.

    The direction comes from short-term momentum (same as TrendFollowing),
    but the SIZE is inversely proportional to Parkinson volatility.
    When vol is high (z > 1), the position shrinks to ±0.25.
    When vol is low (z < -0.5), the position expands to ±1.0.
    When vol is moderate, ±0.5.

    This is how institutional systematic traders actually operate:
    signal determines direction, vol determines size.
    """

    def __init__(
        self,
        feature_columns: list[str],
        *,
        momentum_feature: str = "log_ret_4",
        vol_feature: str = "parkinson_vol_24",
    ) -> None:
        self._mom_idx = _feature_index(feature_columns, momentum_feature)
        self._vol_idx = _feature_index(feature_columns, vol_feature)

    def select_action(self, obs: np.ndarray) -> float:
        mom = obs[self._mom_idx]
        vol = obs[self._vol_idx]

        # Direction from momentum
        if abs(mom) < 0.1:
            return 0.0  # no clear signal

        direction = 1.0 if mom > 0 else -1.0

        # Size inversely from vol (z-scored)
        if vol > 1.0:
            size = 0.25       # high vol -> minimal position
        elif vol > 0.0:
            size = 0.5        # moderate vol -> half position
        else:
            size = 1.0        # low vol -> full position

        return direction * size


class BollingerSizedPolicy(BehaviorPolicy):
    """
    Position size proportional to Bollinger %B displacement.

    Instead of a binary breakout/breakdown signal, this policy uses
    the *distance* from the mid-band to size the position continuously:

    * %B = 0.5 (at mid-band) -> flat
    * %B = 1.0 (at upper band) -> +0.5
    * %B > 1.5 (well above band) -> +1.0
    * %B = 0.0 (at lower band) -> -0.5
    * %B < -0.5 (well below band) -> -1.0

    This naturally produces positions across the full fractional range,
    giving the offline dataset rich coverage of intermediate sizing.
    """

    def __init__(self, feature_columns: list[str]) -> None:
        self._bb_idx = _feature_index(feature_columns, "bollinger_pctb")

    def select_action(self, obs: np.ndarray) -> float:
        bb = obs[self._bb_idx]  # z-scored Bollinger %B

        # Map z-scored %B to position: roughly, bb=0 maps to neutral
        # (since z-scoring centers at the training mean)
        position = np.clip(bb * 0.5, -1.0, 1.0)

        # Snap to nearest quarter to keep positions clean
        return round(position * 4) / 4


class GradualPositionPolicy(BehaviorPolicy):
    """
    Ramps into and out of positions over consecutive bars.

    Instead of jumping from 0 to ±1, this policy increments by ±0.25
    per bar in the direction of momentum.  If momentum reverses, it
    decrements at the same rate.

    This produces a dataset rich in position *transitions* through
    intermediate levels — exactly the state-action territory the
    agent needs evidence about to learn smooth sizing.
    """

    def __init__(
        self,
        feature_columns: list[str],
        step_size: float = 0.25,
        *,
        momentum_feature: str = "log_ret_4",
    ) -> None:
        self._mom_idx = _feature_index(feature_columns, momentum_feature)
        self._step = step_size
        self._current_pos = 0.0

    def reset(self) -> None:
        self._current_pos = 0.0

    def select_action(self, obs: np.ndarray) -> float:
        mom = obs[self._mom_idx]

        if mom > 0.1:
            # Momentum up -> ramp toward long
            self._current_pos = min(1.0, self._current_pos + self._step)
        elif mom < -0.1:
            # Momentum down -> ramp toward short
            self._current_pos = max(-1.0, self._current_pos - self._step)
        else:
            # No momentum -> drift toward flat
            if self._current_pos > 0:
                self._current_pos = max(0.0, self._current_pos - self._step)
            elif self._current_pos < 0:
                self._current_pos = min(0.0, self._current_pos + self._step)

        return self._current_pos


# ── stochastic policies ───────────────────────────────────────────────────


class RandomPolicy(BehaviorPolicy):
    """Uniform random over all position levels.  Maximises action-space coverage."""

    def __init__(
        self,
        rng: np.random.Generator | None = None,
        position_levels: tuple[float, ...] = VALID_POSITIONS_3,
    ) -> None:
        self._rng = rng or np.random.default_rng()
        self._levels = position_levels

    def select_action(self, obs: np.ndarray) -> float:
        return float(self._rng.choice(self._levels))


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
        position_levels: tuple[float, ...] = VALID_POSITIONS_3,
    ) -> None:
        self._base = base
        self._eps = epsilon
        self._rng = rng or np.random.default_rng()
        self._levels = position_levels

    @property
    def name(self) -> str:
        return f"EpsGreedy({self._base.name}, eps={self._eps})"

    def reset(self) -> None:
        self._base.reset()

    def select_action(self, obs: np.ndarray) -> float:
        if self._rng.random() < self._eps:
            return float(self._rng.choice(self._levels))
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

    def select_action(self, obs: np.ndarray) -> float:
        idx = self._rng.choice(len(self._policies), p=self._weights)
        return self._policies[idx].select_action(obs)

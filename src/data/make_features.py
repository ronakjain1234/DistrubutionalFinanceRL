from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd


_DEFAULT_LOG_RET_WINDOWS: Final[list[int]] = [1, 5, 20]
_DEFAULT_VOL_WINDOWS: Final[list[int]] = [5, 20]
_DEFAULT_MA_WINDOWS: Final[list[int]] = [10, 20, 50]

# Supply / demand zone detection parameters (tuned for daily BTC)
_SD_PIVOT_ORDER: Final[int] = 5        # bars on each side to confirm a pivot
_SD_CONFIRM_BARS: Final[int] = 5       # bars after pivot to measure impulse
_SD_IMPULSE_PCT: Final[float] = 0.05   # 5 % move confirms a zone (crypto-scale)
_SD_LOOKBACK: Final[int] = 60          # ~3 months of active zone memory
_SD_PROXIMITY_PCT: Final[float] = 0.03 # within 3 % of zone = "near"
RAW_PATH: Final[str] = "data/raw/btc_daily.parquet"
OUT_PATH: Final[str] = "data/processed/btc_daily_features.parquet"


def _compute_rsi(close, period=14):

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder-style smoothing via EMA.
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _supply_demand_features(close_arr: np.ndarray) -> dict[str, np.ndarray]:
    """
    Detect supply and demand zones from raw close prices and compute
    per-bar proximity features.

    **Demand zone** — a price level where a strong *upward* impulse
    originated (institutional buying absorbed all selling).  When price
    revisits this level, buying is expected to resume.

    **Supply zone** — a price level where a strong *downward* impulse
    originated (institutional selling overwhelmed buying).  When price
    revisits, selling pressure is expected to return.

    Algorithm (no lookahead)
    ------------------------
    1. Pre-scan all bars for *pivot lows* (local minima within ±order bars)
       and *pivot highs* (local maxima).
    2. A pivot low at bar *i* becomes a **demand zone** at price ``close[i]``
       if ``max(close[i : i+confirm]) / close[i] - 1 > impulse_pct``
       (price rallied sharply after the low — confirms institutional demand).
    3. A pivot high becomes a **supply zone** analogously for drops.
    4. At each bar *t*, only zones whose confirmation period ends before *t*
       and that fall within the lookback window are considered "active".
    5. Nearest active zone below (demand) / above (supply) current price
       determines the distance features.

    Returns three arrays of length ``len(close_arr)``:
      * ``sd_dist_demand``  — (close - nearest_demand) / close  (NaN if none)
      * ``sd_dist_supply``  — (nearest_supply - close) / close  (NaN if none)
      * ``sd_zone_signal``  — +1 near demand, −1 near supply, 0 otherwise
    """
    n = len(close_arr)
    c = close_arr

    order = _SD_PIVOT_ORDER
    confirm = _SD_CONFIRM_BARS
    impulse = _SD_IMPULSE_PCT
    lookback = _SD_LOOKBACK
    proximity = _SD_PROXIMITY_PCT

    # ── Step 1: find all pivot lows / highs ────────────────────────────
    demand_zones: list[tuple[int, float]] = []   # (bar_index, price_level)
    supply_zones: list[tuple[int, float]] = []

    for i in range(order, n - order):
        window = c[i - order : i + order + 1]
        if c[i] <= window.min():                 # pivot low
            if i + confirm < n:
                rally = np.max(c[i : i + confirm + 1])
                if (rally - c[i]) / c[i] > impulse:
                    demand_zones.append((i, c[i]))
        if c[i] >= window.max():                 # pivot high
            if i + confirm < n:
                drop = np.min(c[i : i + confirm + 1])
                if (c[i] - drop) / c[i] > impulse:
                    supply_zones.append((i, c[i]))

    # ── Step 2: per-bar proximity features ─────────────────────────────
    dist_demand = np.full(n, np.nan)
    dist_supply = np.full(n, np.nan)
    zone_signal = np.zeros(n)

    for t in range(1, n):
        current = c[t]
        earliest = t - lookback

        # Active demand zones: confirmed before t, within lookback, below price
        active_d = [
            price for (idx, price) in demand_zones
            if earliest <= idx <= t - confirm and price < current
        ]
        if active_d:
            nearest = max(active_d)              # closest below
            dist_demand[t] = (current - nearest) / current

        # Active supply zones: confirmed before t, within lookback, above price
        active_s = [
            price for (idx, price) in supply_zones
            if earliest <= idx <= t - confirm and price > current
        ]
        if active_s:
            nearest = min(active_s)              # closest above
            dist_supply[t] = (nearest - current) / current

        # Composite signal
        if active_d and dist_demand[t] < proximity:
            zone_signal[t] = 1.0                 # near demand → bullish
        elif active_s and dist_supply[t] < proximity:
            zone_signal[t] = -1.0                # near supply → bearish

    # Fill NaN distances with a neutral "far away" value so downstream
    # z-scoring doesn't drop the row.
    dist_demand = np.where(np.isnan(dist_demand), 1.0, dist_demand)
    dist_supply = np.where(np.isnan(dist_supply), 1.0, dist_supply)

    return {
        "sd_dist_demand": dist_demand,
        "sd_dist_supply": dist_supply,
        "sd_zone_signal": zone_signal,
    }


def _make_features(df):
    df = df.sort_values("timestamp").reset_index(drop=True).copy()

    # Base series
    close = df["close"]
    volume = df["volume"]
    log_close = np.log(close.replace(0, np.nan))
    log_ret_1d = log_close.diff()  # log(C_t / C_{t-1})

    # Rolling log returns (additive in log space)
    for w in _DEFAULT_LOG_RET_WINDOWS:
        df[f"log_ret_{w}"] = log_ret_1d.rolling(w).sum()

    # Realized volatility (std of log returns)
    for w in _DEFAULT_VOL_WINDOWS:
        df[f"vol_{w}"] = log_ret_1d.rolling(w).std(ddof=0)

    # Moving average momentum (ratio relative to SMA)
    for w in _DEFAULT_MA_WINDOWS:
        sma = close.rolling(w).mean()
        df[f"ma_ratio_{w}"] = close / sma - 1.0

    # RSI and MACD
    df["rsi_14"] = _compute_rsi(close, period=14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd"] = macd
    df["macd_signal"] = signal
    df["macd_hist"] = macd - signal

    # Volume features
    df["log_volume"] = np.log1p(volume.clip(lower=0))
    df["log_volume_change_1"] = df["log_volume"].diff()

    # Supply / demand zone features (computed from raw close prices)
    sd = _supply_demand_features(close.to_numpy(dtype=np.float64))
    for col_name, arr in sd.items():
        df[col_name] = arr

    # Next-step target: reward at time t is based on return from t to t+1.
    df["next_close"] = df["close"].shift(-1)
    df["log_return_next_1d"] = (np.log(df["next_close"]) - np.log(df["close"])).astype(float)

    feature_cols = [
        f"log_ret_{w}" for w in _DEFAULT_LOG_RET_WINDOWS
    ] + [
        f"vol_{w}" for w in _DEFAULT_VOL_WINDOWS
    ] + [
        f"ma_ratio_{w}" for w in _DEFAULT_MA_WINDOWS
    ] + [
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "log_volume",
        "log_volume_change_1",
        "sd_dist_demand",
        "sd_dist_supply",
        "sd_zone_signal",
    ]

    return df, feature_cols


@dataclass(frozen=True)
class FeatureConfig:
    raw_path: Path = Path(RAW_PATH)
    out_path: Path = Path(OUT_PATH)


def make_feature_panel(cfg: FeatureConfig):
    if not cfg.raw_path.exists():
        raise FileNotFoundError(f"Raw input not found: {cfg.raw_path}")

    df = pd.read_parquet(cfg.raw_path)
    expected = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = expected.difference(set(df.columns))
    if missing:
        raise RuntimeError(f"Raw BTC parquet missing columns: {sorted(missing)}")

    out_df, _ = _make_features(df)
    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(cfg.out_path, index=False)

    print(f"Wrote BTC feature panel to {cfg.out_path}")
    return cfg.out_path


def main():
    cfg = FeatureConfig()
    out_path = make_feature_panel(cfg)
    print(f"Wrote BTC daily features: {out_path}")


if __name__ == "__main__":
    main()


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd


_DEFAULT_LOG_RET_WINDOWS: Final[list[int]] = [1, 5, 20]
_DEFAULT_VOL_WINDOWS: Final[list[int]] = [5, 20]
_DEFAULT_MA_WINDOWS: Final[list[int]] = [10, 20, 50]

# Hourly-specific window sizes
_HOURLY_LOG_RET_WINDOWS: Final[list[int]] = [1, 4, 24, 168]  # 1h, 4h, 1d, 1w
_HOURLY_VOL_WINDOWS: Final[list[int]] = [24, 168]             # 1d, 1w
_HOURLY_MA_WINDOWS: Final[list[int]] = [24, 72, 168]          # 1d, 3d, 1w

# Supply / demand zone detection parameters (tuned for daily BTC)
_SD_PIVOT_ORDER: Final[int] = 5        # bars on each side to confirm a pivot
_SD_CONFIRM_BARS: Final[int] = 5       # bars after pivot to measure impulse
_SD_IMPULSE_PCT: Final[float] = 0.05   # 5 % move confirms a zone (crypto-scale)
_SD_LOOKBACK: Final[int] = 60          # ~3 months of active zone memory
_SD_PROXIMITY_PCT: Final[float] = 0.03 # within 3 % of zone = "near"

# S/D zone parameters tuned for hourly BTC
_SD_HOURLY_PIVOT_ORDER: Final[int] = 12     # 12 hours on each side
_SD_HOURLY_CONFIRM_BARS: Final[int] = 24    # 24 hours to confirm impulse
_SD_HOURLY_IMPULSE_PCT: Final[float] = 0.03 # 3% move (hourly has smaller swings)
_SD_HOURLY_LOOKBACK: Final[int] = 720       # ~30 days of active zone memory
_SD_HOURLY_PROXIMITY_PCT: Final[float] = 0.02  # within 2% of zone

RAW_PATH: Final[str] = "data/raw/btc_daily.parquet"
OUT_PATH: Final[str] = "data/processed/btc_daily_features.parquet"
HOURLY_RAW_PATH: Final[str] = "data/raw/btc_hourly.parquet"
HOURLY_OUT_PATH: Final[str] = "data/processed/btc_hourly_features.parquet"


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


def _supply_demand_features(
    close_arr: np.ndarray,
    *,
    order: int = _SD_PIVOT_ORDER,
    confirm: int = _SD_CONFIRM_BARS,
    impulse: float = _SD_IMPULSE_PCT,
    lookback: int = _SD_LOOKBACK,
    proximity: float = _SD_PROXIMITY_PCT,
) -> dict[str, np.ndarray]:
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


def _make_hourly_features(df):
    """Build feature panel from hourly OHLCV data.

    Returns (df_with_features, feature_col_names). Adds 35 features:
    - Rolling log returns at 1h, 4h, 24h, 168h (1 week)
    - Realized volatility at 24h, 168h
    - MA momentum ratios at 24h, 72h, 168h
    - RSI-14 (14 hours), MACD (12/26/9 hours)
    - Log volume and volume change
    - Calendar encodings: hour_sin, hour_cos, dow_sin, dow_cos
    - vol_ratio (24h/168h), volume_ratio_24h (vol/SMA_24)
    - Return autocorrelation at lags 1, 4, 24
    - OHLC intrabar: close_in_range, bar_body_ratio, log_hl_range
    - Parkinson volatility (24h, 168h), Garman-Klass volatility (24h, 168h)
    - Bollinger %B (24h)
    - Supply/demand zone features (hourly-tuned)
    - Target: log_return_next_1h
    """
    df = df.sort_values("timestamp").reset_index(drop=True).copy()

    close = df["close"]
    volume = df["volume"]
    log_close = np.log(close.replace(0, np.nan))
    log_ret_1h = log_close.diff()

    # Rolling log returns
    for w in _HOURLY_LOG_RET_WINDOWS:
        df[f"log_ret_{w}"] = log_ret_1h.rolling(w).sum()

    # Realized volatility
    for w in _HOURLY_VOL_WINDOWS:
        df[f"vol_{w}"] = log_ret_1h.rolling(w).std(ddof=0)

    # MA momentum ratios
    for w in _HOURLY_MA_WINDOWS:
        sma = close.rolling(w).mean()
        df[f"ma_ratio_{w}"] = close / sma - 1.0

    # RSI-14 (14 hours)
    df["rsi_14"] = _compute_rsi(close, period=14)

    # MACD (12/26/9 hour EMAs)
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

    # ── Hourly-specific alpha features ────────────────────────────────

    # Calendar encodings (cyclical)
    hour_of_day = df["timestamp"].dt.hour
    day_of_week = df["timestamp"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour_of_day / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour_of_day / 24)
    df["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    df["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)

    # Volatility ratio: short-term / long-term realized vol
    vol_24 = log_ret_1h.rolling(24).std(ddof=0)
    vol_168 = log_ret_1h.rolling(168).std(ddof=0)
    df["vol_ratio"] = vol_24 / vol_168.replace(0, np.nan)

    # Volume ratio: current volume vs 24h SMA
    vol_sma_24 = volume.rolling(24).mean()
    df["volume_ratio_24h"] = volume / vol_sma_24.replace(0, np.nan)

    # Return autocorrelation at lags 1, 4, 24 (rolling 48h window)
    for lag in [1, 4, 24]:
        lagged = log_ret_1h.shift(lag)
        df[f"ret_autocorr_{lag}"] = (
            log_ret_1h.rolling(48).corr(lagged)
        )

    # ── OHLC intrabar features ───────────────────────────────────────

    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    hl_range = high - low

    # Close-in-range: where the bar closed relative to its range
    # 1.0 = closed at high (buyers won), 0.0 = closed at low (sellers won)
    df["close_in_range"] = (close - low) / hl_range.replace(0, np.nan)
    df["close_in_range"] = df["close_in_range"].clip(0, 1).fillna(0.5)

    # Bar body ratio: directional commitment vs indecision
    # 1.0 = full Marubozu (strong conviction), 0.0 = doji (indecision)
    df["bar_body_ratio"] = (close - open_).abs() / hl_range.replace(0, np.nan)
    df["bar_body_ratio"] = df["bar_body_ratio"].clip(0, 1).fillna(0.0)

    # Log high-low range: intrabar volatility per candle
    df["log_hl_range"] = np.log((high / low).replace(0, np.nan).clip(lower=1e-10))

    # Parkinson volatility (rolling 24h and 168h)
    # sqrt(1/(4*n*ln2) * sum(log(H/L)^2))
    log_hl_sq = np.log(high / low.replace(0, np.nan)) ** 2
    for w in [24, 168]:
        df[f"parkinson_vol_{w}"] = np.sqrt(
            log_hl_sq.rolling(w).mean() / (4 * np.log(2))
        )

    # Garman-Klass volatility (rolling 24h and 168h)
    # sqrt(1/n * sum(0.5*log(H/L)^2 - (2*ln2-1)*log(C/O)^2))
    log_co_sq = np.log(close / open_.replace(0, np.nan)) ** 2
    gk_term = 0.5 * log_hl_sq - (2 * np.log(2) - 1) * log_co_sq
    for w in [24, 168]:
        df[f"garman_klass_vol_{w}"] = np.sqrt(
            gk_term.rolling(w).mean().clip(lower=0)
        )

    # Bollinger %B (24h window, 2 std bands)
    sma_24 = close.rolling(24).mean()
    std_24 = close.rolling(24).std(ddof=0)
    upper_band = sma_24 + 2 * std_24
    lower_band = sma_24 - 2 * std_24
    band_width = upper_band - lower_band
    df["bollinger_pctb"] = (close - lower_band) / band_width.replace(0, np.nan)

    # Supply / demand zone features (hourly-tuned parameters)
    sd = _supply_demand_features(
        close.to_numpy(dtype=np.float64),
        order=_SD_HOURLY_PIVOT_ORDER,
        confirm=_SD_HOURLY_CONFIRM_BARS,
        impulse=_SD_HOURLY_IMPULSE_PCT,
        lookback=_SD_HOURLY_LOOKBACK,
        proximity=_SD_HOURLY_PROXIMITY_PCT,
    )
    for col_name, arr in sd.items():
        df[col_name] = arr

    # Next-step target: 1-hour forward log return
    df["next_close"] = df["close"].shift(-1)
    df["log_return_next_1h"] = (np.log(df["next_close"]) - np.log(df["close"])).astype(float)

    feature_cols = (
        [f"log_ret_{w}" for w in _HOURLY_LOG_RET_WINDOWS]
        + [f"vol_{w}" for w in _HOURLY_VOL_WINDOWS]
        + [f"ma_ratio_{w}" for w in _HOURLY_MA_WINDOWS]
        + [
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_hist",
            "log_volume",
            "log_volume_change_1",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "vol_ratio",
            "volume_ratio_24h",
            "ret_autocorr_1",
            "ret_autocorr_4",
            "ret_autocorr_24",
            # OHLC intrabar features
            "close_in_range",
            "bar_body_ratio",
            "log_hl_range",
            "parkinson_vol_24",
            "parkinson_vol_168",
            "garman_klass_vol_24",
            "garman_klass_vol_168",
            "bollinger_pctb",
            # S/D zones
            "sd_dist_demand",
            "sd_dist_supply",
            "sd_zone_signal",
        ]
    )

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


@dataclass(frozen=True)
class HourlyFeatureConfig:
    raw_path: Path = Path(HOURLY_RAW_PATH)
    out_path: Path = Path(HOURLY_OUT_PATH)


def make_hourly_feature_panel(cfg: HourlyFeatureConfig):
    if not cfg.raw_path.exists():
        raise FileNotFoundError(f"Raw hourly input not found: {cfg.raw_path}")

    df = pd.read_parquet(cfg.raw_path)
    expected = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = expected.difference(set(df.columns))
    if missing:
        raise RuntimeError(f"Raw BTC hourly parquet missing columns: {sorted(missing)}")

    out_df, _ = _make_hourly_features(df)
    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(cfg.out_path, index=False)

    print(f"Wrote BTC hourly feature panel to {cfg.out_path}  ({len(out_df)} rows)")
    return cfg.out_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build BTC feature panel")
    parser.add_argument(
        "--frequency", choices=["daily", "hourly"], default="daily",
        help="Feature frequency to build (default: daily)",
    )
    args = parser.parse_args()

    if args.frequency == "hourly":
        cfg = HourlyFeatureConfig()
        out_path = make_hourly_feature_panel(cfg)
        print(f"Wrote BTC hourly features: {out_path}")
    else:
        cfg = FeatureConfig()
        out_path = make_feature_panel(cfg)
        print(f"Wrote BTC daily features: {out_path}")


if __name__ == "__main__":
    main()


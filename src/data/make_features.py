"""
Create a BTC feature panel for offline RL.

This module implements the "preprocessing / feature engineering" portion of
Step 2 in the project roadmap:

* rolling log returns over multiple windows
* realized volatility (rolling std of log returns)
* momentum features (moving average ratios, RSI, MACD)
* simple volume features
* creation of the next-step return target (`log_return_next_1d`)

Outputs
-------
* `data/processed/btc_daily_features.parquet` (by default)
* `data/processed/btc_daily_feature_meta.json` (feature column list + params)
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


_DEFAULT_LOG_RET_WINDOWS: Final[list[int]] = [1, 5, 20]
_DEFAULT_VOL_WINDOWS: Final[list[int]] = [5, 20]
_DEFAULT_MA_WINDOWS: Final[list[int]] = [10, 20, 50]


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute RSI using exponential moving averages (EMA smoothing).
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder-style smoothing via EMA.
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _make_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Compute engineered features on a daily OHLCV dataframe.

    Expects columns:
    * timestamp, open, high, low, close, volume
    """
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
    ]

    return df, feature_cols


@dataclass(frozen=True)
class FeatureConfig:
    raw_path: Path
    out_path: Path
    meta_path: Path


def make_feature_panel(cfg: FeatureConfig) -> Path:
    """
    Load raw BTC data, compute features, write the full (un-normalized) panel.
    """
    if not cfg.raw_path.exists():
        raise FileNotFoundError(f"Raw input not found: {cfg.raw_path}")

    df = pd.read_parquet(cfg.raw_path)
    expected = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = expected.difference(set(df.columns))
    if missing:
        raise RuntimeError(f"Raw BTC parquet missing columns: {sorted(missing)}")

    out_df, feature_cols = _make_features(df)
    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(cfg.out_path, index=False)

    meta = {
        "feature_cols": feature_cols,
        "log_ret_windows": _DEFAULT_LOG_RET_WINDOWS,
        "vol_windows": _DEFAULT_VOL_WINDOWS,
        "ma_windows": _DEFAULT_MA_WINDOWS,
        "rsi_period": 14,
        "macd": {"fast": 12, "slow": 26, "signal": 9},
        "target_col": "log_return_next_1d",
        "universe": {"timestamp_tz": "UTC", "frequency": "1D"},
    }
    cfg.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    LOGGER.info("Wrote BTC feature panel to %s", cfg.out_path)
    return cfg.out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create BTC daily feature panel.")
    parser.add_argument("--raw-path", type=str, default="data/raw/btc_daily.parquet", help="Path to cleaned raw BTC parquet.")
    parser.add_argument(
        "--out-path", type=str, default="data/processed/btc_daily_features.parquet", help="Output feature parquet path."
    )
    parser.add_argument(
        "--meta-path",
        type=str,
        default="data/processed/btc_daily_feature_meta.json",
        help="Output JSON with feature column list and engineering parameters.",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging verbosity.")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    cfg = FeatureConfig(
        raw_path=Path(args.raw_path),
        out_path=Path(args.out_path),
        meta_path=Path(args.meta_path),
    )
    out_path = make_feature_panel(cfg)
    print(f"Wrote BTC daily features: {out_path}")


if __name__ == "__main__":
    main()


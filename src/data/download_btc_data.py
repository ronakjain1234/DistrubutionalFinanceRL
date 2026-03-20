"""
Download and clean BTC daily OHLCV data.

This module implements the "data acquisition" portion of Step 2 in the project
roadmap. It downloads a daily BTC series, enforces consistent timestamps,
removes duplicates, and applies a clear missing-data policy.

Default source: `coinbase` (daily BTC OHLCV candles).

Outputs
-------
* `data/raw/btc_daily.parquet` (by default): cleaned daily OHLCV
* `data/raw/btc_daily.csv`: same data in CSV form
* `data/raw/btc_daily_meta.json`: download metadata and assumptions
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import pandas as pd
import time
import urllib.parse
import urllib.request

LOGGER = logging.getLogger(__name__)

_EXPECTED_COLUMNS: Final[set[str]] = {"open", "high", "low", "close", "volume"}


def _parse_date(value: str) -> pd.Timestamp:
    """
    Parse an input date string into a normalized UTC timestamp (midnight).
    """
    ts = pd.to_datetime(value, utc=True)
    return ts.normalize()


def _iso_utc_day(ts: pd.Timestamp) -> str:
    """
    Format a timestamp as ISO UTC 'YYYY-MM-DDTHH:MM:SSZ' for Coinbase params.
    """
    ts = ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _request_json(url: str, *, timeout_s: int = 30, max_retries: int = 5) -> object:
    """
    Small HTTP helper with basic retry for transient failures / rate limiting.
    """
    headers = {
        "User-Agent": "DistrubutionalFinanceRL/Step2BTCDownloader (Educational)",
        "Accept": "application/json",
    }
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                payload = resp.read().decode("utf-8")
                return json.loads(payload)
        except Exception as e:  # noqa: BLE001 - we want to retry multiple failure modes
            last_err = e
            # Simple backoff; Coinbase commonly returns 429 on rate limiting.
            sleep_s = min(2**attempt, 30)
            LOGGER.warning("Request failed (attempt %d/%d): %s. Sleeping %.1fs", attempt, max_retries, e, sleep_s)
            time.sleep(sleep_s)
    assert last_err is not None
    raise RuntimeError(f"Failed request after {max_retries} retries: {url}. Last error: {last_err}") from last_err


def _download_from_coinbase_daily(product_id: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """
    Download daily OHLCV candles from Coinbase Exchange.

    Uses: GET /products/{product_id}/candles
    with: granularity=86400 (1 day)

    Coinbase returns up to ~300 candles per request, so we paginate by chunks.
    """
    base_url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"

    # Daily candles: 300-per-request means ~300 days max per call.
    chunk_days = 295
    cursor = start_ts
    chunks: list[pd.DataFrame] = []

    while cursor <= end_ts:
        chunk_end = min(cursor + pd.Timedelta(days=chunk_days), end_ts)
        params = {
            "granularity": 86400,
            "start": _iso_utc_day(cursor),
            "end": _iso_utc_day(chunk_end),
        }
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        LOGGER.info("Downloading Coinbase candles: %s", url)

        payload = _request_json(url)
        if not isinstance(payload, list) or not payload:
            break

        # Each candle is [time, low, high, open, close, volume]
        rows: list[list[object]] = payload  # type: ignore[assignment]
        df = pd.DataFrame(
            rows,
            columns=["timestamp", "low", "high", "open", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.normalize()
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df = df.drop_duplicates(subset=["timestamp"], keep="last")
        chunks.append(df)

        # Advance by 1 day to avoid overlap.
        cursor = chunk_end + pd.Timedelta(days=1)

    if not chunks:
        raise RuntimeError(f"No Coinbase candles returned for {product_id} in range {start_ts}..{end_ts}")

    out = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["timestamp"], keep="last")
    return out.sort_values("timestamp")


def _load_local_btc_ohlcv(path: Path) -> pd.DataFrame:
    """
    Load a local BTC OHLCV dataset.

    Expected columns (case-insensitive):
    * timestamp/date (daily timestamp)
    * open, high, low, close, volume
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        # Assume CSV
        df = pd.read_csv(path)

    df.columns = [c.lower() for c in df.columns]
    if "date" in df.columns and "timestamp" not in df.columns:
        df = df.rename(columns={"date": "timestamp"})
    if "timestamp" not in df.columns:
        raise RuntimeError(f"Local BTC data must include a 'timestamp' or 'date' column; got {list(df.columns)}")

    missing = _EXPECTED_COLUMNS.difference(set(df.columns))
    if missing:
        raise RuntimeError(f"Local BTC data missing columns: {sorted(missing)}")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.normalize()

    df = df[["timestamp", "open", "high", "low", "close", "volume"]].drop_duplicates(
        subset=["timestamp"], keep="last"
    )
    return df.sort_values("timestamp")


def _reindex_and_fill(
    df: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    *,
    max_ffill_gap_days: int,
) -> pd.DataFrame:
    """
    Reindex to a daily grid and fill "small" gaps using forward-fill.

    Policy
    ------
    * All timestamps are forced to UTC midnight (`freq="D"`).
    * If OHLC is missing because a day is absent, we forward-fill prices for
      gaps of length <= `max_ffill_gap_days`.
    * For those imputed rows, we set `volume=0.0` (volume is not meaningfully
      known for the missing day).
    * Rows with remaining missing OHLC after the above policy are dropped.
    """
    if df.empty:
        raise ValueError("Downloaded dataframe is empty.")

    df = df.set_index("timestamp").sort_index()
    full_index = pd.date_range(start_ts, end_ts, freq="D", tz="UTC")
    df = df.reindex(full_index)

    price_cols = ["open", "high", "low", "close"]
    missing_mask = df["close"].isna()

    # Identify consecutive missing runs to decide which can be forward-filled.
    run_id = (missing_mask != missing_mask.shift()).cumsum()
    imputed_mask = pd.Series(False, index=df.index)

    for _, grp_idx in missing_mask.groupby(run_id).groups.items():
        # grp_idx is an Index of timestamps for this run.
        grp_timestamps = pd.Index(grp_idx)
        if grp_timestamps.empty:
            continue
        if not bool(missing_mask.loc[grp_timestamps].iloc[0]):
            continue
        run_len = int(len(grp_timestamps))
        if run_len <= max_ffill_gap_days:
            imputed_mask.loc[grp_timestamps] = True

    # Apply forward-fill only for selected imputed timestamps.
    df_ffill = df.copy()
    df_ffill[price_cols] = df_ffill[price_cols].ffill()
    df.loc[imputed_mask, price_cols] = df_ffill.loc[imputed_mask, price_cols]
    df.loc[imputed_mask, "volume"] = 0.0

    # Drop any remaining missing OHLC.
    df = df.dropna(subset=price_cols)
    df = df.reset_index().rename(columns={"index": "timestamp"})

    # Basic sanity: non-negative OHLCV.
    for col in price_cols + ["volume"]:
        df[col] = df[col].clip(lower=0)

    return df


@dataclass(frozen=True)
class DownloadConfig:
    symbol: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    out_dir: Path
    max_ffill_gap_days: int = 2


def download_btc_daily(
    cfg: DownloadConfig,
    *,
    raw_df_override: pd.DataFrame | None = None,
) -> Path:
    """
    Download BTC daily OHLCV and write it to the raw data directory.
    """
    out_dir = cfg.out_dir
    parquet_path = out_dir / "btc_daily.parquet"
    csv_path = out_dir / "btc_daily.csv"
    meta_path = out_dir / "btc_daily_meta.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = raw_df_override if raw_df_override is not None else _download_from_coinbase_daily(
        cfg.symbol, cfg.start_date, cfg.end_date
    )
    start_ts, end_ts = cfg.start_date, cfg.end_date
    raw = raw[raw["timestamp"].between(start_ts, end_ts)].copy()
    raw = _reindex_and_fill(
        raw,
        start_ts,
        end_ts,
        max_ffill_gap_days=cfg.max_ffill_gap_days,
    )

    raw.to_parquet(parquet_path, index=False)
    raw.to_csv(csv_path, index=False)

    meta = {
        "symbol": cfg.symbol,
        "source": "coinbase",
        "start_date": str(cfg.start_date.date()),
        "end_date": str(cfg.end_date.date()),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "max_ffill_gap_days": cfg.max_ffill_gap_days,
        "n_rows": int(len(raw)),
        "timestamp_tz": "UTC",
        "columns": list(raw.columns),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    LOGGER.info("Wrote cleaned BTC daily OHLCV to %s", parquet_path)
    return parquet_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and clean BTC daily OHLCV data.")
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC-USD",
        help="Coinbase product id (e.g. BTC-USD).",
    )
    parser.add_argument("--start-date", type=str, default="2016-01-01", help="Start date (inclusive), e.g. 2016-01-01.")
    parser.add_argument("--end-date", type=str, default="2025-12-31", help="End date (inclusive), e.g. 2025-12-31.")
    parser.add_argument("--out-dir", type=str, default="data/raw", help="Output directory for raw parquet/csv.")
    parser.add_argument(
        "--max-ffill-gap-days",
        type=int,
        default=2,
        help="Forward-fill prices for missing days if the gap length is <= this value.",
    )
    parser.add_argument("--input-csv", type=str, default="", help="Optional local CSV path to load instead of downloading.")
    parser.add_argument(
        "--input-parquet", type=str, default="", help="Optional local Parquet path to load instead of downloading."
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging verbosity (DEBUG, INFO, WARNING).")

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if args.input_csv and args.input_parquet:
        raise SystemExit("Provide at most one of --input-csv or --input-parquet.")

    raw_override: pd.DataFrame | None = None
    if args.input_csv:
        raw_override = _load_local_btc_ohlcv(Path(args.input_csv))
    elif args.input_parquet:
        raw_override = _load_local_btc_ohlcv(Path(args.input_parquet))

    cfg = DownloadConfig(
        symbol=args.symbol,
        start_date=_parse_date(args.start_date),
        end_date=_parse_date(args.end_date),
        out_dir=Path(args.out_dir),
        max_ffill_gap_days=int(args.max_ffill_gap_days),
    )

    parquet_path = download_btc_daily(cfg, raw_df_override=raw_override)
    print(f"Downloaded BTC daily OHLCV: {parquet_path}")


if __name__ == "__main__":
    main()


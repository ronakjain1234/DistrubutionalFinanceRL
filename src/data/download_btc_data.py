from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

_EXPECTED_COLUMNS: Final[set[str]] = {"open", "high", "low", "close", "volume"}

SYMBOL: Final[str] = "BTC-USD"
START_DATE: Final[str] = "2017-07-01"
END_DATE: Final[str] = "2025-12-31"
OUT_DIR: Final[str] = "data/raw"


def _parse_date(value):
    ts = pd.to_datetime(value, utc=True)
    return ts.normalize()


def _iso_utc(ts):
    """Format a timestamp as ISO-8601 UTC string for the Coinbase API."""
    ts = ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso_utc_day(ts):
    ts = ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
    ts = ts.normalize()
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _request_json(url, *, timeout_s=30, max_retries=5):
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
        except Exception as e:  # noqa: BLE001
            last_err = e
            sleep_s = min(2**attempt, 30)
            print(
                f"Request failed (attempt {attempt}/{max_retries}): {e}. Sleeping {sleep_s:.1f}s"
            )
            time.sleep(sleep_s)
    assert last_err is not None
    raise RuntimeError(f"Failed request after {max_retries} retries: {url}. Last error: {last_err}") from last_err


def _download_from_coinbase_daily(product_id, start_ts, end_ts):
    base_url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"

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
        print(f"Downloading Coinbase candles: {url}")

        payload = _request_json(url)
        if not isinstance(payload, list) or not payload:
            break

        rows: list[list[object]] = payload  # type: ignore[assignment]
        df = pd.DataFrame(
            rows,
            columns=["timestamp", "low", "high", "open", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.normalize()
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df = df.drop_duplicates(subset=["timestamp"], keep="last")
        chunks.append(df)

        cursor = chunk_end + pd.Timedelta(days=1)

    if not chunks:
        raise RuntimeError(f"No Coinbase candles returned for {product_id} in range {start_ts}..{end_ts}")

    out = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["timestamp"], keep="last")
    return out.sort_values("timestamp")


def _download_from_coinbase_hourly(product_id, start_ts, end_ts):
    """Download hourly OHLCV candles from Coinbase Exchange API.

    Uses granularity=3600 (1 hour). The API returns at most 300 candles per
    request, so we chunk into ~12-day windows (288 hours) with a small delay
    between requests to respect rate limits.
    """
    base_url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"

    chunk_hours = 288  # 12 days × 24 hours, safely under 300-candle limit
    cursor = start_ts
    chunks: list[pd.DataFrame] = []
    request_count = 0

    while cursor <= end_ts:
        chunk_end = min(cursor + pd.Timedelta(hours=chunk_hours), end_ts)
        params = {
            "granularity": 3600,
            "start": _iso_utc(cursor),
            "end": _iso_utc(chunk_end),
        }
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        print(f"Downloading hourly candles: {_iso_utc(cursor)[:10]} -> {_iso_utc(chunk_end)[:10]}  (req #{request_count + 1})")

        payload = _request_json(url)
        request_count += 1

        if not isinstance(payload, list) or not payload:
            cursor = chunk_end + pd.Timedelta(hours=1)
            continue

        rows: list[list[object]] = payload  # type: ignore[assignment]
        df = pd.DataFrame(
            rows,
            columns=["timestamp", "low", "high", "open", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df = df.drop_duplicates(subset=["timestamp"], keep="last")
        chunks.append(df)

        cursor = chunk_end + pd.Timedelta(hours=1)

        # Rate-limit: ~3 requests/sec is safe for Coinbase public API
        time.sleep(0.35)

    if not chunks:
        raise RuntimeError(f"No Coinbase hourly candles returned for {product_id} in range {start_ts}..{end_ts}")

    print(f"Hourly download complete: {request_count} API requests")
    out = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["timestamp"], keep="last")
    return out.sort_values("timestamp")


def _reindex_and_fill_hourly(df, start_ts, end_ts, *, max_ffill_gap_hours):
    """Reindex to a complete hourly grid and forward-fill small gaps."""
    if df.empty:
        raise ValueError("Downloaded hourly dataframe is empty.")

    df = df.set_index("timestamp").sort_index()
    full_index = pd.date_range(start_ts, end_ts, freq="h", tz="UTC")
    df = df.reindex(full_index)

    price_cols = ["open", "high", "low", "close"]
    missing_mask = df["close"].isna()

    run_id = (missing_mask != missing_mask.shift()).cumsum()
    imputed_mask = pd.Series(False, index=df.index)

    for _, grp_idx in missing_mask.groupby(run_id).groups.items():
        grp_timestamps = pd.Index(grp_idx)
        if grp_timestamps.empty:
            continue
        if not bool(missing_mask.loc[grp_timestamps].iloc[0]):
            continue
        run_len = int(len(grp_timestamps))
        if run_len <= max_ffill_gap_hours:
            imputed_mask.loc[grp_timestamps] = True

    df_ffill = df.copy()
    df_ffill[price_cols] = df_ffill[price_cols].ffill()
    df.loc[imputed_mask, price_cols] = df_ffill.loc[imputed_mask, price_cols]
    df.loc[imputed_mask, "volume"] = 0.0

    df = df.dropna(subset=price_cols)
    df = df.reset_index().rename(columns={"index": "timestamp"})

    for col in price_cols + ["volume"]:
        df[col] = df[col].clip(lower=0)

    return df


def _reindex_and_fill(df, start_ts, end_ts, *, max_ffill_gap_days):
    if df.empty:
        raise ValueError("Downloaded dataframe is empty.")

    df = df.set_index("timestamp").sort_index()
    full_index = pd.date_range(start_ts, end_ts, freq="D", tz="UTC")
    df = df.reindex(full_index)

    price_cols = ["open", "high", "low", "close"]
    missing_mask = df["close"].isna()

    run_id = (missing_mask != missing_mask.shift()).cumsum()
    imputed_mask = pd.Series(False, index=df.index)

    for _, grp_idx in missing_mask.groupby(run_id).groups.items():
        grp_timestamps = pd.Index(grp_idx)
        if grp_timestamps.empty:
            continue
        if not bool(missing_mask.loc[grp_timestamps].iloc[0]):
            continue
        run_len = int(len(grp_timestamps))
        if run_len <= max_ffill_gap_days:
            imputed_mask.loc[grp_timestamps] = True

    df_ffill = df.copy()
    df_ffill[price_cols] = df_ffill[price_cols].ffill()
    df.loc[imputed_mask, price_cols] = df_ffill.loc[imputed_mask, price_cols]
    df.loc[imputed_mask, "volume"] = 0.0

    df = df.dropna(subset=price_cols)
    df = df.reset_index().rename(columns={"index": "timestamp"})

    for col in price_cols + ["volume"]:
        df[col] = df[col].clip(lower=0)

    return df


@dataclass(frozen=True)
class DownloadConfig:
    symbol: str = SYMBOL
    start_date: pd.Timestamp = _parse_date(START_DATE)
    end_date: pd.Timestamp = _parse_date(END_DATE)
    out_dir: Path = Path(OUT_DIR)
    max_ffill_gap_days: int = 2


def download_btc_daily(
    cfg: DownloadConfig,
    *,
    raw_df_override: pd.DataFrame | None = None,
):
    out_dir = cfg.out_dir
    parquet_path = out_dir / "btc_daily.parquet"
    csv_path = out_dir / "btc_daily.csv"
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

    print(f"Wrote cleaned BTC daily OHLCV ({len(raw)} rows)")
    print(f"  Parquet: {parquet_path}")
    print(f"  CSV:     {csv_path}")
    return parquet_path


@dataclass(frozen=True)
class HourlyDownloadConfig:
    symbol: str = SYMBOL
    start_date: pd.Timestamp = _parse_date(START_DATE)
    end_date: pd.Timestamp = _parse_date(END_DATE)
    out_dir: Path = Path(OUT_DIR)
    max_ffill_gap_hours: int = 6  # forward-fill up to 6 consecutive missing hours


def download_btc_hourly(
    cfg: HourlyDownloadConfig,
    *,
    raw_df_override: pd.DataFrame | None = None,
):
    out_dir = cfg.out_dir
    parquet_path = out_dir / "btc_hourly.parquet"
    csv_path = out_dir / "btc_hourly.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = raw_df_override if raw_df_override is not None else _download_from_coinbase_hourly(
        cfg.symbol, cfg.start_date, cfg.end_date
    )
    start_ts, end_ts = cfg.start_date, cfg.end_date
    raw = raw[raw["timestamp"].between(start_ts, end_ts)].copy()
    raw = _reindex_and_fill_hourly(
        raw,
        start_ts,
        end_ts,
        max_ffill_gap_hours=cfg.max_ffill_gap_hours,
    )

    raw.to_parquet(parquet_path, index=False)
    raw.to_csv(csv_path, index=False)

    print(f"Wrote cleaned BTC hourly OHLCV ({len(raw)} rows)")
    print(f"  Parquet: {parquet_path}")
    print(f"  CSV:     {csv_path}")
    return parquet_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download BTC OHLCV data from Coinbase")
    parser.add_argument(
        "--frequency", choices=["daily", "hourly"], default="daily",
        help="Candle frequency to download (default: daily)",
    )
    args = parser.parse_args()

    if args.frequency == "hourly":
        cfg = HourlyDownloadConfig()
        parquet_path = download_btc_hourly(cfg)
        print(f"\nDone. BTC hourly OHLCV saved to {parquet_path.parent}")
    else:
        cfg = DownloadConfig()
        parquet_path = download_btc_daily(cfg)
        print(f"\nDone. BTC daily OHLCV saved to {parquet_path.parent}")


if __name__ == "__main__":
    main()

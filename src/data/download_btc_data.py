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


def _parse_date(value: str) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True)
    return ts.normalize()


def _iso_utc_day(ts: pd.Timestamp) -> str:
    ts = ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
    ts = ts.normalize()
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _request_json(url: str, *, timeout_s: int = 30, max_retries: int = 5) -> object:
    """Small HTTP helper with basic retry for transient failures / rate limiting."""
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


def _download_from_coinbase_daily(product_id: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """
    Download daily OHLCV candles from Coinbase Exchange.

    Uses: GET /products/{product_id}/candles
    with: granularity=86400 (1 day)

    Coinbase returns up to ~300 candles per request, so we paginate by chunks.
    """
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
    * All timestamps are forced to UTC midnight (freq="D").
    * If OHLC is missing because a day is absent, we forward-fill prices for
      gaps of length <= max_ffill_gap_days.
    * For those imputed rows, we set volume=0.0 (volume is not meaningfully
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
) -> Path:
    """Download BTC daily OHLCV and write it to the raw data directory."""
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


def main() -> None:
    cfg = DownloadConfig()
    parquet_path = download_btc_daily(cfg)
    print(f"\nDone. BTC daily OHLCV saved to {parquet_path.parent}")


if __name__ == "__main__":
    main()

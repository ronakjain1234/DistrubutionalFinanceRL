"""
Split the engineered BTC feature panel into train/validation/test sets.

This module implements the "split the timeline" and "normalize using training
period statistics only" parts of Step 2 in the project roadmap.

Given an input parquet produced by `make_features.py`, it:
1) selects rows by timestamp boundaries,
2) computes z-score normalization parameters on the training split only,
3) applies the normalization to all splits,
4) writes three parquet files plus a scaler metadata JSON.

Outputs (defaults)
-------------------
* `data/processed/btc_daily_train.parquet`
* `data/processed/btc_daily_val.parquet`
* `data/processed/btc_daily_test.parquet`
* `data/processed/btc_daily_feature_scaler.json`
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

_DEFAULT_SPLITS: Final[dict[str, tuple[str, str]]] = {
    # Matches the README roadmap examples.
    "train": ("2016-01-01", "2020-12-31"),
    "val": ("2021-01-01", "2022-12-31"),
    "test": ("2023-01-01", "2025-12-31"),
}


def _parse_date(value: str) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True)
    return ts.normalize()


def _load_feature_columns(features_df: pd.DataFrame, meta_path: Path | None) -> list[str]:
    """
    Determine which columns are treated as observations for normalization.
    """
    if meta_path is not None and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cols = meta.get("feature_cols")
        if isinstance(cols, list) and all(isinstance(x, str) for x in cols):
            return cols

    # Fallback: everything except obvious non-features.
    excluded = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "next_close",
        "log_return_next_1d",
        "volume",
    }
    return [c for c in features_df.columns if c not in excluded]


@dataclass(frozen=True)
class SplitConfig:
    features_path: Path
    out_dir: Path
    meta_path: Path | None
    feature_cols: list[str] | None
    max_na_rows: int = 0


def split_and_normalize(
    cfg: SplitConfig,
    *,
    splits: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> dict[str, Path]:
    if not cfg.features_path.exists():
        raise FileNotFoundError(f"Features parquet not found: {cfg.features_path}")

    out_dir = cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(cfg.features_path)
    if "timestamp" not in df.columns:
        raise RuntimeError("Expected a 'timestamp' column in the engineered features parquet.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.normalize()
    df = df.sort_values("timestamp")

    feature_cols = cfg.feature_cols or _load_feature_columns(df, cfg.meta_path)

    # Keep only rows with non-null target and non-null features at minimum.
    if "log_return_next_1d" not in df.columns:
        raise RuntimeError("Expected target column 'log_return_next_1d' in features parquet.")

    df = df.dropna(subset=["log_return_next_1d"] + feature_cols)

    # Create splits.
    split_dfs: dict[str, pd.DataFrame] = {}
    for split_name, (start_ts, end_ts) in splits.items():
        mask = df["timestamp"].between(start_ts, end_ts, inclusive="both")
        split_dfs[split_name] = df.loc[mask].copy()
        if split_dfs[split_name].empty:
            print(f"Split {split_name} is empty for bounds {start_ts}..{end_ts}")

    # Fit scaler on training only.
    train_df = split_dfs.get("train")
    if train_df is None or train_df.empty:
        raise RuntimeError("Train split is empty; cannot fit normalization statistics.")

    means = train_df[feature_cols].mean(axis=0)
    stds = train_df[feature_cols].std(axis=0, ddof=0).replace(0.0, 1.0)

    scaler = {
        "feature_cols": feature_cols,
        "means": {k: float(means[k]) for k in feature_cols},
        "stds": {k: float(stds[k]) for k in feature_cols},
        "fitted_on": {
            "train_start": str(splits["train"][0].date()),
            "train_end": str(splits["train"][1].date()),
        },
        "normalization": "z_score",
    }

    scaler_path = out_dir / "btc_daily_feature_scaler.json"
    scaler_path.write_text(json.dumps(scaler, indent=2), encoding="utf-8")

    # Apply normalization and save.
    written: dict[str, Path] = {}
    for split_name, sdf in split_dfs.items():
        norm = sdf.copy()
        norm[feature_cols] = (norm[feature_cols] - means) / stds

        out_path = out_dir / f"btc_daily_{split_name}.parquet"
        norm.to_parquet(out_path, index=False)
        written[split_name] = out_path

    split_meta = {
        "splits": {k: [str(v[0].date()), str(v[1].date())] for k, v in splits.items()},
        "n_rows": {k: int(len(v)) for k, v in split_dfs.items()},
        "dropped_na": "Rows with NaNs in target/features were dropped before splitting.",
    }
    (out_dir / "btc_daily_splits.json").write_text(json.dumps(split_meta, indent=2), encoding="utf-8")

    print(f"Wrote split+normalized parquet files to {out_dir}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Split BTC daily engineered features into time splits + normalize.")
    parser.add_argument("--features-path", type=str, default="data/processed/btc_daily_features.parquet")
    parser.add_argument("--meta-path", type=str, default="data/processed/btc_daily_feature_meta.json")
    parser.add_argument("--out-dir", type=str, default="data/processed")

    # Allow overriding split boundaries from CLI.
    parser.add_argument("--train-start", type=str, default=_DEFAULT_SPLITS["train"][0])
    parser.add_argument("--train-end", type=str, default=_DEFAULT_SPLITS["train"][1])
    parser.add_argument("--val-start", type=str, default=_DEFAULT_SPLITS["val"][0])
    parser.add_argument("--val-end", type=str, default=_DEFAULT_SPLITS["val"][1])
    parser.add_argument("--test-start", type=str, default=_DEFAULT_SPLITS["test"][0])
    parser.add_argument("--test-end", type=str, default=_DEFAULT_SPLITS["test"][1])

    args = parser.parse_args()

    cfg = SplitConfig(
        features_path=Path(args.features_path),
        out_dir=Path(args.out_dir),
        meta_path=Path(args.meta_path) if args.meta_path else None,
        feature_cols=None,
    )

    splits = {
        "train": (_parse_date(args.train_start), _parse_date(args.train_end)),
        "val": (_parse_date(args.val_start), _parse_date(args.val_end)),
        "test": (_parse_date(args.test_start), _parse_date(args.test_end)),
    }

    written = split_and_normalize(cfg, splits=splits)
    for k, p in written.items():
        print(f"{k}: {p}")


if __name__ == "__main__":
    main()


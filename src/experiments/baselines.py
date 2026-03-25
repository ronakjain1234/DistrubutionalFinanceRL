"""
Benchmark policies for experiments (Step 3): buy-and-hold vs learned agents.

The **buy-and-hold** baseline is a constant long position (+1) on BTC, evaluated on the same
processed splits as the RL pipeline (train / val / test parquets from ``split_dataset``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
import pandas as pd

from ..env.portfolio import PortfolioConfig, PortfolioResult, simulate_portfolio

# Default paths (match README / split_dataset.py)
DEFAULT_TRAIN = Path("data/processed/btc_daily_train.parquet")
DEFAULT_VAL = Path("data/processed/btc_daily_val.parquet")
DEFAULT_TEST = Path("data/processed/btc_daily_test.parquet")


@dataclass(frozen=True)
class BuyAndHoldReport:
    """Time series + metrics for one split."""

    split_name: str
    source_path: Path
    timestamps: pd.Series
    equity: np.ndarray
    step_log_returns: np.ndarray
    portfolio: PortfolioResult
    metrics: dict[str, float]


def _compute_drawdown_series(equity: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (equity - peak) / peak


def equity_metrics(
    equity: np.ndarray,
    step_log_returns: np.ndarray,
    *,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """
    Summary statistics for a daily equity curve.

    Assumes ``step_log_returns`` are **per trading day** (``periods_per_year`` defaults to 252).
    """
    eq = np.asarray(equity, dtype=np.float64)
    lr = np.asarray(step_log_returns, dtype=np.float64)
    valid = np.isfinite(lr)
    lr_clean = lr[valid]

    n_steps = len(eq) - 1
    total_return = float(eq[-1] / eq[0] - 1.0) if eq[0] > 0 else float("nan")

    years = n_steps / periods_per_year if n_steps > 0 else 0.0
    if years > 0 and eq[0] > 0 and eq[-1] > 0:
        ann_return = float((eq[-1] / eq[0]) ** (1.0 / years) - 1.0)
    else:
        ann_return = float("nan")

    if lr_clean.size > 1:
        vol = float(np.std(lr_clean, ddof=1) * np.sqrt(periods_per_year))
        mean_lr = float(np.mean(lr_clean))
        sharpe = float((mean_lr / np.std(lr_clean, ddof=1)) * np.sqrt(periods_per_year))
    else:
        vol = float("nan")
        sharpe = float("nan")

    dd = _compute_drawdown_series(eq)
    max_dd = float(np.nanmin(dd))

    return {
        "n_steps": float(n_steps),
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


def run_buy_and_hold_on_split(
    parquet_path: str | Path,
    *,
    split_name: str = "split",
    portfolio_cfg: PortfolioConfig | None = None,
    log_return_column: str = "log_return_next_1d",
    timestamp_column: str = "timestamp",
) -> BuyAndHoldReport:
    """
    Evaluate always-long BTC on one processed feature split.

    Rows without a valid ``log_return_next_1d`` (typically the last row) are dropped.
    """
    path = Path(parquet_path)
    if not path.is_file():
        raise FileNotFoundError(f"Parquet not found: {path}")

    df = pd.read_parquet(path).sort_values(timestamp_column).reset_index(drop=True)
    if log_return_column not in df.columns:
        raise KeyError(
            f"Column {log_return_column!r} missing; run make_features / split_dataset first."
        )

    sub = df.dropna(subset=[log_return_column]).copy()
    log_r = sub[log_return_column].to_numpy(dtype=np.float64)
    price_rel = np.exp(log_r)
    n = price_rel.shape[0]
    positions = np.ones(n, dtype=np.float64)

    base = portfolio_cfg if portfolio_cfg is not None else PortfolioConfig()
    # Benchmark is "always long from the first bar": no initial flat->long turnover fee.
    cfg = replace(base, initial_position=1.0)
    result = simulate_portfolio(price_rel, positions, cfg)

    ts = sub[timestamp_column].reset_index(drop=True)

    m = equity_metrics(result.equity, result.step_log_returns)
    return BuyAndHoldReport(
        split_name=split_name,
        source_path=path.resolve(),
        timestamps=ts,
        equity=result.equity,
        step_log_returns=result.step_log_returns,
        portfolio=result,
        metrics=m,
    )


def run_buy_and_hold_all_splits(
    *,
    train_path: Path = DEFAULT_TRAIN,
    val_path: Path = DEFAULT_VAL,
    test_path: Path = DEFAULT_TEST,
    portfolio_cfg: PortfolioConfig | None = None,
) -> dict[str, BuyAndHoldReport]:
    """Run buy-and-hold on train, val, and test parquets (if present)."""
    out: dict[str, BuyAndHoldReport] = {}
    for name, path in (("train", train_path), ("val", val_path), ("test", test_path)):
        if path.is_file():
            out[name] = run_buy_and_hold_on_split(
                path,
                split_name=name,
                portfolio_cfg=portfolio_cfg,
            )
    return out


def print_buy_and_hold_summary(reports: dict[str, BuyAndHoldReport]) -> None:
    """Pretty-print metrics for each split."""
    for name, rep in reports.items():
        print(f"\n=== Buy-and-hold ({name}) - {rep.source_path.name} ===")
        for k, v in rep.metrics.items():
            if k == "n_steps":
                print(f"  {k}: {int(v)}")
            else:
                print(f"  {k}: {v:.6g}")


def main() -> None:
    reports = run_buy_and_hold_all_splits()
    if not reports:
        print("No processed splits found under data/processed/. Run Step 2 pipeline first.")
        return
    print_buy_and_hold_summary(reports)


if __name__ == "__main__":
    main()

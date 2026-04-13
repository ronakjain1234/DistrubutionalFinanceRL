"""Evaluate all DistCQL × CQL seed combinations as ensembles."""

from __future__ import annotations
import numpy as np
from pathlib import Path

from src.agents.distributional_qnet import DistributionalCQLAgent
from src.agents.cql import load_model as load_cql
from src.experiments.eval_ensemble import EnsemblePolicy, run_rollout, POSITION_LEVELS
from src.experiments.baselines import run_buy_and_hold_on_split

DIST_CQL_MODELS = {
    "s42": "models/dist_cql_alpha05/model_best",
    "s1":  "models/dist_cql_alpha05_seed1/model_best",
    "s2":  "models/dist_cql_alpha05_seed2/model_best",
    "s3":  "models/dist_cql_alpha05_seed3/model_best",
    "s4":  "models/dist_cql_alpha05_seed4/model_best",
    "s5":  "models/dist_cql_alpha05_seed5/model_best",
}

CQL_MODELS = {
    "s42": "models/cql/model_best.d3",
    "s1":  "models/cql_seed1/model_best.d3",
    "s2":  "models/cql_seed2/model_best.d3",
    "s3":  "models/cql_seed3/model_best.d3",
    "s4":  "models/cql_seed4/model_best.d3",
    "s5":  "models/cql_seed5/model_best.d3",
}

TEST_PATH = "data/processed/btc_daily_test.parquet"
VAL_PATH = "data/processed/btc_daily_val.parquet"
LOG_RET_COL = "log_return_next_1d"
PERIODS = 252


def main():
    # Load all models once
    print("Loading models...")
    dist_models = {}
    for name, path in DIST_CQL_MODELS.items():
        print(f"  DistCQL {name}: {path}")
        dist_models[name] = DistributionalCQLAgent.load(path)

    cql_models = {}
    for name, path in CQL_MODELS.items():
        print(f"  CQL {name}: {path}")
        cql_models[name] = load_cql(path)

    # Buy and hold reference
    bh_test = run_buy_and_hold_on_split(TEST_PATH, split_name="test", log_return_column=LOG_RET_COL, periods_per_year=PERIODS)
    bh_val = run_buy_and_hold_on_split(VAL_PATH, split_name="val", log_return_column=LOG_RET_COL, periods_per_year=PERIODS)
    print(f"\nBuy & Hold — val: {bh_val.metrics['total_return']:+.1%}  test: {bh_test.metrics['total_return']:+.1%}")

    # Evaluate all combinations
    print(f"\n{'='*100}")
    print(f"  ENSEMBLE GRID: DistCQL(alpha=0.5) × CQL — average_position")
    print(f"{'='*100}")

    # Header
    cql_keys = list(CQL_MODELS.keys())
    header = f"{'DistCQL \\ CQL':>18s}"
    for ck in cql_keys:
        header += f"  {ck:>14s}"
    print(f"\n--- TEST SET RETURNS ---")
    print(header)

    results = {}
    for dk, d_model in dist_models.items():
        row = f"{'DistCQL '+dk:>18s}"
        for ck, c_model in cql_models.items():
            ensemble = EnsemblePolicy(
                [d_model, c_model], [f"DistCQL_{dk}", f"CQL_{ck}"],
                [False, True], strategy="average_position"
            )
            m_test, _ = run_rollout(ensemble, TEST_PATH, "test", LOG_RET_COL, PERIODS)
            m_val, _ = run_rollout(ensemble, VAL_PATH, "val", LOG_RET_COL, PERIODS)
            results[(dk, ck)] = {"test": m_test, "val": m_val}
            row += f"  {m_test['total_return']:>+13.1%}"
        print(row)

    # Also print Sharpe grid
    print(f"\n--- TEST SET SHARPE ---")
    print(header)
    for dk in dist_models:
        row = f"{'DistCQL '+dk:>18s}"
        for ck in cql_models:
            m = results[(dk, ck)]["test"]
            row += f"  {m['sharpe']:>14.2f}"
        print(row)

    # Print short fraction grid
    print(f"\n--- TEST SET SHORT % ---")
    print(header)
    for dk in dist_models:
        row = f"{'DistCQL '+dk:>18s}"
        for ck in cql_models:
            m = results[(dk, ck)]["test"]
            row += f"  {m['action_frac_short']:>13.0%}"
        print(row)

    # Summary stats
    test_returns = [results[k]["test"]["total_return"] for k in results]
    test_sharpes = [results[k]["test"]["sharpe"] for k in results]
    print(f"\n--- SUMMARY (36 ensemble combinations) ---")
    print(f"  Test Return: mean={np.mean(test_returns):+.1%}  std={np.std(test_returns):.1%}  "
          f"min={np.min(test_returns):+.1%}  max={np.max(test_returns):+.1%}  "
          f"median={np.median(test_returns):+.1%}")
    print(f"  Test Sharpe: mean={np.mean(test_sharpes):.2f}  std={np.std(test_sharpes):.2f}  "
          f"min={np.min(test_sharpes):.2f}  max={np.max(test_sharpes):.2f}  "
          f"median={np.median(test_sharpes):.2f}")
    beat_bh = sum(1 for r in test_returns if r > bh_test.metrics["total_return"])
    print(f"  Beat buy-and-hold ({bh_test.metrics['total_return']:+.1%}): {beat_bh}/36 ({beat_bh/36:.0%})")

    # Val set grid too
    print(f"\n--- VAL SET RETURNS ---")
    print(header)
    for dk in dist_models:
        row = f"{'DistCQL '+dk:>18s}"
        for ck in cql_models:
            m = results[(dk, ck)]["val"]
            row += f"  {m['total_return']:>+13.1%}"
        print(row)

    print()


if __name__ == "__main__":
    main()

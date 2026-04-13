# Hourly 7-Action Training Results (v1)

**Date:** 2026-04-12
**Dataset:** `offline_dataset_hourly_train.npz` (903,302 transitions, 23 policies, 7 actions)
**Data splits:** train 2017-07-01 to 2021-12-31 | val 2022-01-01 to 2023-12-31 | test 2024-01-01 to 2025-12-31

---

## Configuration

### Data Pipeline
- **Source:** Coinbase hourly OHLCV candles (74,506 rows)
- **Features:** 35 features including OHLC intrabar (close_in_range, bar_body_ratio, log_hl_range), efficient volatility (Parkinson, Garman-Klass), Bollinger %B, return autocorrelation, supply/demand zones, calendar encodings
- **Train split:** 39,274 hourly bars
- **Normalization:** z-score fitted on train only, applied to all splits

### Offline Dataset
- **Behavior policies:** 23 total (14 core + 8 epsilon-greedy + 1 mixture)
  - 7 original: BuyAndHold, Random, TrendFollowing, MeanReversion, MACDCrossover, SupplyDemand, VolatilityRegime
  - 4 OHLC-aware: BollingerBreakout, CandlePattern, ParkinsonVolRegime, AutocorrRegime
  - 3 fractional-position: VolSizedTrend, BollingerSized, GradualPosition
  - 8 epsilon-greedy wrappers (eps=0.15) + 1 mixture (14 components)
- **Action space:** 7 discrete levels {-1.0, -0.5, -0.25, 0.0, +0.25, +0.5, +1.0}
- **Reward:** log-return with drawdown penalty (lambda=1.0, threshold=2%)
- **Action coverage in dataset:** short 28.8%, half_short 3.1%, qtr_short 3.0%, flat 24.4%, qtr_long 3.0%, half_long 3.8%, long 33.9%

### Hyperparameters (shared across models)
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| gamma | 0.99 | Increased from 0.95 (daily). Effective horizon = 100 hours (~4 days), comparable to daily agent's 20-day horizon. Prevents myopic churning with transaction costs. |
| batch_size | 256 | Unchanged from daily |
| n_steps (DQN) | 50,000 | Increased from 20K. Dataset is 46x larger than daily. |
| n_steps (CQL/DistCQL) | 80,000 | Increased from 30K for same reason. |
| early_stopping | patience=3 (DQN), 4 (CQL/DistCQL) | On val Sharpe |

### Model-specific settings
- **DQN:** DoubleDQN, hidden=[128,128], lr=3e-4, n_critics=3, dropout=0.1
- **CQL:** alpha=1.0, hidden=[256,256], lr=1e-4, n_critics=3, dropout=0.1
- **DistCQL:** alpha=1.0, alpha_tail=0.5, n_quantiles=51, tail_focus=True, hidden=[256,256], lr=1e-4, n_ensemble=3

---

## Results

### Performance Summary

| Model | Split | Total Return | Ann. Return | Ann. Vol | Sharpe | Max Drawdown |
|-------|-------|-------------|-------------|----------|--------|-------------|
| **DQN** | val | +5.8% | +2.8% | 14.5% | 0.19 | -24.7% |
| **DQN** | **test** | **+44.8%** | **+20.4%** | **13.2%** | **1.40** | **-10.5%** |
| CQL | val | +90.3% | +38.0% | 51.8% | 0.62 | -61.5% |
| CQL | test | -85.7% | -62.1% | 48.6% | -2.00 | -88.6% |
| DistCQL | val | +100.1% | +41.5% | 52.5% | 0.66 | -48.9% |
| DistCQL | test | +2.7% | +1.4% | 48.3% | 0.03 | -54.1% |
| Buy & Hold | val | -9.6% | -4.9% | 54.6% | -0.09 | -67.4% |
| Buy & Hold | test | +107.7% | +44.1% | 49.0% | 0.74 | -34.9% |

### Action Distribution

| Model | Split | short | half_short | qtr_short | flat | qtr_long | half_long | long |
|-------|-------|-------|-----------|-----------|------|----------|-----------|------|
| DQN | val | 0% | 7.6% | 0% | 0% | 92.4% | 0% | 0% |
| DQN | test | 0% | 5.5% | 0% | 0% | 94.5% | 0% | 0% |
| CQL | val | 42.1% | 0% | 0% | 2.1% | 0% | 0% | 55.8% |
| CQL | test | 42.1% | 0% | 0% | 0.5% | 0% | 0% | 57.3% |
| DistCQL | val | 21.2% | 0% | 0% | 1.2% | 0% | 0% | 77.6% |
| DistCQL | test | 11.9% | 0% | 0% | 1.3% | 0% | 0% | 86.9% |

### Early Stopping

| Model | Best Epoch | Best Val Sharpe | Total Epochs Run |
|-------|-----------|----------------|------------------|
| DQN | 8 (step 16K) | 0.19 | 25 |
| CQL | 2 (step 4K) | 0.26 | 40 |
| DistCQL | ~early | 0.66 | 40 |

---

## Analysis

### DQN: Conservative generalization
DQN is the only model that generalizes to the test set. Its Sharpe of 1.40 on test with only -10.5% max drawdown is the best risk-adjusted result across all models and splits, including buy-and-hold.

It achieved this by discovering a **low-exposure strategy**: staying mostly at quarter-long (+0.25) with occasional half-short (-0.5) hedging. This reduces annualized volatility to ~13% (vs ~50% for the other models and buy-and-hold), which drives the strong risk-adjusted return despite a lower absolute return (+44.8% vs buy-and-hold's +107.7%).

DQN is also the **only model using intermediate position sizes** meaningfully. The fractional action space is working as intended here.

### CQL: Severe overfitting
CQL shows the worst val-to-test gap: +90.3% on val, -85.7% on test. It learned a binary short/long switching strategy that happened to work during the 2022-2023 bear market (val period) but catastrophically failed during the 2024-2025 bull market (test period).

The conservatism penalty (alpha=1.0) was insufficient to prevent overfitting to the val-period regime. The Q-gap metric was also high (1.0-1.4), suggesting sharp action preferences that don't generalize.

### DistCQL: Moderate overfitting
DistCQL had the best val performance (+100.1%) but essentially went flat on test (+2.7%). Less catastrophic than CQL, but still a clear overfitting signal. Its action distribution shifted heavily toward long on test (87%) but didn't capture the bull market return that buy-and-hold achieved.

The distributional structure (quantile regression) may have provided some regularization vs vanilla CQL, preventing the catastrophic loss, but wasn't enough to generalize.

### Intermediate action usage
The 7-action space was only meaningfully utilized by DQN. CQL and DistCQL collapsed to binary short/long strategies, ignoring intermediate positions entirely despite:
- 3 fractional-position behavior policies in the dataset
- ~16% of training data covering intermediate actions
- Epsilon-greedy exploration at intermediate levels

This collapse is likely because the Q-function for intermediate actions had less training signal (3% coverage each vs 25-34% for short/long/flat), and the conservative penalties in CQL/DistCQL further suppressed these less-represented actions.

---

## Open Questions for Next Iteration

1. **Why does DQN generalize while CQL/DistCQL overfit?** DQN's simpler objective may act as implicit regularization. CQL's conservative penalty may be creating a false sense of safety on val while learning regime-specific patterns.

2. **Can we increase intermediate action coverage?** The 3% per-level coverage may be too thin. Options: increase weight of fractional policies in the mixture, add more fractional behavior policies, or use importance sampling.

3. **Is CQL alpha too low or too high?** alpha=1.0 with 7 actions penalizes more total probability mass than with 3 actions. May need recalibration.

4. **Should gamma be even higher?** DQN's conservative strategy suggests it's learning to avoid transaction costs, which is good. But CQL/DistCQL's churning between short/long suggests they may not be planning far enough ahead despite gamma=0.99.

5. **Would a larger network help?** 35 features + 7 actions may benefit from more capacity, but capacity without regularization could worsen overfitting for CQL/DistCQL.

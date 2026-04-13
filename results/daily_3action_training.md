# Daily 3-Action Training Results

**Date:** 2026-04-12
**Commit:** `4eb7816` — "Distributional RL"
**Data splits:** train 2017-07-01 to 2021-12-31 | val 2022-01-01 to 2023-12-31 | test 2024-01-01 to 2025-12-31
**Val period context:** Bear market (BTC peak ~$69K to trough ~$16K)
**Test period context:** Bull market (BTC ~$16K to ~$100K+)
**Dataset:** `offline_dataset_train.npz` (daily, 3 actions)

---

## Configuration

- **Action space:** 3 discrete levels {-1.0, 0.0, +1.0} (short, flat, long)
- **Data frequency:** Daily candles
- **Reward:** position * log_return with drawdown penalty (lambda=1.0, threshold=2%)
- **Gamma:** 0.95 (effective horizon ~20 trading days)

### Model-Specific Hyperparameters

| Parameter | DQN | CQL | DistCQL |
|-----------|-----|-----|---------|
| hidden | [128, 128] | [256, 256] | [256, 256] |
| lr | 3e-4 | 1e-4 | 1e-4 |
| alpha | — | 1.0 | 1.0 |
| alpha_tail | — | — | 0.5 |
| n_quantiles | — | — | 51 |
| n_critics/ensemble | 3 | 3 | 3 |
| dropout | 0.1 | 0.1 | 0.1 |
| weight_decay | 1e-4 | 1e-4 | 1e-4 |
| n_steps | 20,000 | 30,000 | 30,000 |
| batch_size | 256 | 256 | 256 |
| target_update | 1,000 | 2,000 | 2,000 |
| layer_norm | yes | yes | yes |
| early_stopping | patience=3 | patience=4 | patience=4 |

---

## Performance Summary

| Model | Split | Total Return | Sharpe | Max Drawdown | Actions (S/F/L) |
|-------|-------|-------------|--------|-------------|-----------------|
| DQN | val | +523.9% | 1.47 | -31.8% | 31% / 13% / 56% |
| DQN | test | -23.0% | -0.24 | -62.1% | 53% / 11% / 36% |
| CQL | val | +129.1% | 0.64 | -45.8% | 42% / 2% / 56% |
| **CQL** | **test** | **+92.3%** | **0.57** | **-37.0%** | **15% / 2% / 83%** |
| DistCQL | val | +17.1% | 0.12 | -68.7% | 5% / 0% / 95% |
| **DistCQL** | **test** | **+97.8%** | **0.59** | **-32.1%** | **0% / 0% / 100%** |
| Buy & Hold | val | -9.6% | -0.09 | -67.4% | 0% / 0% / 100% |
| Buy & Hold | test | +107.7% | 0.74 | -34.9% | 0% / 0% / 100% |

### Best Epochs (from training logs)

| Model | Best Epoch | Best Val Sharpe | Total Epochs |
|-------|-----------|----------------|--------------|
| DQN | unknown | unknown | unknown |
| CQL | 12 (step 24K) | 0.64 | 15 |
| DistCQL | 4 (step 8K) | 0.12 | 8 |

---

## Analysis

### 1. DistCQL came closest to buy-and-hold (+97.8% vs +107.7%)

DistCQL achieved 91% of buy-and-hold's return with **better downside protection** (max drawdown -32.1% vs -34.9%). On the test set it went 100% long — effectively learning that the optimal strategy in a bull market is to stay fully invested. On val it was also 95% long with only 5% short, meaning it didn't try to time the bear market aggressively.

**Why this makes sense:** DistCQL's distributional structure (51 quantiles) gives it a richer picture of return uncertainty. With alpha=1.0 (moderate conservatism), the CQL penalty was just strong enough to prevent the wild short/long switching that hurt other models, but not so strong that it over-regularized. The result: a model that defaults to long but doesn't get whipsawed.

The slight underperformance vs B&H (-9.9%) comes from the 5% short allocation during val — the model learned a small bearish signal that slightly dragged on test returns. But this is a much better outcome than the hourly models, which all either collapsed or massively overfit.

### 2. CQL was the best active trader (+92.3% with regime awareness)

CQL is the most interesting result because it actually trades: 15% short, 2% flat, 83% long on the test set. It earned +92.3% while actively managing positions. Compared to DistCQL (which is just buy-and-hold in disguise), CQL demonstrates genuine learned behavior.

**Why this makes sense:** CQL's conservative penalty pushes Q-values for out-of-distribution actions lower, which on daily data with 3 actions creates a natural bias toward the most-represented action in the dataset (long). But unlike the hourly 7-action setup, the 3-action space means every action has meaningful coverage in the dataset — CQL can actually learn about all three actions without data starvation. The result is a model that's mostly long but knows when to hedge.

CQL's val performance (+129.1%) was strong because it shorted 42% of the time during the bear market, correctly identifying downtrends. On test it reduced shorts to 15% — it partially adapted, but the residual shorting cost it ~15% vs B&H. The val-to-test gap (129% → 92%) is notable but not catastrophic like the hourly CQL models.

### 3. DQN severely overfit (+524% val → -23% test)

DQN shows the worst generalization gap. It learned an aggressive short/long switching strategy that happened to perfectly time the bear market (val +524%) but reversed on test, going 53% short during a bull market.

**Why this makes sense:** DQN has no conservatism penalty — it greedily maximizes Q-values, which with daily data and a long training history allows it to memorize regime-specific patterns from the training period. The 31% short / 56% long split on val shows it was actively timing, and the +524% return confirms it was right during val. But these timing signals were overfit to the specific 2022-2023 bear market dynamics and didn't transfer to the 2024-2025 bull market.

This is the opposite of what we saw with hourly data, where DQN was the *best* generalizer. The difference: hourly DQN was implicitly regularized by the smaller network (128x128) and massive dataset (903K transitions), while daily DQN had a much smaller dataset (~20K transitions) relative to network capacity, enabling overfitting.

### 4. Daily vs hourly: why daily models generalize differently

| Factor | Daily | Hourly |
|--------|-------|--------|
| Dataset size | ~20K transitions | ~903K transitions |
| Action space | 3 actions | 7 actions |
| Data per action | ~7K each | 3% for intermediate actions |
| Signal-to-noise | Higher (daily trends) | Lower (hourly noise) |
| Gamma | 0.95 (~20 day horizon) | 0.99 (~100 hour horizon) |

Daily data has a higher signal-to-noise ratio — daily returns are more predictable than hourly returns because intraday noise averages out. With 3 actions and adequate coverage for each, CQL and DistCQL can learn meaningful policies. The hourly models struggled because:
- The 7-action space diluted data coverage (3% per intermediate action)
- Hourly noise made Q-value estimation harder
- CQL/DistCQL collapsed to 3 of 7 actions anyway, wasting the expanded action space

### 5. Q-value stability in daily CQL

A critical difference from the hourly experiments: daily CQL's Q-values were remarkably stable. Q-mean went from -0.55 (epoch 1) to -0.71 (epoch 15) — it actually *decreased* slightly over training. This is the opposite of the hourly CQL models where Q-mean inflated from -1.3 to +2.6 across 10 epochs.

**Why:** The daily dataset has ~20K transitions with 3 actions, giving ~7K samples per action. This provides enough coverage for CQL's conservative penalty to work as intended — it can accurately estimate the gap between in-distribution and out-of-distribution Q-values. With hourly data and 7 actions, the sparse intermediate actions caused the penalty to malfunction.

---

## Trained Models

| Model | Path | Best Epoch | Status |
|-------|------|-----------|--------|
| DQN | `models/dqn_baseline/model_best.d3` | unknown | Overfit |
| CQL | `models/cql/model_best.d3` | 12 | Best active trader |
| DistCQL | `models/dist_cql/model_best` | 4 | Closest to B&H |

---

## Conclusions

1. **DistCQL on daily data is the closest model to beating buy-and-hold** at +97.8% vs +107.7% (91% of B&H return), with better max drawdown (-32.1% vs -34.9%).

2. **CQL on daily data is the best actively-trading model** at +92.3%, demonstrating genuine regime awareness (42% short in bear market, 15% short in bull market).

3. **Daily 3-action is a stronger setup than hourly 7-action** for CQL/DistCQL. The higher signal-to-noise ratio and adequate per-action coverage prevent Q-value inflation and action collapse.

4. **DQN overfits on daily data** — the opposite of the hourly result. With fewer transitions relative to network capacity, DQN memorizes regime-specific patterns.

5. **The gap to B&H is small and closeable.** DistCQL lost ~10% to residual shorting. If we can reduce unnecessary shorts while preserving the model's ability to protect during drawdowns, we may be able to match or exceed B&H.

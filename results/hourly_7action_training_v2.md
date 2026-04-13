# Hourly 7-Action Training Results (v2)

**Date:** 2026-04-12
**Commit:** `2027218` — "V2 of new changes"
**Models trained:** CQL v2, DistCQL v2

---

## Configuration Changes from v1

### DistCQL v2 hyperparameter changes

| Parameter | v1 | v2 | Rationale |
|-----------|----|----|-----------|
| alpha | 1.0 | 4.0 | 4x stronger CQL conservatism penalty to combat overfitting |
| alpha_tail | 0.5 | 2.0 | 4x stronger tail penalty |
| n_quantiles | 51 | 21 | Fewer quantiles for less capacity / more regularization |
| hidden_units | [256, 256] | [128, 128] | Halved network size to reduce overfitting |
| learning_rate | 1e-4 | 3e-4 | 3x higher LR |
| weight_decay | 0.0 | 0.001 | Added L2 regularization |
| n_steps | 80,000 | 20,000 | 4x fewer training steps |
| early_stopping_patience | 4 | 6 | More patience before stopping |
| use_layer_norm | false | true | Added layer normalization |
| clip_grad_norm | none | 1.0 | Added gradient clipping |
| action_selection | (default) | mean | Explicit mean-based action selection |

### CQL v2

CQL v2 was trained for 20 epochs (20,000 steps). No config file was saved, but training log metrics are available for analysis.

---

## Training Dynamics

### CQL v2 — Val Sharpe Over Epochs

| Epoch | Val Sharpe | Val Return | Max Drawdown | Q-mean | Conservatism Proxy |
|-------|-----------|------------|-------------|--------|-------------------|
| 1 | -1.133 | -70.4% | -82.0% | -1.30 | 1.045 |
| 6 | **0.447** | +61.4% | -69.7% | 0.21 | 0.802 |
| 10 | -0.914 | -61.9% | -73.4% | 1.20 | 0.838 |
| **14** | **0.585** | **+88.7%** | **-48.0%** | 1.85 | 0.780 |
| 17 | 0.469 | +66.3% | -68.1% | 2.25 | 0.786 |
| 20 | -0.404 | -34.6% | -57.0% | 2.65 | 0.801 |

**Best model:** Epoch 14, val Sharpe = 0.585

### DistCQL v2 — Val Sharpe Over Epochs

| Epoch | Val Sharpe | Val Return | Max Drawdown | Q-mean | Quantile Spread | CVaR-10 |
|-------|-----------|------------|-------------|--------|----------------|---------|
| **1** | **0.247** | **+30.0%** | **-51.3%** | -0.47 | 12.0 | -1.09 |
| 2 | 0.096 | +10.6% | -65.9% | 1.02 | 21.3 | -1.13 |
| 3 | -0.377 | -33.2% | -75.2% | 2.71 | 29.3 | -1.36 |
| 4 | -0.162 | -15.9% | -72.1% | 4.33 | 36.4 | -1.73 |
| 5 | -0.530 | -43.1% | -80.0% | 5.91 | 42.3 | -2.31 |
| 6 | -0.347 | -31.1% | -72.5% | 7.55 | 47.9 | -3.08 |
| 7 | -0.573 | -46.4% | -70.7% | 9.27 | 53.5 | -4.03 |

**Best model:** Epoch 1, val Sharpe = 0.247 (model never improved after initialization)

---

## Overfitting Evidence

### 1. CQL v2: Extreme val Sharpe volatility

Val Sharpe oscillates wildly across epochs, ranging from -1.13 to +0.58 — a spread of 1.7 Sharpe units. This is not the behavior of a model learning a stable strategy. Instead, the model cycles through different regime-specific bets:

- **Positive Sharpe epochs** (6, 14, 17, 19): Correspond to long-biased action distributions (55-75% long). These epochs happened to profit from directional moves in the val period.
- **Negative Sharpe epochs** (1, 10, 12, 13): Correspond to short-biased distributions (55-66% short). These caught the wrong side.

The best checkpoint (epoch 14, Sharpe 0.585) is likely a high-water mark in a random walk, not a sign of genuine learning. Evidence: the very next epoch (15) collapses to -0.186 while going 99% long, then epoch 16 stays 99% long at -0.325.

### 2. CQL v2: Monotonically rising Q-values

Q-mean increases steadily from -1.30 (epoch 1) to 2.65 (epoch 20), while val performance does NOT improve monotonically. This is classic Q-value overestimation: the model becomes increasingly confident in its value estimates even as real performance deteriorates. The CQL penalty (measured by conservatism_proxy dropping from 1.05 to 0.80) is failing to counteract this inflation.

### 3. DistCQL v2: Q-value divergence despite 4x alpha

Despite quadrupling the CQL alpha from 1.0 to 4.0, DistCQL Q-means still explode from -0.47 to 9.27 in just 7 epochs — a 20x increase. This is far worse than CQL v2's Q-inflation rate, and it happened in a third of the training time.

The quantile spread (a measure of distributional uncertainty) expands from 12.0 to 53.5 (4.4x), meaning the return distribution becomes increasingly diffuse rather than converging. The model is not learning tighter predictions — it is becoming *more* uncertain as it trains longer.

### 4. DistCQL v2: Best model at epoch 1

The best DistCQL model was saved at the very first checkpoint (step 1000). All 6 subsequent epochs made the model strictly worse. This is a strong signal that the model architecture/hyperparameters cannot productively use more training on this data — additional gradient steps are destructive.

### 5. DistCQL v2: CVaR collapse

CVaR-10 (the mean of the worst 10% of predicted returns) drops from -1.09 to -4.03 across training. This means the model's tail-risk estimates are becoming increasingly extreme and unreliable. For a distributional RL agent meant to be risk-aware, this is counterproductive — the tail risk signal is being distorted by Q-value divergence.

### 6. Action collapse persists (both models)

Both CQL v2 and DistCQL v2 collapse to only 3 of 7 actions: action 0 (full short), action 3 (flat), and action 6 (full long). Actions 1, 2, 4, 5 are used at exactly 0% throughout training. This is identical to the v1 failure mode.

The v2 hyperparameter changes (smaller network, fewer quantiles, more regularization) did not address this. The intermediate actions remain suppressed because:
- They have ~3% coverage each in the dataset vs 25-34% for the dominant actions
- CQL's penalty further suppresses under-represented actions
- Higher alpha (4.0) may have actually *worsened* this by penalizing OOD actions more aggressively

---

## Comparison: v1 vs v2

### CQL

| Metric | v1 | v2 |
|--------|----|----|
| Best val Sharpe | 0.26 (epoch 2) | 0.585 (epoch 14) |
| Best val return | +90.3% | +88.7% |
| Val max drawdown | -61.5% | -48.0% |
| Test return | -85.7% | not yet evaluated |
| Action diversity | 3 of 7 | 3 of 7 |

CQL v2 achieved a higher val Sharpe (0.585 vs 0.26), but this came at epoch 14 of a highly volatile trajectory. The v1 best was at epoch 2 — early and less likely to be overfitting to lucky timing. The v2 result is suspicious: a model that oscillates between -1.13 and +0.58 Sharpe is not stable.

### DistCQL

| Metric | v1 | v2 |
|--------|----|----|
| Best val Sharpe | 0.66 | 0.247 |
| Best val return | +100.1% | +30.0% |
| Val max drawdown | -48.9% | -51.3% |
| Test return | +2.7% | not yet evaluated |
| Q-mean at best epoch | ~moderate | -0.47 |
| Quantile spread at best | ~moderate | 12.0 |
| Action diversity | 3 of 7 | 3 of 7 |

DistCQL v2 performed *worse* on val than v1 despite the regularization changes. The smaller network and higher alpha may have over-regularized, preventing the model from learning anything useful beyond the first epoch. However, the v1 model's 0.66 val Sharpe collapsed to 0.03 on test, so v2's lower val Sharpe is not necessarily bad — it could mean less overfitting if test performance holds up.

---

## Key Takeaways

1. **CQL conservatism penalty is not working.** Even at alpha=4.0, Q-values diverge and the model cycles through unstable strategies. The penalty architecture may need to change (e.g., Lagrangian dual / adaptive alpha rather than fixed).

2. **DistCQL training is destructive after epoch 1.** The model cannot productively use more than 1000 gradient steps. This suggests either the learning rate is too high (3e-4), the distributional loss is unstable with the current quantile count (21), or the alpha/alpha_tail penalties create conflicting gradients.

3. **Action collapse is a data problem, not a model problem.** Three different model architectures across two config versions all collapse to the same 3 actions. The ~3% per-action coverage for intermediate positions is too thin for any offline RL method to learn from, especially under CQL-style penalties that suppress OOD actions.

4. **Val Sharpe alone is an unreliable selection criterion.** CQL v2's best epoch (14) was chosen from a trajectory with 1.7 Sharpe units of noise. A smoothed or ensemble-based selection criterion (e.g., average of top-3 epochs, or val Sharpe + val drawdown jointly) would be more robust.

5. **Test set evaluation is needed.** Neither v2 model has been evaluated on test yet. Given the overfitting patterns, CQL v2's test performance is likely to collapse similarly to v1 (-85.7%). DistCQL v2's lower val performance may translate to less catastrophic test degradation — but this is speculative.

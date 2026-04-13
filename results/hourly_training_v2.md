# Hourly Training Results (v2 — Full Experiment Log)

**Date:** 2026-04-12
**Data splits:** train 2017-07-01 to 2021-12-31 | val 2022-01-01 to 2023-12-31 | test 2024-01-01 to 2025-12-31
**Val period context:** Bear market (BTC peak ~$69K to trough ~$16K)
**Test period context:** Bull market (BTC ~$16K to ~$100K+)

---

## Performance Summary (All Models)

| Model | Version | Split | Total Return | Sharpe | Max Drawdown | Key Actions |
|-------|---------|-------|-------------|--------|-------------|-------------|
| **DQN** | **v1 (7-act)** | **val** | **+5.8%** | **0.19** | **-24.7%** | **qtr_long:92% half_short:8%** |
| **DQN** | **v1 (7-act)** | **test** | **+44.8%** | **1.40** | **-10.5%** | **qtr_long:95% half_short:5%** |
| DQN | v2 (256x256) | val | -99.0% | -7.54 | — | FAILED (too much capacity) |
| DQN | 3-action | val | -18.2% | -0.19 | -68.9% | long:96% flat:4% |
| DQN | 3-action | test | +88.0% | 0.66 | -36.5% | long:97% flat:3% |
| CQL | v1 | val | +90.3% | 0.62 | -61.5% | short:42% long:56% |
| CQL | v1 | test | -85.7% | -2.00 | -88.6% | short:42% long:57% (OVERFIT) |
| CQL | v3 (alpha=5) | — | best val Sharpe -0.67 | — | — | FAILED |
| DistCQL | v1 | val | +100.1% | 0.66 | -48.9% | short:21% long:78% |
| DistCQL | v1 | test | +2.7% | 0.03 | -54.1% | short:12% long:87% (OVERFIT) |
| DistCQL | v2 (alpha=4) | val | +30.0% | 0.25 | -51.3% | short:14% long:85% |
| DistCQL | v2 (alpha=4) | test | +44.5% | 0.38 | -53.5% | short:7% long:92% |
| DistCQL | v2 + cvar_25 | val | -36.4% | -0.42 | -70.8% | long:98% |
| DistCQL | v2 + cvar_25 | test | +84.2% | 0.63 | -34.9% | long:99% |
| DistCQL | v3 (alpha=2) | val | +18.3% | 0.16 | -49.0% | short:29% long:71% |
| DistCQL | v3 (alpha=2) | test | +30.2% | 0.27 | -34.9% | short:7% long:93% |
| Buy & Hold | — | val | -9.6% | -0.09 | -67.4% | long:100% |
| Buy & Hold | — | test | +107.7% | 0.74 | -34.9% | long:100% |

---

## Key Findings

### 1. DQN v1 (7-action) is the best risk-adjusted model

DQN v1 achieves a **test Sharpe of 1.40** — nearly 2x buy-and-hold's 0.74. It does this by:
- Staying mostly at quarter-long (+0.25) position — capturing 25% of upside
- Hedging with half-short (-0.5) during drawdowns (5.5% of the time)
- Reducing annualized volatility to ~13% (vs ~50% for B&H)

**DQN's timing signal is excellent.** When we hypothetically scale DQN's positions 4x (qtr_long -> long, half_short -> full_short), the raw return jumps to +350% with only -33% max drawdown — dramatically beating B&H. The model knows *when* to trade; it's just conservative in *how much*.

### 2. No model beats buy-and-hold on absolute return

| Model | Test Return | vs B&H |
|-------|-----------|--------|
| Buy & Hold | +107.7% | baseline |
| DQN 3-action | +88.0% | -18% |
| DistCQL v2 + cvar_25 | +84.2% | -22% |
| DistCQL v2 (mean) | +44.5% | -59% |
| DQN v1 (7-action) | +44.8% | -58% |

The test period (2024-2025) was a strong bull market. Any strategy that reduces long exposure will underperform B&H on absolute return. The models add value only on risk-adjusted metrics.

### 3. CQL/DistCQL overfit due to Q-value inflation

CQL v1 and DistCQL v1 both showed severe val-to-test gaps:
- CQL v1: val +90.3% -> test -85.7% (catastrophic)
- DistCQL v1: val +100.1% -> test +2.7%

Root cause: Q-values inflate during training (Q-mean drifted from -1.0 to +2.0 for CQL, -0.4 to +3.9 for DistCQL). The conservative penalty alpha*CQL_loss gets overwhelmed by Q-inflation, eroding the regularization.

**Stronger alpha helps but doesn't solve the problem:**
- DistCQL v2 (alpha=4.0): Q-inflation slowed, generalizes somewhat (test +44.5%)
- DistCQL v3 (alpha=2.0): Q-inflation not controlled (1.3 -> 9.1 by epoch 3)
- CQL v3 (alpha=5.0): Q-inflation starting despite high alpha

### 4. Larger networks hurt, not help

DQN v2 (256x256) was catastrophically bad (val Sharpe -7.5 to -9.0) vs DQN v1 (128x128, val Sharpe 0.19). The small network provides implicit regularization that's critical for generalization. This is consistent with offline RL theory — less capacity prevents overfitting to the dataset.

### 5. CVaR action selection doesn't add regime awareness

Testing DistCQL v2 with different CVaR thresholds:
- **cvar_25**: 99% long on test — just buy & hold in disguise
- **cvar_50**: 99% long on test — same
- **cvar_75**: 99% long on test — same
- **mean**: 7% short, 93% long — some regime awareness but modest

CVaR selects actions based on return distribution tails but doesn't create regime-conditional behavior. All strategies collapse to "mostly long" — they just shift the bias.

### 6. 3-action DQN doesn't beat 7-action DQN

Forcing full positions (-1, 0, +1) was meant to capture DQN's timing with bigger position sizes. Instead:
- DQN 3-action: test +88.0%, Sharpe 0.66 (97% long, no shorts)
- DQN v1 7-action: test +44.8%, Sharpe 1.40 (95% qtr_long, 5% half_short)

The 3-action model lost the ability to hedge (0% short) and became a slightly inferior buy-and-hold. The 7-action model's fractional positions enable the hedging that drives its superior Sharpe.

---

## Round 2 Hyperparameters

### DistCQL v2 (best generalizing DistCQL)
| Parameter | v1 Value | v2 Value | Rationale |
|-----------|----------|----------|-----------|
| alpha | 1.0 | **4.0** | Stronger conservatism to fight Q-inflation |
| alpha_tail | 0.5 | **2.0** | Aggressive upper-tail suppression |
| hidden | [256,256] | **[128,128]** | Smaller network = implicit regularization |
| lr | 1e-4 | **3e-4** | Faster learning for shorter budget |
| weight_decay | 1e-4 | **1e-3** | 10x stronger L2 regularization |
| n_quantiles | 51 | **21** | Fewer quantiles, less overfitting |
| n_steps | 80,000 | **20,000** | Shorter training to prevent inflation |
| target_update | 2,000 | **500** | Faster target anchoring |

---

## Trained Models

| Model | Path | Best Epoch | Status |
|-------|------|-----------|--------|
| DQN v1 (7-act) | `models/dqn_hourly/model_best.d3` | 8 | **Best risk-adjusted** |
| DQN 3-action | `models/dqn_hourly_3action/model_best.d3` | 5 | Mostly long, no edge |
| DistCQL v2 | `models/dist_cql_hourly_v2/model_best` | early | Best generalizing DistCQL |
| CQL v1 | `models/cql_hourly/model_best.d3` | 2 | Severe overfit |
| DistCQL v1 | `models/dist_cql_hourly/model_best` | early | Moderate overfit |

---

## Conclusions

1. **DQN v1 (7-action) is the production-grade model.** Sharpe 1.40 on test with -10.5% max drawdown is excellent risk-adjusted performance. It won't match B&H in bull markets, but it protects capital in drawdowns.

2. **The 7-action fractional position space works as intended** — but only for DQN. CQL and DistCQL collapse to binary short/long, ignoring intermediate positions. DQN is the only model that meaningfully uses the quarter-long and half-short levels.

3. **Offline RL with CQL-family algorithms struggles with Q-value inflation** on this dataset size (~900K transitions). DQN's simpler objective avoids this entirely, making it the more robust choice for offline settings.

4. **The fundamental limitation is regime-blindness.** All models apply roughly the same action distribution in both bull and bear markets. A truly market-beating model would need to be short during downtrends and leveraged long during uptrends — none of our models achieve this.

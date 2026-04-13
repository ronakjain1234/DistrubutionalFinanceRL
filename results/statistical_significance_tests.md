# Statistical Significance Analysis

**Date:** 2026-04-13
**Methodology:** Following Henderson et al. (2018) "Deep Reinforcement Learning that Matters"
**Tests:** 2-sample t-test, Kolmogorov-Smirnov test, bootstrap confidence intervals (10k resamples), power analysis
**Seeds per model:** 6 (seeds: 42, 1, 2, 3, 4, 5)
**Baseline:** Buy & Hold = +97.9% test return

---

## 1. Bootstrap 95% Confidence Intervals

### Test Returns

| Model | N | Mean Return | 95% CI Lower | 95% CI Upper |
|-------|---|------------|-------------|-------------|
| **CQL** | 6 | **+133.2%** | +66.0% | +194.2% |
| DistCQL (a=0.5) | 6 | +68.2% | -5.8% | +135.8% |
| DQN | 6 | +19.9% | -20.2% | +60.2% |
| Ensemble (all 36 combos) | 36 | +111.6% | +80.2% | +143.2% |
| Buy & Hold | — | +97.9% | (fixed) | (fixed) |

### Test Sharpe Ratios

| Model | Mean Sharpe | 95% CI Lower | 95% CI Upper |
|-------|------------|-------------|-------------|
| **CQL** | **0.67** | 0.33 | 0.94 |
| DistCQL (a=0.5) | 0.25 | -0.35 | 0.74 |
| DQN | 0.14 | -0.22 | 0.49 |

### Interpretation

- **CQL** is the only model whose 95% CI is entirely above zero for both return and Sharpe. Its CI for returns (+66% to +194%) is wide but confidently positive.
- **DistCQL's CI crosses zero** (-5.8% to +135.8%), meaning we cannot be confident it even produces positive returns across seeds. The high variance from seed sensitivity destroys statistical confidence.
- **DQN's CI is centered near zero** (-20% to +60%), confirming it adds no value on this dataset.
- **The ensemble's CI** (+80% to +143%) is tighter than individual models because N=36 combinations reduces sampling variance.

---

## 2. Does Each Model Beat Buy & Hold? (One-Sample t-Test)

| Model | Mean Return | t-statistic | p-value | Significant (p<0.05)? |
|-------|------------|------------|---------|----------------------|
| CQL | +133.2% | 0.966 | 0.379 | **NO** |
| DistCQL (a=0.5) | +68.2% | -0.758 | 0.482 | **NO** |
| DQN | +19.9% | -3.253 | 0.023 | **YES (worse than B&H)** |
| Ensemble (grid) | +111.6% | 0.849 | 0.402 | **NO** |

### Interpretation

**No model can statistically claim to beat buy-and-hold at the 95% confidence level.** This is the most important finding.

- **CQL's mean (+133.2%) is 35 points above B&H (+97.9%)**, but p=0.38 because the variance across seeds is too high relative to only 6 samples. The one bad seed (seed 5 at -13.1%) drags the significance.
- **DQN is the only significant result** — but it's significantly *worse* than B&H (p=0.023).
- **The ensemble grid** (N=36) still isn't significant because many combinations include bad DistCQL seeds that produce negative returns.

**Why this matters:** Even though CQL's mean return is 35% higher than B&H, we cannot rule out that this difference is due to random chance in seed selection. Henderson et al. (2018) specifically warn about this: "the variance between runs is enough to create statistically different distributions just from varying random seeds."

---

## 3. Pairwise Algorithm Comparisons

### Test Returns

| Comparison | Mean A | Mean B | t-stat | t-test p | KS-stat | KS p | Bootstrap % Diff (95% CI) |
|-----------|--------|--------|--------|---------|---------|------|--------------------------|
| **CQL vs DistCQL** | +133.2% | +68.2% | 1.21 | 0.253 | 0.33 | 0.931 | Not significant |
| **CQL vs DQN** | +133.2% | +19.9% | 2.59 | **0.027** | 0.67 | 0.143 | **Significant (t-test)** |
| **DistCQL vs DQN** | +68.2% | +19.9% | 1.05 | 0.317 | 0.50 | 0.474 | Not significant |

### Test Sharpe Ratios

| Comparison | Mean A | Mean B | t-stat | t-test p | Significant? |
|-----------|--------|--------|--------|---------|-------------|
| CQL vs DistCQL | 0.67 | 0.25 | 1.20 | 0.257 | NO |
| CQL vs DQN | 0.67 | 0.14 | 1.96 | 0.079 | NO |
| DistCQL vs DQN | 0.25 | 0.14 | 0.32 | 0.757 | NO |

### Interpretation

- **CQL vs DQN is the only significant pairwise difference** (t-test p=0.027). CQL is reliably better than DQN.
- **CQL vs DistCQL is NOT significant** (p=0.253). Despite CQL having nearly double the mean return, the variance within each group is too large to distinguish them with only 6 seeds.
- **No Sharpe ratio comparison is significant**, even CQL vs DQN (p=0.079). Sharpe ratios have even more variance across seeds than returns.
- **KS-test finds no significant differences** at all. This is expected with N=6 — the KS-test has low power with small samples.

---

## 4. Power Analysis

| Comparison | % Insignificant | % Positive Significant | % Negative Significant |
|-----------|----------------|----------------------|----------------------|
| CQL vs DistCQL | 78.9% | 21.0% | 0.1% |
| CQL vs DQN | 31.0% | 69.0% | 0.0% |
| DistCQL vs DQN | 80.7% | 19.1% | 0.2% |

### Interpretation

The power analysis simulates: "If we drew new seed samples from the same distributions, how often would we detect a 25% lift?"

- **CQL vs DQN:** We would detect a significant difference 69% of the time. This is the only comparison with reasonable statistical power.
- **CQL vs DistCQL:** Only 21% power — meaning 79% of the time, we'd fail to detect a difference even if CQL is truly better. **We need more seeds.**
- **DistCQL vs DQN:** Only 19% power — these two are practically indistinguishable statistically.

**How many seeds would we need?** As a rough guide, to achieve 80% power for the CQL vs DistCQL comparison (given their observed means and variances), we would need approximately 15-20 seeds per model.

---

## 5. Key Conclusions

### What we can claim with statistical confidence:

1. **CQL is significantly better than DQN** (p=0.027 on returns). This is the only robust pairwise finding.
2. **DQN is significantly worse than buy-and-hold** (p=0.023). DQN should not be used on this dataset.
3. **CQL's confidence interval for returns (+66% to +194%) is entirely positive.** While we can't prove it beats B&H, we can be confident it produces positive returns.

### What we CANNOT claim:

1. **We cannot claim CQL beats buy-and-hold** (p=0.379). The mean is higher (+133% vs +98%), but the variance across seeds is too large for N=6.
2. **We cannot claim CQL is better than DistCQL** (p=0.253). The difference is suggestive but not statistically significant.
3. **We cannot claim DistCQL beats DQN** (p=0.317).
4. **We cannot claim the ensemble beats buy-and-hold** (p=0.402).

### Honest assessment:

The results are *promising but not conclusive*. CQL shows the most consistent positive performance, but with only 6 seeds, the high variance in deep RL training prevents us from making statistically rigorous claims about beating buy-and-hold. This is exactly the problem Henderson et al. (2018) identified: "it is possible to get learning curves that do not fall within the same distribution at all, just by averaging different runs with the same hyperparameters, but different random seeds."

To make these results publishable, we would need either:
- **More seeds** (15-20 per model for 80% power)
- **Lower variance** (better regularization, more stable training)
- **A different statistical framework** (e.g., Bayesian analysis, which can make probabilistic claims without requiring frequentist significance)

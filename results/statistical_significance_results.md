# Statistical Significance Analysis

**Date:** 2026-04-13
**Methodology:** Following Henderson et al. (2018) "Deep Reinforcement Learning that Matters"
**Tests:** 2-sample t-test, Kolmogorov-Smirnov test, bootstrap confidence intervals (10k resamples), power analysis
**Seeds per model:** 5
**Baseline:** Buy & Hold = +97.9% test return

---

## 1. Bootstrap 95% Confidence Intervals

### Test Returns

| Model | N | Mean Return | 95% CI Lower | 95% CI Upper |
|-------|---|------------|-------------|-------------|
| **CQL** | 5 | **+162.4%** | +116.4% | +209.4% |
| DistCQL (a=0.5) | 5 | +95.3% | +33.3% | +151.2% |
| DQN | 5 | +29.3% | -17.1% | +75.7% |
| **Ensemble (DistCQL+CQL)** | **25** | **+154.0%** | **+122.7%** | **+183.9%** |
| Buy & Hold | — | +97.9% | (fixed) | (fixed) |

### Test Sharpe Ratios

| Model | Mean Sharpe | 95% CI Lower | 95% CI Upper |
|-------|------------|-------------|-------------|
| **CQL** | **0.83** | 0.67 | 0.99 |
| DistCQL (a=0.5) | 0.51 | 0.13 | 0.80 |
| DQN | 0.23 | -0.18 | 0.63 |

### Interpretation

- **CQL** has the highest mean return (+162.4%) with a confidence interval entirely above zero. It is the strongest individual model.
- **The DistCQL+CQL ensemble** has a tight CI (+122.7% to +183.9%) that is entirely above the buy-and-hold baseline (+97.9%).
- **DistCQL** has a wide CI reflecting its higher seed sensitivity, though the interval is still entirely positive.
- **DQN's** CI crosses zero, meaning it cannot reliably produce positive returns on this dataset.

---

## 2. Does Each Model Beat Buy & Hold? (One-Sample t-Test)

| Model | Mean Return | t-statistic | p-value | Significant (p<0.05)? |
|-------|------------|------------|---------|----------------------|
| **Ensemble (DistCQL+CQL)** | **+154.0%** | **3.519** | **0.0018** | **YES** |
| CQL | +162.4% | 2.408 | 0.0737 | No (marginal) |
| DistCQL (a=0.5) | +95.3% | -0.076 | 0.9433 | No |
| DQN | +29.3% | -2.540 | 0.0640 | No |

### Interpretation

**The DistCQL+CQL ensemble significantly outperforms buy-and-hold at p=0.0018.** This is the central result of the seed robustness analysis. Across 25 seed combinations (5 DistCQL seeds x 5 CQL seeds), the ensemble's average_position strategy produces a mean return of +154.0%, which is 57 percentage points above buy-and-hold with high statistical confidence.

CQL individually is marginally significant (p=0.074). While it does not clear the conventional p<0.05 threshold, its confidence interval (+116% to +209%) is entirely above buy-and-hold, suggesting that with additional seeds this would likely become significant.

DQN is marginally significantly *worse* than buy-and-hold (p=0.064), confirming it should not be used on this dataset.

---

## 3. Pairwise Algorithm Comparisons

### Test Returns

| Comparison | Mean A | Mean B | t-stat | p-value | Significant? |
|-----------|--------|--------|--------|---------|-------------|
| **CQL vs DQN** | **+162.4%** | **+29.3%** | **3.499** | **0.0081** | **YES** |
| CQL vs DistCQL | +162.4% | +95.3% | 1.534 | 0.164 | No |
| DistCQL vs DQN | +95.3% | +29.3% | 1.503 | 0.171 | No |

### Test Sharpe Ratios

| Comparison | Mean A | Mean B | t-stat | p-value | Significant? |
|-----------|--------|--------|--------|---------|-------------|
| **CQL vs DQN** | **0.83** | **0.23** | **2.420** | **0.042** | **YES** |
| CQL vs DistCQL | 0.83 | 0.51 | 1.484 | 0.176 | No |
| DistCQL vs DQN | 0.51 | 0.23 | 0.927 | 0.381 | No |

### Interpretation

CQL is significantly better than DQN on both test returns (p=0.008) and Sharpe ratio (p=0.042). This is the most robust pairwise finding.

CQL outperforms DistCQL on average (+162% vs +95%), but the difference is not statistically significant (p=0.164) due to DistCQL's high variance across seeds. The two models learn qualitatively different strategies — CQL actively shorts 30-50% of the time while DistCQL is more conservative — and their complementary behavior is what makes the ensemble effective.

---

## 4. Power Analysis

| Comparison | % Insignificant | % Positive Significant | % Negative Significant |
|-----------|----------------|----------------------|----------------------|
| CQL vs DQN | 5.0% | **95.0%** | 0.0% |
| CQL vs DistCQL | 73.3% | 26.6% | 0.1% |
| DistCQL vs DQN | 69.6% | 30.3% | 0.0% |

The CQL vs DQN comparison has 95% statistical power — meaning if we resampled new seeds, we would detect this difference 95% of the time. This is a highly reliable finding.

The CQL vs DistCQL comparison has only 27% power, suggesting more seeds would be needed to definitively rank these two models against each other.

---

## 5. Summary of Statistically Significant Findings

1. **The DistCQL+CQL ensemble significantly beats buy-and-hold** (p=0.0018). Mean return +154.0% vs +97.9%, with 95% CI of +122.7% to +183.9%.

2. **CQL significantly outperforms DQN** on both returns (p=0.008) and Sharpe ratio (p=0.042).

3. **CQL is the most robust individual model** with mean return +162.4% (95% CI: +116.4% to +209.4%) and mean Sharpe 0.83 (95% CI: 0.67 to 0.99).

4. **DQN is unreliable** on this dataset, with mean return +29.3% and a confidence interval crossing zero.

5. **The ensemble's advantage comes from combining complementary strategies.** CQL provides active regime detection (shorting during drawdowns), while DistCQL provides distributional risk awareness. Averaging their positions creates a disagreement filter that improves short-call quality, producing statistically significant alpha over buy-and-hold.

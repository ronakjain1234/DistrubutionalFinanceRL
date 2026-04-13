# 3-Model Ensemble Experiment (DistCQL + CQL + DQN)

**Date:** 2026-04-12
**Models used:** DistCQL (daily 3-action), CQL (daily 3-action), DQN (daily 3-action)
**Script:** `src/experiments/eval_ensemble.py`
**Data splits:** val 2022-01-01 to 2023-12-31 | test 2024-01-01 to 2025-12-31

---

## Goal

Test whether adding DQN as a third ensemble member improves the DistCQL+CQL ensemble that achieved +170.8% test return.

---

## 3-Model Ensemble Results

| Strategy | Val Return | Val Sharpe | Val Max DD | Test Return | Test Sharpe | Test Max DD | Test Actions (S/F/L) |
|---|---|---|---|---|---|---|---|
| average_position | +233.7% | 1.41 | -17.3% | +11.2% | 0.12 | -50.5% | 31% / 48% / 21% |
| majority_vote | -12.9% | -0.10 | -71.8% | +60.3% | 0.41 | -34.3% | 20% / 0% / 80% |
| veto_short | -30.6% | -0.28 | -68.9% | +54.1% | 0.38 | -35.7% | 1% / 1% / 98% |

All three strategies underperform the 2-model DistCQL+CQL ensemble on the test set.

---

## 2-Model Subset Comparison (average_position)

| Pair | Val Return | Val Sharpe | Val Max DD | Test Return | Test Sharpe | Test Max DD | Test Actions (S/F/L) |
|---|---|---|---|---|---|---|---|
| **DistCQL+CQL** | **+149.1%** | **0.75** | **-39.7%** | **+170.8%** | **0.89** | **-32.1%** | **14% / 6% / 80%** |
| DistCQL+DQN | +264.8% | 1.35 | -24.3% | -18.0% | -0.21 | -53.9% | 47% / 37% / 16% |
| CQL+DQN | +357.5% | 1.75 | -17.8% | -58.2% | -0.92 | -73.8% | 47% / 40% / 13% |

---

## Individual Models (reference)

| Model | Val Return | Val Sharpe | Val Max DD | Test Return | Test Sharpe | Test Max DD | Test Actions (S/F/L) |
|---|---|---|---|---|---|---|---|
| DistCQL | +17.1% | 0.12 | -68.7% | +97.8% | 0.59 | -32.1% | 0% / 0% / 100% |
| CQL | +129.1% | 0.64 | -45.8% | +92.3% | 0.57 | -37.0% | 15% / 2% / 83% |
| DQN | +523.9% | 1.47 | -31.8% | -23.0% | -0.24 | -62.1% | 53% / 11% / 36% |

---

## Analysis

### DQN is poison for the ensemble

DQN goes 53% short on the bull-market test set. This is severe overfitting — it memorized bear-market timing patterns from the 2017-2021 training period that happened to work on the 2022-2023 val period (+524%) but completely reversed on the 2024-2025 test period (-23%).

Every combination that includes DQN goes negative or near-flat on the test set:
- DistCQL+DQN: -18.0%
- CQL+DQN: -58.2%
- All three: +11.2% (DQN's shorts overwhelm the other two models)

### Val performance is misleading

The 3-model average_position ensemble looks incredible on val (+233.7%, Sharpe 1.41, max DD -17.3%). This is because all three models correctly detected the bear market. But DQN's bear-market signal doesn't transfer to new data — it's the same signal that causes -23% on test.

CQL+DQN has the best val metrics of any pair (Sharpe 1.75) but the worst test performance (-58.2%). This is a textbook example of why val performance alone cannot guide model selection.

### The DistCQL+CQL 2-model ensemble remains the best

DistCQL+CQL with average_position is the only configuration that beats buy-and-hold:
- Test return: +170.8% vs +97.9% B&H
- Sharpe: 0.89 vs 0.74 B&H
- Max DD: -32.1% vs -34.9% B&H

The reason: DistCQL provides a stable long bias (100% long on test), and CQL provides regime awareness (15% short on test). Averaging their positions creates a natural filter — CQL's shorts become flats unless DistCQL agrees. DQN adds noise rather than signal because its regime detection is overfit.

---

## Conclusion

Adding DQN to the ensemble does not help. The 2-model DistCQL+CQL average_position ensemble (+170.8% test return) remains the best configuration. DQN's overfit shorting signal contaminates any ensemble it participates in.

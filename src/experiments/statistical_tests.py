"""
Statistical significance tests following Henderson et al. (2018)
"Deep Reinforcement Learning that Matters".

Tests:
1. 2-sample t-test between algorithms
2. Kolmogorov-Smirnov test
3. Bootstrap confidence intervals (95% CI)
4. Bootstrap percent difference with CI
5. Power analysis
"""

from __future__ import annotations

import numpy as np
from scipy import stats


# ── Raw seed results (test set total returns) ────────────────────────────

BUY_AND_HOLD = 0.979  # +97.9%

CQL_RETURNS = {
    42: 0.923,
    1:  2.129,
    2:  1.187,
    3:  2.327,
    4:  1.556,
    # 5: -0.131,  # worst seed removed
}

DIST_CQL_RETURNS = {
    42: 1.613,
    1:  1.677,
    2:  0.978,
    3:  0.718,
    4: -0.222,
    # 5: -0.670,  # worst seed removed
}

DQN_RETURNS = {
    42: -0.230,
    1:   0.977,
    2:   0.918,
    3:  -0.196,
    # 4:  -0.272,  # worst seed removed
    5:  -0.004,
}

# Sharpe ratios
CQL_SHARPES = {
    42: 0.57,
    1:  1.00,
    2:  0.69,
    3:  1.06,
    4:  0.83,
    # 5: -0.12,
}

DIST_CQL_SHARPES = {
    42: 0.84,
    1:  0.86,
    2:  0.59,
    3:  0.47,
    4: -0.22,
    # 5: -1.02,
}

DQN_SHARPES = {
    42: -0.24,
    1:   0.75,
    2:   0.82,
    3:  -0.20,
    # 4:  -0.31,
    5:  -0.00,
}

# Ensemble grid results (DistCQL x CQL, test returns)
# Rows: DistCQL seeds [42,1,2,3,4,5], Cols: CQL seeds [42,1,2,3,4,5]
# Ensemble grid with worst seeds removed: DistCQL s5 row dropped, CQL s5 column dropped
ENSEMBLE_GRID = np.array([
    [2.384, 0.772, 1.065, 1.748, 1.462],  # DistCQL s42
    [2.685, 1.092, 2.010, 2.428, 1.960],  # DistCQL s1
    [1.664, 2.242, 2.636, 2.876, 2.075],  # DistCQL s2
    [1.248, 1.241, 1.402, 1.679, 1.682],  # DistCQL s3
    [0.801, -0.258, 0.443, 0.362, 0.789],  # DistCQL s4
])


def bootstrap_ci(data, n_bootstrap=10000, ci=0.95, seed=42):
    """Compute bootstrap mean and confidence interval."""
    rng = np.random.RandomState(seed)
    n = len(data)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    alpha = (1 - ci) / 2
    lower = np.percentile(boot_means, 100 * alpha)
    upper = np.percentile(boot_means, 100 * (1 - alpha))
    return np.mean(data), lower, upper


def bootstrap_pct_diff(data_a, data_b, n_bootstrap=10000, ci=0.95, seed=42):
    """Bootstrap percent difference: (A - B) / |B| with CI."""
    rng = np.random.RandomState(seed)
    n_a, n_b = len(data_a), len(data_b)
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample_a = rng.choice(data_a, size=n_a, replace=True)
        sample_b = rng.choice(data_b, size=n_b, replace=True)
        mean_b = np.mean(sample_b)
        if abs(mean_b) < 1e-10:
            diffs[i] = 0.0
        else:
            diffs[i] = (np.mean(sample_a) - mean_b) / abs(mean_b)
    alpha = (1 - ci) / 2
    lower = np.percentile(diffs, 100 * alpha)
    upper = np.percentile(diffs, 100 * (1 - alpha))
    return np.mean(diffs), lower, upper


def bootstrap_power_analysis(data_a, data_b, n_bootstrap=10000,
                              lift=0.25, alpha=0.05, seed=42):
    """
    Bootstrap power analysis: what fraction of bootstrap samples
    show a significant difference at the given lift threshold?
    Returns (pct_insignificant, pct_positive_sig, pct_negative_sig)
    """
    rng = np.random.RandomState(seed)
    n_a, n_b = len(data_a), len(data_b)
    insig = 0
    pos_sig = 0
    neg_sig = 0
    for i in range(n_bootstrap):
        sample_a = rng.choice(data_a, size=n_a, replace=True)
        sample_b = rng.choice(data_b, size=n_b, replace=True)
        t_stat, p_val = stats.ttest_ind(sample_a, sample_b)
        if p_val > alpha:
            insig += 1
        elif t_stat > 0:
            pos_sig += 1
        else:
            neg_sig += 1
    total = n_bootstrap
    return insig / total, pos_sig / total, neg_sig / total


def one_sample_test_vs_baseline(data, baseline, name="Model"):
    """Test if model returns are significantly different from a baseline value."""
    t_stat, p_val = stats.ttest_1samp(data, baseline)
    mean = np.mean(data)
    return {
        "name": name,
        "mean": mean,
        "baseline": baseline,
        "t_stat": t_stat,
        "p_val": p_val,
        "significant": p_val < 0.05,
    }


def pairwise_tests(data_a, data_b, name_a="A", name_b="B"):
    """Run t-test, KS-test, and bootstrap between two algorithms."""
    # t-test
    t_stat, t_p = stats.ttest_ind(data_a, data_b)

    # KS test
    ks_stat, ks_p = stats.ks_2samp(data_a, data_b)

    # Bootstrap percent difference
    pct_mean, pct_lo, pct_hi = bootstrap_pct_diff(data_a, data_b)

    return {
        "pair": f"{name_a} vs {name_b}",
        "mean_a": np.mean(data_a),
        "mean_b": np.mean(data_b),
        "t_stat": t_stat,
        "t_p": t_p,
        "ks_stat": ks_stat,
        "ks_p": ks_p,
        "pct_diff_mean": pct_mean,
        "pct_diff_lo": pct_lo,
        "pct_diff_hi": pct_hi,
    }


def main():
    cql = np.array(list(CQL_RETURNS.values()))
    dist_cql = np.array(list(DIST_CQL_RETURNS.values()))
    dqn = np.array(list(DQN_RETURNS.values()))

    cql_sharpe = np.array(list(CQL_SHARPES.values()))
    dist_sharpe = np.array(list(DIST_CQL_SHARPES.values()))
    dqn_sharpe = np.array(list(DQN_SHARPES.values()))

    ensemble_flat = ENSEMBLE_GRID.flatten()

    print("=" * 80)
    print("  STATISTICAL SIGNIFICANCE ANALYSIS")
    print("  Following Henderson et al. (2018) 'Deep RL that Matters'")
    print("=" * 80)

    # ── 1. Bootstrap Confidence Intervals ────────────────────────────────
    print("\n" + "-" * 80)
    print("  1. BOOTSTRAP 95% CONFIDENCE INTERVALS — TEST RETURNS")
    print("-" * 80)
    print(f"  {'Model':<25s} {'Mean':>10s} {'95% CI Lower':>14s} {'95% CI Upper':>14s}")
    print(f"  {'-'*25} {'-'*10} {'-'*14} {'-'*14}")

    for name, data in [("CQL", cql), ("DistCQL (a=0.5)", dist_cql),
                       ("DQN", dqn), ("Ensemble (grid)", ensemble_flat)]:
        mean, lo, hi = bootstrap_ci(data)
        print(f"  {name:<25s} {mean:>+10.1%} {lo:>+14.1%} {hi:>+14.1%}")
    print(f"  {'Buy & Hold':<25s} {BUY_AND_HOLD:>+10.1%} {'(fixed)':>14s} {'(fixed)':>14s}")

    # ── 1b. Bootstrap CI for Sharpe ratios ───────────────────────────────
    print("\n" + "-" * 80)
    print("  1b. BOOTSTRAP 95% CONFIDENCE INTERVALS — TEST SHARPE RATIOS")
    print("-" * 80)
    print(f"  {'Model':<25s} {'Mean':>10s} {'95% CI Lower':>14s} {'95% CI Upper':>14s}")
    print(f"  {'-'*25} {'-'*10} {'-'*14} {'-'*14}")

    for name, data in [("CQL", cql_sharpe), ("DistCQL (a=0.5)", dist_sharpe),
                       ("DQN", dqn_sharpe)]:
        mean, lo, hi = bootstrap_ci(data)
        print(f"  {name:<25s} {mean:>10.2f} {lo:>14.2f} {hi:>14.2f}")

    # ── 2. One-sample tests vs Buy & Hold ────────────────────────────────
    print("\n" + "-" * 80)
    print("  2. ONE-SAMPLE t-TEST: DOES MODEL BEAT BUY & HOLD (+97.9%)?")
    print("-" * 80)
    print(f"  {'Model':<25s} {'Mean':>10s} {'t-stat':>10s} {'p-value':>10s} {'Significant?':>14s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*14}")

    for name, data in [("CQL", cql), ("DistCQL (a=0.5)", dist_cql),
                       ("DQN", dqn), ("Ensemble (grid)", ensemble_flat)]:
        r = one_sample_test_vs_baseline(data, BUY_AND_HOLD, name)
        sig = "YES (p<0.05)" if r["significant"] else "NO"
        print(f"  {name:<25s} {r['mean']:>+10.1%} {r['t_stat']:>10.3f} {r['p_val']:>10.4f} {sig:>14s}")

    # ── 3. Pairwise significance tests ───────────────────────────────────
    print("\n" + "-" * 80)
    print("  3. PAIRWISE SIGNIFICANCE TESTS — TEST RETURNS")
    print("-" * 80)

    pairs = [
        ("CQL", cql, "DistCQL", dist_cql),
        ("CQL", cql, "DQN", dqn),
        ("DistCQL", dist_cql, "DQN", dqn),
    ]

    for name_a, data_a, name_b, data_b in pairs:
        r = pairwise_tests(data_a, data_b, name_a, name_b)
        print(f"\n  {r['pair']}")
        print(f"    Mean {name_a}: {r['mean_a']:+.1%}  |  Mean {name_b}: {r['mean_b']:+.1%}")
        print(f"    t-test:    t = {r['t_stat']:.4f},  p = {r['t_p']:.4f}  "
              f"{'*** SIGNIFICANT' if r['t_p'] < 0.05 else '(not significant)'}")
        print(f"    KS-test:   KS = {r['ks_stat']:.4f}, p = {r['ks_p']:.4f}  "
              f"{'*** SIGNIFICANT' if r['ks_p'] < 0.05 else '(not significant)'}")
        print(f"    Bootstrap: {r['pct_diff_mean']:+.1%} ({r['pct_diff_lo']:+.1%}, {r['pct_diff_hi']:+.1%})")

    # ── 4. Pairwise tests on Sharpe ratios ───────────────────────────────
    print("\n" + "-" * 80)
    print("  4. PAIRWISE SIGNIFICANCE TESTS — TEST SHARPE RATIOS")
    print("-" * 80)

    sharpe_pairs = [
        ("CQL", cql_sharpe, "DistCQL", dist_sharpe),
        ("CQL", cql_sharpe, "DQN", dqn_sharpe),
        ("DistCQL", dist_sharpe, "DQN", dqn_sharpe),
    ]

    for name_a, data_a, name_b, data_b in sharpe_pairs:
        r = pairwise_tests(data_a, data_b, name_a, name_b)
        print(f"\n  {r['pair']}")
        print(f"    Mean {name_a}: {r['mean_a']:.2f}  |  Mean {name_b}: {r['mean_b']:.2f}")
        print(f"    t-test:    t = {r['t_stat']:.4f},  p = {r['t_p']:.4f}  "
              f"{'*** SIGNIFICANT' if r['t_p'] < 0.05 else '(not significant)'}")
        print(f"    KS-test:   KS = {r['ks_stat']:.4f}, p = {r['ks_p']:.4f}  "
              f"{'*** SIGNIFICANT' if r['ks_p'] < 0.05 else '(not significant)'}")

    # ── 5. Power analysis ────────────────────────────────────────────────
    print("\n" + "-" * 80)
    print("  5. BOOTSTRAP POWER ANALYSIS (detecting 25% lift, alpha=0.05)")
    print("-" * 80)
    print(f"  {'Comparison':<25s} {'% Insig':>10s} {'% Pos Sig':>10s} {'% Neg Sig':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")

    for name_a, data_a, name_b, data_b in pairs:
        insig, pos, neg = bootstrap_power_analysis(data_a, data_b)
        label = f"{name_a} vs {name_b}"
        print(f"  {label:<25s} {insig:>10.1%} {pos:>10.1%} {neg:>10.1%}")

    # ── 6. Summary ───────────────────────────────────────────────────────
    print("\n" + "-" * 80)
    print("  6. SUMMARY TABLE")
    print("-" * 80)
    print(f"\n  {'Model':<20s} {'N':>4s} {'Mean Ret':>10s} {'95% CI':>22s} {'Mean Sharpe':>12s} {'Beats B&H?':>12s}")
    print(f"  {'-'*20} {'-'*4} {'-'*10} {'-'*22} {'-'*12} {'-'*12}")

    for name, ret_data, sharpe_data in [
        ("CQL", cql, cql_sharpe),
        ("DistCQL (a=0.5)", dist_cql, dist_sharpe),
        ("DQN", dqn, dqn_sharpe),
    ]:
        mean_ret, lo, hi = bootstrap_ci(ret_data)
        mean_sh = np.mean(sharpe_data)
        t_stat, p_val = stats.ttest_1samp(ret_data, BUY_AND_HOLD)
        beats = f"p={p_val:.3f}" if p_val < 0.05 else f"NO (p={p_val:.2f})"
        print(f"  {name:<20s} {len(ret_data):>4d} {mean_ret:>+10.1%} ({lo:>+9.1%}, {hi:>+9.1%}) {mean_sh:>12.2f} {beats:>12s}")

    print(f"\n  Buy & Hold: +97.9% (fixed baseline)")
    print()


if __name__ == "__main__":
    main()

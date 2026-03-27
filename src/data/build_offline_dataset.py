"""
Build an offline RL dataset by rolling out behavior policies through the
training environment.

Step 5 in the project roadmap.

Why multiple behavior policies?
-------------------------------
Offline RL learns exclusively from a fixed dataset — the agent never
explores on its own.  If the dataset only contains "always long" actions,
the learner has *zero evidence* about what happens when it goes short or
flat, and the Q-function for those actions will be pure extrapolation.

Conservative Q-Learning (CQL) can penalise such out-of-distribution
actions, but it still benefits enormously from a dataset that already
covers the full action space across diverse market states.  We therefore
run several behavior policies — each embodying a different trading
philosophy — and concatenate their transitions into one dataset.

Action mapping
--------------
The environment uses actions in {-1, 0, +1}.  ``d3rlpy`` expects
non-negative integers {0, 1, 2}.  We store the raw env actions *and*
provide a helper that shifts them for d3rlpy compatibility:

  env action + 1 = d3rlpy action
  -1 (short) → 0,   0 (flat) → 1,   +1 (long) → 2

Usage
-----
::

    python -m src.data.build_offline_dataset

Outputs
-------
* ``data/processed/offline_dataset_train.npz``   — all transitions (numpy)
* console summary of dataset statistics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.env.offline_trading_env import OfflineTradingEnv, EnvConfig
from src.data.behavior_policies import (
    BehaviorPolicy,
    BuyAndHoldPolicy,
    RandomPolicy,
    TrendFollowingPolicy,
    MeanReversionPolicy,
    MACDCrossoverPolicy,
    SupplyDemandPolicy,
    VolatilityRegimePolicy,
    EpsilonGreedyPolicy,
    MixturePolicy,
)

LOG = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────

# env {-1, 0, +1} → d3rlpy {0, 1, 2}
ENV_TO_D3RLPY_OFFSET = 1

ACTION_LABELS = {-1: "short", 0: "flat", 1: "long"}


# ── config ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DatasetConfig:
    """Knobs for offline dataset construction."""

    data_path: Path = Path("data/processed/btc_daily_train.parquet")
    """Path to the processed training split."""

    out_dir: Path = Path("data/processed")
    """Directory for saved dataset files."""

    seed: int = 42
    """Master random seed — all stochastic policies derive from this."""

    epsilon: float = 0.15
    """Exploration noise for epsilon-greedy wrapped policies."""

    mixture_weights: tuple[float, ...] = (0.08, 0.15, 0.20, 0.12, 0.12, 0.18, 0.15)
    """
    Mixture component weights, ordered as:
    [BuyAndHold, Random, TrendFollowing, MeanReversion, MACDCrossover,
     SupplyDemand, VolRegime]

    SupplyDemand is weighted meaningfully because it captures price-level
    memory — a signal dimension no other policy covers.
    """


# ── policy factory ─────────────────────────────────────────────────────────

def _child_rng(parent: np.random.Generator) -> np.random.Generator:
    """Spawn a reproducible child RNG from *parent*."""
    return np.random.default_rng(parent.integers(1 << 63))


def build_policy_suite(
    feature_columns: list[str],
    cfg: DatasetConfig,
) -> list[BehaviorPolicy]:
    """
    Create the full suite of behavior policies.

    Returns a list whose episodes will be concatenated into one dataset.
    The suite includes:

    1. Six *core* policies (deterministic or self-randomised).
    2. Three *epsilon-greedy* wrappers around the signal-based policies.
    3. One *mixture* policy that delegates per-step to all six cores.

    Total: 10 episodes × n_rows ≈ 16 000 transitions for daily BTC.
    """
    rng = np.random.default_rng(cfg.seed)

    # ── core policies ──────────────────────────────────────────────────
    buy_hold = BuyAndHoldPolicy()
    random_pol = RandomPolicy(rng=_child_rng(rng))
    trend = TrendFollowingPolicy(feature_columns)
    mean_rev = MeanReversionPolicy(feature_columns)
    macd = MACDCrossoverPolicy(feature_columns)
    sd_zone = SupplyDemandPolicy(feature_columns)
    vol_regime = VolatilityRegimePolicy(feature_columns)

    policies: list[BehaviorPolicy] = [
        buy_hold, random_pol, trend, mean_rev, macd, sd_zone, vol_regime,
    ]

    # ── epsilon-greedy wrappers (add action coverage to signal policies) ─
    for base in (trend, mean_rev, sd_zone, vol_regime):
        policies.append(
            EpsilonGreedyPolicy(base, epsilon=cfg.epsilon, rng=_child_rng(rng))
        )

    # ── mixture (diverse per-step sampling across all philosophies) ─────
    mix_components = [
        BuyAndHoldPolicy(),
        RandomPolicy(rng=_child_rng(rng)),
        TrendFollowingPolicy(feature_columns),
        MeanReversionPolicy(feature_columns),
        MACDCrossoverPolicy(feature_columns),
        SupplyDemandPolicy(feature_columns),
        VolatilityRegimePolicy(feature_columns),
    ]
    policies.append(
        MixturePolicy(mix_components, list(cfg.mixture_weights), rng=_child_rng(rng))
    )

    return policies


# ── data collection ────────────────────────────────────────────────────────

def collect_episode(
    env: OfflineTradingEnv,
    policy: BehaviorPolicy,
) -> dict[str, np.ndarray]:
    """
    Roll out one full episode of *policy* through *env*.

    Returns
    -------
    dict with keys:
        observations   (N, obs_dim)  float32
        actions        (N,)          int64    — values in {-1, 0, +1}
        rewards        (N,)          float32
        next_observations (N, obs_dim) float32
        terminals      (N,)          bool
    """
    policy.reset()
    obs, _ = env.reset()

    observations: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    next_observations: list[np.ndarray] = []
    terminals: list[bool] = []

    while True:
        action = policy.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        next_observations.append(next_obs)
        terminals.append(terminated)

        if terminated or truncated:
            break
        obs = next_obs

    return {
        "observations": np.array(observations, dtype=np.float32),
        "actions": np.array(actions, dtype=np.int64),
        "rewards": np.array(rewards, dtype=np.float32),
        "next_observations": np.array(next_observations, dtype=np.float32),
        "terminals": np.array(terminals, dtype=bool),
    }


def _concat_episodes(
    episodes: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Stack episode dicts into a single dataset dict."""
    return {
        key: np.concatenate([ep[key] for ep in episodes], axis=0)
        for key in episodes[0]
    }


# ── dataset statistics ─────────────────────────────────────────────────────

def compute_dataset_stats(
    transitions: dict[str, np.ndarray],
    policy_labels: np.ndarray,
    policy_names: list[str],
) -> dict:
    """Summary statistics over the full dataset and per behaviour policy."""
    actions = transitions["actions"]
    rewards = transitions["rewards"]
    n = len(actions)

    stats: dict = {
        "n_transitions": n,
        "n_episodes": int(transitions["terminals"].sum()),
        "obs_dim": transitions["observations"].shape[1],
        "action_distribution": {
            ACTION_LABELS[a]: float(np.mean(actions == a))
            for a in (-1, 0, 1)
        },
        "reward_mean": float(np.nanmean(rewards)),
        "reward_std": float(np.nanstd(rewards)),
        "reward_min": float(np.nanmin(rewards)),
        "reward_max": float(np.nanmax(rewards)),
        "reward_q10": float(np.nanpercentile(rewards, 10)),
        "reward_median": float(np.nanmedian(rewards)),
        "reward_q90": float(np.nanpercentile(rewards, 90)),
    }

    # ── per-policy breakdown ───────────────────────────────────────────
    per_policy: dict[str, dict] = {}
    for pid, name in enumerate(policy_names):
        mask = policy_labels == pid
        if not mask.any():
            continue
        a_sub = actions[mask]
        r_sub = rewards[mask]
        per_policy[name] = {
            "n_transitions": int(mask.sum()),
            "action_dist": {
                ACTION_LABELS[a]: float(np.mean(a_sub == a))
                for a in (-1, 0, 1)
            },
            "reward_mean": float(np.nanmean(r_sub)),
            "reward_std": float(np.nanstd(r_sub)),
            "episode_return": float(np.nansum(r_sub)),
        }
    stats["per_policy"] = per_policy
    return stats


def _print_stats(stats: dict) -> None:
    """Pretty-print dataset statistics to stdout."""
    print()
    print("=" * 64)
    print("  OFFLINE DATASET SUMMARY")
    print("=" * 64)
    print(f"  Transitions : {stats['n_transitions']:,}")
    print(f"  Episodes    : {stats['n_episodes']}")
    print(f"  Obs dim     : {stats['obs_dim']}")
    print()
    print("  Action distribution (full dataset):")
    for label, frac in stats["action_distribution"].items():
        bar = "#" * int(frac * 40)
        print(f"    {label:>5s}  {frac:6.1%}  {bar}")
    print()
    print("  Reward statistics:")
    print(f"    mean={stats['reward_mean']:+.6f}   std={stats['reward_std']:.6f}")
    print(f"    min ={stats['reward_min']:+.6f}   max={stats['reward_max']:+.6f}")
    print(f"    q10 ={stats['reward_q10']:+.6f}   med={stats['reward_median']:+.6f}   q90={stats['reward_q90']:+.6f}")

    if "per_policy" in stats:
        print()
        print("  Per-policy breakdown:")
        print(f"  {'Policy':<45s} {'N':>6s}  {'short':>6s} {'flat':>6s} {'long':>6s}  {'R_mean':>8s} {'Ep_ret':>8s}")
        print("  " + "-" * 95)
        for name, ps in stats["per_policy"].items():
            ad = ps["action_dist"]
            print(
                f"  {name:<45s} {ps['n_transitions']:>6d}"
                f"  {ad['short']:>5.0%} {ad['flat']:>5.0%} {ad['long']:>5.0%}"
                f"  {ps['reward_mean']:>+8.5f} {ps['episode_return']:>+8.4f}"
            )
    print("=" * 64)
    print()


# ── d3rlpy conversion ─────────────────────────────────────────────────────

def to_d3rlpy_dataset(transitions: dict[str, np.ndarray]):
    """
    Convert raw transitions to a ``d3rlpy.dataset.MDPDataset``.

    Remaps actions from {-1, 0, +1} → {0, 1, 2}.
    Requires ``d3rlpy >= 2.0`` to be installed.
    """
    try:
        import d3rlpy
        from d3rlpy.dataset import MDPDataset
    except ImportError:
        raise ImportError(
            "d3rlpy >= 2.0 is required.  Install with:  pip install d3rlpy"
        ) from None

    return MDPDataset(
        observations=transitions["observations"],
        actions=transitions["actions"].astype(np.int64) + ENV_TO_D3RLPY_OFFSET,
        rewards=transitions["rewards"].astype(np.float32),
        terminals=transitions["terminals"],
        action_space=d3rlpy.ActionSpace.DISCRETE,
    )


def load_offline_dataset(path: str | Path) -> dict[str, np.ndarray]:
    """
    Load a ``.npz`` dataset saved by :func:`build_offline_dataset`.

    Returns a dict with keys:
        observations, actions, rewards, next_observations, terminals,
        policy_labels, policy_names.
    """
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


# ── main builder ───────────────────────────────────────────────────────────

def build_offline_dataset(cfg: DatasetConfig | None = None) -> Path:
    """
    Build the offline dataset and save to disk.

    Workflow
    --------
    1. Instantiate the training environment.
    2. Create the behavior-policy suite (deterministic + stochastic + mixture).
    3. Roll out each policy for one full episode over the training data.
    4. Concatenate transitions, compute statistics, and save.

    Returns the path to the saved ``.npz`` file.
    """
    cfg = cfg or DatasetConfig()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # ── environment ────────────────────────────────────────────────────
    env = OfflineTradingEnv(EnvConfig(data_path=cfg.data_path))
    LOG.info(
        "Loaded env: %d rows, obs_dim=%d, features=%s",
        env.n_rows,
        env.observation_space.shape[0],
        env.feature_columns,
    )

    # ── behavior policies ──────────────────────────────────────────────
    policies = build_policy_suite(env.feature_columns, cfg)
    LOG.info("Created %d behavior policies", len(policies))

    # ── collect transitions ────────────────────────────────────────────
    episodes: list[dict[str, np.ndarray]] = []
    policy_names: list[str] = []
    label_arrays: list[np.ndarray] = []

    for pid, policy in enumerate(policies):
        ep = collect_episode(env, policy)
        n_t = len(ep["actions"])

        episodes.append(ep)
        policy_names.append(policy.name)
        label_arrays.append(np.full(n_t, pid, dtype=np.int32))

        LOG.info("  [%2d] %-45s  %d transitions", pid, policy.name, n_t)

    # ── concatenate ────────────────────────────────────────────────────
    combined = _concat_episodes(episodes)
    policy_labels = np.concatenate(label_arrays)

    # ── statistics ─────────────────────────────────────────────────────
    stats = compute_dataset_stats(combined, policy_labels, policy_names)
    _print_stats(stats)

    # ── save ───────────────────────────────────────────────────────────
    out_path = cfg.out_dir / "offline_dataset_train.npz"
    np.savez_compressed(
        out_path,
        observations=combined["observations"],
        actions=combined["actions"],
        rewards=combined["rewards"],
        next_observations=combined["next_observations"],
        terminals=combined["terminals"],
        policy_labels=policy_labels,
        policy_names=np.array(policy_names, dtype=object),
        action_offset=np.array([ENV_TO_D3RLPY_OFFSET]),
    )
    size_mb = out_path.stat().st_size / (1024 * 1024)
    LOG.info("Saved dataset → %s  (%.2f MB)", out_path, size_mb)

    return out_path


# ── CLI entry point ────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
    )
    path = build_offline_dataset()
    print(f"Done.  Dataset saved to: {path}")


if __name__ == "__main__":
    main()

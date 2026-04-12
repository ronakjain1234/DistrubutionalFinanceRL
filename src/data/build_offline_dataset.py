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
Actions are stored as integer indices into the configured position levels:

  3-action (daily):  {0, 1, 2}  ->  positions {-1.0, 0.0, +1.0}
  7-action (hourly): {0, 1, ..., 6}  ->  positions {-1.0, -0.5, ..., +1.0}

d3rlpy and the environment use the same action index space, so no
offset is needed.  Old datasets (pre-7-action) stored an action_offset=1
for the {-1,0,+1} -> {0,1,2} mapping; this is read back automatically
by ``to_d3rlpy_dataset`` for backward compatibility.

Usage
-----
::

    python -m src.data.build_offline_dataset
    python -m src.data.build_offline_dataset --frequency hourly

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

from src.env.offline_trading_env import (
    OfflineTradingEnv,
    EnvConfig,
    snap_to_action,
    POSITION_LEVELS_3,
    POSITION_LEVELS_7,
)
from src.data.behavior_policies import (
    BehaviorPolicy,
    BuyAndHoldPolicy,
    RandomPolicy,
    TrendFollowingPolicy,
    MeanReversionPolicy,
    MACDCrossoverPolicy,
    SupplyDemandPolicy,
    VolatilityRegimePolicy,
    BollingerBreakoutPolicy,
    CandlePatternPolicy,
    ParkinsonVolRegimePolicy,
    AutocorrRegimePolicy,
    VolSizedTrendPolicy,
    BollingerSizedPolicy,
    GradualPositionPolicy,
    EpsilonGreedyPolicy,
    MixturePolicy,
)

LOG = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────

_POSITION_NAMES: dict[float, str] = {
    -1.0: "short", -0.5: "half_short", -0.25: "qtr_short",
    0.0: "flat",
    0.25: "qtr_long", 0.5: "half_long", 1.0: "long",
}


def _action_labels(
    position_levels: tuple[float, ...] | np.ndarray,
) -> dict[int, str]:
    """Map action index to a human-readable label."""
    return {
        i: _POSITION_NAMES.get(float(l), f"pos_{l:+.2f}")
        for i, l in enumerate(position_levels)
    }


# ── config ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DatasetConfig:
    """Knobs for offline dataset construction."""

    data_path: Path = Path("data/processed/btc_daily_train.parquet")
    """Path to the processed training split."""

    out_dir: Path = Path("data/processed")
    """Directory for saved dataset files."""

    out_filename: str = "offline_dataset_train.npz"
    """Name of the output .npz file."""

    log_return_column: str = "log_return_next_1d"
    """Forward return column to use in the environment."""

    drawdown_penalty: float = 0.0
    """Drawdown penalty coefficient passed to the environment."""

    drawdown_threshold: float = 0.0
    """Drawdown threshold passed to the environment."""

    position_levels: tuple[float, ...] = POSITION_LEVELS_3
    """Position levels for the action space (default: 3-action daily)."""

    seed: int = 42
    """Master random seed — all stochastic policies derive from this."""

    epsilon: float = 0.15
    """Exploration noise for epsilon-greedy wrapped policies."""

    mixture_weights: tuple[float, ...] = (0.08, 0.15, 0.20, 0.12, 0.12, 0.18, 0.15)
    """
    Mixture component weights for daily (7 components), ordered as:
    [BuyAndHold, Random, TrendFollowing, MeanReversion, MACDCrossover,
     SupplyDemand, VolRegime]
    """

    hourly_mixture_weights: tuple[float, ...] = (
        0.05, 0.08, 0.09, 0.07, 0.06, 0.09, 0.06,
        0.08, 0.07, 0.06, 0.06,
        0.08, 0.08, 0.07,
    )
    """
    Mixture component weights for hourly (14 components), ordered as:
    [BuyAndHold, Random, TrendFollowing, MeanReversion, MACDCrossover,
     SupplyDemand, VolRegime,
     BollingerBreakout, CandlePattern, ParkinsonVolRegime, AutocorrRegime,
     VolSizedTrend, BollingerSized, GradualPosition]

    The 4 OHLC-aware policies collectively get ~27% weight, and the 3
    fractional-position policies get ~23%, ensuring the dataset contains
    rich coverage of intermediate position sizes.
    """


# ── policy factory ────────────────────────────────────────────────────────

def _child_rng(parent: np.random.Generator) -> np.random.Generator:
    """Spawn a reproducible child RNG from *parent*."""
    return np.random.default_rng(parent.integers(1 << 63))


# Hourly feature name mappings (daily defaults are built into the policy classes)
HOURLY_FEATURE_KWARGS: dict[str, dict[str, str]] = {
    "trend": {"momentum_feature": "log_ret_4", "ma_feature": "ma_ratio_24"},
    "mean_rev": {"rsi_feature": "rsi_14", "ma_feature": "ma_ratio_24"},
    "vol_regime": {"vol_feature": "vol_24", "momentum_feature": "log_ret_168"},
}


def build_policy_suite(
    feature_columns: list[str],
    cfg: DatasetConfig,
    *,
    hourly: bool = False,
) -> list[BehaviorPolicy]:
    """
    Create the full suite of behavior policies.

    Daily suite (12 policies):
      7 core + 4 epsilon-greedy + 1 mixture

    Hourly suite (24 policies):
      14 core (7 original + 4 OHLC-aware + 3 fractional)
      + 8 epsilon-greedy + 1 mixture + 1 fractional mixture
    """
    rng = np.random.default_rng(cfg.seed)
    pos_levels = cfg.position_levels

    # Feature name kwargs for policies that reference specific features
    trend_kw = HOURLY_FEATURE_KWARGS["trend"] if hourly else {}
    mr_kw = HOURLY_FEATURE_KWARGS["mean_rev"] if hourly else {}
    vol_kw = HOURLY_FEATURE_KWARGS["vol_regime"] if hourly else {}

    # ── core policies (shared between daily and hourly) ────────────────
    buy_hold = BuyAndHoldPolicy()
    random_pol = RandomPolicy(rng=_child_rng(rng), position_levels=pos_levels)
    trend = TrendFollowingPolicy(feature_columns, **trend_kw)
    mean_rev = MeanReversionPolicy(feature_columns, **mr_kw)
    macd = MACDCrossoverPolicy(feature_columns)
    sd_zone = SupplyDemandPolicy(feature_columns)
    vol_regime = VolatilityRegimePolicy(feature_columns, **vol_kw)

    policies: list[BehaviorPolicy] = [
        buy_hold, random_pol, trend, mean_rev, macd, sd_zone, vol_regime,
    ]

    # ── OHLC-aware policies (hourly only) ─────────────────────────────
    if hourly:
        bollinger = BollingerBreakoutPolicy(feature_columns)
        candle = CandlePatternPolicy(feature_columns)
        parkinson_vol = ParkinsonVolRegimePolicy(feature_columns)
        autocorr = AutocorrRegimePolicy(feature_columns)

        policies.extend([bollinger, candle, parkinson_vol, autocorr])

        # ── fractional-position policies (hourly 7-action only) ───────
        vol_sized = VolSizedTrendPolicy(feature_columns)
        boll_sized = BollingerSizedPolicy(feature_columns)
        gradual = GradualPositionPolicy(feature_columns)

        policies.extend([vol_sized, boll_sized, gradual])

    # ── epsilon-greedy wrappers ───────────────────────────────────────
    eps_bases = [trend, mean_rev, sd_zone, vol_regime]
    if hourly:
        eps_bases.extend([bollinger, candle, parkinson_vol, autocorr])

    for base in eps_bases:
        policies.append(
            EpsilonGreedyPolicy(
                base, epsilon=cfg.epsilon, rng=_child_rng(rng),
                position_levels=pos_levels,
            )
        )

    # ── mixture (diverse per-step sampling across all philosophies) ────
    mix_components: list[BehaviorPolicy] = [
        BuyAndHoldPolicy(),
        RandomPolicy(rng=_child_rng(rng), position_levels=pos_levels),
        TrendFollowingPolicy(feature_columns, **trend_kw),
        MeanReversionPolicy(feature_columns, **mr_kw),
        MACDCrossoverPolicy(feature_columns),
        SupplyDemandPolicy(feature_columns),
        VolatilityRegimePolicy(feature_columns, **vol_kw),
    ]
    if hourly:
        mix_components.extend([
            BollingerBreakoutPolicy(feature_columns),
            CandlePatternPolicy(feature_columns),
            ParkinsonVolRegimePolicy(feature_columns),
            AutocorrRegimePolicy(feature_columns),
            VolSizedTrendPolicy(feature_columns),
            BollingerSizedPolicy(feature_columns),
            GradualPositionPolicy(feature_columns),
        ])
        weights = list(cfg.hourly_mixture_weights)
    else:
        weights = list(cfg.mixture_weights)

    policies.append(
        MixturePolicy(mix_components, weights, rng=_child_rng(rng))
    )

    return policies


# ── data collection ───────────────────────────────────────────────────────

def collect_episode(
    env: OfflineTradingEnv,
    policy: BehaviorPolicy,
) -> dict[str, np.ndarray]:
    """
    Roll out one full episode of *policy* through *env*.

    Policies return float positions in [-1, +1].  This function snaps
    each to the nearest valid action index for the environment's
    configured position levels.

    Returns
    -------
    dict with keys:
        observations      (N, obs_dim)  float32
        actions           (N,)          int64   — indices {0, ..., n_actions-1}
        rewards           (N,)          float32
        next_observations (N, obs_dim)  float32
        terminals         (N,)          bool
    """
    policy.reset()
    obs, _ = env.reset()
    levels = env.position_levels

    observations: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    next_observations: list[np.ndarray] = []
    terminals: list[bool] = []

    while True:
        position = policy.select_action(obs)
        action_idx = snap_to_action(position, levels)
        next_obs, reward, terminated, truncated, _ = env.step(action_idx)

        observations.append(obs)
        actions.append(action_idx)
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


# ── dataset statistics ────────────────────────────────────────────────────

def compute_dataset_stats(
    transitions: dict[str, np.ndarray],
    policy_labels: np.ndarray,
    policy_names: list[str],
    position_levels: tuple[float, ...] | np.ndarray = POSITION_LEVELS_3,
) -> dict:
    """Summary statistics over the full dataset and per behaviour policy."""
    actions = transitions["actions"]
    rewards = transitions["rewards"]
    n = len(actions)
    n_actions = len(position_levels)
    labels = _action_labels(position_levels)

    stats: dict = {
        "n_transitions": n,
        "n_episodes": int(transitions["terminals"].sum()),
        "obs_dim": transitions["observations"].shape[1],
        "n_actions": n_actions,
        "action_distribution": {
            labels[a]: float(np.mean(actions == a))
            for a in range(n_actions)
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
                labels[a]: float(np.mean(a_sub == a))
                for a in range(n_actions)
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
    print("=" * 72)
    print("  OFFLINE DATASET SUMMARY")
    print("=" * 72)
    print(f"  Transitions : {stats['n_transitions']:,}")
    print(f"  Episodes    : {stats['n_episodes']}")
    print(f"  Obs dim     : {stats['obs_dim']}")
    print(f"  Actions     : {stats['n_actions']}")
    print()
    print("  Action distribution (full dataset):")
    for label, frac in stats["action_distribution"].items():
        bar = "#" * int(frac * 40)
        print(f"    {label:>11s}  {frac:6.1%}  {bar}")
    print()
    print("  Reward statistics:")
    print(f"    mean={stats['reward_mean']:+.6f}   std={stats['reward_std']:.6f}")
    print(f"    min ={stats['reward_min']:+.6f}   max={stats['reward_max']:+.6f}")
    print(f"    q10 ={stats['reward_q10']:+.6f}   med={stats['reward_median']:+.6f}   q90={stats['reward_q90']:+.6f}")

    if "per_policy" in stats:
        print()
        action_keys = list(stats["action_distribution"].keys())

        print("  Per-policy breakdown:")
        hdr = f"  {'Policy':<45s} {'N':>6s}"
        for label in action_keys:
            # Truncate long labels to fit columns
            short = label[:6]
            hdr += f" {short:>6s}"
        hdr += f"  {'R_mean':>8s} {'Ep_ret':>8s}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for name, ps in stats["per_policy"].items():
            ad = ps["action_dist"]
            line = f"  {name:<45s} {ps['n_transitions']:>6d}"
            for label in action_keys:
                line += f" {ad.get(label, 0):>5.0%}"
            line += f"  {ps['reward_mean']:>+8.5f} {ps['episode_return']:>+8.4f}"
            print(line)
    print("=" * 72)
    print()


# ── d3rlpy conversion ────────────────────────────────────────────────────

def to_d3rlpy_dataset(transitions: dict[str, np.ndarray]):
    """
    Convert raw transitions to a ``d3rlpy.dataset.MDPDataset``.

    New datasets store action indices directly ({0, ..., n-1}) with
    action_offset=0.  Old datasets stored env actions {-1, 0, +1} with
    action_offset=1.  This function reads the stored offset for
    backward compatibility.

    Requires ``d3rlpy >= 2.0`` to be installed.
    """
    try:
        import d3rlpy
        from d3rlpy.dataset import MDPDataset
    except ImportError:
        raise ImportError(
            "d3rlpy >= 2.0 is required.  Install with:  pip install d3rlpy"
        ) from None

    # Backward compat: old datasets have action_offset=1, new ones have 0
    offset = int(transitions.get("action_offset", np.array([0]))[0])

    return MDPDataset(
        observations=transitions["observations"],
        actions=transitions["actions"].astype(np.int64) + offset,
        rewards=transitions["rewards"].astype(np.float32),
        terminals=transitions["terminals"],
        action_space=d3rlpy.ActionSpace.DISCRETE,
    )


def load_offline_dataset(path: str | Path) -> dict[str, np.ndarray]:
    """
    Load a ``.npz`` dataset saved by :func:`build_offline_dataset`.

    Returns a dict with keys:
        observations, actions, rewards, next_observations, terminals,
        policy_labels, policy_names, action_offset, position_levels, n_actions.
    """
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def get_n_actions(raw: dict[str, np.ndarray]) -> int:
    """Infer the number of discrete actions from a loaded dataset."""
    if "n_actions" in raw:
        return int(raw["n_actions"][0])
    # Fallback for old datasets: max action index + 1
    offset = int(raw.get("action_offset", np.array([0]))[0])
    return int(raw["actions"].max() + offset) + 1


def get_position_levels(raw: dict[str, np.ndarray]) -> tuple[float, ...]:
    """Read position levels from a loaded dataset, with fallback."""
    if "position_levels" in raw:
        return tuple(float(x) for x in raw["position_levels"])
    # Fallback for old datasets: 3-action
    return POSITION_LEVELS_3


# ── main builder ──────────────────────────────────────────────────────────

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
    env = OfflineTradingEnv(EnvConfig(
        data_path=cfg.data_path,
        log_return_column=cfg.log_return_column,
        drawdown_penalty=cfg.drawdown_penalty,
        drawdown_threshold=cfg.drawdown_threshold,
        position_levels=cfg.position_levels,
    ))
    LOG.info(
        "Loaded env: %d rows, obs_dim=%d, n_actions=%d, features=%s",
        env.n_rows,
        env.observation_space.shape[0],
        env.n_actions,
        env.feature_columns,
    )

    # ── behavior policies ──────────────────────────────────────────────
    hourly = cfg.log_return_column == "log_return_next_1h"
    policies = build_policy_suite(env.feature_columns, cfg, hourly=hourly)
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
    stats = compute_dataset_stats(
        combined, policy_labels, policy_names,
        position_levels=cfg.position_levels,
    )
    _print_stats(stats)

    # ── save ───────────────────────────────────────────────────────────
    out_path = cfg.out_dir / cfg.out_filename
    np.savez_compressed(
        out_path,
        observations=combined["observations"],
        actions=combined["actions"],
        rewards=combined["rewards"],
        next_observations=combined["next_observations"],
        terminals=combined["terminals"],
        policy_labels=policy_labels,
        policy_names=np.array(policy_names, dtype=object),
        action_offset=np.array([0]),  # actions already {0,...,n-1}
        position_levels=np.asarray(cfg.position_levels),
        n_actions=np.array([len(cfg.position_levels)]),
    )
    size_mb = out_path.stat().st_size / (1024 * 1024)
    LOG.info("Saved dataset -> %s  (%.2f MB)", out_path, size_mb)

    return out_path


# ── CLI entry point ───────────────────────────────────────────────────────

def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
    )

    parser = argparse.ArgumentParser(description="Build offline RL dataset")
    parser.add_argument(
        "--frequency", choices=["daily", "hourly"], default="daily",
        help="Data frequency (default: daily)",
    )
    parser.add_argument(
        "--drawdown_penalty", type=float, default=1.0,
        help="Drawdown penalty coefficient (default: 1.0, 0=disabled)",
    )
    parser.add_argument(
        "--drawdown_threshold", type=float, default=0.02,
        help="Free drawdown allowance before penalty (default: 0.02 = 2%%)",
    )
    args = parser.parse_args()

    if args.frequency == "hourly":
        cfg = DatasetConfig(
            data_path=Path("data/processed/btc_hourly_train.parquet"),
            out_filename="offline_dataset_hourly_train.npz",
            log_return_column="log_return_next_1h",
            drawdown_penalty=args.drawdown_penalty,
            drawdown_threshold=args.drawdown_threshold,
            position_levels=POSITION_LEVELS_7,
        )
    else:
        cfg = DatasetConfig(
            drawdown_penalty=args.drawdown_penalty,
            drawdown_threshold=args.drawdown_threshold,
        )

    path = build_offline_dataset(cfg)
    print(f"Done.  Dataset saved to: {path}")


if __name__ == "__main__":
    main()

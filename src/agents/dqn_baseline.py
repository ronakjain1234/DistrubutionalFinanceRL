"""
Offline DQN / DoubleDQN baseline agent (Step 6).

Wraps ``d3rlpy``'s discrete DQN and DoubleDQN implementations with
project-specific defaults and helpers.

Why DQN as a baseline?
----------------------
DQN trained on offline data is *not* expected to work well — it has no
mechanism to handle extrapolation error.  That is exactly the point: it
serves as a **lower bound** that shows what happens when you naïvely apply
standard value-based RL to a fixed dataset without conservatism (CQL) or
distributional corrections.  If DQN already matches buy-and-hold, the
dataset is easy; if it fails badly, we know the offline corrections in
later steps are earning their keep.

Action mapping reminder
-----------------------
d3rlpy works with discrete actions in {0, 1, 2}.
The environment uses {-1, 0, +1}.
Conversion: ``d3rlpy_action = env_action + 1``.
The offline dataset builder (``build_offline_dataset.to_d3rlpy_dataset``)
handles this automatically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import d3rlpy
from d3rlpy.models.encoders import VectorEncoderFactory

LOG = logging.getLogger(__name__)


# ── Hyperparameter config ─────────────────────────────────────────────────

@dataclass
class DQNBaselineConfig:
    """All tuneable knobs for the DQN / DoubleDQN baseline."""

    # ── Algorithm choice ──────────────────────────────────────────────
    algo: Literal["dqn", "double_dqn"] = "double_dqn"
    """Which algorithm to use.  DoubleDQN reduces overestimation bias."""

    # ── Network architecture ──────────────────────────────────────────
    hidden_units: list[int] = field(default_factory=lambda: [256, 256])
    """Hidden layer sizes for the Q-network."""

    activation: str = "relu"
    """Activation function ('relu', 'tanh', etc.)."""

    # ── Optimisation ──────────────────────────────────────────────────
    learning_rate: float = 3e-4
    """Adam learning rate."""

    batch_size: int = 256
    """Mini-batch size for gradient updates."""

    gamma: float = 0.99
    """Discount factor."""

    n_critics: int = 1
    """Number of Q-networks (>1 gives ensemble, but 1 is standard DQN)."""

    target_update_interval: int = 1000
    """Steps between hard target-network updates."""

    # ── Training budget ───────────────────────────────────────────────
    n_steps: int = 50_000
    """Total gradient steps."""

    n_steps_per_epoch: int = 5_000
    """Steps per epoch (controls logging / checkpoint frequency)."""

    # ── Device ────────────────────────────────────────────────────────
    device: str | None = None
    """
    PyTorch device string.  ``None`` → auto-detect (CUDA if available,
    else CPU).
    """

    # ── Persistence ───────────────────────────────────────────────────
    save_dir: Path = Path("models/dqn_baseline")
    """Directory for saved model checkpoints."""


# ── Factory ───────────────────────────────────────────────────────────────

def _resolve_device(device: str | None) -> str:
    """Pick CUDA if available, otherwise CPU."""
    if device is not None:
        return device
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu:0"


def create_dqn(cfg: DQNBaselineConfig | None = None) -> d3rlpy.algos.QLearningAlgoBase:
    """
    Instantiate a d3rlpy DQN or DoubleDQN with project defaults.

    Returns an **untrained** algorithm object ready for ``.fit()``.
    """
    cfg = cfg or DQNBaselineConfig()
    device = _resolve_device(cfg.device)

    encoder = VectorEncoderFactory(
        hidden_units=cfg.hidden_units,
        activation=cfg.activation,
    )

    common = dict(
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        n_critics=cfg.n_critics,
        target_update_interval=cfg.target_update_interval,
        encoder_factory=encoder,
    )

    if cfg.algo == "double_dqn":
        algo_cfg = d3rlpy.algos.DoubleDQNConfig(**common)
    else:
        algo_cfg = d3rlpy.algos.DQNConfig(**common)

    algo = algo_cfg.create(device=device)
    LOG.info(
        "Created %s  (device=%s, hidden=%s, lr=%s, batch=%d, gamma=%.3f)",
        cfg.algo.upper(), device, cfg.hidden_units,
        cfg.learning_rate, cfg.batch_size, cfg.gamma,
    )
    return algo


# ── Save / load helpers ───────────────────────────────────────────────────

def save_model(algo: d3rlpy.algos.QLearningAlgoBase, path: str | Path) -> Path:
    """Save the full model (weights + config) to *path*.d3"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    algo.save(str(p))
    LOG.info("Model saved → %s", p)
    return p


def load_model(
    path: str | Path,
    device: str | None = None,
) -> d3rlpy.algos.QLearningAlgoBase:
    """
    Load a saved d3rlpy model.

    The model type (DQN / DoubleDQN) is inferred from the saved config.
    """
    device = _resolve_device(device)
    # d3rlpy ≥ 2.x: use class method .from_json for config, then load weights
    # Simplest approach: use d3rlpy.load_learnable
    algo = d3rlpy.load_learnable(str(path), device=device)
    LOG.info("Model loaded ← %s  (device=%s)", path, device)
    return algo

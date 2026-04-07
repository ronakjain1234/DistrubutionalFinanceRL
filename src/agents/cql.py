"""
Discrete Conservative Q-Learning (CQL) agent (Step 7).

Wraps ``d3rlpy``'s ``DiscreteCQL`` with project-specific defaults and
helpers.  Discrete CQL is the right tool for our three-action setup
({-1, 0, +1} → short / flat / long) and acts as our first *true* offline
RL baseline: it has an explicit mechanism to suppress value estimates
for out-of-distribution actions.

Why CQL?
--------
The DQN baseline (Step 6) has no defense against extrapolation error.
On a fixed dataset, its Q-function can become arbitrarily optimistic
about state-action pairs that the behaviour policies rarely visited —
and because it greedily picks the argmax, those over-confident values
directly drive the policy into regions with no historical support.

CQL adds a regulariser to the standard TD loss::

    L_CQL(θ) = L_TD(θ) + α · E_s [ logsumexp_a Q_θ(s,a) - Q_θ(s, a_data) ]

The extra term:

* Pushes **down** Q-values that the network would otherwise push up for
  unseen actions (the logsumexp acts like a soft max over all actions).
* Pushes **up** the Q-values of actions that actually appear in the
  dataset, effectively anchoring them to their Bellman targets.

The trade-off is controlled by ``α``:

* ``α = 0``  → reduces to DoubleDQN (no conservatism).
* ``α ≈ 1``  → gentle anchoring (d3rlpy default).
* ``α >> 1`` → aggressive anchoring; policy will stick close to the
  behaviour distribution even if that sacrifices reward.

In volatile crypto markets, a *slightly* conservative policy that tracks
historically profitable behaviour is usually safer than a bold policy
trained on optimistic extrapolation.

Architecture defaults
---------------------
Mirrors the DQN baseline so comparisons are apples-to-apples:

* Double-DQN target (d3rlpy's ``DiscreteCQL`` subclasses DoubleDQN).
* N-critic ensemble (``n_critics``) — the min over ensemble members
  further dampens overestimation on OOD actions.
* Layer norm + dropout in the MLP encoder (regularisation for the
  relatively small offline dataset).
* AdamW-style weight decay and gradient clipping.

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

import d3rlpy
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.optimizers.optimizers import AdamFactory

LOG = logging.getLogger(__name__)


# ── Hyperparameter config ─────────────────────────────────────────────────

@dataclass
class CQLConfig:
    """All tuneable knobs for the Discrete CQL agent."""

    # ── Conservatism ──────────────────────────────────────────────────
    alpha: float = 1.0
    """
    CQL conservatism weight.  Controls how strongly the agent is pulled
    toward actions seen in the dataset.

    * 0.0  → equivalent to DoubleDQN
    * 0.5  → mild anchoring (often enough when dataset coverage is good)
    * 1.0  → d3rlpy default, balanced
    * 2-4  → aggressive; use when dataset has narrow action coverage
    """

    # ── Network architecture ──────────────────────────────────────────
    hidden_units: list[int] = field(default_factory=lambda: [256, 256])
    """
    Hidden layer sizes for the Q-network.  CQL benefits from slightly
    larger networks than plain DQN because the conservatism penalty
    implicitly shrinks Q-values and a bigger net preserves expressiveness.
    """

    activation: str = "relu"
    """Activation function ('relu', 'tanh', etc.)."""

    use_layer_norm: bool = True
    """
    Apply layer normalisation after each hidden layer.  Recommended for
    offline RL — stabilises training when Q-values fluctuate wildly
    because of the conservative penalty.
    """

    dropout_rate: float | None = 0.1
    """Dropout probability (None = no dropout)."""

    # ── Optimisation ──────────────────────────────────────────────────
    learning_rate: float = 1e-4
    """
    Adam learning rate.  Slightly lower than the DQN baseline because
    CQL's extra loss term can make updates noisier.  d3rlpy's own
    default is 6.25e-5, which is quite conservative; 1e-4 trains faster
    without instability on our ~16k-transition dataset.
    """

    weight_decay: float = 1e-4
    """L2 weight decay (AdamW-style)."""

    clip_grad_norm: float | None = 1.0
    """Max gradient norm for clipping (None = no clipping)."""

    batch_size: int = 256
    """Mini-batch size for gradient updates."""

    gamma: float = 0.95
    """
    Discount factor.  Kept at 0.95 (below the RL-theory default of 0.99)
    to bound the horizon over which Q-estimation errors can accumulate —
    important for offline learning on noisy financial rewards.
    """

    n_critics: int = 3
    """
    Number of Q-networks in the ensemble.  Bootstrapped targets take
    the min over ensemble members, which is a second line of defense
    against Q-value overestimation on OOD actions.
    """

    target_update_interval: int = 2_000
    """
    Steps between hard target-network updates.  Larger than DQN's 1000
    because CQL's loss surface changes more slowly under the
    conservative regulariser — fresher targets just add noise.
    """

    # ── Training budget ───────────────────────────────────────────────
    n_steps: int = 30_000
    """
    Total gradient steps.  50% more than the DQN baseline — CQL needs
    a bit more wall-clock to converge because the conservative term
    initially dominates the TD loss before the Q-values settle.
    """

    n_steps_per_epoch: int = 2_000
    """Steps per epoch (controls logging / checkpoint / early-stop granularity)."""

    # ── Early stopping ────────────────────────────────────────────────
    early_stopping_patience: int = 4
    """
    Stop if val Sharpe doesn't improve for this many epochs.
    Slightly more patient than DQN (3) because CQL's validation curve
    is noisier early in training.  0 = disabled.
    """

    # ── Device ────────────────────────────────────────────────────────
    device: str | None = None
    """
    PyTorch device string.  ``None`` → auto-detect (CUDA if available,
    else CPU).
    """

    # ── Persistence ───────────────────────────────────────────────────
    save_dir: Path = Path("models/cql")
    """Directory for saved model checkpoints."""


# ── Factory ───────────────────────────────────────────────────────────────

def _resolve_device(device: str | None) -> str:
    """Pick CUDA if available, otherwise CPU."""
    if device is not None:
        return device
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu:0"


def create_cql(cfg: CQLConfig | None = None) -> d3rlpy.algos.QLearningAlgoBase:
    """
    Instantiate a d3rlpy ``DiscreteCQL`` with project defaults.

    Returns an **untrained** algorithm object ready for ``.fit()``.
    """
    cfg = cfg or CQLConfig()
    device = _resolve_device(cfg.device)

    encoder = VectorEncoderFactory(
        hidden_units=cfg.hidden_units,
        activation=cfg.activation,
        use_layer_norm=cfg.use_layer_norm,
        dropout_rate=cfg.dropout_rate,
    )

    optim = AdamFactory(
        weight_decay=cfg.weight_decay,
        clip_grad_norm=cfg.clip_grad_norm,
    )

    algo_cfg = d3rlpy.algos.DiscreteCQLConfig(
        learning_rate=cfg.learning_rate,
        optim_factory=optim,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        n_critics=cfg.n_critics,
        target_update_interval=cfg.target_update_interval,
        encoder_factory=encoder,
        alpha=cfg.alpha,
    )

    algo = algo_cfg.create(device=device)
    LOG.info(
        "Created DiscreteCQL  (device=%s, alpha=%.3f, hidden=%s, lr=%.1e, "
        "batch=%d, gamma=%.3f, dropout=%.2f, layer_norm=%s, weight_decay=%.1e, "
        "clip_grad=%.1f, n_critics=%d, target_update=%d)",
        device, cfg.alpha, cfg.hidden_units, cfg.learning_rate,
        cfg.batch_size, cfg.gamma, cfg.dropout_rate or 0.0, cfg.use_layer_norm,
        cfg.weight_decay, cfg.clip_grad_norm or 0.0, cfg.n_critics,
        cfg.target_update_interval,
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
    """Load a saved d3rlpy CQL model."""
    device = _resolve_device(device)
    algo = d3rlpy.load_learnable(str(path), device=device)
    LOG.info("Model loaded ← %s  (device=%s)", path, device)
    return algo

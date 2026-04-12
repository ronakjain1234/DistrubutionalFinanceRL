"""
Distributional Conservative Q-Learning (QR-CQL) agent (Step 8).

Extends CQL with quantile regression to model the full return distribution
per action, enabling risk-sensitive decision making for crypto trading.

Architecture
------------
Combines three mechanisms that reinforce each other in offline RL:

1. **Quantile Regression DQN (QR-DQN)**: Models Q(s,a) as N quantile
   values theta_1(s,a)...theta_N(s,a), trained with quantile Huber loss.
   Captures the full return distribution — not just its mean.

2. **Conservative Q-Learning (CQL)**: Penalizes Q-values for out-of-
   distribution actions. Applied to mean Q-values plus optionally to
   upper-tail quantiles, suppressing optimistic extrapolation.

3. **Ensemble uncertainty**: K independent quantile networks whose
   disagreement signals epistemic uncertainty. Pessimistic aggregation
   (min-of-means) automatically down-weights poorly-supported actions.

Novel enhancements over standard QR-DQN + CQL:

* **Tail-aware CQL penalty**: An additional conservative regularizer on
  the upper-tail quantiles (75th-100th pctile) of OOD actions. This
  prevents the agent from extrapolating optimistic upside for actions not
  well-covered by the dataset — critical in volatile crypto markets where
  tail events dominate.

* **Tail-focused quantile levels**: Non-uniform tau spacing that
  concentrates more quantile levels in the tails (below 15th and above
  85th percentile), improving CVaR and lower-quantile estimation for
  risk-sensitive action selection.

* **Multiple action selection modes**: mean-greedy, CVaR-optimizing,
  lower-quantile-maximizing, and ensemble-uncertainty-penalized selection
  rules — all from the same trained model.

Action mapping
--------------
Network outputs actions in {0, 1, 2} (d3rlpy convention).
Environment uses {-1, 0, +1}.  Conversion: env_action = network_action - 1.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

LOG = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────────────

@dataclass
class DistCQLConfig:
    """Hyperparameters for the Distributional CQL agent."""

    # ── Distributional ────────────────────────────────────────────────
    n_quantiles: int = 51
    """
    Number of quantile levels.  51 gives ~2% resolution including the
    median.  20-51 recommended for our ~16k-transition dataset.
    """

    kappa: float = 1.0
    """Huber loss threshold for quantile regression."""

    tail_quantile_focus: bool = True
    """
    Use non-uniform quantile levels with denser coverage in the tails
    (below 0.15 and above 0.85).  Improves CVaR and lower-quantile
    estimates for risk-sensitive trading decisions.
    """

    # ── CQL conservatism ─────────────────────────────────────────────
    alpha: float = 1.0
    """CQL conservatism weight on mean Q-values."""

    alpha_tail: float = 0.5
    """
    Additional CQL penalty on upper-tail quantiles (75th-100th pctile)
    for OOD actions.  Suppresses optimistic upside extrapolation without
    affecting downside estimates.  Set to 0 to disable.
    """

    # ── Network architecture ─────────────────────────────────────────
    hidden_units: list[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    use_layer_norm: bool = True
    dropout_rate: float = 0.1

    # ── Ensemble ─────────────────────────────────────────────────────
    n_ensemble: int = 3
    """Number of independent quantile networks."""

    ensemble_penalty: float = 0.0
    """
    Coefficient for ensemble disagreement penalty during action selection.
    0 = pessimistic min-of-means (default).
    >0 = mean - penalty * std (soft uncertainty penalty).
    """

    # ── Optimisation ─────────────────────────────────────────────────
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    clip_grad_norm: float = 1.0
    batch_size: int = 256
    gamma: float = 0.95
    target_update_interval: int = 2_000

    # ── Training budget ──────────────────────────────────────────────
    n_steps: int = 30_000
    n_steps_per_epoch: int = 2_000

    # ── Early stopping ───────────────────────────────────────────────
    early_stopping_patience: int = 4

    # ── Device ───────────────────────────────────────────────────────
    device: str | None = None

    # ── Persistence ──────────────────────────────────────────────────
    save_dir: Path = Path("models/dist_cql")

    # ── Action selection ─────────────────────────────────────────────
    action_selection: str = "mean"
    """
    Default action selection for predict():
      "mean"        — argmax of pessimistic mean Q (min over ensemble)
      "cvar_10"     — maximise CVaR at the 10th percentile
      "cvar_25"     — maximise CVaR at the 25th percentile
      "quantile_10" — maximise the 10th-percentile quantile value
    """

    seed: int = 42


# ── Helpers ────────────────────────────────────────────────────────────────

def _resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device.replace("cpu:0", "cpu").replace("cuda:0", "cuda"))
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_quantile_taus(n_quantiles: int, tail_focus: bool = False) -> np.ndarray:
    """
    Generate quantile levels tau_1 ... tau_N in (0, 1).

    Uniform:       tau_i = (2i - 1) / (2N)
    Tail-focused:  ~30% of levels in each tail for better risk estimates.
    """
    if not tail_focus:
        return np.array(
            [(2 * i - 1) / (2 * n_quantiles) for i in range(1, n_quantiles + 1)],
            dtype=np.float32,
        )

    n_lower = max(1, int(0.30 * n_quantiles))
    n_upper = max(1, int(0.30 * n_quantiles))
    n_center = n_quantiles - n_lower - n_upper
    if n_center < 1:
        return np.linspace(0.01, 0.99, n_quantiles, dtype=np.float32)

    lower = np.linspace(0.01, 0.15, n_lower, dtype=np.float32)
    center = np.linspace(0.17, 0.83, n_center, dtype=np.float32)
    upper = np.linspace(0.85, 0.99, n_upper, dtype=np.float32)

    taus = np.concatenate([lower, center, upper])
    taus.sort()
    return taus


# ── Quantile Network ──────────────────────────────────────────────────────

class QuantileNetwork(nn.Module):
    """
    MLP that outputs N quantile values per action.

    Forward: (batch, obs_dim) -> (batch, n_actions, n_quantiles)
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        n_quantiles: int,
        hidden_units: list[int],
        use_layer_norm: bool = True,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles

        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_units:
            layers.append(nn.Linear(in_dim, h))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions * n_quantiles))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).view(-1, self.n_actions, self.n_quantiles)


# ── Distributional CQL Agent ─────────────────────────────────────────────

class DistributionalCQLAgent:
    """
    Full distributional CQL agent: training, inference, persistence.

    Implements ``predict(obs) -> actions`` for compatibility with the
    evaluation harness in ``eval_policies.py``.
    """

    def __init__(
        self,
        cfg: DistCQLConfig,
        obs_dim: int,
        n_actions: int = 3,
    ):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = _resolve_device(cfg.device)

        self.taus = torch.tensor(
            make_quantile_taus(cfg.n_quantiles, cfg.tail_quantile_focus),
            dtype=torch.float32,
            device=self.device,
        )

        # Ensemble of quantile networks + frozen targets
        self.networks: list[QuantileNetwork] = []
        self.target_networks: list[QuantileNetwork] = []
        self.optimizers: list[torch.optim.AdamW] = []

        for _ in range(cfg.n_ensemble):
            net = QuantileNetwork(
                obs_dim, n_actions, cfg.n_quantiles,
                cfg.hidden_units, cfg.use_layer_norm, cfg.dropout_rate,
            ).to(self.device)

            target = copy.deepcopy(net)
            target.requires_grad_(False)

            optim = torch.optim.AdamW(
                net.parameters(), lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )

            self.networks.append(net)
            self.target_networks.append(target)
            self.optimizers.append(optim)

        self._step_count = 0

        LOG.info(
            "Created DistributionalCQL  (device=%s, quantiles=%d, ensemble=%d, "
            "alpha=%.2f, alpha_tail=%.2f, hidden=%s, lr=%.1e, tail_focus=%s)",
            self.device, cfg.n_quantiles, cfg.n_ensemble,
            cfg.alpha, cfg.alpha_tail, cfg.hidden_units,
            cfg.learning_rate, cfg.tail_quantile_focus,
        )

    # ── Target network management ────────────────────────────────────

    def _hard_update_targets(self) -> None:
        for net, target in zip(self.networks, self.target_networks):
            target.load_state_dict(net.state_dict())

    # ── Loss functions ───────────────────────────────────────────────

    def _quantile_huber_loss(
        self,
        current: torch.Tensor,   # (batch, N)
        target: torch.Tensor,    # (batch, N)
    ) -> torch.Tensor:
        """Pairwise quantile Huber loss."""
        kappa = self.cfg.kappa

        # delta[b, i, j] = target[b, j] - current[b, i]
        delta = target.unsqueeze(1) - current.unsqueeze(2)  # (B, N, N)
        abs_delta = delta.abs()

        huber = torch.where(
            abs_delta <= kappa,
            0.5 * delta.pow(2),
            kappa * (abs_delta - 0.5 * kappa),
        )

        # weight = |tau_i - I(delta < 0)|
        indicator = (delta < 0).float()
        taus = self.taus.view(1, -1, 1)
        weight = (taus - indicator).abs()

        # Sum over target quantiles, mean over current quantiles and batch
        return (weight * huber / kappa).sum(dim=2).mean(dim=1).mean()

    def _cql_loss(
        self,
        all_quantiles: torch.Tensor,  # (B, n_actions, N)
        batch_actions: torch.Tensor,  # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        CQL regularisation on mean and upper-tail Q-values.

        Returns (mean_cql_loss, tail_cql_loss).
        """
        # Mean Q per action
        q_mean = all_quantiles.mean(dim=-1)                          # (B, A)
        logsumexp = torch.logsumexp(q_mean, dim=1)                   # (B,)
        q_data = q_mean.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
        mean_cql = (logsumexp - q_data).mean()

        # Upper-tail CQL (75th-100th pctile mean)
        tail_cql = torch.tensor(0.0, device=self.device)
        if self.cfg.alpha_tail > 0:
            n_upper = max(1, self.cfg.n_quantiles // 4)
            q_upper = all_quantiles[:, :, -n_upper:].mean(dim=-1)
            lse_upper = torch.logsumexp(q_upper, dim=1)
            q_upper_data = q_upper.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
            tail_cql = (lse_upper - q_upper_data).mean()

        return mean_cql, tail_cql

    # ── Single training step ─────────────────────────────────────────

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """One gradient update across all ensemble members."""
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        terminals = batch["terminals"]

        sum_qr = sum_cql_m = sum_cql_t = sum_total = 0.0

        for k in range(self.cfg.n_ensemble):
            net = self.networks[k]
            target_net = self.target_networks[k]
            optimizer = self.optimizers[k]
            net.train()

            # Current quantiles for taken action: (B, N)
            all_q = net(obs)                          # (B, A, N)
            idx = actions.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.cfg.n_quantiles)
            current_q = all_q.gather(1, idx).squeeze(1)

            # Target quantiles (Double-DQN action selection)
            with torch.no_grad():
                next_q_target = target_net(next_obs)          # (B, A, N)
                next_q_online = net(next_obs)
                next_a = next_q_online.mean(dim=-1).argmax(dim=1)  # (B,)
                idx_next = next_a.unsqueeze(1).unsqueeze(2).expand(
                    -1, 1, self.cfg.n_quantiles
                )
                target_vals = next_q_target.gather(1, idx_next).squeeze(1)

                bellman = rewards.unsqueeze(1) + (
                    self.cfg.gamma * (1.0 - terminals.unsqueeze(1)) * target_vals
                )

            qr_loss = self._quantile_huber_loss(current_q, bellman)
            cql_m, cql_t = self._cql_loss(all_q, actions)
            loss = qr_loss + self.cfg.alpha * cql_m + self.cfg.alpha_tail * cql_t

            optimizer.zero_grad()
            loss.backward()
            if self.cfg.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(net.parameters(), self.cfg.clip_grad_norm)
            optimizer.step()

            sum_qr += qr_loss.item()
            sum_cql_m += cql_m.item()
            sum_cql_t += cql_t.item()
            sum_total += loss.item()

        self._step_count += 1
        if self._step_count % self.cfg.target_update_interval == 0:
            self._hard_update_targets()

        K = self.cfg.n_ensemble
        return {
            "qr_loss": sum_qr / K,
            "cql_mean_loss": sum_cql_m / K,
            "cql_tail_loss": sum_cql_t / K,
            "total_loss": sum_total / K,
        }

    # ── Full training loop ───────────────────────────────────────────

    def fit(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        terminals: np.ndarray,
        *,
        n_steps: int | None = None,
        n_steps_per_epoch: int | None = None,
        epoch_callback: Callable | None = None,
        show_progress: bool = True,
    ) -> list[dict[str, float]]:
        """
        Train on an offline dataset.

        Parameters
        ----------
        epoch_callback
            ``fn(agent, epoch, total_step) -> bool``.
            Return ``True`` to signal early stop.
        """
        n_steps = n_steps or self.cfg.n_steps
        n_steps_per_epoch = n_steps_per_epoch or self.cfg.n_steps_per_epoch

        # Clean NaN rewards
        bad = ~np.isfinite(rewards)
        if bad.any():
            LOG.warning("Replacing %d NaN rewards with 0.0", bad.sum())
            rewards = rewards.copy()
            rewards[bad] = 0.0

        obs_t = torch.tensor(observations, dtype=torch.float32, device=self.device)
        act_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rew_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        nobs_t = torch.tensor(next_observations, dtype=torch.float32, device=self.device)
        term_t = torch.tensor(terminals, dtype=torch.float32, device=self.device)
        n_data = len(observations)

        torch.manual_seed(self.cfg.seed)

        history: list[dict[str, float]] = []
        epoch = 0

        pbar = tqdm(range(1, n_steps + 1), desc="DistCQL", disable=not show_progress)
        for step in pbar:
            idx = torch.randint(0, n_data, (self.cfg.batch_size,))
            batch = {
                "obs": obs_t[idx],
                "actions": act_t[idx],
                "rewards": rew_t[idx],
                "next_obs": nobs_t[idx],
                "terminals": term_t[idx],
            }

            losses = self.train_step(batch)
            history.append(losses)

            if step % 100 == 0:
                pbar.set_postfix(
                    loss=f"{losses['total_loss']:.4f}",
                    qr=f"{losses['qr_loss']:.4f}",
                    cql=f"{losses['cql_mean_loss']:.4f}",
                )

            if step % n_steps_per_epoch == 0:
                epoch += 1
                if epoch_callback is not None:
                    should_stop = epoch_callback(self, epoch, step)
                    if should_stop is True:
                        LOG.info("Early stop signalled at epoch %d (step %d).", epoch, step)
                        break

        return history

    # ── Inference helpers ────────────────────────────────────────────

    @torch.no_grad()
    def _ensemble_quantiles(self, obs: torch.Tensor) -> torch.Tensor:
        """Mean quantile values across ensemble: (B, A, N)."""
        stacked = torch.stack(
            [net.eval()(obs) for net in self.networks], dim=0
        )
        return stacked.mean(dim=0)

    @torch.no_grad()
    def _ensemble_mean_q(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Pessimistic mean-Q: min over ensemble of per-member mean-Q.

        Returns (B, A).
        """
        member_means = torch.stack(
            [net.eval()(obs).mean(dim=-1) for net in self.networks], dim=0
        )  # (K, B, A)
        return member_means.min(dim=0).values

    # ── Public predict interface ─────────────────────────────────────

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Select actions. Returns (B,) int64 in {0, 1, 2}.

        Compatible with ``eval_policies.rollout_policy``.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        obs = torch.tensor(x, dtype=torch.float32, device=self.device)

        strategy = self.cfg.action_selection

        if strategy == "mean":
            if self.cfg.ensemble_penalty > 0:
                member_means = torch.stack(
                    [net.eval()(obs).mean(dim=-1) for net in self.networks], dim=0
                )
                q_select = member_means.mean(0) - self.cfg.ensemble_penalty * member_means.std(0)
            else:
                q_select = self._ensemble_mean_q(obs)
            actions = q_select.argmax(dim=1)

        elif strategy.startswith("cvar_"):
            alpha = int(strategy.split("_")[1]) / 100.0
            quantiles = self._ensemble_quantiles(obs)
            taus_np = self.taus.cpu().numpy()
            mask = torch.tensor(taus_np <= alpha, device=self.device)
            if not mask.any():
                mask[0] = True
            cvar = quantiles[:, :, mask].mean(dim=-1)
            actions = cvar.argmax(dim=1)

        elif strategy.startswith("quantile_"):
            tau_target = int(strategy.split("_")[1]) / 100.0
            quantiles = self._ensemble_quantiles(obs)
            taus_np = self.taus.cpu().numpy()
            idx = int(np.argmin(np.abs(taus_np - tau_target)))
            actions = quantiles[:, :, idx].argmax(dim=1)

        else:
            raise ValueError(f"Unknown action_selection: {strategy!r}")

        return actions.cpu().numpy().astype(np.int64)

    @torch.no_grad()
    def predict_value(self, x: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Mean Q-value for (obs, action) pairs.

        Compatible with Q-stats diagnostics.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        obs = torch.tensor(x, dtype=torch.float32, device=self.device)
        act = torch.tensor(actions, dtype=torch.long, device=self.device)

        quantiles = self._ensemble_quantiles(obs)          # (B, A, N)
        q_mean = quantiles.mean(dim=-1)                    # (B, A)
        return q_mean.gather(1, act.unsqueeze(1)).squeeze(1).cpu().numpy()

    @torch.no_grad()
    def predict_quantiles(self, x: np.ndarray) -> np.ndarray:
        """Full quantile distribution: (B, A, N)."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        obs = torch.tensor(x, dtype=torch.float32, device=self.device)
        return self._ensemble_quantiles(obs).cpu().numpy()

    def get_taus(self) -> np.ndarray:
        return self.taus.cpu().numpy()

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        cfg_dict = {}
        for k, v in self.cfg.__dict__.items():
            cfg_dict[k] = str(v) if isinstance(v, Path) else v
        (p / "config.json").write_text(json.dumps(cfg_dict, indent=2))

        meta = {
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "step_count": self._step_count,
        }
        (p / "meta.json").write_text(json.dumps(meta, indent=2))
        np.save(str(p / "taus.npy"), self.taus.cpu().numpy())

        for k, (net, target) in enumerate(zip(self.networks, self.target_networks)):
            torch.save(net.state_dict(), p / f"network_{k}.pt")
            torch.save(target.state_dict(), p / f"target_{k}.pt")

        LOG.info("Agent saved -> %s", p)
        return p

    @classmethod
    def load(
        cls, path: str | Path, device: str | None = None,
    ) -> "DistributionalCQLAgent":
        p = Path(path)

        cfg_dict = json.loads((p / "config.json").read_text())
        if "save_dir" in cfg_dict:
            cfg_dict["save_dir"] = Path(cfg_dict["save_dir"])
        if device is not None:
            cfg_dict["device"] = device
        cfg = DistCQLConfig(**cfg_dict)

        meta = json.loads((p / "meta.json").read_text())
        agent = cls(cfg, obs_dim=meta["obs_dim"], n_actions=meta["n_actions"])
        agent._step_count = meta["step_count"]

        for k in range(cfg.n_ensemble):
            agent.networks[k].load_state_dict(
                torch.load(p / f"network_{k}.pt", map_location=agent.device, weights_only=True)
            )
            agent.target_networks[k].load_state_dict(
                torch.load(p / f"target_{k}.pt", map_location=agent.device, weights_only=True)
            )

        LOG.info("Agent loaded <- %s", p)
        return agent

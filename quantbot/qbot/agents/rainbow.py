"""Agent Rainbow (Hessel et al., 2018) + Munchausen (Vieillard et al., 2020).

Extensions intégrées, chacune désactivable pour permettre une véritable étude d'ablation :

  1. Double Q-learning     — supprime le biais d'optimisme de max_a Q
  2. Dueling                — sépare V(s) et A(s,a)
  3. Prioritized replay     — échantillonne les transitions informatives
  4. Retours n-step         — propagation rapide du crédit
  5. NoisyNet               — exploration apprise, spécifique à l'état
  6. Distributional (QR/C51)— apprend la loi du retour, pas seulement sa moyenne
  7. Munchausen             — bonus d'entropie implicite, gain quasi gratuit

Pourquoi le point 6 compte particulièrement ici : en finance, deux actions de même
espérance mais de distributions différentes ne sont PAS interchangeables. Modéliser les
quantiles permet en plus de basculer la politique sur un critère CVaR, c'est-à-dire de
maximiser le pire décile plutôt que la moyenne.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AgentConfig
from ..utils.logging import get_logger
from .networks import QNetwork
from .replay import ObsReconstructor, PrioritizedReplayBuffer

log = get_logger("agents.rainbow")


def resolve_device(spec: str = "auto") -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


class RainbowAgent:
    """Agent Q distributionnel à espace d'actions discret."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        n_features: int,
        window: int,
        n_portfolio: int,
        cfg: Optional[AgentConfig] = None,
        seed: int = 0,
    ):
        self.cfg = cfg or AgentConfig()
        self.obs_dim, self.n_actions = obs_dim, n_actions
        self.n_features, self.window, self.n_portfolio = n_features, window, n_portfolio
        self.device = resolve_device(self.cfg.device)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        # L'algo "dqn" désactive toutes les extensions : sert de référence d'ablation.
        vanilla = self.cfg.algo == "dqn"
        self.double_q = self.cfg.double_q and not vanilla
        self.noisy = self.cfg.noisy and not vanilla
        self.prioritized = self.cfg.prioritized and not vanilla
        self.munchausen = self.cfg.munchausen and not vanilla
        self.n_step = 1 if vanilla else max(int(self.cfg.n_step), 1)
        self.distributional = "none" if vanilla else self.cfg.distributional

        net_kwargs = dict(
            obs_dim=obs_dim, n_actions=n_actions, n_features=n_features, window=window,
            extra_dim=n_portfolio, encoder=self.cfg.encoder,
            hidden_sizes=tuple(self.cfg.hidden_sizes), encoder_hidden=self.cfg.encoder_hidden,
            tcn_channels=tuple(self.cfg.tcn_channels), tcn_kernel=self.cfg.tcn_kernel,
            dropout=self.cfg.dropout, layer_norm=self.cfg.layer_norm,
            noisy=self.noisy, sigma0=self.cfg.noisy_sigma0,
            dueling=self.cfg.dueling and not vanilla, distributional=self.distributional,
            n_quantiles=self.cfg.n_quantiles, n_atoms=self.cfg.n_atoms,
            v_min=self.cfg.v_min, v_max=self.cfg.v_max,
        )
        self.online = QNetwork(**net_kwargs).to(self.device)
        self.target = copy.deepcopy(self.online).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.target.eval()

        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=self.cfg.lr, eps=self.cfg.adam_eps,
            weight_decay=self.cfg.weight_decay,
        )

        self.buffer = PrioritizedReplayBuffer(
            capacity=self.cfg.buffer_size, n_portfolio=n_portfolio,
            alpha=self.cfg.per_alpha, beta0=self.cfg.per_beta0,
            beta_steps=self.cfg.per_beta_steps, eps=self.cfg.per_eps,
            prioritized=self.prioritized,
        )
        self.reconstructor: Optional[ObsReconstructor] = None
        self.gamma = float(self.cfg.gamma)
        self.train_steps = 0
        self.env_steps = 0

    # ---------------------------------------------------------------------------------
    def bind_features(self, features: np.ndarray) -> None:
        """Associe la matrice de features utilisée pour reconstruire les observations.

        À rappeler à chaque changement de segment (fold de walk-forward) : sans cela, le
        tampon reconstruirait des observations à partir des features d'un AUTRE segment,
        ce qui mélange silencieusement les périodes.
        """
        self.reconstructor = ObsReconstructor(
            features, self.window, include_portfolio=self.n_portfolio > 0
        )

    def _obs_tensor(self, t: np.ndarray, portfolio: np.ndarray) -> torch.Tensor:
        if self.reconstructor is None:
            raise RuntimeError("bind_features() n'a pas été appelé.")
        obs = self.reconstructor(t, portfolio)
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device)

    # ---------------------------------------------------------------------------------
    def epsilon(self) -> float:
        """ε résiduel : avec NoisyNet il est quasi nul, l'exploration venant du réseau."""
        if self.noisy:
            return 0.0
        c = self.cfg
        frac = min(self.env_steps / max(c.eps_decay_steps, 1), 1.0)
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    @torch.no_grad()
    def act(self, obs: np.ndarray, greedy: bool = False, cvar_alpha: Optional[float] = None) -> int:
        """Choisit une action.

        `cvar_alpha` bascule la politique sur un critère averse au risque : on maximise
        le CVaR du retour au lieu de son espérance. Utile en production quand le mandat
        impose une contrainte de perte extrême plutôt qu'un objectif de rendement moyen.
        """
        if not greedy and not self.noisy and self.rng.random() < self.epsilon():
            return int(self.rng.integers(0, self.n_actions))

        self.online.eval() if greedy else self.online.train()
        if self.noisy and not greedy:
            self.online.reset_noise()
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device).unsqueeze(0)
        scores = (self.online.risk_measure(x, cvar_alpha) if cvar_alpha
                  else self.online.q_values(x))
        return int(scores.argmax(dim=1).item())

    @torch.no_grad()
    def act_batch(self, obs: np.ndarray, cvar_alpha: Optional[float] = None) -> np.ndarray:
        """Inférence vectorisée — utilisée par le backtest pour éviter T appels réseau."""
        self.online.eval()
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        scores = (self.online.risk_measure(x, cvar_alpha) if cvar_alpha
                  else self.online.q_values(x))
        return scores.argmax(dim=1).cpu().numpy()

    # ---------------------------------------------------------------------------------
    def _quantile_huber_loss(
        self, pred: torch.Tensor, target: torch.Tensor, taus: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perte de régression quantile de Huber (Dabney et al., 2018).

        pred   : (B, N)  quantiles prédits pour l'action jouée
        target : (B, N') quantiles cibles
        La pondération |τ_i - 1{u<0}| est ce qui force le quantile i à converger vers le
        i-ème quantile de la loi du retour, et non vers sa moyenne.
        """
        u = target.unsqueeze(1) - pred.unsqueeze(2)              # (B, N, N')
        k = self.cfg.huber_kappa
        abs_u = u.abs()
        huber = torch.where(abs_u <= k, 0.5 * u.pow(2), k * (abs_u - 0.5 * k))
        weight = (taus.view(1, -1, 1) - (u.detach() < 0).float()).abs()
        loss_elem = weight * huber / k
        loss = loss_elem.mean(dim=2).sum(dim=1)                  # (B,)
        td_error = u.detach().abs().mean(dim=(1, 2))             # pour la priorisation
        return loss, td_error

    def _munchausen_terms(
        self, q_target_cur: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calcule τ·ln π(a|s) sur l'état courant et la politique softmax sur l'état suivant.

        Le terme de Munchausen ajoute la log-probabilité de l'action RÉELLEMENT jouée à la
        récompense. Effet : le bonus implicite d'entropie élargit l'écart entre action
        optimale et sous-optimales, ce qui accélère nettement la convergence — pour
        quelques lignes de code et aucun paramètre réseau supplémentaire.
        """
        tau = self.cfg.m_tau
        q = q_target_cur
        v = q.max(dim=1, keepdim=True).values
        log_pi = (q - v) - tau * torch.logsumexp((q - v) / tau, dim=1, keepdim=True)
        log_pi_a = log_pi.gather(1, actions.unsqueeze(1)).squeeze(1)
        return log_pi_a, log_pi

    # ---------------------------------------------------------------------------------
    def learn(self) -> Optional[Dict[str, float]]:
        """Une étape de descente de gradient. Retourne les métriques, ou None si le
        tampon n'est pas encore suffisamment rempli."""
        if len(self.buffer) < max(self.cfg.learn_start, self.cfg.batch_size):
            return None

        batch = self.buffer.sample(self.cfg.batch_size, self.train_steps, self.rng)
        obs = self._obs_tensor(batch["t"], batch["portfolio"])
        next_obs = self._obs_tensor(batch["next_t"], batch["next_portfolio"])
        actions = torch.as_tensor(batch["action"], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch["reward"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device)
        weights = torch.as_tensor(batch["weights"], dtype=torch.float32, device=self.device)
        n_steps = torch.as_tensor(batch["n_steps"], dtype=torch.float32, device=self.device)
        gamma_n = self.gamma ** n_steps

        self.online.train()
        if self.noisy:
            self.online.reset_noise()
            self.target.reset_noise()

        if self.distributional == "qr":
            loss_per_sample, td_error = self._loss_qr(obs, next_obs, actions, rewards, dones, gamma_n)
        elif self.distributional == "c51":
            loss_per_sample, td_error = self._loss_c51(obs, next_obs, actions, rewards, dones, gamma_n)
        else:
            loss_per_sample, td_error = self._loss_dqn(obs, next_obs, actions, rewards, dones, gamma_n)

        loss = (loss_per_sample * weights).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip)
        self.optimizer.step()

        self.buffer.update_priorities(batch["idx"], td_error.cpu().numpy())
        self.train_steps += 1
        self._maybe_update_target()

        return {
            "loss": float(loss.item()),
            "td_error": float(td_error.mean().item()),
            "grad_norm": float(grad_norm),
            "beta": self.buffer.beta(self.train_steps),
        }

    # --- variantes de perte ------------------------------------------------------------
    def _loss_qr(self, obs, next_obs, actions, rewards, dones, gamma_n):
        quantiles = self.online(obs)                                          # (B, A, N)
        pred = quantiles.gather(
            1, actions.view(-1, 1, 1).expand(-1, 1, quantiles.shape[2])
        ).squeeze(1)                                                          # (B, N)

        with torch.no_grad():
            next_q_target = self.target(next_obs)                             # (B, A, N)
            if self.munchausen:
                # Politique douce sur s' : on mélange les quantiles pondérés par π(a|s'),
                # au lieu de ne garder que l'argmax (extension distributionnelle de M-DQN).
                q_next_mean = next_q_target.mean(dim=2)
                tau = self.cfg.m_tau
                v_next = q_next_mean.max(dim=1, keepdim=True).values
                log_pi_next = (q_next_mean - v_next) - tau * torch.logsumexp(
                    (q_next_mean - v_next) / tau, dim=1, keepdim=True)
                pi_next = log_pi_next.div(tau).softmax(dim=1)                 # (B, A)
                target_quant = (pi_next.unsqueeze(2) *
                                (next_q_target - log_pi_next.unsqueeze(2))).sum(dim=1)

                q_cur_target = self.target(obs).mean(dim=2)
                log_pi_a, _ = self._munchausen_terms(q_cur_target, actions)
                munch = self.cfg.m_alpha * log_pi_a.clamp(min=self.cfg.m_clip, max=0.0)
                rewards = rewards + munch
            else:
                if self.double_q:
                    next_actions = self.online(next_obs).mean(dim=2).argmax(dim=1)
                else:
                    next_actions = next_q_target.mean(dim=2).argmax(dim=1)
                target_quant = next_q_target.gather(
                    1, next_actions.view(-1, 1, 1).expand(-1, 1, next_q_target.shape[2])
                ).squeeze(1)

            target = rewards.unsqueeze(1) + gamma_n.unsqueeze(1) * (1.0 - dones.unsqueeze(1)) * target_quant

        return self._quantile_huber_loss(pred, target, self.online.taus)

    def _loss_c51(self, obs, next_obs, actions, rewards, dones, gamma_n):
        n_atoms = self.cfg.n_atoms
        support = self.online.support
        delta_z = self.online.delta_z
        v_min, v_max = self.cfg.v_min, self.cfg.v_max

        dist = self.online(obs)                                                # (B, A, atoms)
        pred = dist.gather(1, actions.view(-1, 1, 1).expand(-1, 1, n_atoms)).squeeze(1)

        with torch.no_grad():
            next_dist = self.target(next_obs)                                  # (B, A, atoms)
            q_next = (next_dist * support.view(1, 1, -1)).sum(2)               # (B, A)

            if self.munchausen:
                tau = self.cfg.m_tau
                v_next = q_next.max(dim=1, keepdim=True).values
                log_pi_next = (q_next - v_next) - tau * torch.logsumexp(
                    (q_next - v_next) / tau, dim=1, keepdim=True)              # (B, A)
                pi_next = log_pi_next.div(tau).softmax(dim=1)

                q_cur = (self.target(obs) * support.view(1, 1, -1)).sum(2)
                log_pi_a, _ = self._munchausen_terms(q_cur, actions)
                rewards = rewards + self.cfg.m_alpha * log_pi_a.clamp(min=self.cfg.m_clip, max=0.0)

                # Support décalé PAR ACTION (le terme -τ·ln π dépend de a), donc projection
                # sur (B, A, atoms) puis mélange pondéré par la politique douce.
                tz = (rewards.view(-1, 1, 1)
                      + gamma_n.view(-1, 1, 1) * (1.0 - dones.view(-1, 1, 1))
                      * (support.view(1, 1, -1) - tau * log_pi_next.unsqueeze(2)))
                tz = tz.clamp(v_min, v_max)
                proj = self._project_atoms(tz, next_dist, n_atoms, v_min, delta_z)  # (B, A, atoms)
                target = (pi_next.unsqueeze(2) * proj).sum(dim=1)
                target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)
            else:
                if self.double_q:
                    next_actions = (self.online(next_obs) * support.view(1, 1, -1)).sum(2).argmax(1)
                else:
                    next_actions = q_next.argmax(1)
                next_p = next_dist.gather(
                    1, next_actions.view(-1, 1, 1).expand(-1, 1, n_atoms)).squeeze(1)
                tz = (rewards.unsqueeze(1)
                      + gamma_n.unsqueeze(1) * (1.0 - dones.unsqueeze(1)) * support.view(1, -1))
                target = self._project_atoms(tz.clamp(v_min, v_max).unsqueeze(1),
                                             next_p.unsqueeze(1), n_atoms, v_min, delta_z).squeeze(1)

        log_pred = torch.log(pred.clamp_min(1e-8))
        loss = -(target * log_pred).sum(dim=1)
        return loss, loss.detach()

    @staticmethod
    def _project_atoms(tz: torch.Tensor, probs: torch.Tensor, n_atoms: int,
                       v_min: float, delta_z: float) -> torch.Tensor:
        """Projette une loi décalée-dilatée sur le support fixe (Bellemare et al., 2017).

        tz, probs : (..., atoms). La masse de chaque atome déplacé est répartie sur les
        deux atomes du support qui l'encadrent, proportionnellement à la distance.
        """
        b = (tz - v_min) / delta_z
        lo, up = b.floor().long(), b.ceil().long()
        # Sans cette correction, un b exactement entier verrait sa masse annulée
        # (up - b = b - lo = 0) : une fuite de probabilité silencieuse.
        lo = torch.where((up > 0) & (lo == up), lo - 1, lo).clamp(0, n_atoms - 1)
        up = torch.where((lo < n_atoms - 1) & (lo == up), up + 1, up).clamp(0, n_atoms - 1)

        out = torch.zeros_like(probs)
        out.scatter_add_(-1, lo, probs * (up.float() - b))
        out.scatter_add_(-1, up, probs * (b - lo.float()))
        return out

    def _loss_dqn(self, obs, next_obs, actions, rewards, dones, gamma_n):
        q = self.online(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next_target = self.target(next_obs)
            if self.munchausen:
                # M-DQN (Vieillard et al., 2020) : récompense augmentée de α·τ·ln π(a|s),
                # et cible en espérance sous la politique douce plutôt qu'en max.
                tau = self.cfg.m_tau
                v_next = q_next_target.max(dim=1, keepdim=True).values
                log_pi_next = (q_next_target - v_next) - tau * torch.logsumexp(
                    (q_next_target - v_next) / tau, dim=1, keepdim=True)
                pi_next = log_pi_next.div(tau).softmax(dim=1)
                next_q = (pi_next * (q_next_target - log_pi_next)).sum(dim=1)

                log_pi_a, _ = self._munchausen_terms(self.target(obs), actions)
                rewards = rewards + self.cfg.m_alpha * log_pi_a.clamp(min=self.cfg.m_clip, max=0.0)
            else:
                if self.double_q:
                    next_actions = self.online(next_obs).argmax(dim=1)
                else:
                    next_actions = q_next_target.argmax(dim=1)
                next_q = q_next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards + gamma_n * (1.0 - dones) * next_q
        td = target - q
        k = self.cfg.huber_kappa
        loss = torch.where(td.abs() <= k, 0.5 * td.pow(2), k * (td.abs() - 0.5 * k))
        return loss, td.detach().abs()

    # ---------------------------------------------------------------------------------
    def _maybe_update_target(self) -> None:
        if self.cfg.target_soft_tau:
            tau = self.cfg.target_soft_tau
            with torch.no_grad():
                for tp, op in zip(self.target.parameters(), self.online.parameters()):
                    tp.mul_(1.0 - tau).add_(op, alpha=tau)
        elif self.train_steps % max(self.cfg.target_update_interval, 1) == 0:
            self.target.load_state_dict(self.online.state_dict())

    # ---------------------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "cfg": self.cfg.__dict__,
                "dims": {
                    "obs_dim": self.obs_dim, "n_actions": self.n_actions,
                    "n_features": self.n_features, "window": self.window,
                    "n_portfolio": self.n_portfolio,
                },
                "train_steps": self.train_steps, "env_steps": self.env_steps,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path, device: Optional[str] = None) -> "RainbowAgent":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg_raw = dict(ckpt["cfg"])
        for k, v in cfg_raw.items():
            if isinstance(v, list):
                cfg_raw[k] = tuple(v)
        cfg = AgentConfig(**cfg_raw)
        if device:
            cfg.device = device
        d = ckpt["dims"]
        agent = cls(d["obs_dim"], d["n_actions"], d["n_features"], d["window"],
                    d["n_portfolio"], cfg)
        agent.online.load_state_dict(ckpt["online"])
        agent.target.load_state_dict(ckpt["target"])
        agent.train_steps = ckpt.get("train_steps", 0)
        agent.env_steps = ckpt.get("env_steps", 0)
        agent.online.eval()
        return agent

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.online.parameters())

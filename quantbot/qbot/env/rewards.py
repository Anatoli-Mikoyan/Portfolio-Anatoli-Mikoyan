"""Fonctions de récompense.

Le choix de la récompense DÉFINIT la stratégie apprise. Maximiser le PnL brut produit un
agent qui prend un levier maximal et se fait détruire au premier régime défavorable :
c'est mathématiquement le comportement optimal pour cet objectif. Les récompenses
ajustées du risque encodent directement l'objectif réel d'un gérant — un ratio
rendement/risque stable — dans le signal d'apprentissage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


class RewardFunction:
    """Interface commune. `reset()` est appelé à chaque début d'épisode."""

    def reset(self) -> None:  # pragma: no cover - trivial
        pass

    def __call__(self, net_ret: float, **kwargs) -> float:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class PnLReward(RewardFunction):
    """Rendement net simple, mis à l'échelle."""
    scale: float = 100.0

    def __call__(self, net_ret: float, **kwargs) -> float:
        return float(net_ret * self.scale)


@dataclass
class LogPnLReward(RewardFunction):
    """Rendement logarithmique = maximisation du taux de croissance géométrique (Kelly).

    Contrairement au rendement arithmétique, il pénalise automatiquement la volatilité
    (par le terme -σ²/2) et rend la ruine infiniment coûteuse : log(0) = -∞.
    """
    scale: float = 100.0

    def __call__(self, net_ret: float, **kwargs) -> float:
        return float(np.log1p(max(net_ret, -0.999)) * self.scale)


@dataclass
class DifferentialSharpeReward(RewardFunction):
    """Differential Sharpe Ratio (Moody & Saffell, 1998).

    Dérivée du ratio de Sharpe par rapport à la fenêtre d'observation, calculable EN LIGNE
    à chaque pas. C'est ce qui permet d'optimiser un objectif de type Sharpe dans un cadre
    RL, alors que le Sharpe classique n'est défini que sur une période complète.

        A_t = A_{t-1} + η·(R_t - A_{t-1})          (moment d'ordre 1, EWMA)
        B_t = B_{t-1} + η·(R_t² - B_{t-1})         (moment d'ordre 2, EWMA)
        D_t = (B_{t-1}·ΔA - ½·A_{t-1}·ΔB) / (B_{t-1} - A_{t-1}²)^{3/2}

    Propriété clé pour le RL : à variance donnée, D_t croît avec le rendement ; mais à
    rendement donné, D_t DÉCROÎT quand la volatilité récente augmente. L'agent est donc
    incité à produire des gains réguliers plutôt que de gros gains erratiques.

    Note : en régime stationnaire E[D_t] = 0 — c'est normal, D_t mesure la VARIATION du
    Sharpe, pas son niveau. Le signal d'apprentissage est porté par le signe et
    l'amplitude instantanés, pas par la somme cumulée.
    """
    eta: float = 0.01
    scale: float = 1.0
    warmup: int = 30
    clip: float = 10.0
    a: float = field(default=0.0, init=False)
    b: float = field(default=0.0, init=False)
    n: int = field(default=0, init=False)
    _sum: float = field(default=0.0, init=False)
    _sum_sq: float = field(default=0.0, init=False)

    def reset(self) -> None:
        self.a, self.b, self.n = 0.0, 0.0, 0
        self._sum, self._sum_sq = 0.0, 0.0

    def __call__(self, net_ret: float, **kwargs) -> float:
        r = float(net_ret)

        # --- Amorçage : on estime A et B par les moments empiriques avant d'émettre un DSR.
        # Démarrer à A=B=0 rendrait le dénominateur (B - A²)^{3/2} quasi nul et produirait
        # des récompenses de plusieurs ordres de grandeur trop grandes sur les premiers pas,
        # ce qui déstabilise durablement l'apprentissage.
        if self.n < self.warmup:
            self._sum += r
            self._sum_sq += r * r
            self.n += 1
            if self.n == self.warmup:
                self.a = self._sum / self.warmup
                self.b = self._sum_sq / self.warmup
            return float(np.clip(r * self.scale, -self.clip, self.clip))

        d_a = r - self.a
        d_b = r * r - self.b
        var = self.b - self.a * self.a
        if var <= 1e-14:
            reward = r * self.scale
        else:
            reward = (self.b * d_a - 0.5 * self.a * d_b) / (var ** 1.5) * self.scale * self.eta

        self.a += self.eta * d_a
        self.b += self.eta * d_b
        self.n += 1
        return float(np.clip(reward, -self.clip, self.clip))


@dataclass
class VolScaledReward(RewardFunction):
    """Rendement divisé par la volatilité récente : un « Sharpe instantané ».

    Effet secondaire important : la récompense devient homogène entre régimes calmes et
    agités, ce qui stabilise énormément l'apprentissage (les gradients ne sont plus
    dominés par les périodes de crise).
    """
    scale: float = 100.0
    span: int = 60
    _ewm_var: float = field(default=0.0, init=False)
    _n: int = field(default=0, init=False)

    def reset(self) -> None:
        self._ewm_var, self._n = 0.0, 0

    def __call__(self, net_ret: float, **kwargs) -> float:
        alpha = 2.0 / (self.span + 1.0)
        r = float(net_ret)
        self._ewm_var = (1 - alpha) * self._ewm_var + alpha * r * r
        self._n += 1
        vol = np.sqrt(max(self._ewm_var, 1e-12))
        if self._n < 20:
            return float(r * self.scale)
        return float(np.clip(r / vol, -10.0, 10.0) * self.scale / 100.0)


@dataclass
class DrawdownPenalizedReward(RewardFunction):
    """Rendement net moins une pénalité sur l'AGGRAVATION du drawdown.

    On ne pénalise que l'incrément de drawdown, pas son niveau : pénaliser le niveau
    reviendrait à punir l'agent en permanence pour une erreur passée qu'il ne peut plus
    corriger, ce qui brouille l'attribution de crédit.
    """
    scale: float = 100.0
    penalty: float = 0.25
    _peak: float = field(default=1.0, init=False)
    _equity: float = field(default=1.0, init=False)
    _dd: float = field(default=0.0, init=False)

    def reset(self) -> None:
        self._peak, self._equity, self._dd = 1.0, 1.0, 0.0

    def __call__(self, net_ret: float, **kwargs) -> float:
        self._equity *= (1.0 + float(net_ret))
        self._peak = max(self._peak, self._equity)
        dd = self._equity / self._peak - 1.0
        worsening = max(self._dd - dd, 0.0)     # > 0 uniquement si le drawdown s'aggrave
        self._dd = dd
        return float((net_ret - self.penalty * worsening) * self.scale)


def build_reward(cfg) -> RewardFunction:
    """Fabrique la fonction de récompense à partir de EnvConfig."""
    kind = cfg.reward
    if kind == "pnl":
        return PnLReward(scale=cfg.reward_scale)
    if kind == "log_pnl":
        return LogPnLReward(scale=cfg.reward_scale)
    if kind == "dsr":
        # scale ramène le DSR (≈1e-2 par pas) dans une plage O(1), adaptée à un critique neuronal.
        return DifferentialSharpeReward(eta=cfg.dsr_eta, scale=cfg.reward_scale)
    if kind == "vol_scaled":
        return VolScaledReward(scale=cfg.reward_scale)
    if kind == "dd_penalized":
        return DrawdownPenalizedReward(scale=cfg.reward_scale, penalty=cfg.drawdown_penalty)
    raise ValueError(f"Fonction de récompense inconnue : {kind}")

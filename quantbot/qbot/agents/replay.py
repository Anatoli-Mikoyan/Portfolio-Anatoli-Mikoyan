"""Mémoire de rejeu : n-step + priorisation par erreur TD.

Deux problèmes résolus ici.

**1. Priorisation (Schaul et al., 2016).** Échantillonner uniformément le tampon fait
perdre l'essentiel du signal : en trading, 95 % des transitions sont des barres sans
intérêt, et les rares transitions informatives (retournements, chocs de volatilité)
sont noyées. On échantillonne donc proportionnellement à |erreur TD|^α, corrigé par des
poids d'importance-sampling pour ne pas biaiser le point fixe de l'apprentissage.

**2. Mémoire.** Une observation vaut ici `window × n_features` flottants — typiquement
2 000 à 4 000 valeurs. Stocker `obs` ET `next_obs` en clair pour 300 000 transitions
demanderait plusieurs gigaoctets. Comme toute observation est entièrement déterminée par
(indice temporel, état de portefeuille), on ne stocke que ce couple et on reconstruit la
fenêtre à la volée : ~65 octets par transition au lieu de ~17 ko, soit un facteur 250.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np


# =======================================================================================
# Arbre de somme : échantillonnage proportionnel en O(log n)
# =======================================================================================
class SumTree:
    """Arbre binaire dont chaque nœud contient la somme de ses enfants.

    Permet un tirage proportionnel aux priorités en O(log n) au lieu de O(n) — sans quoi
    la priorisation coûterait plus cher que l'apprentissage lui-même.
    """

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity, dtype=np.float64)

    def total(self) -> float:
        return float(self.tree[1])

    def max_priority(self) -> float:
        leaves = self.tree[self.capacity:]
        m = float(leaves.max()) if leaves.size else 0.0
        return m if m > 0 else 1.0

    def min_priority(self) -> float:
        leaves = self.tree[self.capacity:]
        positive = leaves[leaves > 0]
        return float(positive.min()) if positive.size else 1.0

    def update(self, idx: int, priority: float) -> None:
        i = idx + self.capacity
        self.tree[i] = priority
        i //= 2
        while i >= 1:
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]
            i //= 2

    def update_batch(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        pos = indices + self.capacity
        self.tree[pos] = priorities
        pos = np.unique(pos // 2)
        while pos.size and pos[0] >= 1:
            self.tree[pos] = self.tree[2 * pos] + self.tree[2 * pos + 1]
            pos = np.unique(pos // 2)
            pos = pos[pos >= 1]

    def sample(self, values: np.ndarray) -> np.ndarray:
        """Descente vectorisée : pour chaque valeur dans [0, total), l'index de la feuille."""
        idx = np.ones(values.shape[0], dtype=np.int64)
        v = values.copy()
        while idx[0] < self.capacity:
            left = 2 * idx
            go_right = v > self.tree[left]
            v = np.where(go_right, v - self.tree[left], v)
            idx = left + go_right.astype(np.int64)
        return idx - self.capacity


# =======================================================================================
# Accumulateur n-step
# =======================================================================================
@dataclass
class Transition:
    t: int
    portfolio: np.ndarray
    action: int
    reward: float
    next_t: int
    next_portfolio: np.ndarray
    done: bool
    n: int = 1


class NStepAccumulator:
    """Agrège n transitions en une seule à retour n-step.

        R^(n) = Σ_{k=0}^{n-1} γ^k r_{t+k}

    Compromis biais/variance : n=1 donne un apprentissage à faible variance mais lent à
    propager le crédit ; n grand propage vite mais introduit du biais off-policy. n=3..5
    est le régime standard du Rainbow et convient bien aux horizons de trading.
    """

    def __init__(self, n_step: int, gamma: float):
        self.n_step, self.gamma = max(int(n_step), 1), float(gamma)
        self.buf: Deque[Transition] = deque(maxlen=self.n_step)

    def reset(self) -> None:
        self.buf.clear()

    def push(self, tr: Transition) -> Optional[Transition]:
        self.buf.append(tr)
        if len(self.buf) < self.n_step:
            return None
        return self._collapse()

    def flush(self) -> list[Transition]:
        """Vide la file en fin d'épisode : les transitions restantes ont un n plus court."""
        out = []
        while self.buf:
            out.append(self._collapse())
            self.buf.popleft()
        return out

    def _collapse(self) -> Transition:
        first = self.buf[0]
        reward, gamma_k = 0.0, 1.0
        last = first
        for tr in self.buf:
            reward += gamma_k * tr.reward
            gamma_k *= self.gamma
            last = tr
            if tr.done:
                break
        return Transition(
            t=first.t, portfolio=first.portfolio, action=first.action, reward=reward,
            next_t=last.next_t, next_portfolio=last.next_portfolio, done=last.done,
            n=int(round(np.log(max(gamma_k, 1e-12)) / np.log(self.gamma))) if self.gamma < 1 else len(self.buf),
        )


# =======================================================================================
# Tampon priorisé
# =======================================================================================
class PrioritizedReplayBuffer:
    """Tampon circulaire à priorisation proportionnelle et stockage compact."""

    def __init__(
        self,
        capacity: int,
        n_portfolio: int,
        alpha: float = 0.6,
        beta0: float = 0.4,
        beta_steps: int = 200_000,
        eps: float = 1e-6,
        prioritized: bool = True,
    ):
        # Capacité arrondie à la puissance de 2 supérieure (contrainte de l'arbre de somme).
        self.capacity = 1 << (int(capacity) - 1).bit_length()
        self.alpha, self.beta0, self.beta_steps, self.eps = alpha, beta0, max(beta_steps, 1), eps
        self.prioritized = prioritized

        self.t = np.zeros(self.capacity, dtype=np.int32)
        self.next_t = np.zeros(self.capacity, dtype=np.int32)
        self.portfolio = np.zeros((self.capacity, n_portfolio), dtype=np.float32)
        self.next_portfolio = np.zeros((self.capacity, n_portfolio), dtype=np.float32)
        self.action = np.zeros(self.capacity, dtype=np.int64)
        self.reward = np.zeros(self.capacity, dtype=np.float32)
        self.done = np.zeros(self.capacity, dtype=np.float32)
        self.n_steps = np.ones(self.capacity, dtype=np.int32)

        self.tree = SumTree(self.capacity)
        self.pos = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(self, tr: Transition) -> None:
        i = self.pos
        self.t[i], self.next_t[i] = tr.t, tr.next_t
        self.portfolio[i] = tr.portfolio
        self.next_portfolio[i] = tr.next_portfolio
        self.action[i], self.reward[i], self.done[i] = tr.action, tr.reward, float(tr.done)
        self.n_steps[i] = tr.n
        # Priorité maximale pour toute nouvelle transition : elle doit être vue au moins
        # une fois avant d'être éventuellement reléguée.
        self.tree.update(i, self.tree.max_priority() if self.prioritized else 1.0)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def beta(self, step: int) -> float:
        """β annelé de β0 vers 1 : la correction d'IS doit être complète en fin
        d'entraînement, quand la convergence exige un estimateur non biaisé."""
        frac = min(step / self.beta_steps, 1.0)
        return self.beta0 + frac * (1.0 - self.beta0)

    def sample(self, batch_size: int, step: int, rng: np.random.Generator) -> dict:
        if self.size < batch_size:
            raise ValueError(f"Tampon insuffisant : {self.size} < {batch_size}")

        if not self.prioritized:
            idx = rng.integers(0, self.size, batch_size)
            weights = np.ones(batch_size, dtype=np.float32)
        else:
            total = self.tree.total()
            # Échantillonnage stratifié : une valeur par segment de largeur total/B,
            # ce qui réduit fortement la variance par rapport à B tirages i.i.d.
            segment = total / batch_size
            values = (rng.random(batch_size) + np.arange(batch_size)) * segment
            idx = self.tree.sample(np.clip(values, 0, total - 1e-9))
            idx = np.clip(idx, 0, max(self.size - 1, 0))

            probs = self.tree.tree[idx + self.capacity] / max(total, 1e-12)
            beta = self.beta(step)
            weights = (self.size * np.maximum(probs, 1e-12)) ** (-beta)
            weights = (weights / weights.max()).astype(np.float32)

        return {
            "idx": idx,
            "t": self.t[idx], "next_t": self.next_t[idx],
            "portfolio": self.portfolio[idx], "next_portfolio": self.next_portfolio[idx],
            "action": self.action[idx], "reward": self.reward[idx],
            "done": self.done[idx], "n_steps": self.n_steps[idx],
            "weights": weights,
        }

    def update_priorities(self, idx: np.ndarray, td_errors: np.ndarray) -> None:
        if not self.prioritized:
            return
        priorities = (np.abs(td_errors) + self.eps) ** self.alpha
        self.tree.update_batch(np.asarray(idx, dtype=np.int64),
                               np.asarray(priorities, dtype=np.float64))


class ObsReconstructor:
    """Reconstruit les observations complètes à partir des codes compacts."""

    def __init__(self, features: np.ndarray, window: int, include_portfolio: bool = True):
        self.features = np.asarray(features, dtype=np.float32)
        self.window = int(window)
        self.include_portfolio = include_portfolio
        self.n_features = self.features.shape[1]

    def __call__(self, t: np.ndarray, portfolio: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=np.int64)
        # Indices de fenêtre vectorisés : (B, W) -> (B, W, F) -> (B, W*F)
        offsets = np.arange(-self.window + 1, 1)
        rows = np.clip(t[:, None] + offsets[None, :], 0, self.features.shape[0] - 1)
        windows = self.features[rows].reshape(t.shape[0], -1)
        if not self.include_portfolio:
            return windows
        return np.concatenate([windows, np.asarray(portfolio, dtype=np.float32)], axis=1)

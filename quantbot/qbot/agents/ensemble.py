"""Ensemble d'agents entraînés sur des graines différentes.

Motivation empirique forte : la variance inter-graines du RL profond est ÉNORME. Un même
algorithme, mêmes données, mêmes hyperparamètres, graines différentes, peut produire un
Sharpe de 1.8 ou de -0.3. Publier (ou trader) le résultat d'une seule graine revient à
publier le maximum d'un échantillon en le présentant comme sa moyenne.

Deux bénéfices concrets ici :
  1. moyenner les Q réduit la variance de la politique ;
  2. le DÉSACCORD entre agents est un signal d'incertitude épistémique exploitable :
     quand les agents ne sont pas d'accord, le bon comportement est de rester plat.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

from .rainbow import RainbowAgent


class EnsembleAgent:
    """Agrège plusieurs `RainbowAgent` en une politique unique."""

    def __init__(self, agents: Sequence[RainbowAgent], agreement_threshold: float = 0.0):
        if not agents:
            raise ValueError("Ensemble vide.")
        self.agents: List[RainbowAgent] = list(agents)
        self.agreement_threshold = float(agreement_threshold)
        self.n_actions = self.agents[0].n_actions
        # Convention : l'action « neutre » est celle de position nulle, au milieu de la grille.
        self.flat_action = self.n_actions // 2

    def bind_features(self, features: np.ndarray) -> None:
        for a in self.agents:
            a.bind_features(features)

    @torch.no_grad()
    def _scores(self, obs: np.ndarray, cvar_alpha: Optional[float] = None) -> np.ndarray:
        """(n_agents, batch, n_actions)"""
        x2d = np.atleast_2d(np.asarray(obs, dtype=np.float32))
        out = []
        for agent in self.agents:
            agent.online.eval()
            x = torch.as_tensor(x2d, device=agent.device)
            q = (agent.online.risk_measure(x, cvar_alpha) if cvar_alpha
                 else agent.online.q_values(x))
            out.append(q.cpu().numpy())
        return np.stack(out, axis=0)

    def act(self, obs: np.ndarray, greedy: bool = True, cvar_alpha: Optional[float] = None) -> int:
        return int(self.act_batch(np.atleast_2d(obs), cvar_alpha)[0])

    def act_batch(self, obs: np.ndarray, cvar_alpha: Optional[float] = None) -> np.ndarray:
        scores = self._scores(obs, cvar_alpha)
        mean_q = scores.mean(axis=0)
        actions = mean_q.argmax(axis=1)

        if self.agreement_threshold > 0 and len(self.agents) > 1:
            votes = scores.argmax(axis=2)                       # (n_agents, batch)
            agreement = (votes == actions[None, :]).mean(axis=0)
            # Désaccord = incertitude épistémique : on se met à plat plutôt que de deviner.
            actions = np.where(agreement >= self.agreement_threshold, actions, self.flat_action)
        return actions.astype(np.int64)

    def disagreement(self, obs: np.ndarray) -> np.ndarray:
        """Fraction d'agents en désaccord avec la décision d'ensemble — mesure d'incertitude."""
        scores = self._scores(obs)
        consensus = scores.mean(axis=0).argmax(axis=1)
        votes = scores.argmax(axis=2)
        return 1.0 - (votes == consensus[None, :]).mean(axis=0)

    # ---------------------------------------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for i, agent in enumerate(self.agents):
            agent.save(directory / f"agent_{i}.pt")
        return directory

    @classmethod
    def load(cls, directory: str | Path, device: Optional[str] = None,
             agreement_threshold: float = 0.0) -> "EnsembleAgent":
        directory = Path(directory)
        paths = sorted(directory.glob("agent_*.pt"))
        if not paths:
            raise FileNotFoundError(f"Aucun agent trouvé dans {directory}")
        return cls([RainbowAgent.load(p, device=device) for p in paths], agreement_threshold)

"""Socle de la détection de régime (cahier des charges §7).

Le piège central de cette couche, et la raison pour laquelle elle mérite un socle
explicite : **la plupart des implémentations de détection de régime regardent le futur
sans que personne ne s'en aperçoive.**

Un HMM ajusté sur toute la série puis interrogé par Viterbi ou par lissage
avant-arrière donne, pour chaque date t, l'état le plus probable *sachant l'ensemble de
l'échantillon, y compris ce qui s'est passé après t*. Le résultat est superbe — les
régimes sont nets, les transitions parfaitement placées — et parfaitement inutilisable :
en production, on ne dispose que du passé.

Chaque détecteur expose donc deux méthodes clairement séparées :

  * `filter(X)`  — CAUSAL. N'utilise que les observations ≤ t. C'est la seule utilisable
    en backtest comme en production.
  * `smooth(X)`  — NON CAUSAL. Utilise toute la série. Réservé à l'analyse rétrospective,
    et volontairement nommé pour qu'on ne puisse pas l'appeler par mégarde.

`RegimeDetector.smooth` est en outre marqué par `leaks_future = True`, et le backtest
refuse de consommer une série de régimes produite par lissage.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class LookaheadError(RuntimeError):
    """Levée lorsqu'une série de régimes non causale est utilisée dans un backtest."""


@dataclass
class RegimeSeries:
    """Série d'états de régime, accompagnée de sa provenance.

    Le drapeau `causal` n'est pas décoratif : il circule avec les données et permet à
    l'aval de refuser une série qui regarderait le futur.
    """
    states: pd.Series
    proba: Optional[pd.DataFrame] = None
    causal: bool = True
    detector: str = ""
    labels: Dict[int, str] = field(default_factory=dict)

    def require_causal(self) -> "RegimeSeries":
        if not self.causal:
            raise LookaheadError(
                f"Série de régimes produite par {self.detector}.smooth() : elle utilise "
                "des observations postérieures à chaque date. Utiliser filter() pour "
                "tout backtest ou toute exécution."
            )
        return self

    def name_of(self, state: int) -> str:
        return self.labels.get(int(state), f"état {int(state)}")

    def distribution(self) -> pd.Series:
        counts = self.states.value_counts(normalize=True).sort_index()
        counts.index = [self.name_of(s) for s in counts.index]
        return counts

    def transition_rate(self) -> float:
        """Fréquence des changements d'état — un détecteur trop nerveux est inexploitable."""
        return float((self.states.diff() != 0).mean())


# =======================================================================================
class RegimeDetector(ABC):
    """Interface commune à toutes les approches comparées au §7."""

    n_states: int
    name: str = "detector"

    @abstractmethod
    def fit(self, X: pd.DataFrame) -> "RegimeDetector":
        """Ajuste le détecteur. À n'appeler QUE sur le segment d'entraînement."""

    @abstractmethod
    def filter(self, X: pd.DataFrame) -> RegimeSeries:
        """Inférence causale : l'état à t n'utilise que les observations ≤ t."""

    def smooth(self, X: pd.DataFrame) -> RegimeSeries:
        """Inférence rétrospective. NON CAUSALE — analyse uniquement.

        Par défaut, identique au filtrage : les détecteurs sans mémoire (clustering,
        règles) n'ont pas de version lissée, leur sortie ne dépend que de l'instant t.
        """
        out = self.filter(X)
        return RegimeSeries(states=out.states, proba=out.proba, causal=out.causal,
                            detector=self.name, labels=out.labels)

    # ---------------------------------------------------------------------------------
    def label_states(self, X: pd.DataFrame, states: pd.Series) -> Dict[int, str]:
        """Nomme les états à partir de leurs caractéristiques moyennes.

        Un HMM produit des états numérotés arbitrairement : sans nommage, impossible de
        savoir si l'état 2 est « tendance haussière » ou « panique ». Le nommage
        s'appuie sur les colonnes de régime quand elles existent.
        """
        labels: Dict[int, str] = {}
        vol_col = next((c for c in ("vol_pctile", "vol_20", "vol_10", "yz_vol") if c in X.columns), None)
        trend_col = next((c for c in ("trend_strength", "adx", "di_diff") if c in X.columns), None)

        for state in sorted(pd.unique(states.dropna())):
            mask = states == state
            if mask.sum() == 0:
                continue
            parts: List[str] = []
            if vol_col:
                v = float(X.loc[mask, vol_col].mean())
                overall = float(X[vol_col].mean())
                parts.append("vol haute" if v > overall else "vol basse")
            if trend_col:
                t = float(X.loc[mask, trend_col].abs().mean())
                overall_t = float(X[trend_col].abs().mean())
                parts.append("tendance" if t > overall_t else "range")
            labels[int(state)] = " / ".join(parts) if parts else f"état {int(state)}"
        return labels

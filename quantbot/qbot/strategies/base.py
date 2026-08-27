"""Socle des stratégies quantitatives.

Principe directeur imposé par le cahier des charges (§8) : *aucune stratégie n'est
supposée rentable*. Chacune est une **hypothèse falsifiable** sur le comportement du
marché, qui doit survivre à la batterie de validation avant d'être considérée.

Ce socle rend cette exigence structurelle plutôt que déclarative :

  * chaque stratégie DOIT énoncer son hypothèse (`hypothesis`) et la condition dans
    laquelle elle est censée échouer (`fails_when`) ;
  * chaque stratégie DOIT exposer une grille de paramètres explicite, ce qui permet de
    COMPTER les essais et donc d'alimenter honnêtement le Deflated Sharpe ;
  * le signal est un flottant dans [-1, 1] daté à la clôture de la barre `t`, jamais
    au-delà — la convention de décalage vit dans `run_backtest`, une seule fois.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, Iterator, List, Sequence

import numpy as np
import pandas as pd


@dataclass
class Strategy(ABC):
    """Interface commune. Les sous-classes sont des dataclasses de paramètres."""

    @property
    @abstractmethod
    def hypothesis(self) -> str:
        """L'affirmation testable sur le marché. Si elle est fausse, la stratégie perd."""

    @property
    @abstractmethod
    def fails_when(self) -> str:
        """Le régime dans lequel cette stratégie est censée perdre.

        L'exiger sert à deux choses : forcer à réfléchir avant de coder, et donner à la
        couche de régime (§7) un critère d'activation qui ne soit pas purement empirique.
        """

    @abstractmethod
    def raw_signal(self, df: pd.DataFrame) -> pd.Series:
        """Signal brut, dans [-1, 1], strictement causal."""

    # ---------------------------------------------------------------------------------
    @property
    def name(self) -> str:
        params = ",".join(f"{k}={v}" for k, v in self.params.items())
        return f"{type(self).__name__}({params})"

    @property
    def params(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @property
    def warmup(self) -> int:
        """Barres nécessaires avant que le signal soit défini. À surcharger."""
        numeric = [v for v in self.params.values() if isinstance(v, (int, float)) and v > 1]
        return int(max(numeric, default=20) * 3)

    def signal(self, df: pd.DataFrame) -> pd.Series:
        """Signal final : borné, sans NaN, et neutre pendant la période de chauffe."""
        raw = self.raw_signal(df)
        out = raw.reindex(df.index).astype(float)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
        # Neutralisation explicite du warm-up : un signal calculé sur une fenêtre
        # incomplète n'est pas « approximatif », il est faux.
        out.iloc[: min(self.warmup, len(out))] = 0.0
        return out.rename(type(self).__name__)

    # ---------------------------------------------------------------------------------
    @classmethod
    @abstractmethod
    def param_grid(cls) -> Dict[str, Sequence[Any]]:
        """Grille de paramètres à balayer.

        La garder PETITE est une décision statistique, pas une paresse : chaque
        combinaison supplémentaire est un essai de plus qui dégrade le Deflated Sharpe.
        """

    @classmethod
    def enumerate(cls) -> List["Strategy"]:
        """Instancie toutes les combinaisons de la grille."""
        grid = cls.param_grid()
        keys = list(grid)
        return [cls(**dict(zip(keys, values))) for values in product(*(grid[k] for k in keys))]

    @classmethod
    def n_trials(cls) -> int:
        grid = cls.param_grid()
        n = 1
        for values in grid.values():
            n *= len(values)
        return n

    def describe(self) -> str:  # pragma: no cover - affichage
        return (f"{self.name}\n"
                f"  hypothèse : {self.hypothesis}\n"
                f"  échoue si : {self.fails_when}")

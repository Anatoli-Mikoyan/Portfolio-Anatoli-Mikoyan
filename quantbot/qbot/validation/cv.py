"""Validation croisée pour séries financières (López de Prado, ch. 7).

La K-fold standard est INVALIDE en finance, pour deux raisons cumulatives :

1. **Fuite temporelle** — entraîner sur 2020 et tester sur 2018 revient à utiliser le
   futur pour prédire le passé.
2. **Fuite par chevauchement** — même en respectant l'ordre, un label construit sur un
   horizon de 24 barres à la fin du train partage ses rendements avec le début du test.
   Le modèle a donc déjà « vu » une partie de la réponse.

Deux corrections, toutes deux implémentées ici :
  * **Purge** : on retire du train tout label dont l'horizon [t0, t1] recoupe le test.
  * **Embargo** : on retire en plus une marge après le test, car l'autocorrélation
    sérielle des features fait fuir de l'information au-delà de la seule zone de
    chevauchement des labels.
"""
from __future__ import annotations

from itertools import combinations
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _as_int_index(index: pd.Index, values) -> np.ndarray:
    return np.asarray(index.searchsorted(values), dtype=np.int64)


def purge_train_indices(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    t1_positions: Optional[np.ndarray],
    embargo: int,
    n_samples: int,
) -> np.ndarray:
    """Retire du train les observations qui recouvrent le test, plus l'embargo."""
    if test_idx.size == 0:
        return train_idx
    test_start, test_end = int(test_idx.min()), int(test_idx.max())

    if t1_positions is not None:
        # Purge : une observation i est écartée si son horizon [i, t1_i] touche le test.
        overlap = (t1_positions[train_idx] >= test_start) & (train_idx <= test_end)
        train_idx = train_idx[~overlap]
    else:
        train_idx = train_idx[(train_idx < test_start) | (train_idx > test_end)]

    if embargo > 0:
        # Embargo appliqué APRÈS le test uniquement : l'information circule vers l'avant.
        embargo_end = min(test_end + embargo, n_samples - 1)
        train_idx = train_idx[(train_idx <= test_end) | (train_idx > embargo_end)]
    return train_idx


class PurgedKFold:
    """K-fold à blocs contigus, avec purge et embargo."""

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01, t1: Optional[pd.Series] = None):
        if n_splits < 2:
            raise ValueError("n_splits doit être >= 2")
        self.n_splits, self.embargo_pct, self.t1 = int(n_splits), float(embargo_pct), t1

    def split(self, index: pd.Index) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(index)
        embargo = int(n * self.embargo_pct)
        t1_pos = _as_int_index(index, self.t1.reindex(index).ffill().to_numpy()) if self.t1 is not None else None

        bounds = np.linspace(0, n, self.n_splits + 1).astype(int)
        all_idx = np.arange(n)
        for k in range(self.n_splits):
            test_idx = all_idx[bounds[k]: bounds[k + 1]]
            train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=True)
            yield purge_train_indices(train_idx, test_idx, t1_pos, embargo, n), test_idx

    def get_n_splits(self) -> int:
        return self.n_splits


class CombinatorialPurgedCV:
    """Combinatorial Purged Cross-Validation (CPCV, López de Prado ch. 12).

    Le walk-forward ne fournit qu'UN SEUL chemin historique : on estime donc un Sharpe
    sur un unique tirage, sans aucune idée de sa variance. La CPCV découpe l'échantillon
    en N groupes, teste toutes les combinaisons de k groupes, et reconstitue

        φ = C(N, k) · k / N   chemins de backtest distincts

    On obtient une DISTRIBUTION de Sharpe au lieu d'un point — ce qui permet enfin de
    répondre à « ce résultat est-il robuste ou est-ce un coup de chance ? ».

    Exemple : N=6, k=2 -> 15 combinaisons -> 5 chemins complets.
    """

    def __init__(self, n_groups: int = 6, n_test_groups: int = 2,
                 embargo_pct: float = 0.01, t1: Optional[pd.Series] = None):
        if n_test_groups >= n_groups:
            raise ValueError("n_test_groups doit être < n_groups")
        self.n_groups, self.n_test_groups = int(n_groups), int(n_test_groups)
        self.embargo_pct, self.t1 = float(embargo_pct), t1

    @property
    def n_splits(self) -> int:
        from math import comb
        return comb(self.n_groups, self.n_test_groups)

    @property
    def n_paths(self) -> int:
        """Nombre de chemins de backtest complets reconstructibles."""
        return self.n_splits * self.n_test_groups // self.n_groups

    def split(self, index: pd.Index) -> Iterator[Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]]:
        n = len(index)
        embargo = int(n * self.embargo_pct)
        t1_pos = _as_int_index(index, self.t1.reindex(index).ffill().to_numpy()) if self.t1 is not None else None

        bounds = np.linspace(0, n, self.n_groups + 1).astype(int)
        groups = [np.arange(bounds[i], bounds[i + 1]) for i in range(self.n_groups)]
        all_idx = np.arange(n)

        for combo in combinations(range(self.n_groups), self.n_test_groups):
            test_idx = np.sort(np.concatenate([groups[g] for g in combo]))
            train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=True)
            # Purge et embargo appliqués séparément pour CHAQUE bloc de test, car les
            # blocs sont disjoints : un embargo global sur-purgerait inutilement.
            for g in combo:
                train_idx = purge_train_indices(train_idx, groups[g], t1_pos, embargo, n)
            yield train_idx, test_idx, combo

    def assemble_paths(self, fold_results: Sequence[Tuple[Tuple[int, ...], dict]]) -> List[dict]:
        """Recompose les chemins de backtest à partir des résultats de chaque combinaison.

        Chaque groupe g apparaît dans exactement C(N-1, k-1) combinaisons ; on distribue
        ces occurrences sur `n_paths` chemins de sorte que chaque chemin couvre l'ensemble
        des N groupes exactement une fois.
        """
        by_group: dict[int, list] = {g: [] for g in range(self.n_groups)}
        for combo, payload in fold_results:
            for g in combo:
                by_group[g].append(payload)

        paths: List[dict] = []
        for p in range(self.n_paths):
            path = {}
            for g in range(self.n_groups):
                bucket = by_group[g]
                if p < len(bucket):
                    path[g] = bucket[p]
            paths.append(path)
        return paths


def train_valid_test_split(
    index: pd.Index, train: float = 0.6, valid: float = 0.2, embargo_pct: float = 0.01
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Découpage chronologique simple avec embargo entre les segments.

    Le test doit être touché UNE SEULE FOIS, en toute fin de projet. Chaque nouvelle
    consultation du test transforme celui-ci en jeu de validation, et le résultat rapporté
    devient un estimateur biaisé de la performance réelle.
    """
    n = len(index)
    embargo = int(n * embargo_pct)
    i_train = int(n * train)
    i_valid = int(n * (train + valid))
    return (
        np.arange(0, i_train),
        np.arange(min(i_train + embargo, n), i_valid),
        np.arange(min(i_valid + embargo, n), n),
    )

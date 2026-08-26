"""Analyse walk-forward : ré-entraînement périodique et évaluation hors échantillon.

C'est le protocole qui se rapproche le plus de la réalité de production : on n'entraîne
jamais qu'avec le passé, on trade la période suivante, puis on ré-entraîne. Chaque barre
de la courbe d'équité produite ici a été générée par un modèle qui ne l'avait jamais vue.

Deux variantes :
  * **glissante** (rolling) — fenêtre d'entraînement de taille fixe. Oublie le passé
    lointain : préférable quand le marché change de régime, ce qui est la norme en FX.
  * **ancrée** (anchored) — fenêtre qui s'étend depuis le début. Plus de données, mais
    dilue le régime récent dans des dynamiques obsolètes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..backtest.metrics import PerformanceReport, compute_report, sharpe_ratio
from ..utils.logging import get_logger

log = get_logger("validation.walkforward")


@dataclass
class Fold:
    idx: int
    train_slice: slice
    valid_slice: slice
    test_slice: slice
    train_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class FoldResult:
    fold: Fold
    returns: pd.Series
    positions: pd.Series
    report: Optional[PerformanceReport] = None
    extra: Dict = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    folds: List[FoldResult]
    oos_returns: pd.Series
    oos_positions: pd.Series
    report: PerformanceReport
    bars_per_year: float

    @property
    def fold_sharpes(self) -> np.ndarray:
        return np.array([
            sharpe_ratio(f.returns.to_numpy(), self.bars_per_year) for f in self.folds
        ])

    def consistency(self) -> Dict[str, float]:
        """Régularité inter-folds — plus informative que le Sharpe agrégé.

        Une stratégie dont le Sharpe agrégé vaut 1.5 mais qui est portée par 1 fold sur 8
        n'est pas exploitable : elle sera abandonnée avant que le fold gagnant n'arrive.
        """
        s = self.fold_sharpes
        return {
            "n_folds": int(s.size),
            "mean_sharpe": float(s.mean()),
            "median_sharpe": float(np.median(s)),
            "std_sharpe": float(s.std(ddof=1)) if s.size > 1 else 0.0,
            "pct_positive": float((s > 0).mean()),
            "worst_fold": float(s.min()),
            "best_fold": float(s.max()),
            # t de Student sur les Sharpes de folds : significativité de la régularité
            "t_stat": float(s.mean() / (s.std(ddof=1) / np.sqrt(s.size)))
            if s.size > 1 and s.std(ddof=1) > 1e-12 else 0.0,
        }


def make_folds(
    n: int,
    index: pd.Index,
    train_bars: int,
    test_bars: int,
    valid_pct: float = 0.15,
    anchored: bool = False,
    embargo_bars: int = 0,
) -> List[Fold]:
    """Construit la séquence de fenêtres (train / valid / test) sans chevauchement."""
    folds: List[Fold] = []
    start = 0
    k = 0
    while True:
        train_end = start + train_bars
        n_valid = int(train_bars * valid_pct)
        # La validation est prélevée à la FIN du train : c'est la période la plus proche
        # du test, donc la plus représentative du régime à venir.
        valid_start = train_end - n_valid
        test_start = train_end + embargo_bars
        test_end = test_start + test_bars
        if test_end > n:
            break

        folds.append(
            Fold(
                idx=k,
                train_slice=slice(0 if anchored else start, valid_start),
                valid_slice=slice(valid_start, train_end),
                test_slice=slice(test_start, test_end),
                train_start=index[0 if anchored else start],
                test_start=index[test_start],
                test_end=index[min(test_end - 1, n - 1)],
            )
        )
        start += test_bars
        k += 1
    return folds


def run_walkforward(
    data: pd.DataFrame,
    fit_predict: Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame, Fold], Tuple[pd.Series, pd.Series]],
    train_bars: int = 20_000,
    test_bars: int = 5_000,
    valid_pct: float = 0.15,
    anchored: bool = False,
    embargo_bars: int = 0,
    bars_per_year: float = 6240.0,
    n_trials: int = 1,
) -> WalkForwardResult:
    """Exécute le walk-forward.

    `fit_predict(train_df, valid_df, test_df, fold)` doit :
      1. n'utiliser QUE `train_df` (et `valid_df` pour la sélection de modèle),
      2. retourner (rendements nets sur le test, positions sur le test).

    Toute la responsabilité de la causalité incombe à cette fonction ; le walk-forward
    garantit seulement que les fenêtres ne se chevauchent pas.
    """
    n = len(data)
    folds = make_folds(n, data.index, train_bars, test_bars, valid_pct, anchored, embargo_bars)
    if not folds:
        raise ValueError(
            f"Aucun fold constructible : {n} barres pour train={train_bars} + test={test_bars}. "
            "Réduire train_bars/test_bars ou fournir plus de données."
        )

    log.info("Walk-forward %s : %d folds (train=%d, test=%d barres)",
             "ancré" if anchored else "glissant", len(folds), train_bars, test_bars)

    results: List[FoldResult] = []
    for fold in folds:
        train_df = data.iloc[fold.train_slice]
        valid_df = data.iloc[fold.valid_slice]
        test_df = data.iloc[fold.test_slice]
        log.info("  fold %d | train %s -> %s | test %s -> %s",
                 fold.idx, fold.train_start.date(), valid_df.index[-1].date(),
                 fold.test_start.date(), fold.test_end.date())

        rets, pos = fit_predict(train_df, valid_df, test_df, fold)
        rep = compute_report(rets.to_numpy(), bars_per_year, n_trials=1,
                             positions=pos.to_numpy())
        results.append(FoldResult(fold=fold, returns=rets, positions=pos, report=rep))
        log.info("    -> sharpe OOS = %6.3f | rendement = %+7.2f%% | maxDD = %6.2f%%",
                 rep.sharpe, 100 * rep.total_return, 100 * rep.max_drawdown)

    oos_returns = pd.concat([r.returns for r in results]).sort_index()
    oos_positions = pd.concat([r.positions for r in results]).sort_index()
    oos_returns = oos_returns[~oos_returns.index.duplicated(keep="first")]
    oos_positions = oos_positions[~oos_positions.index.duplicated(keep="first")]

    report = compute_report(
        oos_returns.to_numpy(), bars_per_year, n_trials=n_trials,
        positions=oos_positions.to_numpy(),
    )
    return WalkForwardResult(results, oos_returns, oos_positions, report, bars_per_year)

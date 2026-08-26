"""Probabilité de sur-apprentissage du backtest (PBO) par CSCV.

Bailey, Borwein, López de Prado & Zhu (2014), *The Probability of Backtest Overfitting*.

Question posée : « Si je sélectionne la meilleure configuration IN-SAMPLE, quelle est la
probabilité qu'elle se classe SOUS LA MÉDIANE out-of-sample ? »

Réponse sous la forme d'un nombre entre 0 et 1 :
    PBO ≈ 0.0-0.2  -> la sélection capture un vrai signal
    PBO ≈ 0.5      -> la sélection est équivalente à un tirage au sort
    PBO > 0.5      -> la sélection est ANTI-prédictive : le meilleur IS est
                      systématiquement mauvais OOS, signature d'un sur-apprentissage massif

Résultat théorique majeur de l'article : quand le nombre de configurations testées N
augmente, PBO tend vers 1 même si AUCUNE configuration n'a de véritable edge. Autrement
dit, chercher assez longtemps garantit de trouver un backtest magnifique et sans valeur.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Optional

import numpy as np
from scipy import stats


def _default_perf(returns_matrix: np.ndarray) -> np.ndarray:
    """Sharpe non annualisé par colonne (l'annualisation ne change aucun classement)."""
    mu = returns_matrix.mean(axis=0)
    sd = returns_matrix.std(axis=0, ddof=1)
    return np.divide(mu, sd, out=np.zeros_like(mu), where=sd > 1e-14)


@dataclass
class PBOResult:
    pbo: float
    n_combinations: int
    n_strategies: int
    logits: np.ndarray
    oos_ranks: np.ndarray
    is_perf_selected: np.ndarray
    oos_perf_selected: np.ndarray
    degradation_slope: float
    degradation_intercept: float
    prob_oos_loss: float

    def __str__(self) -> str:  # pragma: no cover - affichage
        verdict = ("signal réel" if self.pbo < 0.2 else
                   "douteux" if self.pbo < 0.4 else
                   "équivalent au hasard" if self.pbo < 0.6 else
                   "SUR-APPRENTISSAGE MASSIF")
        return "\n".join([
            "┌─ PROBABILITÉ DE SUR-APPRENTISSAGE (CSCV) ────────────────────┐",
            f"│ Configurations testées   {self.n_strategies:>10,}                        │",
            f"│ Combinaisons CSCV        {self.n_combinations:>10,}                        │",
            f"│ PBO                      {self.pbo:>10.3f}   -> {verdict:<20}│",
            f"│ Pente de dégradation     {self.degradation_slope:>10.3f}   (< 1 = perte OOS)     │",
            f"│ P(perte OOS | best IS)   {self.prob_oos_loss:>10.3f}                        │",
            f"│ Perf. IS moyenne         {self.is_perf_selected.mean():>10.4f}                        │",
            f"│ Perf. OOS moyenne        {self.oos_perf_selected.mean():>10.4f}                        │",
            "└──────────────────────────────────────────────────────────────┘",
        ])


def compute_pbo(
    returns_matrix: np.ndarray,
    n_partitions: int = 16,
    perf_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    max_combinations: Optional[int] = 5000,
    seed: int = 0,
) -> PBOResult:
    """Calcule la PBO par validation croisée combinatoirement symétrique.

    `returns_matrix` : (T observations, N configurations). Chaque colonne est la série de
    rendements d'une configuration testée — grille d'hyperparamètres, graines, variantes
    de features, etc. TOUT ce qui a été essayé doit y figurer, sinon la PBO est sous-estimée.
    """
    m = np.asarray(returns_matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError("returns_matrix doit être 2D (T, N)")
    t, n = m.shape
    if n < 2:
        raise ValueError("Au moins 2 configurations sont nécessaires pour estimer la PBO.")
    if n_partitions % 2 != 0:
        raise ValueError("n_partitions doit être pair (partition symétrique).")
    if t < n_partitions * 2:
        raise ValueError(f"Trop peu d'observations ({t}) pour {n_partitions} partitions.")

    perf_fn = perf_fn or _default_perf
    if n < 20:
        # Vérifié par simulation : sous H0 (aucun edge) E[PBO] vaut 0.50 dès N>=20, mais
        # remonte à ~0.63 pour N=5 — le rang OOS ne prend alors que N valeurs distinctes,
        # ce qui biaise l'estimateur vers le haut. En dessous de 20 configurations,
        # interpréter la PBO comme un indice qualitatif, pas comme une probabilité.
        import warnings

        warnings.warn(
            f"PBO estimée sur seulement {n} configurations : l'estimateur est biaisé vers "
            "le haut sous cette taille (E[PBO] ~ 0.63 au lieu de 0.50 sous H0). "
            "Fournir >= 20 configurations pour une lecture quantitative.",
            RuntimeWarning, stacklevel=2,
        )

    # Découpage en S blocs contigus (l'ordre temporel est préservé dans chaque bloc).
    bounds = np.linspace(0, t, n_partitions + 1).astype(int)
    blocks = [np.arange(bounds[i], bounds[i + 1]) for i in range(n_partitions)]

    combos = list(combinations(range(n_partitions), n_partitions // 2))
    if max_combinations and len(combos) > max_combinations:
        # C(16,8) = 12 870 ; au-delà on échantillonne, l'estimateur reste non biaisé.
        rng = np.random.default_rng(seed)
        picks = rng.choice(len(combos), size=max_combinations, replace=False)
        combos = [combos[i] for i in sorted(picks)]

    logits, ranks, is_sel, oos_sel = [], [], [], []
    for combo in combos:
        train_idx = np.concatenate([blocks[i] for i in combo])
        test_mask = np.ones(n_partitions, dtype=bool)
        test_mask[list(combo)] = False
        test_idx = np.concatenate([blocks[i] for i in np.flatnonzero(test_mask)])

        r_is = perf_fn(m[train_idx])
        r_oos = perf_fn(m[test_idx])

        best = int(np.argmax(r_is))
        # Rang relatif OOS de la configuration choisie IS, dans (0, 1).
        rank = float(stats.rankdata(r_oos)[best]) / (n + 1.0)
        rank = min(max(rank, 1e-9), 1 - 1e-9)

        logits.append(np.log(rank / (1.0 - rank)))
        ranks.append(rank)
        is_sel.append(r_is[best])
        oos_sel.append(r_oos[best])

    logits = np.asarray(logits)
    is_sel = np.asarray(is_sel)
    oos_sel = np.asarray(oos_sel)

    # Dégradation IS -> OOS : une pente < 1 signifie que la performance ne se transporte pas.
    if is_sel.size > 2 and is_sel.std() > 1e-12:
        slope, intercept = np.polyfit(is_sel, oos_sel, 1)
    else:
        slope, intercept = 0.0, 0.0

    return PBOResult(
        pbo=float((logits <= 0).mean()),
        n_combinations=len(combos),
        n_strategies=n,
        logits=logits,
        oos_ranks=np.asarray(ranks),
        is_perf_selected=is_sel,
        oos_perf_selected=oos_sel,
        degradation_slope=float(slope),
        degradation_intercept=float(intercept),
        prob_oos_loss=float((oos_sel <= 0).mean()),
    )

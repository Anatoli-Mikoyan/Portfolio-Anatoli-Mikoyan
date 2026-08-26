"""Poids d'échantillons pour labels chevauchants (López de Prado, ch. 4).

Problème structurel du ML financier : deux labels dont les horizons se recouvrent
partagent les mêmes rendements. Ils ne sont donc PAS i.i.d., alors que tous les
algorithmes d'apprentissage (et tous les bootstraps) le supposent. Conséquence : la
taille effective de l'échantillon est très inférieure à sa taille nominale, et le modèle
surapprend en croyant disposer de beaucoup plus d'information qu'il n'en a réellement.

Correctifs implémentés ici : unicité moyenne, pondération par attribution de rendement,
décroissance temporelle et bootstrap séquentiel.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def num_concurrent_events(bar_index: pd.DatetimeIndex, t1: pd.Series) -> pd.Series:
    """Nombre de labels « actifs » (non clôturés) à chaque barre."""
    t1 = t1.dropna()
    if t1.empty:
        return pd.Series(0.0, index=bar_index)
    start = bar_index.searchsorted(t1.index)
    end = bar_index.searchsorted(pd.DatetimeIndex(t1))
    counts = np.zeros(len(bar_index) + 1, dtype=float)
    np.add.at(counts, start, 1.0)
    np.add.at(counts, np.minimum(end + 1, len(bar_index)), -1.0)
    return pd.Series(np.cumsum(counts)[:-1], index=bar_index, name="concurrency")


def average_uniqueness(bar_index: pd.DatetimeIndex, t1: pd.Series) -> pd.Series:
    """Unicité moyenne d'un label = moyenne de 1/concurrence sur sa durée de vie.

    Un label seul sur son intervalle vaut 1.0 ; un label partagé avec 9 autres vaut 0.1.
    La somme des unicités donne la taille EFFECTIVE de l'échantillon.
    """
    conc = num_concurrent_events(bar_index, t1)
    inv = (1.0 / conc.replace(0.0, np.nan)).fillna(0.0)
    cum = np.concatenate([[0.0], np.cumsum(inv.to_numpy())])

    t1 = t1.dropna()
    start = bar_index.searchsorted(t1.index)
    end = np.minimum(bar_index.searchsorted(pd.DatetimeIndex(t1)) + 1, len(bar_index))
    span = np.maximum(end - start, 1)
    return pd.Series((cum[end] - cum[start]) / span, index=t1.index, name="tW")


def return_attribution_weights(
    bar_index: pd.DatetimeIndex, t1: pd.Series, close: pd.Series, normalize: bool = True
) -> pd.Series:
    """Poids ∝ |somme des log-rendements attribués|, corrigés de la concurrence.

    Deux effets combinés : les observations rares pèsent plus (unicité), et les
    observations associées à de gros mouvements pèsent plus (attribution de rendement).
    """
    conc = num_concurrent_events(bar_index, t1)
    log_ret = np.log(close).diff().reindex(bar_index).fillna(0.0)
    contrib = (log_ret / conc.replace(0.0, np.nan)).fillna(0.0)
    cum = np.concatenate([[0.0], np.cumsum(contrib.to_numpy())])

    t1 = t1.dropna()
    start = bar_index.searchsorted(t1.index)
    end = np.minimum(bar_index.searchsorted(pd.DatetimeIndex(t1)) + 1, len(bar_index))
    w = np.abs(cum[end] - cum[start])
    out = pd.Series(w, index=t1.index, name="w")
    if normalize and out.sum() > 0:
        out = out * len(out) / out.sum()
    return out


def time_decay_weights(tw: pd.Series, last_weight: float = 0.5) -> pd.Series:
    """Décroissance linéaire en unicité cumulée.

    `last_weight` = poids de l'observation la PLUS ANCIENNE (1.0 = pas de décroissance,
    0 = les plus vieilles ne comptent plus, < 0 = elles sont carrément supprimées).
    Justification : les marchés changent de régime, les données de 2012 ne décrivent plus
    la microstructure de 2026.
    """
    cum = tw.sort_index().cumsum()
    total = float(cum.iloc[-1]) if len(cum) else 1.0
    if last_weight >= 0:
        slope = (1.0 - last_weight) / total
    else:
        slope = 1.0 / ((last_weight + 1.0) * total)
    const = 1.0 - slope * total
    decay = const + slope * cum
    return decay.clip(lower=0.0)


def indicator_matrix(bar_index: pd.DatetimeIndex, t1: pd.Series) -> np.ndarray:
    """Matrice binaire (barres x labels) : 1 si le label i est actif à la barre t."""
    t1 = t1.dropna()
    m = np.zeros((len(bar_index), len(t1)), dtype=np.float32)
    start = bar_index.searchsorted(t1.index)
    end = np.minimum(bar_index.searchsorted(pd.DatetimeIndex(t1)) + 1, len(bar_index))
    for j, (a, b) in enumerate(zip(start, end)):
        m[a:b, j] = 1.0
    return m


def sequential_bootstrap(ind_m: np.ndarray, n_samples: Optional[int] = None,
                         rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Bootstrap séquentiel : tire les observations avec une probabilité proportionnelle
    à leur unicité **conditionnellement aux tirages déjà effectués**.

    Le bootstrap standard rééchantillonne des observations redondantes et produit des
    ensembles (bagging, random forest) dont les arbres sont bien plus corrélés qu'on ne
    le croit — d'où une sous-estimation massive de la variance out-of-sample.
    """
    rng = rng or np.random.default_rng(0)
    n_samples = n_samples or ind_m.shape[1]
    phi: list[int] = []
    running = np.zeros(ind_m.shape[0], dtype=np.float64)

    for _ in range(n_samples):
        denom = running + 1.0                      # concurrence si l'on ajoutait ce label
        avg_u = (ind_m / denom[:, None]).sum(axis=0)
        counts = ind_m.sum(axis=0)
        avg_u = np.divide(avg_u, np.maximum(counts, 1.0))
        total = avg_u.sum()
        prob = avg_u / total if total > 0 else np.full(ind_m.shape[1], 1.0 / ind_m.shape[1])
        pick = int(rng.choice(ind_m.shape[1], p=prob))
        phi.append(pick)
        running += ind_m[:, pick]
    return np.asarray(phi, dtype=np.int64)


def build_sample_weights(
    bar_index: pd.DatetimeIndex,
    t1: pd.Series,
    close: pd.Series,
    use_return_attribution: bool = True,
    time_decay_last: Optional[float] = 0.5,
) -> pd.DataFrame:
    """Compose les trois corrections en un jeu de poids final normalisé."""
    tw = average_uniqueness(bar_index, t1)
    w = (return_attribution_weights(bar_index, t1, close)
         if use_return_attribution else tw.copy())
    if time_decay_last is not None:
        w = w * time_decay_weights(tw, time_decay_last).reindex(w.index).fillna(0.0)
    if w.sum() > 0:
        w = w * len(w) / w.sum()
    return pd.DataFrame({"tW": tw, "w": w})

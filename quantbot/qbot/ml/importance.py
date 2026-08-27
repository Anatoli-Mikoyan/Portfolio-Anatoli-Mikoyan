"""Importance des features (cahier des charges §6, López de Prado ch. 8).

Trois méthodes, dans un ordre de fiabilité croissante :

**MDI** (Mean Decrease Impurity) — lue directement dans les arbres. Gratuite, mais
in-sample et biaisée vers les variables à forte cardinalité. À traiter comme un indice,
jamais comme une preuve.

**MDA** (Mean Decrease Accuracy) — on permute une colonne et on mesure la dégradation
hors échantillon, sur une CV purgée. C'est la mesure honnête.

**MDA par CLUSTERS** — indispensable en finance. Deux features fortement corrélées se
substituent l'une à l'autre : permuter la première laisse le modèle s'appuyer sur la
seconde, et les deux ressortent avec une importance nulle alors qu'ensemble elles portent
tout le signal. On regroupe donc les features corrélées et on permute le groupe entier.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from ..validation import PurgedKFold
from .dataset import MetaDataset
from .models import build_model, fit_model

log = get_logger("ml.importance")


# =======================================================================================
def mdi_importance(dataset: MetaDataset, model_name: str = "forest", **kwargs) -> pd.Series:
    """Importance par diminution d'impureté. Rapide, in-sample, indicative seulement."""
    model = build_model(model_name, **kwargs)
    fit_model(model, dataset.X.to_numpy(float), dataset.y.to_numpy(int),
              dataset.sample_weight.to_numpy(float))
    if not hasattr(model, "feature_importances_"):
        raise ValueError(f"{model_name} n'expose pas feature_importances_.")
    imp = pd.Series(model.feature_importances_, index=dataset.X.columns, name="mdi")
    return imp.sort_values(ascending=False)


# =======================================================================================
def _score(model, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Log-vraisemblance négative pondérée (plus c'est haut, mieux c'est).

    Préférée à l'exactitude : elle réagit aux probabilités elles-mêmes, alors qu'une
    exactitude à seuil ignore complètement l'ampleur de l'erreur.
    """
    from sklearn.metrics import log_loss

    p = model.predict_proba(X)[:, 1]
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -float(log_loss(y, p, sample_weight=w, labels=[0, 1]))


def mda_importance(
    dataset: MetaDataset,
    model_name: str = "forest",
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    n_repeats: int = 3,
    groups: Optional[Dict[str, List[str]]] = None,
    seed: int = 0,
    **kwargs,
) -> pd.DataFrame:
    """Importance par permutation, sur validation croisée purgée.

    `groups` permet de permuter des ENSEMBLES de colonnes simultanément (voir
    `cluster_features`), ce qui neutralise les effets de substitution.
    """
    X = dataset.X.to_numpy(dtype=float)
    y = dataset.y.to_numpy(dtype=int)
    w = dataset.sample_weight.to_numpy(dtype=float)
    columns = list(dataset.X.columns)
    col_pos = {c: i for i, c in enumerate(columns)}
    groups = groups or {c: [c] for c in columns}

    rng = np.random.default_rng(seed)
    cv = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct, t1=dataset.t1)
    records: Dict[str, List[float]] = {g: [] for g in groups}
    baselines: List[float] = []

    for train_idx, test_idx in cv.split(dataset.X.index):
        if train_idx.size < 100 or test_idx.size < 30 or len(np.unique(y[train_idx])) < 2:
            continue
        model = build_model(model_name, **kwargs)
        fit_model(model, X[train_idx], y[train_idx], w[train_idx])
        base = _score(model, X[test_idx], y[test_idx], w[test_idx])
        baselines.append(base)

        for gname, members in groups.items():
            idxs = [col_pos[c] for c in members if c in col_pos]
            if not idxs:
                continue
            losses = []
            for _ in range(n_repeats):
                X_perm = X[test_idx].copy()
                order = rng.permutation(X_perm.shape[0])
                # Permutation SOLIDAIRE des colonnes du groupe : les permuter séparément
                # détruirait leur structure de dépendance et surestimerait l'importance.
                X_perm[:, idxs] = X_perm[np.ix_(order, idxs)]
                losses.append(base - _score(model, X_perm, y[test_idx], w[test_idx]))
            records[gname].append(float(np.mean(losses)))

    if not baselines:
        raise ValueError("Aucun fold exploitable pour la MDA.")

    rows = []
    for gname, values in records.items():
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            continue
        rows.append({
            "feature": gname,
            "mda": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            # t de Student : une importance moyenne élevée mais instable entre folds
            # n'est pas exploitable.
            "t_stat": float(arr.mean() / (arr.std(ddof=1) / np.sqrt(arr.size)))
            if arr.size > 1 and arr.std(ddof=1) > 1e-12 else 0.0,
            "n_members": len(groups[gname]),
        })
    out = pd.DataFrame(rows).sort_values("mda", ascending=False).reset_index(drop=True)
    out.attrs["baseline_score"] = float(np.mean(baselines))
    return out


# =======================================================================================
def cluster_features(X: pd.DataFrame, n_clusters: Optional[int] = None,
                     threshold: float = 0.5) -> Dict[str, List[str]]:
    """Regroupe les features par corrélation (distance = sqrt((1-rho)/2)).

    Cette distance est une vraie métrique sur les corrélations : deux features
    parfaitement corrélées sont à distance 0, deux features indépendantes à 0.71, deux
    features parfaitement anticorrélées à 1.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    # `.to_numpy()` renvoie une vue en lecture seule sous pandas 3 : la copie est requise
    # avant toute écriture en place (fill_diagonal).
    corr = np.array(X.corr().fillna(0.0).to_numpy(), dtype=float, copy=True)
    np.fill_diagonal(corr, 1.0)
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0

    link = linkage(squareform(dist, checks=False), method="average")
    labels = (fcluster(link, t=n_clusters, criterion="maxclust") if n_clusters
              else fcluster(link, t=threshold, criterion="distance"))

    groups: Dict[str, List[str]] = {}
    for col, lab in zip(X.columns, labels):
        groups.setdefault(f"C{lab}", []).append(col)
    # Nom lisible : le cluster prend le nom de son premier membre, plus le compte.
    return {(members[0] if len(members) == 1 else f"{members[0]}+{len(members) - 1}"): members
            for members in groups.values()}


def clustered_mda(dataset: MetaDataset, threshold: float = 0.5, **kwargs) -> pd.DataFrame:
    """MDA par clusters — la version à utiliser quand les features sont corrélées."""
    groups = cluster_features(dataset.X, threshold=threshold)
    log.info("%d features regroupées en %d clusters", dataset.X.shape[1], len(groups))
    return mda_importance(dataset, groups=groups, **kwargs)


def select_features(importance: pd.DataFrame, min_t: float = 2.0,
                    max_features: Optional[int] = None) -> List[str]:
    """Retient les features dont l'importance est STABLE entre folds.

    Le critère porte sur le t de Student, pas sur l'importance moyenne : une feature
    dont l'importance varie beaucoup d'un fold à l'autre décrit une particularité de
    période, pas une régularité de marché.
    """
    kept = importance[(importance["mda"] > 0) & (importance["t_stat"] >= min_t)]
    kept = kept.sort_values("mda", ascending=False)
    if max_features:
        kept = kept.head(max_features)
    names: List[str] = []
    for _, row in kept.iterrows():
        names.append(str(row["feature"]))
    return names

"""Trois approches de détection de régime, comparables entre elles (§7).

Le cahier demande de « proposer plusieurs approches et de les comparer ». Elles sont
ordonnées par complexité, comme pour le zoo de modèles :

  1. **Règles** — percentile de volatilité croisé avec la force de tendance. Aucun
     apprentissage, aucun risque de sur-apprentissage, immédiatement interprétable.
     C'est la référence à battre.
  2. **Clustering** — k-moyennes ou mélange gaussien sur les features de régime.
     Apprend la structure sans supposer de dynamique temporelle.
  3. **HMM** — seul modèle à représenter explicitement la PERSISTANCE des régimes et
     les probabilités de transition. C'est ce qui le rend pertinent ici : un régime de
     marché n'est pas un tirage indépendant à chaque barre.

Le filtrage causal est implémenté à la main pour les trois, y compris pour le HMM dont
les méthodes fournies par les bibliothèques (`predict` par Viterbi, `predict_proba` par
lissage avant-arrière) utilisent toutes la série entière.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from .base import RegimeDetector, RegimeSeries

log = get_logger("regime.detectors")


def _warn_if_normalized(X: pd.DataFrame, detector: str) -> None:
    """Alerte si les features ont subi un z-score glissant court.

    Ce n'est pas une précaution théorique : mesuré sur ce dépôt, un marché à deux régimes
    de volatilité 2 % et 40 % — trivialement séparables — donne un ARI de 0.011 (le
    hasard) avec des features z-scorées sur 300 barres, contre 0.947 avec des features de
    niveau. La normalisation glissante efface le niveau, qui EST l'information de régime.
    """
    from .features import looks_rolling_normalized

    if looks_rolling_normalized(X):
        log.warning(
            "%s : les features semblent z-scorées sur fenêtre glissante courte. "
            "La détection de régime a besoin de NIVEAUX — utiliser "
            "qbot.regime.build_regime_matrix(). Voir qbot/regime/features.py.",
            detector,
        )


# =======================================================================================
class RuleBasedDetector(RegimeDetector):
    """Quatre régimes par croisement volatilité × tendance. Aucun paramètre appris.

    Volontairement primitif. S'il n'est pas battu par le HMM, c'est que la complexité
    supplémentaire ne sert à rien — et cela arrive plus souvent qu'on ne le croit.
    """

    name = "rules"

    def __init__(self, vol_col: str = "vol_pctile", trend_col: str = "trend_strength",
                 vol_threshold: float = 0.5, trend_threshold: float = 0.3):
        self.vol_col, self.trend_col = vol_col, trend_col
        self.vol_threshold, self.trend_threshold = vol_threshold, trend_threshold
        self.n_states = 4
        self._labels = {
            0: "vol basse / range", 1: "vol basse / tendance",
            2: "vol haute / range", 3: "vol haute / tendance",
        }

    def fit(self, X: pd.DataFrame) -> "RuleBasedDetector":
        missing = [c for c in (self.vol_col, self.trend_col) if c not in X.columns]
        if missing:
            raise ValueError(f"Colonnes de régime absentes : {missing}")
        return self

    def filter(self, X: pd.DataFrame) -> RegimeSeries:
        # Les deux entrées sont déjà des statistiques glissantes causales.
        high_vol = (X[self.vol_col] > self.vol_threshold).astype(int)
        trending = (X[self.trend_col].abs() > self.trend_threshold).astype(int)
        states = (2 * high_vol + trending).rename("regime")
        return RegimeSeries(states=states, causal=True, detector=self.name,
                            labels=dict(self._labels))


# =======================================================================================
class ClusteringDetector(RegimeDetector):
    """k-moyennes ou mélange gaussien sur les features de régime.

    Sans dynamique temporelle : l'état à t ne dépend que des features à t. C'est à la
    fois sa faiblesse (régimes instables d'une barre à l'autre) et sa force (aucune
    possibilité structurelle de fuite du futur).
    """

    name = "clustering"

    def __init__(self, n_states: int = 4, method: str = "kmeans", seed: int = 0):
        self.n_states, self.method, self.seed = int(n_states), method, seed
        self._model = None
        self._scaler = None
        self._labels: Dict[int, str] = {}

    def fit(self, X: pd.DataFrame) -> "ClusteringDetector":
        _warn_if_normalized(X, self.name)
        from sklearn.cluster import KMeans
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler().fit(X.to_numpy(dtype=float))
        Z = self._scaler.transform(X.to_numpy(dtype=float))
        if self.method == "kmeans":
            self._model = KMeans(n_clusters=self.n_states, n_init=10,
                                 random_state=self.seed).fit(Z)
        elif self.method == "gmm":
            self._model = GaussianMixture(n_components=self.n_states, covariance_type="diag",
                                          random_state=self.seed, max_iter=200).fit(Z)
        else:
            raise ValueError(f"method inconnue : {self.method}")
        self._labels = self.label_states(X, pd.Series(self._predict(Z), index=X.index))
        return self

    def _predict(self, Z: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict(Z), dtype=int)

    def filter(self, X: pd.DataFrame) -> RegimeSeries:
        if self._model is None:
            raise RuntimeError("ClusteringDetector non ajusté.")
        Z = self._scaler.transform(X.to_numpy(dtype=float))
        states = pd.Series(self._predict(Z), index=X.index, name="regime")
        proba = None
        if hasattr(self._model, "predict_proba"):
            proba = pd.DataFrame(self._model.predict_proba(Z), index=X.index)
        return RegimeSeries(states=states, proba=proba, causal=True, detector=self.name,
                            labels=dict(self._labels))


# =======================================================================================
class HMMDetector(RegimeDetector):
    """Modèle de Markov caché gaussien, avec filtrage causal implémenté à la main.

    Le HMM est le seul des trois à modéliser la persistance : sa matrice de transition
    dit explicitement qu'un régime a tendance à durer. C'est exactement le fait stylisé
    qu'on cherche à capturer, et ce que le clustering ignore par construction.

    **Point critique.** `hmmlearn.predict()` applique Viterbi sur la séquence entière et
    `predict_proba()` un lissage avant-arrière : les deux utilisent le futur. On ne s'en
    sert donc que pour l'ajustement, et l'inférence en ligne est réimplémentée par la
    récursion avant (forward), qui n'utilise que les observations ≤ t.
    """

    name = "hmm"

    def __init__(self, n_states: int = 3, covariance_type: str = "diag",
                 n_iter: int = 100, seed: int = 0):
        self.n_states, self.covariance_type = int(n_states), covariance_type
        self.n_iter, self.seed = int(n_iter), seed
        self._model = None
        self._scaler = None
        self._labels: Dict[int, str] = {}

    # ---------------------------------------------------------------------------------
    def fit(self, X: pd.DataFrame) -> "HMMDetector":
        _warn_if_normalized(X, self.name)
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError as exc:                            # pragma: no cover
            raise ImportError("HMMDetector requiert hmmlearn (pip install hmmlearn).") from exc
        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler().fit(X.to_numpy(dtype=float))
        Z = self._scaler.transform(X.to_numpy(dtype=float))
        self._model = GaussianHMM(n_components=self.n_states,
                                  covariance_type=self.covariance_type,
                                  n_iter=self.n_iter, random_state=self.seed)
        self._model.fit(Z)
        log.info("HMM ajusté : %d états, persistance moyenne %.3f",
                 self.n_states, float(np.mean(np.diag(self._model.transmat_))))
        self._labels = self.label_states(X, self.filter(X).states)
        return self

    # ---------------------------------------------------------------------------------
    def _log_emissions(self, Z: np.ndarray) -> np.ndarray:
        """Log-vraisemblance d'émission par état, calculée depuis les attributs publics.

        Reproduit le calcul plutôt que d'appeler `_compute_log_likelihood`, qui est une
        API privée susceptible de changer d'une version à l'autre.
        """
        means = self._model.means_                       # (K, D)
        covars = self._model.covars_                     # (K, D, D) ou (K, D)
        var = np.array([np.diag(c) if c.ndim == 2 else c for c in covars], dtype=float)
        var = np.maximum(var, 1e-12)

        diff = Z[:, None, :] - means[None, :, :]         # (T, K, D)
        quad = np.sum(diff ** 2 / var[None, :, :], axis=2)
        norm = np.sum(np.log(2.0 * np.pi * var), axis=1)  # (K,)
        return -0.5 * (quad + norm[None, :])

    def _forward_filter(self, Z: np.ndarray) -> np.ndarray:
        """Récursion avant normalisée : alpha_t ∝ b_t ⊙ (A^T alpha_{t-1}).

        Retourne P(état_t | observations 1..t) — la quantité effectivement disponible à
        l'instant t en production. Travail en log puis normalisation par pas pour éviter
        le sous-écoulement numérique sur longues séries.
        """
        log_b = self._log_emissions(Z)
        A = np.maximum(self._model.transmat_, 1e-300)
        pi = np.maximum(self._model.startprob_, 1e-300)

        T, K = log_b.shape
        alpha = np.empty((T, K), dtype=float)
        log_pi = np.log(pi)
        log_A = np.log(A)

        prev = log_pi + log_b[0]
        prev -= prev.max()
        a = np.exp(prev)
        alpha[0] = a / a.sum()

        for t in range(1, T):
            # transition puis émission, en log pour la stabilité
            pred = np.log(np.maximum(alpha[t - 1] @ A, 1e-300))
            cur = pred + log_b[t]
            cur -= cur.max()
            a = np.exp(cur)
            alpha[t] = a / a.sum()
        return alpha

    # ---------------------------------------------------------------------------------
    def filter(self, X: pd.DataFrame) -> RegimeSeries:
        if self._model is None:
            raise RuntimeError("HMMDetector non ajusté.")
        Z = self._scaler.transform(X.to_numpy(dtype=float))
        alpha = self._forward_filter(Z)
        states = pd.Series(alpha.argmax(axis=1), index=X.index, name="regime")
        return RegimeSeries(states=states, proba=pd.DataFrame(alpha, index=X.index),
                            causal=True, detector=self.name, labels=dict(self._labels))

    def smooth(self, X: pd.DataFrame) -> RegimeSeries:
        """Lissage avant-arrière — utilise TOUTE la série. Analyse rétrospective seulement.

        Fourni délibérément, et délibérément marqué non causal : c'est la version qui
        produit de jolis graphiques de régimes et des backtests faux.
        """
        if self._model is None:
            raise RuntimeError("HMMDetector non ajusté.")
        Z = self._scaler.transform(X.to_numpy(dtype=float))
        posteriors = self._model.predict_proba(Z)
        return RegimeSeries(
            states=pd.Series(posteriors.argmax(axis=1), index=X.index, name="regime"),
            proba=pd.DataFrame(posteriors, index=X.index),
            causal=False, detector=self.name, labels=dict(self._labels),
        )

    @property
    def persistence(self) -> np.ndarray:
        """Probabilité de rester dans chaque état — le paramètre qui distingue le HMM."""
        if self._model is None:
            raise RuntimeError("HMMDetector non ajusté.")
        return np.diag(self._model.transmat_).copy()


def build_detector(kind: str, **kwargs) -> RegimeDetector:
    builders = {
        "rules": RuleBasedDetector,
        "kmeans": lambda **kw: ClusteringDetector(method="kmeans", **kw),
        "gmm": lambda **kw: ClusteringDetector(method="gmm", **kw),
        "hmm": HMMDetector,
    }
    if kind not in builders:
        raise ValueError(f"Détecteur inconnu : {kind}. Disponibles : {list(builders)}")
    return builders[kind](**kwargs)

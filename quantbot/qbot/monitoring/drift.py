"""Détection de dérive des distributions (cahier des charges §17).

Le problème que ce module résout est le plus insidieux du ML en production. Un modèle
ne « tombe pas en panne » : il continue de répondre, avec la même assurance, alors que
les données qu'on lui présente ne ressemblent plus à celles sur lesquelles il a été
entraîné. La perte est progressive, attribuée au hasard, et constatée trop tard.

Trois familles de dérive, qu'il faut distinguer parce qu'elles n'appellent pas la même
réaction :

  * **Dérive de covariables** — P(X) change (la volatilité double, le spread s'élargit,
    l'heure de la séance change). Le modèle est hors distribution. Réaction : réduire la
    taille, éventuellement passer à plat.
  * **Dérive de concept** — P(Y|X) change : les mêmes features n'ont plus la même
    conséquence. Invisible sur X seul ; se détecte par la divergence entre performance
    attendue et réalisée (voir `reconciliation.py`).
  * **Dérive de prior** — P(Y) change (le marché passe de tendanciel à sans tendance).

Ce module traite la première et fournit le détecteur séquentiel (Page-Hinkley) utilisé
par la seconde.

Un avertissement statistique qui coûte cher quand on l'ignore : **les tests à deux
échantillons supposent des observations indépendantes**. Les features financières sont
fortement autocorrélées ; une fenêtre glissante de 250 barres horaires ne contient pas
250 informations indépendantes. Appliquer un Kolmogorov-Smirnov brut sur de telles
séries produit des p-values ridiculement petites en permanence — donc des alertes que
l'équipe finit par ignorer. On corrige ici par la taille d'échantillon effective
(Bayley & Hammersley), ce qui rend le test utilisable au quotidien.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import special, stats

from ..utils.logging import get_logger

log = get_logger("monitoring.drift")

__all__ = [
    "population_stability_index", "kl_divergence", "jensen_shannon_distance",
    "ks_two_sample", "ks_one_sample", "effective_sample_size", "PageHinkley", "FeatureDrift",
    "DriftReport", "ReferenceDistribution", "DriftMonitor",
]

# Seuils PSI usuels du risk management (crédit puis, par extension, ML en production).
PSI_STABLE = 0.10
PSI_SHIFTED = 0.25


# =======================================================================================
# Distances entre distributions discrétisées
# =======================================================================================
def _smoothed_probabilities(counts: np.ndarray, pseudo: float = 0.5) -> np.ndarray:
    """Probabilités lissées par un a priori de Jeffreys (0.5 par case).

    Sans lissage, une case vide côté production rend le PSI et la divergence KL infinis :
    le tableau de bord n'affiche plus que `inf` et perd toute capacité à classer les
    features par gravité. Le lissage additif borne la statistique tout en conservant son
    ordonnancement.
    """
    c = np.asarray(counts, dtype=float)
    total = c.sum() + pseudo * c.size
    if total <= 0:
        return np.full(c.size, 1.0 / max(c.size, 1))
    return (c + pseudo) / total


def population_stability_index(p_ref: np.ndarray, p_live: np.ndarray) -> float:
    """PSI = Σ (p_live - p_ref) · ln(p_live / p_ref).

    C'est la divergence de Kullback-Leibler symétrisée par addition (divergence de
    Jeffreys). Interprétation industrielle : < 0.10 stable, 0.10-0.25 décalage modéré,
    > 0.25 décalage significatif.

    Attention : le PSI dépend du nombre de cases. Comparer deux PSI n'a de sens que s'ils
    ont été calculés avec le même découpage — d'où le fait que le découpage soit figé
    dans `ReferenceDistribution` au moment de l'entraînement, et jamais recalculé.
    """
    a = np.asarray(p_ref, dtype=float)
    b = np.asarray(p_live, dtype=float)
    if a.shape != b.shape or a.size == 0:
        return float("nan")
    return float(np.sum((b - a) * np.log(b / a)))


def kl_divergence(p_ref: np.ndarray, p_live: np.ndarray) -> float:
    """KL(live ‖ ref) en nats : coût d'encoder le flux live avec le code de référence."""
    a = np.asarray(p_ref, dtype=float)
    b = np.asarray(p_live, dtype=float)
    return float(np.sum(b * np.log(b / a)))


def jensen_shannon_distance(p_ref: np.ndarray, p_live: np.ndarray) -> float:
    """Racine de la divergence de Jensen-Shannon (base 2) : une vraie métrique de [0, 1].

    Contrairement au PSI, elle est bornée. Utile pour comparer la gravité de dérives
    entre features aux découpages différents, ou pour agréger en un score global.
    """
    a = np.asarray(p_ref, dtype=float)
    b = np.asarray(p_live, dtype=float)
    m = 0.5 * (a + b)
    div = 0.5 * np.sum(a * np.log2(a / m)) + 0.5 * np.sum(b * np.log2(b / m))
    return float(np.sqrt(max(div, 0.0)))


# =======================================================================================
# Tests à deux échantillons, corrigés de l'autocorrélation
# =======================================================================================
def effective_sample_size(x: np.ndarray) -> float:
    """Taille d'échantillon effective d'une série autocorrélée.

    n_eff = n · (1 - ρ₁) / (1 + ρ₁), avec ρ₁ l'autocorrélation de rang 1.

    Pour ρ₁ = 0.9 — parfaitement banal pour une volatilité glissante — 250 observations
    n'en valent que 13. Utiliser 250 dans un test de Kolmogorov-Smirnov reviendrait à
    multiplier la puissance par ~4.4 et à déclarer une dérive significative en
    permanence. Cette correction est la différence entre un système d'alerte que
    l'équipe lit et un système qu'elle coupe.
    """
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 8:
        return float(n)
    sd = v.std(ddof=1)
    if sd < 1e-14:
        return 1.0
    rho = float(np.corrcoef(v[:-1], v[1:])[0, 1])
    if not np.isfinite(rho):
        rho = 0.0
    rho = float(np.clip(rho, -0.99, 0.99))
    return float(np.clip(n * (1.0 - rho) / (1.0 + rho), 2.0, float(n)))


def ks_two_sample(ref: np.ndarray, live: np.ndarray,
                  autocorr_correction: bool = True) -> tuple[float, float]:
    """Statistique de Kolmogorov-Smirnov à deux échantillons, p-value corrigée.

    Retourne (D, p). La statistique D est celle du test standard ; seule la p-value est
    recalculée avec les tailles effectives, via la loi asymptotique de Kolmogorov.

    À réserver au cas où les DEUX échantillons sont des séries temporelles observées.
    Pour comparer une fenêtre de production à une référence dont on connaît la fonction
    de répartition, `ks_one_sample` est à la fois plus juste et plus puissant.
    """
    a = np.asarray(ref, dtype=float)
    b = np.asarray(live, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 5 or b.size < 5:
        return float("nan"), float("nan")

    d = float(stats.ks_2samp(a, b).statistic)
    if autocorr_correction:
        n1, n2 = effective_sample_size(a), effective_sample_size(b)
    else:
        n1, n2 = float(a.size), float(b.size)
    n_e = n1 * n2 / (n1 + n2)
    p = float(special.kolmogorov(math.sqrt(n_e) * d))
    return d, float(np.clip(p, 0.0, 1.0))


def ks_one_sample(sample: np.ndarray, cdf, n_effective: Optional[float] = None) -> tuple[float, float]:
    """Kolmogorov-Smirnov d'un échantillon contre une fonction de répartition connue.

    C'est le test approprié pour la surveillance de dérive : la référence n'est pas un
    second échantillon aléatoire, c'est une loi estimée une fois pour toutes à
    l'entraînement et figée avec le modèle. La traiter comme un échantillon aléatoire
    reviendrait à payer deux fois l'incertitude d'estimation, donc à perdre en puissance.

    `n_effective` permet d'injecter la taille d'échantillon corrigée de
    l'autocorrélation ; par défaut elle est estimée sur l'échantillon lui-même.
    """
    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 5:
        return float("nan"), float("nan")
    xs = np.sort(x)
    f = np.clip(np.asarray(cdf(xs), dtype=float), 0.0, 1.0)
    i = np.arange(1, n + 1, dtype=float)
    d = float(max(np.max(i / n - f), np.max(f - (i - 1.0) / n)))
    n_eff = float(n_effective) if n_effective is not None else effective_sample_size(x)
    p = float(special.kolmogorov(math.sqrt(max(n_eff, 1.0)) * d))
    return d, float(np.clip(p, 0.0, 1.0))


# =======================================================================================
# Détecteur séquentiel de changement de moyenne
# =======================================================================================
@dataclass
class PageHinkley:
    """Test de Page-Hinkley standardisé : détection en ligne d'un décrochage de moyenne.

    Principe : on cumule les écarts à la moyenne courante, diminués d'une tolérance δ qui
    fait dériver le cumul vers le bas tant que rien ne change. Dès qu'un régime nouveau
    s'installe, le cumul remonte et son écart au minimum historique franchit λ.

    Pourquoi celui-ci plutôt qu'un seuil sur moyenne glissante : le Page-Hinkley est le
    test séquentiel du rapport de vraisemblance appliqué à un saut de moyenne. À taux de
    fausse alarme égal, il détecte plus vite — et sur un compte en production, le délai
    de détection est exactement ce qu'on paie.

    **Les écarts sont standardisés** par l'écart-type courant (Welford). Sans cela, λ
    devrait être recalibré pour chaque variable surveillée — rendement horaire, latence
    en millisecondes, PSI — et se tromperait d'un facteur mille. Ici λ s'exprime en
    écarts-types cumulés et garde le même sens partout : λ = 25 signifie « il faudrait
    25 σ de dérive cumulée pour l'expliquer par le hasard ».

    Le détecteur est bidirectionnel : on surveille aussi bien l'effondrement du rendement
    (dérive de concept) qu'une explosion suspecte — laquelle, en pratique, signale
    presque toujours un bug de données plutôt qu'un don du ciel.

    **Deux modes, et confondre les deux annule le détecteur.**

      * *Adaptatif* (défaut) : la moyenne et l'écart-type de référence sont estimés au
        fil de l'eau. C'est ce qu'il faut quand aucune référence n'existe — latence,
        PSI, spread : on cherche une rupture par rapport à « ce qui se passait avant ».
      * *Référence fixe* (`ref_mean` / `ref_std` fournis) : la référence vient d'ailleurs
        — typiquement des moments du backtest. **Obligatoire pour surveiller la
        performance** : en mode adaptatif, la moyenne courante suit la série et absorbe
        exactement la dégradation qu'on cherche à voir. Le détecteur reste alors muet
        pendant que la stratégie s'effondre, en toute logique et en toute inutilité.

    **Calibration de (δ, λ).** Un détecteur non calibré est pire qu'aucun détecteur :
    il produit soit du bruit qu'on apprend à ignorer, soit un silence rassurant et faux.
    La règle classique fixe δ à la moitié de la dérive qu'on veut repérer ; λ se déduit
    ensuite du budget de fausses alarmes via l'approximation de Siegmund du temps moyen
    avant alarme sous H₀ (`arl0`, exposée ici pour être lue, pas devinée) :

        ARL₀ ≈ [exp(2δ(λ + 1.166)) − 2δ(λ + 1.166) − 1] / (2δ²)

    et le délai moyen de détection d'une dérive d'amplitude μ vaut environ λ / (μ − δ).
    Les valeurs par défaut (δ = 0.25, λ = 15) visent une dérive de 0.5 σ : environ une
    fausse alarme tous les 12 000 pas — soit près de deux ans en barres horaires — pour
    un délai médian de l'ordre de 50 barres. Les deux chiffres sont vérifiés par
    simulation dans `tests/test_monitoring.py`.
    """
    delta: float = 0.25       # tolérance en σ : moitié de la dérive à détecter
    threshold: float = 15.0   # λ : seuil d'alarme, en σ cumulés
    burn_in: int = 30         # observations d'échauffement avant toute alarme
    ref_mean: Optional[float] = None   # référence connue : voir ci-dessous
    ref_std: Optional[float] = None

    n: int = field(default=0, init=False)
    mean: float = field(default=0.0, init=False)
    _m2: float = field(default=0.0, init=False)
    _cum_up: float = field(default=0.0, init=False)
    _min_up: float = field(default=0.0, init=False)
    _cum_dn: float = field(default=0.0, init=False)
    _max_dn: float = field(default=0.0, init=False)
    triggered: bool = field(default=False, init=False)
    direction: str = field(default="", init=False)

    @property
    def std(self) -> float:
        return float(np.sqrt(self._m2 / (self.n - 1))) if self.n > 1 else 0.0

    def update(self, value: float) -> bool:
        """Ajoute une observation. Retourne True si une rupture est détectée."""
        x = float(value)
        if not np.isfinite(x):
            return self.triggered

        # Welford : moyenne et variance courantes en un seul passage, sans stocker la série.
        self.n += 1
        delta_x = x - self.mean
        self.mean += delta_x / self.n
        self._m2 += delta_x * (x - self.mean)

        if self.ref_std is not None and self.ref_mean is not None:
            # Référence fixe : aucun échauffement n'est nécessaire, la loi sous H₀ est
            # connue dès la première observation.
            sd = float(self.ref_std)
            if sd < 1e-14:
                return self.triggered
            z = (x - float(self.ref_mean)) / sd
        else:
            sd = self.std
            if self.n <= self.burn_in or sd < 1e-14:
                # Pendant l'échauffement on alimente les moments sans cumuler : les
                # premiers écarts, mesurés contre une moyenne encore instable,
                # produiraient des fausses alarmes systématiques.
                return self.triggered
            z = delta_x / sd        # écart à la moyenne AVANT mise à jour : non biaisé
        self._cum_up += z - self.delta
        self._min_up = min(self._min_up, self._cum_up)
        self._cum_dn += z + self.delta
        self._max_dn = max(self._max_dn, self._cum_dn)

        if self._cum_up - self._min_up > self.threshold:
            self.triggered, self.direction = True, "hausse"
        elif self._max_dn - self._cum_dn > self.threshold:
            self.triggered, self.direction = True, "baisse"
        return self.triggered

    @property
    def statistic(self) -> float:
        """Statistique courante, en σ cumulés : comparable directement à `threshold`."""
        return float(max(self._cum_up - self._min_up, self._max_dn - self._cum_dn))

    def arl0(self) -> float:
        """Temps moyen avant fausse alarme sous H₀ (approximation de Siegmund, bilatéral).

        À lire comme un budget : « ce détecteur criera au loup une fois toutes les N
        barres même si rien ne se passe ». Si N est inférieur à l'horizon de décision,
        le seuil est trop bas et l'alerte sera ignorée.
        """
        if self.delta <= 1e-9:
            return float("inf")
        b = 2.0 * self.delta * (self.threshold + 1.166)
        one_sided = (math.exp(min(b, 700.0)) - b - 1.0) / (2.0 * self.delta ** 2)
        return float(one_sided / 2.0)

    def expected_delay(self, magnitude: float) -> float:
        """Délai moyen de détection d'une dérive d'amplitude `magnitude` (en σ)."""
        gap = abs(float(magnitude)) - self.delta
        return float(self.threshold / gap) if gap > 1e-9 else float("inf")

    @classmethod
    def calibrate(cls, magnitude: float, arl0: float = 10_000.0,
                  ref_mean: Optional[float] = None, ref_std: Optional[float] = None,
                  burn_in: int = 30) -> "PageHinkley":
        """Construit un détecteur pour une dérive donnée et un budget de fausses alarmes.

        `magnitude` est l'amplitude de dérive à détecter, en écarts-types PAR
        OBSERVATION. C'est le paramètre qu'il faut penser, et il est facile à traduire :
        une chute du Sharpe annualisé de ΔS correspond à une dérive de
        ΔS / √(barres par an) écarts-types par barre. Pour ΔS = 2 en barres horaires
        (6240 par an), cela fait 0.025 σ — un signal minuscule, et c'est bien pour cela
        que la détection prend du temps. `expected_delay` donne le prix exact.

        δ est fixé à magnitude / 2 (règle classique du CUSUM, quasi optimale au sens du
        délai moyen), puis λ est résolu numériquement dans l'approximation de Siegmund
        pour atteindre `arl0` (bilatéral).
        """
        mag = abs(float(magnitude))
        if mag < 1e-12:
            raise ValueError("L'amplitude à détecter doit être strictement positive.")
        delta = mag / 2.0
        target = 2.0 * float(arl0) * 2.0 * delta ** 2      # (e^b − b − 1) recherché

        lo, hi = 1e-9, 700.0
        for _ in range(200):                               # bissection : f est croissante
            mid = 0.5 * (lo + hi)
            if math.exp(min(mid, 700.0)) - mid - 1.0 < target:
                lo = mid
            else:
                hi = mid
        lam = max(0.5 * (lo + hi) / (2.0 * delta) - 1.166, 1.0)
        return cls(delta=delta, threshold=lam, burn_in=burn_in,
                   ref_mean=ref_mean, ref_std=ref_std)

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self._m2 = 0.0
        self._cum_up = self._min_up = self._cum_dn = self._max_dn = 0.0
        self.triggered = False
        self.direction = ""


# =======================================================================================
# Référence figée à l'entraînement
# =======================================================================================
@dataclass
class FeatureDrift:
    name: str
    psi: float
    kl: float
    js: float
    ks_stat: float
    ks_pvalue: float
    ref_mean: float
    live_mean: float
    ref_std: float
    live_std: float
    z_shift: float          # décalage de moyenne en écarts-types de référence
    n_live: int
    n_effective: float
    psi_warn: float = PSI_STABLE
    psi_critical: float = PSI_SHIFTED
    calibrated: bool = False

    @property
    def verdict(self) -> str:
        if not np.isfinite(self.psi):
            return "indéterminé"
        if self.psi >= self.psi_critical:
            return "critique"
        if self.psi >= self.psi_warn:
            return "modéré"
        return "stable"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["verdict"] = self.verdict
        return d


@dataclass
class DriftReport:
    features: List[FeatureDrift]
    n_live: int
    psi_warn: float = PSI_STABLE
    psi_critical: float = PSI_SHIFTED

    @property
    def worst(self) -> Optional[FeatureDrift]:
        finite = [f for f in self.features if np.isfinite(f.psi)]
        return max(finite, key=lambda f: f.psi) if finite else None

    @property
    def n_critical(self) -> int:
        return sum(1 for f in self.features if f.verdict == "critique")

    @property
    def n_moderate(self) -> int:
        return sum(1 for f in self.features if f.verdict == "modéré")

    @property
    def calibrated(self) -> bool:
        return any(f.calibrated for f in self.features)

    @property
    def global_score(self) -> float:
        """Score global borné dans [0, 1] : moyenne des distances de Jensen-Shannon.

        Le PSI n'est pas moyennable (non borné, une seule feature aberrante l'écrase).
        La distance JS l'est, ce qui donne un indicateur unique lisible sur le tableau
        de bord sans masquer les cas extrêmes — que `n_critical` continue de signaler.
        """
        vals = [f.js for f in self.features if np.isfinite(f.js)]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def status(self) -> str:
        if self.n_critical:
            return "critique"
        if self.n_moderate:
            return "modéré"
        return "stable"

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame([f.to_dict() for f in self.features])
        return df.sort_values("psi", ascending=False).reset_index(drop=True) if not df.empty else df

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_live": self.n_live,
            "status": self.status,
            "global_score": self.global_score,
            "n_critical": self.n_critical,
            "n_moderate": self.n_moderate,
            "worst": self.worst.name if self.worst else None,
            "worst_psi": self.worst.psi if self.worst else float("nan"),
            "calibrated": self.calibrated,
            "features": [f.to_dict() for f in self.features],
        }

    def __str__(self) -> str:  # pragma: no cover - affichage
        df = self.to_frame()
        origine = "seuils calibrés sur l'entraînement" if self.calibrated \
                  else "SEUILS INDUSTRIELS NON CALIBRÉS"
        head = f"Dérive : {self.status.upper()} — score global {self.global_score:.4f} " \
               f"({self.n_critical} critiques, {self.n_moderate} modérées, n={self.n_live}) " \
               f"[{origine}]"
        if df.empty:
            return head
        cols = ["name", "psi", "psi_critical", "js", "ks_pvalue", "z_shift", "verdict"]
        top = df.head(10)[[c for c in cols if c in df.columns]]
        return head + "\n" + top.to_string(index=False, float_format=lambda v: f"{v:8.4f}")


@dataclass
class ReferenceDistribution:
    """Photographie de la distribution des features au moment de l'entraînement.

    Point de méthode essentiel : la référence est **figée et versionnée avec le modèle**.
    Comparer la fenêtre live aux 250 barres précédentes ne détecterait qu'un changement
    brutal ; une dérive lente — celle qui tue un modèle en six mois — passerait
    inaperçue puisque la référence dériverait avec elle. On compare donc toujours à ce
    que le modèle a réellement appris.
    """
    feature_names: List[str]
    edges: Dict[str, List[float]]        # bornes intérieures des cases, par feature
    ref_counts: Dict[str, List[float]]
    mean: Dict[str, float]
    std: Dict[str, float]
    quantiles: Dict[str, List[float]]    # 5 / 25 / 50 / 75 / 95
    n_ref: int = 0
    n_bins: int = 10
    model_id: str = ""
    created: str = ""
    psi_warn_by_feature: Dict[str, float] = field(default_factory=dict)
    psi_critical_by_feature: Dict[str, float] = field(default_factory=dict)
    calibration_window: int = 0
    n_calibration_windows: int = 0

    # -- construction -------------------------------------------------------------------
    @classmethod
    def fit(cls, X: pd.DataFrame, n_bins: int = 10, model_id: str = "") -> "ReferenceDistribution":
        from datetime import datetime, timezone

        if X.empty:
            raise ValueError("Matrice de référence vide.")
        names = [str(c) for c in X.columns]
        edges: Dict[str, List[float]] = {}
        counts: Dict[str, List[float]] = {}
        mean: Dict[str, float] = {}
        std: Dict[str, float] = {}
        quants: Dict[str, List[float]] = {}

        for name in names:
            v = X[name].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                v = np.zeros(1)
            # Découpage par quantiles : chaque case porte le même poids de référence, ce
            # qui maximise la sensibilité du PSI et évite les cases vides côté référence.
            cut = np.unique(np.quantile(v, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]))
            if cut.size <= 1:
                # Un seul point de coupe distinct : la feature est constante, ou quasi
                # (un drapeau rare, un indicateur saturé). Il ne suffit PAS de poser une seule
                # borne : toute la masse de référence et toute la masse de production
                # tomberaient dans la même case, et une feature devenue vivante — le cas
                # typique d'un indicateur gelé qui se remet à bouger, ou d'un flux figé
                # qui reprend — afficherait un PSI nul. On encadre donc la valeur pour
                # que « en dessous », « à la valeur » et « au-dessus » soient distingués.
                v0 = float(cut[0]) if cut.size else float(v[0])
                pad = max(abs(v0), 1.0) * 1e-6
                cut = np.array([v0 - pad, v0, v0 + pad])
            edges[name] = [float(e) for e in cut]
            counts[name] = np.bincount(np.searchsorted(cut, v, side="right"),
                                       minlength=cut.size + 1).astype(float).tolist()
            mean[name] = float(v.mean())
            std[name] = float(v.std(ddof=1)) if v.size > 1 else 0.0
            quants[name] = [float(q) for q in np.quantile(v, [0.05, 0.25, 0.50, 0.75, 0.95])]

        return cls(
            feature_names=names, edges=edges, ref_counts=counts, mean=mean, std=std,
            quantiles=quants, n_ref=int(len(X)), n_bins=int(n_bins), model_id=model_id,
            created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    # -- calibration ---------------------------------------------------------------------
    def calibrate(self, X: pd.DataFrame, window: int = 250, step: Optional[int] = None,
                  warn_q: float = 0.95, crit_q: float = 0.99,
                  min_windows: int = 8, floor: float = 0.05) -> "ReferenceDistribution":
        """Calibre les seuils de dérive sur les données d'ENTRAÎNEMENT elles-mêmes.

        Sans cette étape, la couche de dérive est inutilisable — et de façon invisible.
        Les seuils industriels du PSI (0.10 / 0.25) viennent du scoring de crédit, où
        l'on compare deux grandes populations stables. Ici on compare une fenêtre de
        250 barres, qui vit dans UN régime, à une référence groupée qui en mélange des
        dizaines. Mesuré sur ce dépôt, en appliquant 0.25 à des fenêtres tirées du jeu
        d'entraînement lui-même : **44 % des mesures dépassent le seuil, et 27 features
        sur 61 sont déclarées « critiques »** — sur des données que le modèle a apprises.
        Un tel taux d'alerte garantit que plus personne ne lit les alertes.

        La calibration remplace la question « ce PSI dépasse-t-il une constante venue
        d'un autre métier ? » par la seule qui ait un sens : **« cette fenêtre est-elle
        plus atypique que 99 % de celles sur lesquelles le modèle a été entraîné ? »**

        Les seuils sont calculés PAR FEATURE, et c'est indispensable : une feature de
        calendrier comme `month_cos` est quasi constante sur 250 barres (dix jours) alors
        que la référence couvre des mois — son PSI in-sample dépasse 5, et elle serait
        signalée en permanence sous un seuil unique. Une feature stationnaire, elle,
        mérite un seuil bien plus serré que 0.25.

        `floor` empêche qu'une feature exceptionnellement stable hérite d'un seuil si bas
        que le bruit d'échantillonnage le franchisse.
        """
        step = int(step or max(window // 4, 1))
        starts = list(range(0, max(len(X) - window + 1, 0), step))
        if len(starts) < min_windows:
            log.warning(
                "Calibration impossible : %d fenêtres disponibles pour %d requises. "
                "Les seuils industriels (%.2f / %.2f) restent en vigueur — ils sont "
                "connus pour être trop sensibles sur des séries financières.",
                len(starts), min_windows, PSI_STABLE, PSI_SHIFTED)
            return self

        per_feature: Dict[str, List[float]] = {n: [] for n in self.feature_names}
        for s0 in starts:
            for f in self.compare(X.iloc[s0:s0 + window], autocorr_correction=False).features:
                if np.isfinite(f.psi):
                    per_feature[f.name].append(f.psi)

        self.psi_warn_by_feature = {
            n: float(max(np.quantile(v, warn_q), floor)) for n, v in per_feature.items() if v}
        self.psi_critical_by_feature = {
            n: float(max(np.quantile(v, crit_q), floor)) for n, v in per_feature.items() if v}
        self.calibration_window = int(window)
        self.n_calibration_windows = len(starts)

        crit = np.array(list(self.psi_critical_by_feature.values()))
        log.info("Seuils de dérive calibrés sur %d fenêtres de %d barres : "
                 "seuil critique médian %.3f (min %.3f, max %.3f)",
                 len(starts), window, float(np.median(crit)), float(crit.min()),
                 float(crit.max()))
        return self

    # -- comparaison --------------------------------------------------------------------
    def compare(self, live: pd.DataFrame, autocorr_correction: bool = True) -> DriftReport:
        """Compare une fenêtre de production à la référence, feature par feature."""
        results: List[FeatureDrift] = []
        n_live = int(len(live))
        for name in self.feature_names:
            if name not in live.columns:
                continue
            v = live[name].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            cut = np.asarray(self.edges[name], dtype=float)
            ref_counts = np.asarray(self.ref_counts[name], dtype=float)
            warn_t = self.psi_warn_by_feature.get(name, PSI_STABLE)
            crit_t = self.psi_critical_by_feature.get(name, PSI_SHIFTED)
            is_cal = name in self.psi_critical_by_feature

            if v.size == 0:
                results.append(FeatureDrift(name, float("nan"), float("nan"), float("nan"),
                                            float("nan"), float("nan"), self.mean[name],
                                            float("nan"), self.std[name], float("nan"),
                                            float("nan"), 0, 0.0,
                                            warn_t, crit_t, is_cal))
                continue

            live_counts = np.bincount(np.searchsorted(cut, v, side="right"),
                                      minlength=cut.size + 1).astype(float)
            p_ref = _smoothed_probabilities(ref_counts)
            p_live = _smoothed_probabilities(live_counts)

            # Kolmogorov-Smirnov à UN échantillon contre la fonction de répartition de
            # référence : celle-ci est connue et figée, ce n'est pas un tirage aléatoire.
            # La taille effective est celle de la fenêtre live, corrigée de son
            # autocorrélation — c'est la seule source d'aléa du test.
            n_eff = effective_sample_size(v) if autocorr_correction else float(v.size)
            ks_d, ks_p = ks_one_sample(v, lambda z, _n=name: self.cdf(_n, z), n_eff)

            sd = self.std[name] if self.std[name] > 1e-12 else float("nan")
            results.append(FeatureDrift(
                name=name,
                psi=population_stability_index(p_ref, p_live),
                kl=kl_divergence(p_ref, p_live),
                js=jensen_shannon_distance(p_ref, p_live),
                ks_stat=ks_d, ks_pvalue=ks_p,
                ref_mean=self.mean[name], live_mean=float(v.mean()),
                ref_std=self.std[name], live_std=float(v.std(ddof=1)) if v.size > 1 else 0.0,
                z_shift=float((v.mean() - self.mean[name]) / sd) if np.isfinite(sd) else float("nan"),
                n_live=int(v.size), n_effective=n_eff,
                psi_warn=warn_t, psi_critical=crit_t, calibrated=is_cal,
            ))
        return DriftReport(features=results, n_live=n_live)

    def _cdf_knots(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Nœuds (x, F(x)) de la fonction de répartition de référence.

        On ne conserve pas les données d'entraînement — volume, et surtout un modèle
        déployé ne doit pas embarquer l'historique sur lequel il a été ajusté. Les bornes
        de cases sont les quantiles k/n_bins de la référence : les relier linéairement
        redonne une répartition qui approche l'originale à la résolution du découpage.
        C'est volontairement approché, et c'est pourquoi le KS n'est ici qu'un signal
        secondaire : le PSI, lui, est calculé sur les comptages exacts.
        """
        cut = np.asarray(self.edges[name], dtype=float)
        counts = np.asarray(self.ref_counts[name], dtype=float)
        total = max(counts.sum(), 1.0)
        cdf_at_cuts = np.cumsum(counts)[:-1] / total          # F aux bornes intérieures
        span = float(cut[-1] - cut[0])
        pad = span * 0.05 if span > 0 else max(abs(float(cut[0])), 1.0) * 0.05 + 1e-9
        knots_x = np.concatenate([[cut[0] - pad], cut, [cut[-1] + pad]])
        knots_p = np.concatenate([[0.0], cdf_at_cuts, [1.0, 1.0]])[: knots_x.size]
        order = np.argsort(knots_x, kind="stable")
        return knots_x[order], np.maximum.accumulate(knots_p[order])

    def cdf(self, name: str, x: np.ndarray | float) -> np.ndarray:
        """Fonction de répartition de référence, évaluée en x (interpolation linéaire)."""
        knots_x, knots_p = self._cdf_knots(name)
        return np.clip(np.interp(np.asarray(x, dtype=float), knots_x, knots_p), 0.0, 1.0)

    # -- persistance --------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ReferenceDistribution":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)


# =======================================================================================
# Surveillance en continu
# =======================================================================================
class DriftMonitor:
    """Tampon glissant de features de production + verdict de dérive à la demande."""

    def __init__(self, reference: ReferenceDistribution, window: int = 250,
                 min_samples: int = 100, psi_warn: float = PSI_STABLE,
                 psi_critical: float = PSI_SHIFTED):
        self.reference = reference
        self.window = int(window)
        self.min_samples = int(min_samples)
        self.psi_warn = float(psi_warn)
        self.psi_critical = float(psi_critical)
        self._rows: List[np.ndarray] = []
        self._names = list(reference.feature_names)
        self.n_seen = 0

    def push(self, row: Sequence[float] | Dict[str, float] | pd.Series) -> None:
        """Enregistre une observation de features (une barre)."""
        if isinstance(row, dict):
            vec = np.array([float(row.get(n, np.nan)) for n in self._names], dtype=float)
        elif isinstance(row, pd.Series):
            vec = np.array([float(row.get(n, np.nan)) for n in self._names], dtype=float)
        else:
            vec = np.asarray(row, dtype=float).ravel()
            if vec.size != len(self._names):
                raise ValueError(
                    f"{vec.size} valeurs reçues pour {len(self._names)} features de référence.")
        self._rows.append(vec)
        self.n_seen += 1
        if len(self._rows) > self.window:
            del self._rows[:-self.window]

    @property
    def ready(self) -> bool:
        return len(self._rows) >= self.min_samples

    def frame(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame(columns=self._names)
        return pd.DataFrame(np.vstack(self._rows), columns=self._names)

    def report(self) -> Optional[DriftReport]:
        """Verdict de dérive, ou None tant que la fenêtre n'est pas assez remplie.

        Rendre None plutôt qu'un rapport bruité est délibéré : une alerte de dérive
        calculée sur 20 barres serait fausse la moitié du temps et détruirait la
        crédibilité du système auprès de ceux qui doivent y réagir.
        """
        if not self.ready:
            return None
        rep = self.reference.compare(self.frame())
        if not self.reference.psi_critical_by_feature:
            # Aucune calibration disponible : on retombe sur les seuils de configuration.
            rep.psi_warn = self.psi_warn
            rep.psi_critical = self.psi_critical
            for f in rep.features:
                f.psi_warn, f.psi_critical = self.psi_warn, self.psi_critical
        return rep

    def reset(self) -> None:
        self._rows.clear()

"""Dimensionnement de position.

Le modèle décide de la DIRECTION ; le dimensionnement décide de la SURVIE. Un edge réel
mal dimensionné mène à la ruine avec probabilité 1 — c'est un résultat mathématique, pas
une opinion : au-delà du Kelly optimal, le taux de croissance à long terme devient négatif
alors même que l'espérance de gain reste positive.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def kelly_fraction(win_rate: float, payoff_ratio: float) -> float:
    """Critère de Kelly discret : f* = p - (1-p)/b.

    p = probabilité de gain, b = ratio gain moyen / perte moyenne.
    Maximise le taux de croissance géométrique du capital.
    """
    p = float(np.clip(win_rate, 0.0, 1.0))
    b = max(float(payoff_ratio), 1e-9)
    return float(p - (1.0 - p) / b)


def continuous_kelly(mean_return: float, variance: float) -> float:
    """Kelly continu : f* = μ / σ².

    Pour des rendements approximativement gaussiens, c'est la fraction de capital qui
    maximise E[log(richesse)].
    """
    return float(mean_return / max(variance, 1e-12))


def fractional_kelly(full_kelly: float, fraction: float = 0.25, cap: float = 1.0) -> float:
    """Kelly fractionnaire — la seule version utilisable en pratique.

    Trois raisons de ne JAMAIS trader le Kelly plein :
      1. μ et σ sont estimés avec erreur ; surestimer μ de 30 % suffit à passer du côté
         destructeur de la courbe de croissance.
      2. Le Kelly plein produit des drawdowns de l'ordre de 50 % — humainement et
         commercialement intenable.
      3. Le demi-Kelly conserve 75 % du taux de croissance pour la moitié de la volatilité.
    """
    return float(np.clip(full_kelly * fraction, -cap, cap))


def vol_target_size(
    target_vol: float, realized_vol: float, max_leverage: float = 1.0, min_vol: float = 1e-6
) -> float:
    """Levier tel que la volatilité de la position atteigne `target_vol`.

    Effet majeur et sous-estimé : le vol targeting rend la distribution des rendements de
    la stratégie beaucoup plus proche d'une gaussienne, ce qui améliore mécaniquement le
    Sharpe et réduit les queues — indépendamment de toute qualité prédictive du signal.
    """
    return float(np.clip(target_vol / max(realized_vol, min_vol), 0.0, max_leverage))


def risk_parity_weights(
    cov: np.ndarray,
    budget: Optional[np.ndarray] = None,
    max_iter: int = 1000,
    tol: float = 1e-12,
) -> np.ndarray:
    """Poids à contribution au risque égale (multi-actifs).

    Résout w_i · (Σw)_i = b_i · σ(w) par descente coordonnée cyclique
    (Griveau-Billion, Richard & Roncalli, 2013). À chaque coordonnée, l'équation devient
    un simple trinôme du second degré dont on prend la racine positive :

        Σ_ii · w_i² + (Σ_{j≠i} w_j Σ_ij) · w_i - b_i · σ(w) = 0

    Cette formulation converge de façon monotone. L'itération multiplicative naïve
    w_i <- b_i / (Σw)_i, souvent citée, possède le bon point fixe mais oscille et peut
    s'arrêter sur des contributions inégales — ce qui donne un faux sentiment de sécurité.
    """
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    if cov.shape != (n, n):
        raise ValueError("cov doit être carrée")
    b = np.ones(n) / n if budget is None else np.asarray(budget, float) / np.sum(budget)

    w = np.ones(n) / n
    for _ in range(max_iter):
        w_old = w.copy()
        sigma = float(np.sqrt(max(w @ cov @ w, 1e-300)))
        for i in range(n):
            a = float(cov[i, i])
            if a <= 1e-300:                      # actif sans risque : hors du budget
                w[i] = 0.0
                continue
            c = float(cov[i] @ w) - a * w[i]     # exposition croisée aux autres actifs
            disc = c * c + 4.0 * a * b[i] * sigma
            w[i] = (-c + np.sqrt(max(disc, 0.0))) / (2.0 * a)
        if np.abs(w - w_old).max() < tol:
            break
    total = w.sum()
    return w / total if total > 0 else np.ones(n) / n


def position_from_signal(
    signal: float,
    realized_vol: float,
    target_vol: float = 0.10,
    kelly_frac: float = 0.25,
    edge_estimate: Optional[float] = None,
    max_position: float = 1.0,
    confidence: float = 1.0,
) -> float:
    """Combine signal directionnel, vol targeting, Kelly fractionnaire et confiance.

    `confidence` ∈ [0,1] doit venir d'une source d'incertitude réelle : probabilité du
    meta-label, désaccord d'ensemble, écart interquantile du critique distributionnel.
    Utiliser une confiance constante revient à ignorer l'information la plus utile
    produite par le modèle.
    """
    base = float(np.clip(signal, -1.0, 1.0))
    lever = vol_target_size(target_vol, realized_vol, max_leverage=max_position)
    size = base * lever * float(np.clip(confidence, 0.0, 1.0))

    if edge_estimate is not None:
        k = fractional_kelly(continuous_kelly(edge_estimate, max(realized_vol, 1e-6) ** 2), kelly_frac)
        size *= float(np.clip(abs(k), 0.0, 1.0))
    return float(np.clip(size, -max_position, max_position))


def lots_from_exposure(
    exposure: float,
    equity: float,
    price: float,
    contract_size: float = 100_000.0,
    lot_step: float = 0.01,
    min_lot: float = 0.01,
    max_lot: float = 100.0,
) -> float:
    """Convertit une exposition en fraction du capital vers un volume MetaTrader (lots).

    Sur EURUSD, 1 lot standard = 100 000 unités de devise de base. Une exposition de 0.5
    sur un compte de 10 000 € correspond donc à 5 000 € de notionnel, soit 0.05 lot.
    """
    if equity <= 0 or price <= 0 or contract_size <= 0:
        return 0.0
    notional = abs(exposure) * equity
    raw_lots = notional / (contract_size * price) if price > 0 else 0.0
    # Arrondi INFÉRIEUR au pas de lot : ne jamais dépasser l'exposition demandée.
    lots = np.floor(raw_lots / lot_step) * lot_step
    if lots < min_lot:
        return 0.0
    return float(np.sign(exposure) * min(lots, max_lot))

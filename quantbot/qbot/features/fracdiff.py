"""Différenciation fractionnaire (López de Prado, ch. 5).

Problème : les prix sont non-stationnaires (un modèle ne peut pas apprendre dessus), mais
les différencier entièrement (rendements) détruit toute la mémoire de la série — or c'est
justement cette mémoire qui porte le signal prédictif.

La différenciation fractionnaire d'ordre d ∈ ]0,1[ trouve le point minimal où la série
devient stationnaire tout en conservant le maximum de mémoire.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def frac_diff_weights(d: float, thresh: float = 1e-4, max_size: int = 10_000) -> np.ndarray:
    """Poids du développement binomial (1-B)^d, tronqués sous `thresh`."""
    w = [1.0]
    k = 1
    while k < max_size:
        w_k = -w[-1] * (d - k + 1.0) / k
        if abs(w_k) < thresh:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1], dtype=float)   # ordre chronologique : w[-1] pondère l'obs courante


def frac_diff_ffd(series: pd.Series | np.ndarray, d: float, thresh: float = 1e-4) -> pd.Series:
    """Différenciation fractionnaire à fenêtre fixe (FFD) — poids constants, donc
    variance homogène dans le temps, contrairement à la version à fenêtre étendue."""
    if isinstance(series, np.ndarray):
        series = pd.Series(series)
    w = frac_diff_weights(d, thresh)
    width = len(w)
    values = series.to_numpy(dtype=float)
    if width > values.shape[0]:
        return pd.Series(np.full(values.shape[0], np.nan), index=series.index)

    # Convolution causale : out[i] = sum_k w[k] * x[i - width + 1 + k]
    windows = np.lib.stride_tricks.sliding_window_view(values, width)
    out = np.full(values.shape[0], np.nan)
    out[width - 1:] = windows @ w
    return pd.Series(out, index=series.index, name=f"ffd_{d:.3f}")


def adf_stat(x: np.ndarray, max_lag: int = 8) -> float:
    """Statistique t de Dickey-Fuller augmentée, implémentée en OLS pur (pas de statsmodels).

    Régression : Δy_t = φ y_{t-1} + Σ γ_i Δy_{t-i} + c + ε_t ; on retourne t(φ).
    Valeurs critiques usuelles (constante, sans tendance) : -3.43 (1%), -2.86 (5%), -2.57 (10%).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.shape[0] < max_lag + 20:
        return np.nan

    dy = np.diff(x)
    n = dy.shape[0] - max_lag
    if n <= max_lag + 2:
        return np.nan

    y = dy[max_lag:]
    cols = [x[max_lag:-1], np.ones(n)]
    for i in range(1, max_lag + 1):
        cols.append(dy[max_lag - i: -i] if i > 0 else dy[max_lag:])
    X = np.column_stack(cols)

    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:  # pragma: no cover
        return np.nan
    resid = y - X @ beta
    dof = max(n - X.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:  # pragma: no cover
        return np.nan
    se = np.sqrt(max(sigma2 * xtx_inv[0, 0], 1e-300))
    return float(beta[0] / se)


def find_min_ffd(
    series: pd.Series,
    d_grid: Optional[np.ndarray] = None,
    thresh: float = 1e-4,
    adf_target: float = -2.86,
) -> Tuple[float, float, float]:
    """Cherche le plus petit d rendant la série stationnaire au seuil de 5 %.

    Retourne (d, statistique ADF, corrélation avec la série d'origine).
    La corrélation quantifie la mémoire conservée : on veut d minimal ET corrélation élevée.
    """
    d_grid = d_grid if d_grid is not None else np.arange(0.0, 1.01, 0.05)
    base = np.log(series.to_numpy(dtype=float))
    base_s = pd.Series(base, index=series.index)

    best = (1.0, np.nan, 0.0)
    for d in d_grid:
        ffd = frac_diff_ffd(base_s, float(d), thresh).dropna()
        if ffd.shape[0] < 100:
            continue
        stat = adf_stat(ffd.to_numpy())
        if not np.isfinite(stat):
            continue
        aligned = base_s.reindex(ffd.index)
        corr = float(np.corrcoef(aligned.to_numpy(), ffd.to_numpy())[0, 1]) if ffd.std() > 0 else 0.0
        if stat < adf_target:
            return float(d), float(stat), corr
        best = (float(d), float(stat), corr)
    return best

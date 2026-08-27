"""Facteurs d'annualisation déduits de l'index temporel plutôt que codés en dur."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Nombre de barres par an pour les timeframes usuels (marché FX ~ 24h x 5j, 252 jours de bourse).
_FX_HOURS_PER_YEAR = 24 * 5 * 52.0
_EQUITY_DAYS_PER_YEAR = 252.0


_TIMEFRAME_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H2": 120,
                      "H4": 240, "H8": 480, "D1": 1440, "W1": 10080, "MN1": 43200}


def bars_per_year_for_timeframe(timeframe: str, asset_class: str = "fx") -> float:
    """Barres par an déduites d'un libellé de timeframe MT5 ("H1", "M15"…).

    Utile quand aucune série n'est disponible — au démarrage d'un serveur, par exemple,
    où la configuration est connue mais pas encore les données. Un timeframe inconnu
    retombe sur la convention actions (252), et c'est délibérément visible : un facteur
    d'annualisation faux se propage à TOUTES les métriques annualisées sans jamais
    déclencher d'erreur.
    """
    minutes = _TIMEFRAME_MINUTES.get(str(timeframe).upper())
    if minutes is None:
        return _EQUITY_DAYS_PER_YEAR
    if minutes >= 1440:
        return _EQUITY_DAYS_PER_YEAR * (1440.0 / minutes)
    hours_per_year = _FX_HOURS_PER_YEAR if asset_class == "fx" else _EQUITY_DAYS_PER_YEAR * 6.5
    return max(hours_per_year * 60.0 / minutes, 1.0)


def infer_bars_per_year(index: pd.DatetimeIndex, asset_class: str = "fx") -> float:
    """Estime le nombre de barres par an à partir de l'espacement médian de l'index."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return _EQUITY_DAYS_PER_YEAR
    # `index.view("int64")` renverrait les entiers de la résolution SOUS-JACENTE, qui n'est
    # pas garantie en nanosecondes : depuis pandas 3, `date_range` et `read_csv` produisent
    # par défaut du datetime64[us]. Diviser par 1e9 donnait alors un pas mille fois trop
    # petit, donc un nombre de barres par an mille fois trop grand — et un Sharpe annualisé
    # multiplié par √1000 ≈ 31.6, sans la moindre erreur visible. `total_seconds()` connaît
    # l'unité de l'index et reste juste quelle qu'elle soit.
    deltas = (index[1:] - index[:-1]).total_seconds().to_numpy(dtype=float)
    deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
    if deltas.size == 0:
        return _EQUITY_DAYS_PER_YEAR
    step_s = float(np.median(deltas))
    if step_s >= 86400 * 0.9:  # barres journalières ou plus
        return _EQUITY_DAYS_PER_YEAR * (86400.0 / step_s)
    hours = step_s / 3600.0
    seconds_per_year = (_FX_HOURS_PER_YEAR if asset_class == "fx" else _EQUITY_DAYS_PER_YEAR * 6.5) * 3600.0
    return max(seconds_per_year / step_s, 1.0) if hours > 0 else _EQUITY_DAYS_PER_YEAR


def ann_factor(bars_per_year: float) -> float:
    """Facteur multiplicatif d'annualisation d'un Sharpe calculé par barre."""
    return float(np.sqrt(max(bars_per_year, 1.0)))

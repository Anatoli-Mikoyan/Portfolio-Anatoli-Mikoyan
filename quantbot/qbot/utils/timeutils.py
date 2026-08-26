"""Facteurs d'annualisation déduits de l'index temporel plutôt que codés en dur."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Nombre de barres par an pour les timeframes usuels (marché FX ~ 24h x 5j, 252 jours de bourse).
_FX_HOURS_PER_YEAR = 24 * 5 * 52.0
_EQUITY_DAYS_PER_YEAR = 252.0


def infer_bars_per_year(index: pd.DatetimeIndex, asset_class: str = "fx") -> float:
    """Estime le nombre de barres par an à partir de l'espacement médian de l'index."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return _EQUITY_DAYS_PER_YEAR
    deltas = np.diff(index.view("int64")) / 1e9  # secondes
    deltas = deltas[deltas > 0]
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

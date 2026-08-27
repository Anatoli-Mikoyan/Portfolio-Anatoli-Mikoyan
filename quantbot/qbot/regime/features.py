"""Features destinées à la détection de régime — et pourquoi elles diffèrent des autres.

Les features du modèle prédictif sont normalisées par un z-score glissant : c'est ce qui
les rend stationnaires et comparables entre époques. Pour la détection de régime, cette
même normalisation est **destructrice**, et le piège est subtil.

Un régime dure typiquement plusieurs centaines de barres. Un z-score glissant sur 300
barres mesure « à quel point la volatilité actuelle est inhabituelle par rapport aux 300
dernières barres » — donc, à l'intérieur d'un régime long, il converge vers zéro quel que
soit le NIVEAU absolu de volatilité de ce régime. La normalisation supprime exactement
l'information que le détecteur cherche.

Mesure faite sur ce dépôt, marché synthétique à deux régimes de volatilité 2 % et 40 %
annualisés (donc trivialement séparables), régimes durant en moyenne 833 barres :

    features z-scorées sur 300 barres  ->  ARI = 0.011  (le hasard)
    features de niveau                 ->  ARI = 0.947

Règle qui en découle : **ne jamais normaliser les features de régime sur une fenêtre plus
courte que la durée typique d'un régime.** Les features produites ici sont soit des
niveaux (log-volatilité), soit des quantités déjà bornées par construction (percentiles
sur fenêtre longue, R² de tendance, exposant de Hurst).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..features.regime import rolling_autocorr, trend_strength, variance_ratio
from ..features.technical import adx, atr, hurst_exponent, realized_vol
from ..utils.logging import get_logger

log = get_logger("regime.features")

DEFAULT_VOL_WINDOWS = (20, 60, 240)


def build_regime_matrix(
    prices: pd.DataFrame,
    vol_windows: Sequence[int] = DEFAULT_VOL_WINDOWS,
    pctile_window: int = 1000,
    trend_window: int = 120,
    dropna: bool = True,
) -> pd.DataFrame:
    """Matrice de features adaptée à la détection de régime.

    `pctile_window` doit dépasser nettement la durée attendue d'un régime — sinon le
    percentile subit le même effacement que le z-score glissant.
    """
    close, high, low = prices["close"], prices["high"], prices["low"]
    out = pd.DataFrame(index=prices.index)

    # --- NIVEAUX de volatilité, en logarithme -------------------------------------------
    # Le log rend l'échelle additive (un doublement de vol vaut le même écart partout)
    # sans détruire le niveau, contrairement à une normalisation glissante courte.
    for w in vol_windows:
        vol = realized_vol(close, w).replace(0.0, np.nan)
        out[f"log_vol_{w}"] = np.log(vol)

    # Rapport de volatilités court/long : détecte les CHANGEMENTS de régime, pas le niveau.
    short, long_ = min(vol_windows), max(vol_windows)
    out["vol_ratio_short_long"] = (realized_vol(close, short)
                                   / realized_vol(close, long_).replace(0.0, np.nan))

    # --- quantités déjà bornées par construction -----------------------------------------
    vol_ref = realized_vol(close, min(vol_windows))
    out["vol_pctile"] = vol_ref.rolling(pctile_window, min_periods=pctile_window // 4).rank(pct=True)
    out["trend_strength"] = trend_strength(close, trend_window)
    out["hurst"] = hurst_exponent(close, window=128) - 0.5
    out["ac_1"] = rolling_autocorr(close, 1, 120)
    out["vr_5"] = variance_ratio(close, q=5, window=252) - 1.0

    adx_df = adx(high, low, close, 14)
    out["adx"] = adx_df["adx"] / 100.0

    # Position dans le drawdown : proxy simple mais robuste du régime risk-on / risk-off.
    roll_max = close.rolling(pctile_window, min_periods=2).max()
    out["dd_from_high"] = close / roll_max - 1.0

    # Amplitude intra-barre relative, en logarithme : niveau, pas z-score.
    out["log_range"] = np.log((atr(high, low, close, 14) / close).replace(0.0, np.nan))

    out = out.replace([np.inf, -np.inf], np.nan)
    if dropna:
        out = out.dropna()
    log.info("Matrice de régime : %d lignes x %d colonnes (features de NIVEAU)", *out.shape)
    return out


def looks_rolling_normalized(X: pd.DataFrame, window: int = 300, tol: float = 0.25) -> bool:
    """Détecte une matrice qui a subi un z-score glissant court.

    Signature : moyenne glissante proche de 0 et écart-type glissant proche de 1 sur
    presque toutes les colonnes. Sert de garde-fou — passer de telles features à un
    détecteur de régime produit silencieusement des états aléatoires.
    """
    if len(X) < window * 2:
        return False
    numeric = X.select_dtypes(include=[np.number])
    if numeric.empty:
        return False
    mu = numeric.rolling(window).mean().abs().mean()
    sd = numeric.rolling(window).std(ddof=0).mean()
    centered = (mu < tol).mean()
    unit = ((sd - 1.0).abs() < tol).mean()
    return bool(centered > 0.7 and unit > 0.7)

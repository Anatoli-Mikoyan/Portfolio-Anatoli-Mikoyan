"""Features de microstructure de marché (López de Prado, ch. 19).

Ces mesures estiment ce qu'on ne voit pas directement dans l'OHLCV : le coût réel de
liquidité, l'asymétrie d'information et la pression des ordres. Elles apportent une
information orthogonale aux indicateurs de prix classiques — c'est précisément ce qu'on
cherche quand on empile des features (sinon on ne fait qu'ajouter du bruit corrélé).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def roll_spread(close: pd.Series, window: int = 50) -> pd.Series:
    """Estimateur de Roll (1984) du spread effectif : 2*sqrt(-cov(Δp_t, Δp_{t-1})).

    Repose sur le bounce bid-ask : les allers-retours entre bid et ask créent une
    autocovariance négative des variations de prix, dont l'amplitude mesure le spread.
    """
    dp = close.diff()
    cov = dp.rolling(window).cov(dp.shift(1))
    return 2.0 * np.sqrt(np.maximum(-cov, 0.0)) / close


def corwin_schultz_spread(high: pd.Series, low: pd.Series, window: int = 1) -> pd.Series:
    """Estimateur high-low de Corwin & Schultz (2012).

    Idée : sur deux barres consécutives, la variance est proportionnelle à la durée
    tandis que le spread ne l'est pas — on peut donc les séparer analytiquement.
    """
    hl = np.log(high / low) ** 2
    beta = (hl + hl.shift(1)).rolling(window).mean() if window > 1 else (hl + hl.shift(1))
    h2 = high.rolling(2).max()
    l2 = low.rolling(2).min()
    gamma = np.log(h2 / l2) ** 2

    den = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / den - np.sqrt(gamma / den)
    alpha = alpha.clip(lower=0.0)
    return 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))


def amihud_illiquidity(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """Ratio d'Amihud (2002) : |rendement| / volume en notionnel.

    Mesure combien le prix bouge par unité de flux — c'est l'impact de marché tel qu'il
    sera réellement subi à l'exécution."""
    ret = np.log(close).diff().abs()
    dollar_vol = (close * volume).replace(0.0, np.nan)
    return (ret / dollar_vol).rolling(window).mean() * 1e6


def kyle_lambda(close: pd.Series, volume: pd.Series, window: int = 50) -> pd.Series:
    """Lambda de Kyle : pente de Δp régressé sur le volume signé (impact linéaire).

    Estimée en forme fermée par régression glissante sans constante : λ = Σxy / Σx².
    """
    dp = close.diff()
    signed_vol = np.sign(dp).fillna(0.0) * volume
    num = (dp * signed_vol).rolling(window).sum()
    den = (signed_vol ** 2).rolling(window).sum().replace(0.0, np.nan)
    return (num / den) * 1e6


def vpin(close: pd.Series, volume: pd.Series, window: int = 50) -> pd.Series:
    """VPIN simplifié (Easley, López de Prado & O'Hara) : déséquilibre de volume normalisé.

    Un VPIN élevé signale un flux toxique / informé — historiquement un précurseur des
    épisodes de flash-crash et un bon filtre pour couper l'exposition.
    """
    dp = close.diff()
    sign = np.sign(dp).replace(0.0, np.nan).ffill().fillna(1.0)
    buy_vol = volume.where(sign > 0, 0.0)
    sell_vol = volume.where(sign < 0, 0.0)
    imbalance = (buy_vol - sell_vol).rolling(window).sum().abs()
    total = volume.rolling(window).sum().replace(0.0, np.nan)
    return imbalance / total


def parkinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """Volatilité de Parkinson : ~5x plus efficace que l'écart-type close-to-close."""
    hl2 = np.log(high / low) ** 2
    return np.sqrt(hl2.rolling(window).mean() / (4.0 * np.log(2.0)))


def garman_klass_vol(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series, window: int = 20) -> pd.Series:
    term = 0.5 * np.log(h / l) ** 2 - (2.0 * np.log(2.0) - 1.0) * np.log(c / o) ** 2
    return np.sqrt(term.rolling(window).mean().clip(lower=0.0))


def rogers_satchell_vol(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series, window: int = 20) -> pd.Series:
    """Estimateur de Rogers-Satchell : insensible au drift, contrairement à Garman-Klass."""
    term = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    return np.sqrt(term.rolling(window).mean().clip(lower=0.0))


def yang_zhang_vol(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series, window: int = 20) -> pd.Series:
    """Yang-Zhang : combine overnight, open-to-close et Rogers-Satchell.
    C'est l'estimateur de volatilité OHLC de variance minimale."""
    log_oc = np.log(o / c.shift(1))
    log_co = np.log(c / o)
    sigma_o = log_oc.rolling(window).var(ddof=0)
    sigma_c = log_co.rolling(window).var(ddof=0)
    sigma_rs = (rogers_satchell_vol(o, h, l, c, window) ** 2)
    k = 0.34 / (1.34 + (window + 1.0) / (window - 1.0))
    return np.sqrt((sigma_o + k * sigma_c + (1.0 - k) * sigma_rs).clip(lower=0.0))


def build_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    out = pd.DataFrame(index=df.index)
    out["roll_spread"] = roll_spread(c) * 1e4
    out["cs_spread"] = corwin_schultz_spread(h, l) * 1e4
    out["amihud"] = np.log1p(amihud_illiquidity(c, v).clip(lower=0.0))
    out["kyle_lambda"] = kyle_lambda(c, v)
    out["vpin"] = vpin(c, v)
    out["park_vol"] = parkinson_vol(h, l)
    out["gk_vol"] = garman_klass_vol(o, h, l, c)
    out["yz_vol"] = yang_zhang_vol(o, h, l, c)
    # Ratio vol intra-barre / vol close-to-close : > 1 => bruit de microstructure dominant
    cc_vol = np.log(c).diff().rolling(20).std(ddof=0).replace(0.0, np.nan)
    out["vol_ratio_park_cc"] = out["park_vol"] / cc_vol
    if "spread" in df.columns:
        out["spread_rel"] = (df["spread"] / c) * 1e4
        # Un spread relatif CONSTANT a un z-score nul par définition : il est exactement
        # sur sa moyenne. Le `replace(0.0, np.nan)` d'origine produisait au contraire des
        # NaN sur toute la colonne, et le `dropna` du pipeline vidait alors la matrice
        # entière — soixante-quatre features correctes emportées par une seule.
        #
        # Le cas n'est pas théorique et il était masqué : les données d'entraînement
        # passent par un CSV, dont l'arrondi donnait à ce spread constant un écart-type
        # de 1,8e-17. Non nul, donc jamais remplacé, donc divisé — et `spread_z` ne
        # valait alors que du bruit de virgule flottante amplifié d'un facteur 1e17.
        # En service, le spread reconstruit est exactement constant, l'écart-type est
        # exactement nul, et la colonne mourait pour de bon.
        moyenne = out["spread_rel"].rolling(200).mean()
        ecart = out["spread_rel"].rolling(200).std(ddof=0)
        z = (out["spread_rel"] - moyenne) / ecart.mask(ecart <= 1e-12)
        out["spread_z"] = z.mask((ecart <= 1e-12) & moyenne.notna(), 0.0)
    return out

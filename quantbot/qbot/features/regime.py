"""Détection de régime de marché.

Un agent RL entraîné sur un marché en tendance et déployé sur un marché en range
échouera — pas parce que le modèle est mauvais, mais parce que la distribution a changé.
Donner le régime en entrée permet à l'agent de conditionner sa politique dessus au lieu
de moyenner sur des dynamiques incompatibles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def variance_ratio(close: pd.Series, q: int = 5, window: int = 252) -> pd.Series:
    """Test du ratio de variance de Lo & MacKinlay.

    VR(q) = Var(r_q) / (q * Var(r_1)).
    VR > 1 => tendance persistante ; VR < 1 => retour à la moyenne ; VR ≈ 1 => marche aléatoire.
    """
    r1 = np.log(close).diff()
    rq = np.log(close).diff(q)
    v1 = r1.rolling(window).var(ddof=0)
    vq = rq.rolling(window).var(ddof=0)
    return vq / (q * v1).replace(0.0, np.nan)


def rolling_autocorr(close: pd.Series, lag: int = 1, window: int = 100) -> pd.Series:
    r = np.log(close).diff()
    return r.rolling(window).corr(r.shift(lag))


def vol_percentile(close: pd.Series, vol_window: int = 20, rank_window: int = 500) -> pd.Series:
    """Percentile de la volatilité courante dans sa propre distribution passée.

    Feature bornée [0,1] et stationnaire par construction : bien plus exploitable par un
    réseau qu'une volatilité brute dont l'échelle dérive dans le temps.
    """
    vol = np.log(close).diff().rolling(vol_window).std(ddof=0)
    return vol.rolling(rank_window).rank(pct=True)


def trend_strength(close: pd.Series, window: int = 60) -> pd.Series:
    """R² de la régression linéaire du log-prix sur le temps, signé par la pente.

    Proche de +1 : tendance haussière propre ; -1 : baissière propre ; 0 : pas de tendance.
    """
    logp = np.log(close.to_numpy(dtype=float))
    n = logp.shape[0]
    out = np.full(n, np.nan)
    if n <= window:
        return pd.Series(out, index=close.index, name="trend_strength")

    x = np.arange(window, dtype=float)
    x_c = x - x.mean()
    sxx = float(x_c @ x_c)
    windows = np.lib.stride_tricks.sliding_window_view(logp, window)
    y_mean = windows.mean(axis=1, keepdims=True)
    y_c = windows - y_mean
    sxy = y_c @ x_c
    syy = np.einsum("ij,ij->i", y_c, y_c)
    slope = sxy / sxx
    r2 = np.where(syy > 1e-18, (sxy ** 2) / (sxx * np.maximum(syy, 1e-18)), 0.0)
    out[window - 1:] = np.sign(slope) * r2
    return pd.Series(out, index=close.index, name="trend_strength")


def plugin_entropy(close: pd.Series, window: int = 250, word_len: int = 4) -> pd.Series:
    """Entropie plug-in de la séquence binarisée des rendements (López de Prado, ch. 18).

    On encode le signe des rendements en bits, puis on estime l'entropie de Shannon sur
    les mots de `word_len` bits. Une marche aléatoire donne 1.0 (les 2^k mots sont
    équiprobables) ; toute structure sérielle exploitable — momentum, mean-reversion,
    saisonnalité intraday — fait chuter la valeur.

    Note d'implémentation : discrétiser par quantiles de la MÊME fenêtre donnerait un
    histogramme uniforme par construction, donc une entropie constante égale à 1 — une
    feature morte. Le codage par signe évite complètement ce piège.
    """
    r = np.log(close).diff().to_numpy(dtype=float)
    bits = (r > 0).astype(np.int64)
    n = r.shape[0]
    out = np.full(n, np.nan)
    n_words = 2 ** word_len
    max_ent = np.log(n_words)
    powers = 2 ** np.arange(word_len - 1, -1, -1)

    if n <= window + word_len:
        return pd.Series(out, index=close.index, name="entropy")

    # Encodage vectorisé de tous les mots glissants de longueur `word_len`.
    words = np.lib.stride_tricks.sliding_window_view(bits, word_len) @ powers
    n_valid = words.shape[0]
    for i in range(window, n_valid):
        counts = np.bincount(words[i - window: i], minlength=n_words).astype(float)
        p = counts / counts.sum()
        p = p[p > 0]
        out[i + word_len - 1] = float(-(p * np.log(p)).sum() / max_ent)
    return pd.Series(out, index=close.index, name="entropy")


# Alias rétro-compatible
shannon_entropy = plugin_entropy


def drawdown_state(close: pd.Series, window: int = 250) -> pd.DataFrame:
    """Position relative au plus-haut glissant : proxy simple mais puissant du régime
    « risk-on / risk-off » (les corrélations et les vols changent en drawdown)."""
    roll_max = close.rolling(window, min_periods=2).max()
    dd = close / roll_max - 1.0
    roll_min = close.rolling(window, min_periods=2).min()
    return pd.DataFrame({
        "dd_from_high": dd,
        "pos_in_range": (close - roll_min) / (roll_max - roll_min).replace(0.0, np.nan),
    })


def build_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    from .technical import hurst_exponent

    c = df["close"]
    out = pd.DataFrame(index=df.index)
    out["vr_5"] = variance_ratio(c, q=5, window=252) - 1.0
    out["vr_20"] = variance_ratio(c, q=20, window=252) - 1.0
    out["ac_1"] = rolling_autocorr(c, 1, 100)
    out["ac_5"] = rolling_autocorr(c, 5, 100)
    out["vol_pctile"] = vol_percentile(c)
    out["trend_strength"] = trend_strength(c)
    out["hurst"] = hurst_exponent(c, window=128) - 0.5
    out["entropy"] = plugin_entropy(c)
    return pd.concat([out, drawdown_state(c)], axis=1)


def build_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Encodage cyclique du temps (sin/cos) : évite la discontinuité artificielle
    entre 23h et 0h que produirait un encodage linéaire."""
    out = pd.DataFrame(index=index)
    hour = index.hour + index.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * index.dayofweek / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * index.dayofweek / 7.0)
    out["month_sin"] = np.sin(2 * np.pi * (index.month - 1) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * (index.month - 1) / 12.0)
    # Sessions FX (heures UTC) : la dynamique de liquidité y est radicalement différente
    out["sess_asia"] = ((hour >= 0) & (hour < 8)).astype(float)
    out["sess_london"] = ((hour >= 7) & (hour < 16)).astype(float)
    out["sess_ny"] = ((hour >= 12) & (hour < 21)).astype(float)
    out["sess_overlap"] = ((hour >= 12) & (hour < 16)).astype(float)
    return out

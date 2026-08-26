"""Indicateurs techniques, implémentés en numpy/pandas purs (aucune dépendance TA-Lib).

Règle absolue respectée partout : **causalité**. Chaque valeur à l'instant t n'utilise que
des informations disponibles à t. Aucun `center=True`, aucun `shift(-k)`, aucun fit global.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------------------
# Blocs élémentaires
# ---------------------------------------------------------------------------------------
def log_returns(close: pd.Series, periods: int = 1) -> pd.Series:
    return np.log(close).diff(periods)


def realized_vol(close: pd.Series, window: int, annualize: float = 1.0) -> pd.Series:
    return log_returns(close).rolling(window).std(ddof=0) * np.sqrt(annualize)


def ewma(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range en lissage de Wilder (équivalent EMA de span 2n-1)."""
    return true_range(high, low, close).ewm(alpha=1.0 / window, adjust=False).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / window, adjust=False).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ewma(close, fast) - ewma(close, slow)
    sig = ewma(line, signal)
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def bollinger(close: pd.Series, window: int = 20, k: float = 2.0) -> pd.DataFrame:
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std(ddof=0)
    upper, lower = ma + k * sd, ma - k * sd
    width = (upper - lower) / ma.replace(0.0, np.nan)
    # %B : position normalisée dans la bande, indicateur borné donc bien conditionné
    pct_b = (close - lower) / (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame({"bb_width": width, "bb_pct": pct_b})


def donchian(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.DataFrame:
    """Canal de Donchian décalé d'une barre : le plus haut des N barres PRÉCÉDENTES.

    Sans le `shift(1)`, la barre courante participe à son propre extrême, ce qui crée un
    look-ahead subtil et très courant (le fameux breakout « toujours gagnant » en backtest).
    """
    hh = high.rolling(window).max().shift(1)
    ll = low.rolling(window).min().shift(1)
    rng = (hh - ll).replace(0.0, np.nan)
    return pd.DataFrame({
        "dc_pos": (close - ll) / rng,
        "dc_break_up": (close > hh).astype(float),
        "dc_break_dn": (close < ll).astype(float),
    })


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.DataFrame:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(high, low, close).ewm(alpha=1.0 / window, adjust=False).mean()
    tr_safe = tr.replace(0.0, np.nan)
    plus_di = 100.0 * pd.Series(plus_dm, index=high.index).ewm(alpha=1.0 / window, adjust=False).mean() / tr_safe
    minus_di = 100.0 * pd.Series(minus_dm, index=high.index).ewm(alpha=1.0 / window, adjust=False).mean() / tr_safe
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return pd.DataFrame({
        "adx": dx.ewm(alpha=1.0 / window, adjust=False).mean(),
        "di_diff": (plus_di - minus_di) / 100.0,
    })


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14, smooth: int = 3) -> pd.Series:
    ll = low.rolling(window).min()
    hh = high.rolling(window).max()
    k = 100.0 * (close - ll) / (hh - ll).replace(0.0, np.nan)
    return k.rolling(smooth).mean()


def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    tp = (high + low + close) / 3.0
    ma = tp.rolling(window).mean()
    md = (tp - ma).abs().rolling(window).mean()
    return (tp - ma) / (0.015 * md.replace(0.0, np.nan))


def zscore(series: pd.Series, window: int) -> pd.Series:
    mu = series.rolling(window).mean()
    sd = series.rolling(window).std(ddof=0)
    return (series - mu) / sd.replace(0.0, np.nan)


def hurst_exponent(close: pd.Series, window: int = 128, lags: tuple[int, ...] = (2, 4, 8, 16, 32)) -> pd.Series:
    """Exposant de Hurst glissant par la méthode des moments d'ordre 2.

    H > 0.5 => série persistante (le momentum a du sens).
    H < 0.5 => série anti-persistante (le mean-reversion a du sens).
    H ≈ 0.5 => marche aléatoire (ni l'un ni l'autre : ne pas trader).
    """
    logp = np.log(close.to_numpy(dtype=float))
    n = logp.shape[0]
    out = np.full(n, np.nan)
    max_lag = max(lags)
    if window <= max_lag * 2:
        return pd.Series(out, index=close.index, name="hurst")

    log_lags = np.log(np.asarray(lags, dtype=float))
    for i in range(window, n):
        seg = logp[i - window: i]
        taus = []
        for lag in lags:
            diff = seg[lag:] - seg[:-lag]
            taus.append(np.sqrt(np.mean(diff * diff)) + 1e-12)
        slope = np.polyfit(log_lags, np.log(np.asarray(taus)), 1)[0]
        out[i] = slope
    return pd.Series(out, index=close.index, name="hurst")


def build_technical_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Assemble le bloc technique complet à partir de la configuration."""
    o, h, l, c, v = (df["open"], df["high"], df["low"], df["close"], df["volume"])
    feats: dict[str, pd.Series] = {}

    for w in cfg.returns_windows:
        feats[f"ret_{w}"] = log_returns(c, w)
    for w in cfg.vol_windows:
        feats[f"vol_{w}"] = realized_vol(c, w)
        feats[f"ret_over_vol_{w}"] = log_returns(c, w) / (realized_vol(c, w) * np.sqrt(w)).replace(0.0, np.nan)
    for w in cfg.rsi_windows:
        feats[f"rsi_{w}"] = rsi(c, w) / 100.0 - 0.5
    for w in cfg.ema_windows:
        feats[f"ema_dist_{w}"] = (c / ewma(c, w) - 1.0)

    a = atr(h, l, c, cfg.atr_window)
    feats["atr_rel"] = a / c
    feats["range_rel"] = (h - l) / c
    feats["body_rel"] = (c - o) / c
    feats["upper_wick"] = (h - np.maximum(o, c)) / c
    feats["lower_wick"] = (np.minimum(o, c) - l) / c
    feats["gap"] = (o / c.shift(1) - 1.0)

    out = pd.DataFrame(feats, index=df.index)
    out = pd.concat([out, macd(c), bollinger(c, cfg.bb_window),
                     donchian(h, l, c, cfg.donchian_window), adx(h, l, c, cfg.adx_window)], axis=1)
    out["stoch"] = stochastic(h, l, c) / 100.0 - 0.5
    out["cci"] = cci(h, l, c) / 100.0
    out["vol_z"] = zscore(np.log1p(v), 60)
    out["vol_ratio"] = v / v.rolling(60).mean().replace(0.0, np.nan)
    return out

"""Méthode de la triple barrière (López de Prado, ch. 3).

Le labeling naïf « le prix monte-t-il dans N barres ? » est incompatible avec la façon
dont on trade réellement : en pratique on sort sur un take-profit, un stop-loss OU une
expiration, la première atteinte l'emportant. Labelliser autrement crée un décalage
structurel entre ce que le modèle apprend et ce que la stratégie exécute.

Les barrières sont dimensionnées en unités de volatilité et non en pips fixes : un stop de
20 pips n'a pas le même sens à 4 % et à 15 % de volatilité annualisée.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

log = get_logger("labeling.triple_barrier")


def get_vol_target(close: pd.Series, span: int = 100, lookback: int = 1) -> pd.Series:
    """Volatilité EWMA des rendements — sert d'unité de mesure aux barrières."""
    ret = close / close.shift(lookback) - 1.0
    return ret.ewm(span=span).std()


def get_vertical_barriers(index: pd.DatetimeIndex, t_events: pd.DatetimeIndex, n_bars: int) -> pd.Series:
    """Barrière temporelle : l'horodatage situé `n_bars` barres après chaque événement.

    Les événements dont la barrière dépasserait la fin de l'échantillon reçoivent NaT :
    les tronquer à la dernière barre créerait des labels calculés sur un horizon plus
    court que prévu, donc systématiquement biaisés vers zéro.
    """
    positions = index.searchsorted(t_events)
    end_pos = positions + n_bars
    valid = end_pos < len(index)
    values = np.where(valid, index.to_numpy()[np.minimum(end_pos, len(index) - 1)], np.datetime64("NaT"))
    return pd.Series(pd.DatetimeIndex(values, tz=index.tz), index=t_events, name="t1")


def apply_triple_barrier(
    close: pd.Series,
    events: pd.DataFrame,
    pt_sl: Tuple[float, float],
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Retourne l'horodatage de premier contact de chaque barrière.

    Si `high`/`low` sont fournis, on teste le contact en INTRA-BARRE (réaliste : un stop
    est touché au plus bas de la barre, pas à sa clôture). Se limiter aux clôtures
    sous-estime systématiquement la fréquence des stops et flatte le backtest.

    Implémentation en indexation positionnelle numpy : le slicing pandas par label dans
    une boucle est à la fois lent et source d'erreurs de dtype/timezone.
    """
    idx = close.index
    c = close.to_numpy(dtype=float)
    hi = high.to_numpy(dtype=float) if high is not None else None
    lo = low.to_numpy(dtype=float) if low is not None else None

    pt_mult, sl_mult = pt_sl
    ev = events.dropna(subset=["t1"])
    starts = idx.searchsorted(ev.index)
    ends = idx.searchsorted(ev["t1"].to_numpy())
    sides = (ev["side"].to_numpy(dtype=float) if "side" in ev.columns
             else np.ones(len(ev), dtype=float))
    trgts = ev["trgt"].to_numpy(dtype=float)

    n = len(idx)
    pt_pos = np.full(len(ev), -1, dtype=np.int64)
    sl_pos = np.full(len(ev), -1, dtype=np.int64)

    for k in range(len(ev)):
        i0, i1 = int(starts[k]), int(min(ends[k], n - 1))
        trgt, side = float(trgts[k]), float(sides[k])
        if i1 <= i0 or not np.isfinite(trgt) or trgt <= 0:
            continue

        entry = c[i0]
        sl_seg = slice(i0 + 1, i1 + 1)   # la barre d'entrée elle-même ne peut pas déclencher
        if hi is not None and lo is not None:
            up = (hi[sl_seg] / entry - 1.0) * side if side > 0 else -(lo[sl_seg] / entry - 1.0)
            dn = (lo[sl_seg] / entry - 1.0) * side if side > 0 else -(hi[sl_seg] / entry - 1.0)
        else:
            up = dn = (c[sl_seg] / entry - 1.0) * side

        if pt_mult > 0:
            hit = np.flatnonzero(up > pt_mult * trgt)
            if hit.size:
                pt_pos[k] = i0 + 1 + int(hit[0])
        if sl_mult > 0:
            hit = np.flatnonzero(dn < -sl_mult * trgt)
            if hit.size:
                sl_pos[k] = i0 + 1 + int(hit[0])

    def _to_ts(positions: np.ndarray) -> pd.Series:
        values = np.where(positions >= 0, idx.to_numpy()[np.maximum(positions, 0)], np.datetime64("NaT"))
        return pd.Series(pd.DatetimeIndex(values, tz=idx.tz), index=ev.index)

    return pd.DataFrame({"t1": ev["t1"], "pt": _to_ts(pt_pos), "sl": _to_ts(sl_pos)})


def get_events(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    pt_sl: Tuple[float, float] = (1.5, 1.0),
    trgt: Optional[pd.Series] = None,
    min_ret: float = 0.0,
    vertical_bars: Optional[int] = 20,
    side: Optional[pd.Series] = None,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Construit la table d'événements : instant de sortie effectif `t1` et cible `trgt`."""
    if trgt is None:
        trgt = get_vol_target(close)
    trgt = trgt.reindex(t_events).dropna()
    trgt = trgt[trgt > min_ret]
    if trgt.empty:
        raise ValueError("Aucun événement ne dépasse le seuil de rendement minimal.")

    t1 = (get_vertical_barriers(close.index, trgt.index, vertical_bars)
          if vertical_bars else pd.Series(pd.NaT, index=trgt.index))

    if side is None:
        side_ = pd.Series(1.0, index=trgt.index)
        pt_sl_ = (pt_sl[0], pt_sl[0])   # barrières symétriques quand le sens est inconnu
    else:
        side_ = side.reindex(trgt.index).dropna()
        trgt = trgt.reindex(side_.index)
        t1 = t1.reindex(side_.index)
        pt_sl_ = pt_sl

    events = pd.DataFrame({"t1": t1, "trgt": trgt, "side": side_}).dropna(subset=["trgt"])
    touches = apply_triple_barrier(close, events, pt_sl_, high=high, low=low)

    # t1 final = première barrière touchée, quelle qu'elle soit
    events["t1"] = touches[[c for c in ("pt", "sl", "t1") if c in touches.columns]].min(axis=1, skipna=True)
    events["t1_pt"] = touches.get("pt", pd.Series(pd.NaT, index=events.index))
    events["t1_sl"] = touches.get("sl", pd.Series(pd.NaT, index=events.index))
    if side is None:
        events = events.drop(columns=["side"])
    events = events.dropna(subset=["t1"])

    # Les événements situés trop près de la fin de l'échantillon n'ont pas d'horizon
    # futur : leur barrière verticale est écrasée sur eux-mêmes. Les garder produirait
    # un rendement identiquement nul et un label arbitraire.
    valid = events["t1"] > events.index
    if (~valid).any():
        log.debug("%d événements sans horizon futur retirés (fin d'échantillon).", int((~valid).sum()))
    return events[valid]


def get_bins(events: pd.DataFrame, close: pd.Series, zero_label_on_vertical: bool = True) -> pd.DataFrame:
    """Transforme les événements en labels.

    - Sans `side` : label ∈ {-1, +1} = direction du rendement réalisé (modèle primaire).
    - Avec `side` : **meta-labeling**, label ∈ {0, 1} = « faut-il suivre ce signal ? ».
      Le meta-labeling est l'astuce centrale du livre : au lieu de demander au ML de
      prédire la direction (tâche à très faible ratio signal/bruit), on lui demande de
      filtrer les signaux d'un modèle primaire — un problème binaire bien plus facile,
      qui améliore surtout la précision et donc le profit factor.
    """
    events_ = events.dropna(subset=["t1"])
    # Indexation positionnelle : `Series.values` sur un index tz-aware renvoie du
    # datetime64 naïf, ce qui casse tout réalignement par label.
    idx = close.index
    c = close.to_numpy(dtype=float)
    i_in = idx.searchsorted(events_.index)
    i_out = np.minimum(idx.searchsorted(events_["t1"].to_numpy()), len(idx) - 1)

    out = pd.DataFrame(index=events_.index)
    out["ret"] = c[i_out] / c[i_in] - 1.0
    out["t1"] = events_["t1"]
    out["bars_held"] = (i_out - i_in).astype(int)

    if "side" in events_.columns:
        out["ret"] = out["ret"] * events_["side"].to_numpy(float)
        out["side"] = events_["side"]
        out["bin"] = (out["ret"] > 0).astype(int)          # meta-label : 1 = prendre le trade
        if zero_label_on_vertical and "t1_pt" in events_.columns:
            # Sortie par expiration sans avoir touché de barrière => pas de conviction : 0.
            expired = events_["t1_pt"].isna() & events_["t1_sl"].isna()
            out.loc[expired & (out["ret"] <= 0), "bin"] = 0
    else:
        out["bin"] = np.sign(out["ret"])
        out.loc[out["ret"] == 0, "bin"] = 0

    return out


def cusum_filter(series: pd.Series, threshold: float | pd.Series) -> pd.DatetimeIndex:
    """Filtre CUSUM symétrique : échantillonne uniquement les mouvements significatifs.

    Sans filtre, on labellise chaque barre — y compris des milliers de barres où il ne se
    passe rien. Cela dilue le signal, gonfle artificiellement la taille de l'échantillon
    et rend les labels massivement redondants. Le CUSUM ne déclenche que lorsque la
    dérive cumulée dépasse un seuil, produisant des événements quasi indépendants.
    """
    t_events, s_pos, s_neg = [], 0.0, 0.0
    diff = np.log(series).diff().dropna()
    thr = threshold if isinstance(threshold, pd.Series) else pd.Series(threshold, index=diff.index)
    thr = thr.reindex(diff.index).ffill()

    for t, value in diff.items():
        h = float(thr.get(t, np.nan))
        if not np.isfinite(h) or h <= 0:
            continue
        s_pos = max(0.0, s_pos + float(value))
        s_neg = min(0.0, s_neg + float(value))
        if s_neg < -h:
            s_neg = 0.0
            t_events.append(t)
        elif s_pos > h:
            s_pos = 0.0
            t_events.append(t)
    return pd.DatetimeIndex(t_events)

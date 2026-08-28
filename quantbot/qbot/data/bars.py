"""Barres pilotées par l'information (López de Prado, *Advances in Financial ML*, ch. 2).

Les barres temporelles échantillonnent le marché à une cadence qui n'a rien à voir avec
l'arrivée de l'information : elles sur-échantillonnent les périodes mortes et sous-échantillonnent
les périodes actives. Résultat : des rendements fortement hétéroscédastiques et non-normaux,
ce qui dégrade tout modèle statistique en aval.

Les barres tick/volume/dollar/imbalance échantillonnent à information constante et
produisent des rendements nettement plus proches de l'i.i.d.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .loader import validate_ohlcv


def _aggregate(df: pd.DataFrame, group_id: np.ndarray) -> pd.DataFrame:
    """Agrège les lignes d'entrée en barres OHLCV selon un identifiant de groupe."""
    work = df.copy()
    work["__g"] = group_id
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    if "spread" in work.columns:
        agg["spread"] = "mean"
    out = work.groupby("__g").agg(agg)
    # L'horodatage de la barre est celui de sa DERNIÈRE observation : la barre n'est
    # exploitable qu'une fois close, sinon on introduit du look-ahead.
    out.index = work.groupby("__g").apply(lambda g: g.index[-1], include_groups=False)
    out.index.name = df.index.name or "time"
    out = out.sort_index()
    validate_ohlcv(out)
    return out


def _threshold_groups(values: np.ndarray, threshold: float) -> np.ndarray:
    """Numérote les groupes : un nouveau groupe démarre quand la somme cumulée dépasse le seuil."""
    groups = np.empty(values.shape[0], dtype=np.int64)
    acc, gid = 0.0, 0
    for i, v in enumerate(values):
        acc += v
        groups[i] = gid
        if acc >= threshold:
            acc, gid = 0.0, gid + 1
    return groups


def tick_bars(df: pd.DataFrame, ticks_per_bar: int) -> pd.DataFrame:
    """Une barre tous les `ticks_per_bar` enregistrements."""
    if ticks_per_bar < 1:
        raise ValueError("ticks_per_bar doit être >= 1")
    return _aggregate(df, np.arange(len(df)) // int(ticks_per_bar))


def volume_bars(df: pd.DataFrame, volume_per_bar: float) -> pd.DataFrame:
    """Une barre par tranche de volume échangé."""
    return _aggregate(df, _threshold_groups(df["volume"].to_numpy(float), float(volume_per_bar)))


def dollar_bars(df: pd.DataFrame, dollars_per_bar: float) -> pd.DataFrame:
    """Une barre par tranche de notionnel échangé — la variante la plus robuste aux
    variations de prix et aux splits sur longue période."""
    notional = df["close"].to_numpy(float) * df["volume"].to_numpy(float)
    return _aggregate(df, _threshold_groups(notional, float(dollars_per_bar)))


def imbalance_bars(
    df: pd.DataFrame,
    expected_imbalance: Optional[float] = None,
    target_bar_size: int = 50,
    kind: str = "tick",
) -> pd.DataFrame:
    """Barres de déséquilibre (tick / volume imbalance bars).

    On échantillonne dès que |somme des signes de ticks (pondérés)| dépasse l'espérance
    courante du déséquilibre. Ces barres se déclenchent quand un flux directionnel
    informé arrive sur le marché — précisément ce qu'un modèle veut voir.
    """
    close = df["close"].to_numpy(float)
    dprice = np.diff(close, prepend=close[0])
    signs = np.sign(dprice)
    # règle du tick : un tick à variation nulle hérite du signe précédent
    for i in range(1, signs.shape[0]):
        if signs[i] == 0:
            signs[i] = signs[i - 1]
    if signs[0] == 0:
        signs[0] = 1.0

    weights = signs if kind == "tick" else signs * df["volume"].to_numpy(float)

    if expected_imbalance is None:
        # La somme signée se comporte comme une marche aléatoire : |S_L| croît en sqrt(L).
        # Pour viser des barres de ~`target_bar_size` observations, le seuil doit donc être
        # calibré en sqrt(L) * ecart-type des poids, et non linéairement en L.
        scale = float(np.std(weights)) or 1.0
        expected_imbalance = max(np.sqrt(float(target_bar_size)) * scale, 1e-9)

    groups = np.empty(weights.shape[0], dtype=np.int64)
    acc, gid = 0.0, 0
    for i, w in enumerate(weights):
        acc += w
        groups[i] = gid
        if abs(acc) >= expected_imbalance:
            acc, gid = 0.0, gid + 1
    return _aggregate(df, groups)


def build_bars(df: pd.DataFrame, bar_type: str = "time", threshold: Optional[float] = None) -> pd.DataFrame:
    """Point d'entrée unique piloté par la configuration."""
    bar_type = (bar_type or "time").lower()
    if bar_type == "time":
        return df
    if threshold is None:
        # Défaut raisonnable : ~1 barre pour 50 observations d'entrée.
        if bar_type == "tick":
            threshold = 50
        elif bar_type == "volume":
            threshold = float(df["volume"].sum() / max(len(df) / 50, 1))
        elif bar_type == "dollar":
            threshold = float((df["close"] * df["volume"]).sum() / max(len(df) / 50, 1))
    if bar_type == "tick":
        return tick_bars(df, int(threshold))
    if bar_type == "volume":
        return volume_bars(df, float(threshold))
    if bar_type == "dollar":
        return dollar_bars(df, float(threshold))
    if bar_type == "imbalance":
        return imbalance_bars(df, expected_imbalance=threshold)
    raise ValueError(f"bar_type inconnu : {bar_type}")

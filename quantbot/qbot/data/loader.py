"""Chargement et validation des séries OHLCV.

Le contrat est strict : index `DatetimeIndex` trié, unique, croissant, colonnes en minuscules.
Toute violation est une source classique de fuite d'information (look-ahead) ou de double
comptage, donc on échoue tôt et bruyamment plutôt que de « réparer » silencieusement.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

log = get_logger("data.loader")

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

_TF_TO_PANDAS = {
    "M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
    "H1": "1h", "H4": "4h", "D1": "1D", "W1": "1W",
}

_CANONICAL = {
    "date": "time", "datetime": "time", "timestamp": "time", "<date>": "time",
    "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
    "tickvol": "volume", "tick_volume": "volume", "real_volume": "volume",
    "vol": "volume", "adj close": "close", "adj_close": "close",
    "<open>": "open", "<high>": "high", "<low>": "low", "<close>": "close",
    "<tickvol>": "volume", "<vol>": "volume", "<spread>": "spread",
}


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={k: v for k, v in _CANONICAL.items() if k in df.columns})
    return df


def load_ohlcv(
    path: str | Path,
    start: Optional[str] = None,
    end: Optional[str] = None,
    tz: Optional[str] = "UTC",
) -> pd.DataFrame:
    """Charge un CSV/parquet MetaTrader, Dukascopy ou générique en DataFrame OHLCV validé."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier de données introuvable : {path}")

    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else None
        df = pd.read_csv(path, sep=sep, engine="python")

    df = _canonicalize_columns(df)

    # MetaTrader exporte parfois <DATE> et <TIME> séparés.
    if "time" not in df.columns and {"<date>", "<time>"}.issubset(set(df.columns)):
        df["time"] = df["<date>"].astype(str) + " " + df["<time>"].astype(str)
    if "time" not in df.columns:
        date_like = [c for c in df.columns if "date" in c or "time" in c]
        if not date_like:
            raise ValueError(f"Aucune colonne temporelle trouvée dans {path} (colonnes : {list(df.columns)})")
        df = df.rename(columns={date_like[0]: "time"})

    df["time"] = pd.to_datetime(df["time"], utc=(tz == "UTC"), errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    if tz == "UTC" and df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    if "volume" not in df.columns:
        log.warning("Aucune colonne volume : remplie à 1.0 (les features de volume seront neutres).")
        df["volume"] = 1.0

    keep = [c for c in OHLCV_COLUMNS + ["spread"] if c in df.columns]
    df = df[keep]
    df = df[~df.index.duplicated(keep="last")]

    if start is not None:
        df = df[df.index >= pd.Timestamp(start, tz=df.index.tz)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end, tz=df.index.tz)]

    validate_ohlcv(df)
    log.info("Chargé %s : %d barres de %s à %s", path.name, len(df), df.index[0], df.index[-1])
    return df


def validate_ohlcv(df: pd.DataFrame, columns: Iterable[str] = OHLCV_COLUMNS) -> None:
    """Vérifie les invariants d'une série OHLCV ; lève ValueError au premier problème."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes OHLCV manquantes : {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("L'index doit être un DatetimeIndex.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("L'index temporel doit être strictement croissant (risque de look-ahead).")
    if df.index.has_duplicates:
        raise ValueError("L'index contient des horodatages dupliqués.")
    if len(df) == 0:
        raise ValueError("Série vide après filtrage.")

    px = df[["open", "high", "low", "close"]].to_numpy(dtype=float)
    if not np.isfinite(px).all():
        raise ValueError("Prix non finis (NaN/inf) détectés.")
    if (px <= 0).any():
        raise ValueError("Prix négatifs ou nuls détectés.")
    hi, lo = df["high"].to_numpy(float), df["low"].to_numpy(float)
    op, cl = df["open"].to_numpy(float), df["close"].to_numpy(float)
    bad = (hi < lo) | (hi < op - 1e-12) | (hi < cl - 1e-12) | (lo > op + 1e-12) | (lo > cl + 1e-12)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} barres incohérentes (high/low n'encadrent pas open/close).")


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Ré-échantillonne vers un timeframe supérieur (label et closed à gauche, sans fuite)."""
    rule = _TF_TO_PANDAS.get(timeframe.upper(), timeframe)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    if "spread" in df.columns:
        agg["spread"] = "mean"
    out = df.resample(rule, label="left", closed="left").agg(agg).dropna(subset=["open", "close"])
    validate_ohlcv(out)
    return out

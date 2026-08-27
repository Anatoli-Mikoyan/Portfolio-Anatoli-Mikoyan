"""Bibliothèque de stratégies candidates (cahier des charges §8).

Cinq familles, volontairement classiques. Le but n'est pas l'originalité : c'est
d'obtenir des hypothèses SIMPLES, indépendantes et falsifiables, dont on peut mesurer
l'edge individuellement avant d'envisager de les combiner (§15) ou de les allouer par RL
(§10). Allouer du capital entre des stratégies dont aucune n'a d'edge démontré revient à
répartir du bruit.

Toutes utilisent les indicateurs de `qbot.features.technical`, déjà couverts par les
tests de causalité.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from ..features.technical import atr, ewma, realized_vol
from .base import Strategy


# =======================================================================================
@dataclass
class TrendFollowing(Strategy):
    """Croisement de moyennes exponentielles, normalisé par la volatilité."""

    fast: int = 20
    slow: int = 100
    vol_window: int = 60

    @property
    def hypothesis(self) -> str:
        return ("Les tendances persistent au-delà de ce que produirait une marche "
                "aléatoire : l'écart entre moyenne courte et moyenne longue prédit le "
                "signe du rendement futur.")

    @property
    def fails_when(self) -> str:
        return "Marché en range : les croisements se multiplient et chaque faux départ coûte le spread."

    @property
    def warmup(self) -> int:
        return int(max(self.slow * 3, self.vol_window * 2))

    def raw_signal(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        spread = ewma(close, self.fast) - ewma(close, self.slow)
        # Normalisation par la volatilité : sans elle, l'amplitude du signal serait
        # dictée par le régime de volatilité et non par la force de la tendance.
        scale = (realized_vol(close, self.vol_window) * close).replace(0.0, np.nan)
        return np.tanh(spread / (scale * np.sqrt(self.slow)))

    @classmethod
    def param_grid(cls) -> Dict[str, Sequence[Any]]:
        return {"fast": (10, 20), "slow": (60, 100, 200), "vol_window": (60,)}


# =======================================================================================
@dataclass
class TimeSeriesMomentum(Strategy):
    """Momentum absolu : signe du rendement passé, mis à l'échelle par la volatilité.

    C'est l'anomalie la mieux documentée de la littérature (Moskowitz, Ooi & Pedersen,
    2012) — et une référence redoutable : beaucoup de modèles ML sophistiqués ne la battent pas.
    """

    lookback: int = 60
    vol_window: int = 60
    threshold: float = 0.0

    @property
    def hypothesis(self) -> str:
        return ("Le rendement des N dernières barres prédit positivement le rendement "
                "suivant (autocorrélation positive à cet horizon).")

    @property
    def fails_when(self) -> str:
        return "Retournements brutaux de tendance : le momentum est structurellement en retard."

    @property
    def warmup(self) -> int:
        return int(max(self.lookback, self.vol_window) * 2)

    def raw_signal(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        ret = np.log(close).diff(self.lookback)
        vol = realized_vol(close, self.vol_window) * np.sqrt(self.lookback)
        z = ret / vol.replace(0.0, np.nan)
        signal = np.tanh(z)
        # Zone morte autour de zéro : évite de trader un momentum indiscernable du bruit.
        return signal.where(z.abs() > self.threshold, 0.0)

    @classmethod
    def param_grid(cls) -> Dict[str, Sequence[Any]]:
        return {"lookback": (20, 60, 120), "vol_window": (60,), "threshold": (0.0, 0.5)}


# =======================================================================================
@dataclass
class MeanReversion(Strategy):
    """Retour à la moyenne sur z-score, filtré par l'absence de tendance.

    Le filtre de tendance n'est pas décoratif : trader la réversion dans une tendance
    forte revient à se placer systématiquement du mauvais côté du mouvement.
    """

    window: int = 20
    entry_z: float = 1.5
    trend_filter: int = 200

    @property
    def hypothesis(self) -> str:
        return ("Les écarts extrêmes à la moyenne mobile se corrigent, hors période de "
                "tendance établie.")

    @property
    def fails_when(self) -> str:
        return "Tendance persistante ou cassure de range : le prix continue et le stop est touché."

    @property
    def warmup(self) -> int:
        return int(max(self.window, self.trend_filter) * 2)

    def raw_signal(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        ma = close.rolling(self.window).mean()
        sd = close.rolling(self.window).std(ddof=0).replace(0.0, np.nan)
        z = (close - ma) / sd

        signal = -np.tanh(z / self.entry_z)
        if self.trend_filter:
            long_ma = ewma(close, self.trend_filter)
            # L'écart au filtre long est un écart de NIVEAU : il doit être rapporté à la
            # volatilité sur le MÊME horizon, soit sigma_barre * sqrt(horizon). Le diviser
            # par la volatilité d'une seule barre gonfle l'indicateur d'un facteur
            # sqrt(200) ≈ 14 et éteint le signal en permanence.
            horizon_vol = realized_vol(close, 60) * np.sqrt(self.trend_filter)
            trend_strength = (close / long_ma - 1.0).abs() / horizon_vol.replace(0.0, np.nan)
            # Extinction progressive quand la tendance domine, plutôt qu'un interrupteur :
            # un filtre binaire crée des discontinuités et donc du turnover inutile.
            signal = signal * np.exp(-trend_strength.clip(lower=0.0))
        return signal.where(z.abs() > self.entry_z, 0.0)

    @classmethod
    def param_grid(cls) -> Dict[str, Sequence[Any]]:
        return {"window": (10, 20, 50), "entry_z": (1.0, 2.0), "trend_filter": (200,)}


# =======================================================================================
@dataclass
class DonchianBreakout(Strategy):
    """Cassure de canal, confirmée par une expansion de volatilité.

    Le canal est décalé d'une barre (`shift(1)`) : sans cela la barre courante
    participerait à son propre extrême et toute cassure serait détectée par construction.
    """

    channel: int = 55
    exit_channel: int = 20
    atr_window: int = 14
    atr_mult: float = 0.5

    @property
    def hypothesis(self) -> str:
        return ("Une sortie de range accompagnée d'une expansion de volatilité se "
                "prolonge (rupture d'équilibre entre acheteurs et vendeurs).")

    @property
    def fails_when(self) -> str:
        return "Fausses cassures en marché sans direction : entrée au plus haut, sortie au plus bas."

    @property
    def warmup(self) -> int:
        return int(max(self.channel, self.exit_channel, self.atr_window) * 3)

    def raw_signal(self, df: pd.DataFrame) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        upper = high.rolling(self.channel).max().shift(1)
        lower = low.rolling(self.channel).min().shift(1)

        a = atr(high, low, close, self.atr_window)
        # Confirmation : la cassure doit dépasser le bord du canal d'une fraction d'ATR.
        long_break = close > (upper + self.atr_mult * a)
        short_break = close < (lower - self.atr_mult * a)

        state = pd.Series(np.nan, index=close.index)
        state[long_break] = 1.0
        state[short_break] = -1.0

        # Sortie sur canal opposé plus court : une position ouverte le reste jusqu'à
        # invalidation, sinon le signal oscillerait à chaque barre sous le seuil.
        exit_up = high.rolling(self.exit_channel).max().shift(1)
        exit_dn = low.rolling(self.exit_channel).min().shift(1)
        state[(close < exit_dn) & (~long_break) & (~short_break)] = 0.0
        state[(close > exit_up) & (~long_break) & (~short_break)] = 0.0
        return state.ffill()

    @classmethod
    def param_grid(cls) -> Dict[str, Sequence[Any]]:
        return {"channel": (20, 55), "exit_channel": (10, 20), "atr_window": (14,),
                "atr_mult": (0.0, 0.5)}


# =======================================================================================
@dataclass
class VolatilitySqueeze(Strategy):
    """Compression puis expansion de volatilité, direction donnée par la tendance courte.

    L'hypothèse est en deux temps : la volatilité est fortement autocorrélée (fait
    stylisé robuste), et une compression anormale se résout par un mouvement directionnel.
    """

    short_vol: int = 20
    long_vol: int = 100
    squeeze_pct: float = 0.25
    direction_window: int = 10

    @property
    def hypothesis(self) -> str:
        return ("Une volatilité anormalement basse relativement à son propre historique "
                "précède une expansion, dont la direction suit la micro-tendance en cours.")

    @property
    def fails_when(self) -> str:
        return "L'expansion se produit dans le sens opposé à la micro-tendance (faux départ)."

    @property
    def warmup(self) -> int:
        return int(self.long_vol * 3)

    def raw_signal(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        v_short = realized_vol(close, self.short_vol)
        # Percentile de la vol courte dans sa propre distribution passée : borné,
        # stationnaire, et donc comparable entre actifs et entre époques.
        rank = v_short.rolling(self.long_vol).rank(pct=True)
        squeezed = rank < self.squeeze_pct

        direction = np.sign(np.log(close).diff(self.direction_window))
        intensity = ((self.squeeze_pct - rank) / self.squeeze_pct).clip(lower=0.0, upper=1.0)
        return (direction * intensity).where(squeezed, 0.0)

    @classmethod
    def param_grid(cls) -> Dict[str, Sequence[Any]]:
        return {"short_vol": (10, 20), "long_vol": (100,), "squeeze_pct": (0.2, 0.35),
                "direction_window": (5, 10)}


STRATEGY_CLASSES = (
    TrendFollowing, TimeSeriesMomentum, MeanReversion, DonchianBreakout, VolatilitySqueeze,
)


def all_strategies() -> list[Strategy]:
    """Instancie la grille complète de toutes les familles.

    Le nombre retourné EST le nombre d'essais à déclarer au Deflated Sharpe.
    """
    out: list[Strategy] = []
    for cls in STRATEGY_CLASSES:
        out.extend(cls.enumerate())
    return out


def default_strategies() -> list[Strategy]:
    """Une configuration par famille, paramètres médians — pour un usage sans balayage."""
    return [
        TrendFollowing(fast=20, slow=100),
        TimeSeriesMomentum(lookback=60, threshold=0.5),
        MeanReversion(window=20, entry_z=2.0),
        DonchianBreakout(channel=55, exit_channel=20, atr_mult=0.5),
        VolatilitySqueeze(short_vol=20, squeeze_pct=0.25, direction_window=10),
    ]

"""Construction du jeu d'apprentissage supervisé (cahier des charges §9).

Le méta-modèle ne prédit PAS la direction du marché. Il répond à une question beaucoup
plus facile : « ce signal-là, émis par cette stratégie, dans ce régime, vaut-il la peine
d'être suivi ? »

C'est le **meta-labeling** de López de Prado (ch. 3), et le gain n'est pas cosmétique :
prédire la direction sur des marchés à rapport signal/bruit de l'ordre de 1/20 est un
problème quasi impossible, alors que filtrer les signaux d'une stratégie dont la
direction est déjà donnée est un problème binaire ordinaire. Le méta-modèle améliore
surtout la PRÉCISION, donc le profit factor, au prix d'un rappel plus faible — ce qui
est exactement le bon compromis quand chaque trade coûte le spread.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..labeling import build_sample_weights, get_bins, get_events, get_vol_target
from ..strategies.base import Strategy
from ..utils.logging import get_logger

log = get_logger("ml.dataset")


@dataclass
class MetaDataset:
    """Jeu supervisé aligné : features, labels binaires, poids et horizons."""
    X: pd.DataFrame
    y: pd.Series                 # 1 = le trade aurait été gagnant, 0 = perdant
    sample_weight: pd.Series
    t1: pd.Series                # fin de vie de chaque label — indispensable pour la purge
    side: pd.Series              # direction imposée par la stratégie primaire
    returns: pd.Series           # rendement réalisé du trade, orienté par le side

    def __len__(self) -> int:
        return len(self.y)

    @property
    def base_rate(self) -> float:
        """Taux de réussite de la stratégie primaire — la référence à battre.

        Un méta-modèle dont la précision n'excède pas ce taux n'apporte rien : autant
        suivre tous les signaux."""
        return float(self.y.mean())

    def split(self, train_frac: float = 0.7) -> Tuple["MetaDataset", "MetaDataset"]:
        cut = int(len(self.y) * train_frac)
        return self._slice(slice(0, cut)), self._slice(slice(cut, None))

    def _slice(self, s: slice) -> "MetaDataset":
        return MetaDataset(
            X=self.X.iloc[s], y=self.y.iloc[s], sample_weight=self.sample_weight.iloc[s],
            t1=self.t1.iloc[s], side=self.side.iloc[s], returns=self.returns.iloc[s],
        )


def build_meta_dataset(
    strategy: Strategy,
    features: pd.DataFrame,
    prices: pd.DataFrame,
    pt_sl: Tuple[float, float] = (1.5, 1.0),
    vertical_bars: int = 24,
    min_signal: float = 0.1,
    vol_span: int = 100,
    time_decay_last: Optional[float] = 0.5,
) -> MetaDataset:
    """Assemble le jeu de méta-labels pour une stratégie primaire.

    Les événements sont les barres où la stratégie VEUT trader — pas toutes les barres.
    Labelliser chaque barre diluerait le signal dans des milliers d'observations où il ne
    se passe rien, et gonflerait artificiellement la taille de l'échantillon.
    """
    idx = features.index.intersection(prices.index)
    features, prices = features.loc[idx], prices.loc[idx]

    signal = strategy.signal(prices)
    active = signal[signal.abs() >= min_signal]
    if active.empty:
        raise ValueError(f"{strategy.name} n'émet aucun signal au-dessus de {min_signal}.")

    side = pd.Series(np.sign(active.to_numpy()), index=active.index, name="side")
    close = prices["close"]
    trgt = get_vol_target(close, span=vol_span)

    events = get_events(
        close, pd.DatetimeIndex(side.index), pt_sl=pt_sl, trgt=trgt,
        vertical_bars=vertical_bars, side=side,
        high=prices.get("high"), low=prices.get("low"),
    )
    bins = get_bins(events, close)
    if bins.empty:
        raise ValueError(f"Aucun label exploitable pour {strategy.name}.")

    weights = build_sample_weights(prices.index, events["t1"], close,
                                   time_decay_last=time_decay_last)

    common = bins.index.intersection(features.index).intersection(weights.index)
    X = features.loc[common].copy()
    # La force du signal primaire est elle-même une feature : un méta-modèle doit savoir
    # distinguer un signal marginal d'un signal franc.
    X["primary_signal"] = active.reindex(common).astype(float)
    X["primary_abs"] = X["primary_signal"].abs()

    log.info("%s : %d événements, taux de base %.1f%%, unicité moyenne %.2f",
             type(strategy).__name__, len(common), 100 * bins.loc[common, "bin"].mean(),
             weights.loc[common, "tW"].mean())

    return MetaDataset(
        X=X,
        y=bins.loc[common, "bin"].astype(int),
        sample_weight=weights.loc[common, "w"],
        t1=events.loc[common, "t1"],
        side=side.reindex(common),
        returns=bins.loc[common, "ret"],
    )

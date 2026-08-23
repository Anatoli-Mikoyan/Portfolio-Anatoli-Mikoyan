"""Contrat des sources de donnees.

Le moteur ne connait que ``DataProvider`` et ``RawSeries``. Ajouter Polygon,
Alpha Vantage ou un dump interne consiste a implementer une classe dans ce
paquet ; rien d'autre ne bouge.

Le contrat impose une declaration explicite de trois proprietes que les sources
laissent habituellement implicites, et dont l'oubli produit un backtest faux :

* ``bar_label`` -- le timestamp marque-t-il le debut ou la fin de la barre ?
* ``timezone`` -- dans quel fuseau l'index est-il exprime ?
* ``is_preadjusted`` -- les prix sont-ils deja retro-ajustes ?

Aucune valeur par defaut n'est fournie pour ces trois champs : une source qui
ne sait pas repondre est une source qu'on ne peut pas utiliser serieusement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Final, final

import pandas as pd

from ...errors import SchemaError
from ..corporate_actions import CorporateActions
from ..types import BarLabel, Frequency, ensure_utc

__all__ = ["DataProvider", "DataRequest", "RawSeries"]

_REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")

#: Instance partagee : ``CorporateActions`` est immuable, un seul objet vide suffit.
_NO_ACTIONS: Final = CorporateActions()


@final
@dataclass(frozen=True, slots=True)
class DataRequest:
    """Demande de donnees adressee a une source."""

    symbol: str
    frequency: Frequency
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", ensure_utc(self.start, what="DataRequest.start"))
        object.__setattr__(self, "end", ensure_utc(self.end, what="DataRequest.end"))
        if self.end <= self.start:
            raise ValueError(f"Intervalle vide ou inverse : {self.start} -> {self.end}")
        if not self.symbol.strip():
            raise ValueError("Symbole vide")

    @property
    def cache_key(self) -> str:
        return f"{self.symbol.upper()}__{self.frequency.value}"


@final
@dataclass(frozen=True, slots=True)
class RawSeries:
    """Payload brut d'une source, avant normalisation.

    ``frame`` porte un ``DatetimeIndex`` et les colonnes ``open, high, low,
    close, volume``. Sa validation structurelle est faite ici, le plus tot
    possible : une source qui renvoie n'importe quoi doit echouer a la
    frontiere, pas trois couches plus loin.
    """

    symbol: str
    frequency: Frequency
    frame: pd.DataFrame
    bar_label: BarLabel
    timezone: str
    provider: str
    actions: CorporateActions = _NO_ACTIONS
    is_preadjusted: bool = False
    """La source a-t-elle deja retro-ajuste les prix ? Si oui, l'ajustement
    point-in-time est irrecuperable et le normaliseur refuse la serie."""

    def __post_init__(self) -> None:
        frame = self.frame
        missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise SchemaError(
                f"{self.provider}/{self.symbol} : colonnes manquantes {missing}. "
                f"Recu : {list(frame.columns)}"
            )
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise SchemaError(
                f"{self.provider}/{self.symbol} : index de type "
                f"{type(frame.index).__name__}, attendu DatetimeIndex"
            )
        if frame.empty:
            raise SchemaError(f"{self.provider}/{self.symbol} : serie vide")

    def __repr__(self) -> str:
        return (
            f"RawSeries({self.provider}/{self.symbol} {self.frequency.value}, "
            f"{len(self.frame)} lignes, label={self.bar_label.value}, tz={self.timezone})"
        )


class DataProvider(ABC):
    """Source de donnees de marche."""

    #: Identifiant court, utilise dans les chemins de cache et les logs.
    name: str

    @abstractmethod
    def fetch(self, request: DataRequest) -> RawSeries:
        """Recupere la serie demandee. Leve ``ProviderError`` en cas d'echec."""

    def supports(self, frequency: Frequency) -> bool:  # noqa: ARG002
        """Toutes les frequences par defaut ; a restreindre selon la source."""
        return True

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"

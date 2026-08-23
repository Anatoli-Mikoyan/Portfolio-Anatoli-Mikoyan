"""Vocabulaire canonique de la couche donnees.

Deux invariants portes par ce module et verifies partout ailleurs :

1. **Tout instant est timezone-aware, en UTC.** Aucun ``datetime`` naif ne
   franchit la frontiere de la couche donnees. Un datetime naif est une bombe a
   retardement : selon la machine, le meme backtest decale d'une heure deux fois
   par an.
2. **Une barre est labellisee par sa CLOTURE.** Une barre journaliere du
   2020-01-02 porte le timestamp ``2020-01-02T21:00Z`` (16:00 America/New_York),
   pas ``2020-01-02T00:00``. C'est la seule convention pour laquelle l'enonce
   << toute l'information de la barre T est connue a l'instant T >> est vrai.
   Beaucoup de providers labellisent a l'ouverture : le normaliseur convertit,
   et une conversion manquante offrirait une seance entiere de futur gratuit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Final

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "OHLCV_FIELDS",
    "UTC",
    "Bar",
    "BarLabel",
    "Field",
    "Frequency",
    "ensure_utc",
    "to_epoch_ns",
]

UTC: Final = UTC

#: Bornes de plausibilite d'un timestamp de marche, en nanosecondes depuis
#: l'epoch.
#:
#: La borne basse est fixee au 1er janvier 1971, et non a une date "large"
#: comme 1900, parce qu'elle vise un bug precis : lire des microsecondes (ou
#: des millisecondes) comme des nanosecondes divise la valeur par 1 000 (ou
#: 1 000 000) et projette n'importe quelle date moderne dans **janvier 1970**.
#: Une borne a 1900 laisserait donc passer exactement l'erreur qu'elle pretend
#: attraper. Le cout est de ne pas pouvoir representer une seance anterieure a
#: 1971 -- sans consequence pour un moteur qui exige un calendrier de seances
#: et des volumes.
_MIN_PLAUSIBLE_NS: Final = 31_536_000_000_000_000  # 1971-01-01
_MAX_PLAUSIBLE_NS: Final = 7_258_118_400_000_000_000  # 2200-01-01


class Frequency(Enum):
    """Pas temporel d'une serie de barres."""

    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    DAY_1 = "1d"
    WEEK_1 = "1wk"

    @property
    def delta(self) -> timedelta:
        return _FREQUENCY_DELTA[self]

    @property
    def is_intraday(self) -> bool:
        return self.delta < timedelta(days=1)

    @property
    def periods_per_year(self) -> float:
        """Nombre de barres par an, pour l'annualisation des metriques.

        Base sur 252 seances de 6h30 pour l'intraday actions : utiliser 365
        jours calendaires gonflerait mecaniquement tout Sharpe annualise.
        """
        if not self.is_intraday:
            return 252.0 if self is Frequency.DAY_1 else 52.0
        session = timedelta(hours=6, minutes=30)
        return 252.0 * (session / self.delta)

    @classmethod
    def parse(cls, raw: str) -> Frequency:
        try:
            return cls(raw)
        except ValueError:
            valid = ", ".join(item.value for item in cls)
            raise ValueError(f"Frequence inconnue : {raw!r}. Valeurs admises : {valid}") from None


_FREQUENCY_DELTA: Final[dict[Frequency, timedelta]] = {
    Frequency.MINUTE_1: timedelta(minutes=1),
    Frequency.MINUTE_5: timedelta(minutes=5),
    Frequency.MINUTE_15: timedelta(minutes=15),
    Frequency.MINUTE_30: timedelta(minutes=30),
    Frequency.HOUR_1: timedelta(hours=1),
    Frequency.DAY_1: timedelta(days=1),
    Frequency.WEEK_1: timedelta(weeks=1),
}


class BarLabel(Enum):
    """Convention de labellisation temporelle d'un provider.

    ``OPEN`` : le timestamp marque le debut de la barre (yfinance journalier,
    la plupart des exports CSV). ``CLOSE`` : le timestamp marque la fin de la
    barre. Le moteur ne manipule que du ``CLOSE`` ; ``OPEN`` n'existe que le
    temps de la normalisation.
    """

    OPEN = "open"
    CLOSE = "close"
    SESSION_DATE = "session_date"
    """Le timestamp identifie une SEANCE (une date), pas un instant. Cas des
    barres journalieres yfinance, datees a minuit dans le fuseau de la place.
    Le normaliseur les projette sur la cloture reelle via le calendrier."""


class Field(Enum):
    """Colonne d'une serie OHLCV."""

    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"

    @property
    def is_price(self) -> bool:
        return self is not Field.VOLUME


OHLCV_FIELDS: Final[tuple[Field, ...]] = (
    Field.OPEN,
    Field.HIGH,
    Field.LOW,
    Field.CLOSE,
    Field.VOLUME,
)


def ensure_utc(moment: datetime, *, what: str = "timestamp") -> datetime:
    """Valide qu'un datetime est aware et le convertit en UTC.

    Refuse les datetimes naifs plutot que de supposer une timezone.
    """
    if moment.tzinfo is None or moment.tzinfo.utcoffset(moment) is None:
        raise ValueError(
            f"{what} naif ({moment!r}) : la couche donnees exige des datetimes "
            "timezone-aware. Precise explicitement la timezone de la source."
        )
    return moment.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class Bar:
    """Une barre OHLCV close. Immuable.

    ``timestamp`` est l'instant de cloture : toute l'information portee par
    cet objet est connue du marche a partir de cet instant, et pas avant.
    """

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("Bar.timestamp doit etre timezone-aware")

    def value(self, field: Field) -> float:
        return float(getattr(self, field.value))

    @property
    def typical_price(self) -> float:
        """(H + L + C) / 3 : reference de prix moins sensible au bruit du close."""
        return (self.high + self.low + self.close) / 3.0

    @property
    def is_coherent(self) -> bool:
        return (
            self.low <= min(self.open, self.close)
            and max(self.open, self.close) <= self.high
            and self.low <= self.high
            and self.volume >= 0.0
        )


def to_epoch_ns(index: pd.DatetimeIndex) -> NDArray[np.int64]:
    """Convertit un index pandas en nanosecondes UTC depuis l'epoch.

    Passe systematiquement par ``as_unit("ns")`` : depuis pandas 2, un
    ``DatetimeIndex`` construit a partir d'objets ``datetime`` Python porte une
    resolution en **microsecondes**. Lire ses entiers bruts comme des
    nanosecondes divise toute la serie par mille et la ramene vers 1970 -- sans
    exception, sans avertissement, et avec des barres toujours parfaitement
    ordonnees. Le controle de plausibilite ci-dessous existe pour que cette
    erreur ne puisse plus passer inapercue.
    """
    if index.tz is None:
        raise ValueError(
            "Index temporel naif : la conversion en epoch exige une timezone explicite."
        )
    # ``.to_numpy()`` plutot que ``.asi8`` : meme resultat, mais typee dans
    # pandas-stubs et stable entre versions majeures de pandas.
    values = index.tz_convert("UTC").as_unit("ns").to_numpy(dtype="datetime64[ns]")
    values = np.asarray(values.astype(np.int64), dtype=np.int64)
    if values.size and (
        int(values.min()) < _MIN_PLAUSIBLE_NS or int(values.max()) > _MAX_PLAUSIBLE_NS
    ):
        first = datetime.fromtimestamp(int(values[0]) / 1e9, tz=UTC)
        raise ValueError(
            f"Timestamps hors de la plage plausible 1971-2200 (premier : "
            f"{first.isoformat()}). Cause la plus probable : confusion d'unite "
            "entre microsecondes, millisecondes et nanosecondes lors de la "
            "conversion de l'index temporel."
        )
    return values

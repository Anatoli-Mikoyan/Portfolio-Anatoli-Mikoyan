"""Calendriers de seances.

Sert a deux choses precises :

* **Etiqueter les barres a l'heure de cloture reelle** (16:00 ET, 13:00 les
  demi-seances), ce qui conditionne la validite temporelle de tout le moteur.
* **Distinguer un jour ferie d'un trou de donnees.** Sans calendrier, une
  seance absente est indiscernable d'une donnee manquante : soit on ignore de
  vrais trous, soit on signale 10 faux positifs par an.

Implementation sans dependance (pas de ``pandas_market_calendars``) : les
regles NYSE sont algorithmiques et stables, et une dependance de plus sur un
chemin critique n'est pas justifiee.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from typing import Final, final
from zoneinfo import ZoneInfo

from .types import UTC, Frequency

__all__ = [
    "AlwaysOpenCalendar",
    "TradingCalendar",
    "XNYSCalendar",
    "get_calendar",
]


def _easter_sunday(year: int) -> date:
    """Algorithme gregorien anonyme (Meeus/Jones/Butcher)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month, day = divmod(h + m - 7 * n + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-ieme ``weekday`` (0=lundi) du mois. ``n=-1`` pour le dernier."""
    if n > 0:
        first = date(year, month, 1)
        offset = (weekday - first.weekday()) % 7
        return first + timedelta(days=offset + 7 * (n - 1))
    last_day = (date(year + month // 12, month % 12 + 1, 1)) - timedelta(days=1)
    offset = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=offset)


def _observed(day: date) -> date | None:
    """Regle NYSE de report d'un ferie a date fixe.

    Samedi -> vendredi precedent ; dimanche -> lundi suivant. Exception : un
    1er janvier tombant un samedi n'est pas reporte au 31 decembre.
    """
    if day.weekday() == 5:
        if day.month == 1 and day.day == 1:
            return None
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


class TradingCalendar(ABC):
    """Contrat minimal attendu par le normaliseur et le controle qualite."""

    name: str
    timezone: ZoneInfo

    @abstractmethod
    def is_session(self, day: date) -> bool:
        """La date est-elle une seance de negociation ?"""

    @abstractmethod
    def close_time(self, day: date) -> time:
        """Heure de cloture locale de la seance (gere les demi-seances)."""

    @abstractmethod
    def open_time(self, day: date) -> time:
        """Heure d'ouverture locale de la seance."""

    def sessions(self, start: date, end: date) -> tuple[date, ...]:
        """Seances dans ``[start, end]`` inclus."""
        if end < start:
            raise ValueError(f"Intervalle inverse : {start} > {end}")
        out: list[date] = []
        day = start
        while day <= end:
            if self.is_session(day):
                out.append(day)
            day += timedelta(days=1)
        return tuple(out)

    def session_close_utc(self, day: date) -> datetime:
        """Instant de cloture de la seance, en UTC."""
        local = datetime.combine(day, self.close_time(day), tzinfo=self.timezone)
        return local.astimezone(UTC)

    def session_open_utc(self, day: date) -> datetime:
        local = datetime.combine(day, self.open_time(day), tzinfo=self.timezone)
        return local.astimezone(UTC)

    def expected_closes(
        self, start: date, end: date, frequency: Frequency
    ) -> tuple[datetime, ...]:
        """Timestamps de cloture attendus sur l'intervalle, pour la frequence.

        Reference du detecteur de trous : ce qui manque ici et pas dans la
        serie est un trou, pas un ferie.
        """
        sessions = self.sessions(start, end)
        if frequency is Frequency.DAY_1:
            return tuple(self.session_close_utc(day) for day in sessions)
        if frequency is Frequency.WEEK_1:
            # Une barre hebdomadaire clot a la derniere seance de sa semaine ISO.
            by_week: dict[tuple[int, int], date] = {}
            for day in sessions:
                by_week[day.isocalendar()[:2]] = day
            return tuple(self.session_close_utc(day) for _, day in sorted(by_week.items()))
        step = frequency.delta
        out: list[datetime] = []
        for day in sessions:
            cursor = self.session_open_utc(day) + step
            session_end = self.session_close_utc(day)
            while cursor <= session_end:
                out.append(cursor)
                cursor += step
        return tuple(out)


@final
class AlwaysOpenCalendar(TradingCalendar):
    """Marche 24/7 (crypto). Toute date est une seance, cloture a minuit UTC."""

    def __init__(self) -> None:
        self.name = "24/7"
        self.timezone = ZoneInfo("UTC")

    # `day` fait partie du contrat TradingCalendar : un marche 24/7 l'ignore,
    # mais la signature doit rester substituable.
    def is_session(self, day: date) -> bool:  # noqa: ARG002
        return True

    def close_time(self, day: date) -> time:  # noqa: ARG002
        return time(23, 59, 59)

    def open_time(self, day: date) -> time:  # noqa: ARG002
        return time(0, 0)


# Fermetures exceptionnelles NYSE (deuils nationaux, 11 septembre, Sandy...).
# Liste volontairement non exhaustive : le controle qualite signale les seances
# manquantes inconnues plutot que de les absorber en silence.
_ADHOC_CLOSURES: Final[frozenset[date]] = frozenset(
    {
        date(2001, 9, 11),
        date(2001, 9, 12),
        date(2001, 9, 13),
        date(2001, 9, 14),
        date(2004, 6, 11),  # deuil Reagan
        date(2007, 1, 2),  # deuil Ford
        date(2012, 10, 29),  # Sandy
        date(2012, 10, 30),
        date(2018, 12, 5),  # deuil G. H. W. Bush
        date(2025, 1, 9),  # deuil Carter
    }
)


@final
class XNYSCalendar(TradingCalendar):
    """Calendrier NYSE / Nasdaq (actions US)."""

    def __init__(self) -> None:
        self.name = "XNYS"
        self.timezone = ZoneInfo("America/New_York")

    def is_session(self, day: date) -> bool:
        if day.weekday() >= 5:
            return False
        if day in _ADHOC_CLOSURES:
            return False
        return day not in _nyse_holidays(day.year)

    def close_time(self, day: date) -> time:
        return time(13, 0) if day in _nyse_half_days(day.year) else time(16, 0)

    def open_time(self, day: date) -> time:  # noqa: ARG002 - contrat de l'interface
        return time(9, 30)


@lru_cache(maxsize=256)
def _nyse_holidays(year: int) -> frozenset[date]:
    """Feries NYSE observes pour une annee."""
    candidates: list[date | None] = [
        _observed(date(year, 1, 1)),  # Jour de l'an
        _nth_weekday(year, 1, 0, 3) if year >= 1998 else None,  # Martin Luther King
        _nth_weekday(year, 2, 0, 3),  # Presidents' Day
        _easter_sunday(year) - timedelta(days=2),  # Vendredi saint
        _nth_weekday(year, 5, 0, -1),  # Memorial Day
        _observed(date(year, 6, 19)) if year >= 2022 else None,  # Juneteenth
        _observed(date(year, 7, 4)),  # Independence Day
        _nth_weekday(year, 9, 0, 1),  # Labor Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
        _observed(date(year, 12, 25)),  # Noel
    ]
    return frozenset(day for day in candidates if day is not None)


@lru_cache(maxsize=256)
def _nyse_half_days(year: int) -> frozenset[date]:
    """Demi-seances : cloture a 13:00 ET.

    Compte pour l'etiquetage temporel : une barre du 26 novembre 2021 clot a
    18:00 UTC, pas 21:00. Un ecart de 3h suffit a desaligner une jointure
    multi-actifs intraday.
    """
    days: set[date] = set()
    thanksgiving = _nth_weekday(year, 11, 3, 4)
    days.add(thanksgiving + timedelta(days=1))  # Lendemain de Thanksgiving

    for fixed in (date(year, 7, 4), date(year, 12, 25)):
        eve = fixed - timedelta(days=1)
        if eve.weekday() < 5 and eve not in _nyse_holidays(year):
            days.add(eve)
    return frozenset(day for day in days if day.weekday() < 5)


_CALENDARS: Final[dict[str, type[TradingCalendar]]] = {
    "XNYS": XNYSCalendar,
    "NYSE": XNYSCalendar,
    "NASDAQ": XNYSCalendar,
    "24/7": AlwaysOpenCalendar,
    "CRYPTO": AlwaysOpenCalendar,
}


def get_calendar(name: str) -> TradingCalendar:
    """Fabrique un calendrier depuis son nom de configuration."""
    key = name.strip().upper()
    if key not in _CALENDARS:
        raise KeyError(f"Calendrier inconnu : {name!r}. Connus : {sorted(set(_CALENDARS))}")
    return _CALENDARS[key]()

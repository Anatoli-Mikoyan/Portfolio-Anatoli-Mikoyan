"""Calendrier de seances : verifie contre des faits NYSE connus."""

from __future__ import annotations

from datetime import date, time

import pytest

from quant_engine.data.calendar import (
    AlwaysOpenCalendar,
    XNYSCalendar,
    _easter_sunday,
    get_calendar,
)
from quant_engine.data.types import Frequency


@pytest.fixture
def nyse() -> XNYSCalendar:
    return XNYSCalendar()


@pytest.mark.parametrize(
    ("year", "expected"),
    [(2020, date(2020, 4, 12)), (2021, date(2021, 4, 4)), (2024, date(2024, 3, 31)),
     (2025, date(2025, 4, 20)), (2038, date(2038, 4, 25))],
)
def test_paques_gregorien(year: int, expected: date) -> None:
    assert _easter_sunday(year) == expected


@pytest.mark.parametrize(
    ("year", "expected"),
    [(2021, 252), (2022, 251), (2023, 250), (2024, 252), (2025, 250)],
)
def test_nombre_de_seances_annuel(nyse: XNYSCalendar, year: int, expected: int) -> None:
    """Decomptes officiels NYSE. Un ecart signale une regle de ferie fausse."""
    assert len(nyse.sessions(date(year, 1, 1), date(year, 12, 31))) == expected


@pytest.mark.parametrize(
    "closed",
    [
        date(2024, 1, 1),    # Jour de l'an
        date(2024, 1, 15),   # Martin Luther King
        date(2024, 2, 19),   # Presidents' Day
        date(2024, 3, 29),   # Vendredi saint
        date(2024, 5, 27),   # Memorial Day
        date(2024, 6, 19),   # Juneteenth
        date(2024, 7, 4),    # Independence Day
        date(2024, 9, 2),    # Labor Day
        date(2024, 11, 28),  # Thanksgiving
        date(2024, 12, 25),  # Noel
        date(2012, 10, 29),  # Ouragan Sandy
        date(2025, 1, 9),    # Deuil national Carter
    ],
)
def test_jours_de_fermeture(nyse: XNYSCalendar, closed: date) -> None:
    assert not nyse.is_session(closed)


def test_juneteenth_absent_avant_2022(nyse: XNYSCalendar) -> None:
    """Le ferie n'existe qu'a partir de 2022 : l'appliquer retroactivement
    ferait apparaitre un faux trou de donnees chaque annee anterieure."""
    assert nyse.is_session(date(2021, 6, 18))
    assert not nyse.is_session(date(2022, 6, 20))


def test_mlk_absent_avant_1998(nyse: XNYSCalendar) -> None:
    assert nyse.is_session(date(1997, 1, 20))
    assert not nyse.is_session(date(1998, 1, 19))


def test_report_de_ferie_le_week_end(nyse: XNYSCalendar) -> None:
    # 4 juillet 2020 tombe un samedi -> reporte au vendredi 3.
    assert not nyse.is_session(date(2020, 7, 3))
    # 25 decembre 2021 tombe un samedi -> reporte au vendredi 24.
    assert not nyse.is_session(date(2021, 12, 24))
    # Exception : un 1er janvier samedi n'est PAS reporte au 31 decembre.
    assert nyse.is_session(date(2021, 12, 31))


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (date(2024, 11, 29), time(13, 0)),  # lendemain de Thanksgiving
        (date(2024, 7, 3), time(13, 0)),    # veille du 4 juillet
        (date(2024, 12, 24), time(13, 0)),  # veille de Noel
        (date(2024, 3, 15), time(16, 0)),   # seance ordinaire
    ],
)
def test_demi_seances(nyse: XNYSCalendar, day: date, expected: time) -> None:
    assert nyse.close_time(day) == expected


def test_cloture_utc_suit_lheure_dete(nyse: XNYSCalendar) -> None:
    """16:00 New York vaut 21:00 UTC en hiver et 20:00 UTC en ete.

    Figer un decalage constant desaligne toutes les barres pendant la moitie de
    l'annee -- invisible sur un actif seul, destructeur des qu'on croise deux
    marches.
    """
    assert nyse.session_close_utc(date(2024, 1, 3)).hour == 21
    assert nyse.session_close_utc(date(2024, 7, 10)).hour == 20


def test_calendrier_24_7() -> None:
    always = AlwaysOpenCalendar()
    assert always.is_session(date(2024, 1, 1))
    assert always.is_session(date(2024, 3, 30))  # un samedi
    assert len(always.sessions(date(2024, 1, 1), date(2024, 1, 31))) == 31


def test_clotures_attendues_en_journalier(nyse: XNYSCalendar) -> None:
    closes = nyse.expected_closes(date(2024, 1, 1), date(2024, 1, 31), Frequency.DAY_1)
    # 23 jours ouvres en janvier 2024, moins le 1er (jour de l'an) et le 15 (MLK).
    assert len(closes) == 21
    assert all(stamp.tzinfo is not None for stamp in closes)
    assert closes == tuple(sorted(closes))


def test_clotures_attendues_en_intraday(nyse: XNYSCalendar) -> None:
    closes = nyse.expected_closes(date(2024, 3, 4), date(2024, 3, 4), Frequency.HOUR_1)
    # Seance 9h30-16h00 : 6 barres horaires pleines (10h30 ... 15h30) + rien a 16h30.
    assert len(closes) == 6


def test_fabrique_de_calendrier() -> None:
    assert isinstance(get_calendar("xnys"), XNYSCalendar)
    assert isinstance(get_calendar("CRYPTO"), AlwaysOpenCalendar)
    with pytest.raises(KeyError, match="Calendrier inconnu"):
        get_calendar("EURONEXT")


def test_intervalle_inverse_refuse(nyse: XNYSCalendar) -> None:
    with pytest.raises(ValueError, match="Intervalle inverse"):
        nyse.sessions(date(2024, 5, 1), date(2024, 4, 1))

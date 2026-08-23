"""Vocabulaire canonique : frequences, timestamps, barres."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from quant_engine.data.types import UTC, Bar, Field, Frequency, ensure_utc, to_epoch_ns


def test_parse_de_frequence() -> None:
    assert Frequency.parse("1d") is Frequency.DAY_1
    with pytest.raises(ValueError, match="Frequence inconnue"):
        Frequency.parse("3d")


def test_periodes_par_an() -> None:
    assert Frequency.DAY_1.periods_per_year == 252.0
    assert Frequency.WEEK_1.periods_per_year == 52.0
    # 6h30 de seance : 6,5 barres horaires par jour, 252 jours.
    assert Frequency.HOUR_1.periods_per_year == pytest.approx(1638.0)


def test_lannualisation_intraday_utilise_les_heures_de_seance() -> None:
    """Annualiser une serie horaire sur 24h x 365j gonfle mecaniquement tout
    Sharpe : le facteur d'annualisation serait surestime d'un facteur ~2,4."""
    naif = 365 * 24
    assert Frequency.HOUR_1.periods_per_year < naif / 2


def test_ensure_utc_refuse_un_datetime_naif() -> None:
    with pytest.raises(ValueError, match="naif"):
        ensure_utc(datetime(2024, 1, 1))


def test_ensure_utc_convertit() -> None:
    paris = timezone(timedelta(hours=2))
    converted = ensure_utc(datetime(2024, 6, 1, 14, 0, tzinfo=paris))
    assert converted.hour == 12
    assert converted.tzinfo is UTC


def test_bar_refuse_un_timestamp_naif() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        Bar(datetime(2024, 1, 1), 1.0, 2.0, 0.5, 1.5, 100.0)


def test_coherence_dune_barre() -> None:
    stamp = datetime(2024, 1, 1, tzinfo=UTC)
    assert Bar(stamp, 10.0, 12.0, 9.0, 11.0, 5.0).is_coherent
    assert not Bar(stamp, 10.0, 9.0, 11.0, 10.5, 5.0).is_coherent
    assert not Bar(stamp, 10.0, 12.0, 9.0, 11.0, -1.0).is_coherent


def test_valeur_par_champ() -> None:
    bar = Bar(datetime(2024, 1, 1, tzinfo=UTC), 10.0, 12.0, 9.0, 11.0, 5.0)
    assert bar.value(Field.HIGH) == 12.0
    assert bar.typical_price == pytest.approx((12.0 + 9.0 + 11.0) / 3)


def test_to_epoch_ns_force_la_nanoseconde() -> None:
    """Regression : pandas >= 2 construit des index en MICROsecondes.

    Lire leurs entiers bruts comme des nanosecondes divise toute la serie par
    mille et la projette vers 1970 -- sans exception, avec des barres toujours
    parfaitement ordonnees et un backtest qui tourne normalement.
    """
    index = pd.DatetimeIndex([datetime(2020, 1, 2, 21, tzinfo=UTC)])
    assert getattr(index.dtype, "unit", None) == "us", "pandas construit ici en microsecondes"
    values = to_epoch_ns(index)
    assert int(values[0]) == 1_577_998_800_000_000_000
    assert values.dtype == np.int64


def test_to_epoch_ns_refuse_un_index_naif() -> None:
    with pytest.raises(ValueError, match="naif"):
        to_epoch_ns(pd.DatetimeIndex([datetime(2020, 1, 2)]))


@pytest.mark.parametrize(
    ("valeur_ns", "unite_reelle"),
    [
        (1_577_998_800_000_000, "microsecondes lues comme nanosecondes"),
        (1_577_998_800_000, "millisecondes lues comme nanosecondes"),
    ],
)
def test_to_epoch_ns_detecte_une_confusion_dunite(valeur_ns: int, unite_reelle: str) -> None:
    """Un index deja abime doit etre refuse, pas propage.

    Ces deux valeurs correspondent au 2 janvier 2020 exprime en microsecondes
    puis en millisecondes. Relues comme des nanosecondes, elles donnent des
    dates de janvier 1970 : ordonnees, finies, superficiellement valides.
    """
    corrupted = pd.DatetimeIndex(pd.to_datetime([valeur_ns], unit="ns", utc=True))
    with pytest.raises(ValueError, match="unite"):
        to_epoch_ns(corrupted)

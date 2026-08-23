"""Normalisation : recalage temporel, reparations, refus.

Le test le plus important du fichier est ``test_label_de_seance_projete_sur_la
_cloture`` : un mauvais recalage donne une seance entiere d'avance a toutes
les strategies, sans lever aucune erreur.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from quant_engine.data import (
    BarLabel,
    Field,
    FindingKind,
    Frequency,
    NormalizationPolicy,
    QualityPolicy,
    RawSeries,
    normalize,
)
from quant_engine.data.types import UTC
from quant_engine.errors import SchemaError


def frame_of(index: pd.DatetimeIndex, closes: list[float] | None = None) -> pd.DataFrame:
    n = len(index)
    close = np.asarray(closes if closes is not None else [100.0 + i for i in range(n)], dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(n, 1e6),
        },
        index=index,
    )


def raw_of(
    index: pd.DatetimeIndex,
    *,
    label: BarLabel,
    timezone: str = "UTC",
    closes: list[float] | None = None,
    **kwargs: object,
) -> RawSeries:
    return RawSeries(
        symbol="T",
        frequency=Frequency.DAY_1,
        frame=frame_of(index, closes),
        bar_label=label,
        timezone=timezone,
        provider="test",
        **kwargs,  # type: ignore[arg-type]
    )


LENIENT_NOW = datetime(2030, 1, 1, tzinfo=UTC)
POLICY = NormalizationPolicy(raise_on_blocking=False, quality=QualityPolicy(min_bars=1))


def test_label_de_seance_projete_sur_la_cloture() -> None:
    """yfinance date une barre journaliere a minuit, heure de la place.

    La laisser telle quelle signifie qu'a minuit, avant meme l'ouverture, le
    moteur connait deja le close du soir : une seance entiere de futur offerte
    a toutes les strategies, sans la moindre erreur levee.
    """
    index = pd.DatetimeIndex(
        pd.to_datetime(["2024-03-14", "2024-03-15"]), name="timestamp"
    ).tz_localize("America/New_York")
    data = normalize(
        raw_of(index, label=BarLabel.SESSION_DATE, timezone="America/New_York"),
        POLICY, now=LENIENT_NOW,
    )
    # 16:00 New York en mars (heure d'ete) = 20:00 UTC.
    assert data.start == datetime(2024, 3, 14, 20, 0, tzinfo=UTC)
    assert data.end == datetime(2024, 3, 15, 20, 0, tzinfo=UTC)


def test_label_douverture_decale_dune_periode() -> None:
    index = pd.DatetimeIndex(
        pd.to_datetime(["2024-03-14 00:00", "2024-03-15 00:00"]), name="timestamp"
    ).tz_localize("UTC")
    data = normalize(raw_of(index, label=BarLabel.OPEN), POLICY, now=LENIENT_NOW)
    assert data.start == datetime(2024, 3, 15, tzinfo=UTC)


def test_label_de_cloture_inchange() -> None:
    index = pd.DatetimeIndex(
        pd.to_datetime(["2024-03-14 20:00", "2024-03-15 20:00"]), name="timestamp"
    ).tz_localize("UTC")
    data = normalize(raw_of(index, label=BarLabel.CLOSE), POLICY, now=LENIENT_NOW)
    assert data.start == datetime(2024, 3, 14, 20, tzinfo=UTC)


def test_index_naif_localise_selon_la_source() -> None:
    index = pd.DatetimeIndex(pd.to_datetime(["2024-03-14", "2024-03-15"]), name="timestamp")
    data = normalize(
        raw_of(index, label=BarLabel.SESSION_DATE, timezone="America/New_York"),
        POLICY, now=LENIENT_NOW,
    )
    assert data.start.tzinfo is not None
    assert data.start.hour == 20


def test_resolution_nanoseconde_preservee() -> None:
    """Regression du bug de resolution : pandas construit en microsecondes.

    Sans conversion explicite, toute la serie se retrouvait projetee en
    janvier 1970 -- ordonnee, finie, et parfaitement silencieuse.
    """
    index = pd.DatetimeIndex(
        [datetime(2024, 3, 14, 20, tzinfo=UTC), datetime(2024, 3, 15, 20, tzinfo=UTC)],
        name="timestamp",
    )
    data = normalize(raw_of(index, label=BarLabel.CLOSE), POLICY, now=LENIENT_NOW)
    assert data.start.year == 2024
    assert int(data.timestamps[0]) == 1_710_446_400_000_000_000


def test_serie_preajustee_refusee() -> None:
    index = pd.DatetimeIndex(
        pd.to_datetime(["2024-03-14 20:00", "2024-03-15 20:00"])
    ).tz_localize("UTC")
    with pytest.raises(SchemaError, match="retro-ajustes"):
        normalize(raw_of(index, label=BarLabel.CLOSE, is_preadjusted=True), POLICY)


def test_serie_preajustee_acceptee_sur_optin_mais_signalee() -> None:
    index = pd.DatetimeIndex(
        pd.to_datetime(["2024-03-14 20:00", "2024-03-15 20:00"])
    ).tz_localize("UTC")
    policy = NormalizationPolicy(
        allow_preadjusted=True, raise_on_blocking=False, quality=QualityPolicy(min_bars=1)
    )
    data = normalize(raw_of(index, label=BarLabel.CLOSE, is_preadjusted=True), policy,
                     now=LENIENT_NOW)
    assert data.quality is not None
    assert data.quality.has(FindingKind.PREADJUSTED_SOURCE)


def test_doublons_resolus_et_traces() -> None:
    index = pd.DatetimeIndex(
        pd.to_datetime(["2024-03-14 20:00", "2024-03-14 20:00", "2024-03-15 20:00"])
    ).tz_localize("UTC")
    data = normalize(raw_of(index, label=BarLabel.CLOSE, closes=[100.0, 200.0, 300.0]),
                     POLICY, now=LENIENT_NOW)
    assert len(data) == 2
    assert data.raw(Field.CLOSE)[0] == 200.0  # keep='last'
    assert data.quality is not None
    assert data.quality.has(FindingKind.DEDUPLICATED)


def test_doublons_bloquants_si_demande() -> None:
    index = pd.DatetimeIndex(
        pd.to_datetime(["2024-03-14 20:00", "2024-03-14 20:00"])
    ).tz_localize("UTC")
    policy = NormalizationPolicy(on_duplicate="fail", quality=QualityPolicy(min_bars=1))
    with pytest.raises(SchemaError, match="dupliques"):
        normalize(raw_of(index, label=BarLabel.CLOSE), policy, now=LENIENT_NOW)


def test_index_desordonne_reordonne() -> None:
    index = pd.DatetimeIndex(
        pd.to_datetime(["2024-03-15 20:00", "2024-03-14 20:00"])
    ).tz_localize("UTC")
    data = normalize(raw_of(index, label=BarLabel.CLOSE, closes=[200.0, 100.0]),
                     POLICY, now=LENIENT_NOW)
    assert data.raw(Field.CLOSE).tolist() == [100.0, 200.0]


def test_barre_en_cours_ecartee() -> None:
    """Une barre dont la cloture est future est partielle : son high, son low
    et son close bougeront encore. La backtester revient a connaitre la fin de
    la seance des son debut."""
    now = datetime(2024, 3, 15, 12, tzinfo=UTC)
    index = pd.DatetimeIndex(
        pd.to_datetime(["2024-03-14 20:00", "2024-03-15 20:00"])
    ).tz_localize("UTC")
    data = normalize(raw_of(index, label=BarLabel.CLOSE), POLICY, now=now)
    assert len(data) == 1
    assert data.quality is not None
    assert data.quality.has(FindingKind.TRIMMED_INCOMPLETE_BAR)


def test_valeurs_invalides_ecartees() -> None:
    index = pd.DatetimeIndex(
        pd.to_datetime(["2024-03-13 20:00", "2024-03-14 20:00", "2024-03-15 20:00"])
    ).tz_localize("UTC")
    data = normalize(
        raw_of(index, label=BarLabel.CLOSE, closes=[100.0, float("nan"), 102.0]),
        POLICY, now=LENIENT_NOW,
    )
    assert len(data) == 2
    assert data.quality is not None
    assert data.quality.has(FindingKind.DROPPED_ROWS)


def test_forward_fill_signale_bruyamment() -> None:
    """Le ffill cree des barres a rendement nul : la volatilite mesuree baisse
    et tout Sharpe calcule dessus est surestime. Il reste possible, jamais
    silencieux."""
    index = pd.DatetimeIndex(
        pd.to_datetime(["2024-03-13 20:00", "2024-03-14 20:00", "2024-03-15 20:00"])
    ).tz_localize("UTC")
    policy = NormalizationPolicy(
        on_nan="ffill", raise_on_blocking=False, quality=QualityPolicy(min_bars=1)
    )
    data = normalize(
        raw_of(index, label=BarLabel.CLOSE, closes=[100.0, float("nan"), 102.0]),
        policy, now=LENIENT_NOW,
    )
    assert len(data) == 3
    assert data.raw(Field.CLOSE)[1] == 100.0
    assert data.quality is not None
    finding = data.quality.of_kind(FindingKind.FORWARD_FILLED)
    assert finding and "surestime" in finding[0].detail


def test_ohlc_incoherent_bloquant_par_defaut() -> None:
    index = pd.DatetimeIndex(
        pd.to_datetime(["2024-03-14 20:00", "2024-03-15 20:00"])
    ).tz_localize("UTC")
    frame = frame_of(index)
    frame.loc[frame.index[1], "high"] = 1.0  # high < low
    raw = RawSeries(
        symbol="T", frequency=Frequency.DAY_1, frame=frame,
        bar_label=BarLabel.CLOSE, timezone="UTC", provider="test",
    )
    with pytest.raises(SchemaError, match="OHLC incoherent"):
        normalize(raw, NormalizationPolicy(quality=QualityPolicy(min_bars=1)), now=LENIENT_NOW)


def test_serie_vide_apres_nettoyage() -> None:
    index = pd.DatetimeIndex(
        pd.to_datetime(["2024-03-14 20:00", "2024-03-15 20:00"])
    ).tz_localize("UTC")
    with pytest.raises(SchemaError, match="aucune barre exploitable"):
        normalize(
            raw_of(index, label=BarLabel.CLOSE, closes=[float("nan"), float("nan")]),
            POLICY, now=LENIENT_NOW,
        )


def test_colonnes_manquantes_refusees() -> None:
    index = pd.DatetimeIndex(pd.to_datetime(["2024-03-14 20:00"])).tz_localize("UTC")
    with pytest.raises(SchemaError, match="colonnes manquantes"):
        RawSeries(
            symbol="T", frequency=Frequency.DAY_1,
            frame=pd.DataFrame({"open": [1.0]}, index=index),
            bar_label=BarLabel.CLOSE, timezone="UTC", provider="test",
        )


def test_index_non_temporel_refuse() -> None:
    with pytest.raises(SchemaError, match="DatetimeIndex"):
        RawSeries(
            symbol="T", frequency=Frequency.DAY_1,
            frame=pd.DataFrame(
                {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]}
            ),
            bar_label=BarLabel.CLOSE, timezone="UTC", provider="test",
        )


def test_volume_nul_optionnellement_ecarte() -> None:
    index = pd.DatetimeIndex(
        pd.to_datetime(["2024-03-13 20:00", "2024-03-14 20:00", "2024-03-15 20:00"])
    ).tz_localize("UTC")
    frame = frame_of(index)
    frame.loc[frame.index[1], "volume"] = 0.0
    raw = RawSeries(
        symbol="T", frequency=Frequency.DAY_1, frame=frame,
        bar_label=BarLabel.CLOSE, timezone="UTC", provider="test",
    )
    policy = NormalizationPolicy(
        drop_zero_volume=True, raise_on_blocking=False, quality=QualityPolicy(min_bars=1)
    )
    assert len(normalize(raw, policy, now=LENIENT_NOW)) == 2

"""Detecteurs de qualite : chaque anomalie est fabriquee, puis doit etre vue."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pytest
from conftest import build_market_data, build_raw

from quant_engine.data import (
    CorporateActions,
    DataQualityReport,
    FindingKind,
    Frequency,
    NormalizationPolicy,
    QualityPolicy,
    Severity,
    Split,
    SyntheticSpec,
    normalize,
    run_quality_checks,
)
from quant_engine.data.calendar import XNYSCalendar
from quant_engine.data.quality import _looks_like_split
from quant_engine.data.types import UTC
from quant_engine.errors import DataQualityError


def stamps(n: int) -> np.ndarray:
    base = datetime(2020, 1, 1, 21, tzinfo=UTC)
    return np.array(
        [int((base + timedelta(days=i)).timestamp() * 1e9) for i in range(n)],
        dtype=np.int64,
    )


def check(close: np.ndarray, **kwargs: Any) -> DataQualityReport:
    n = close.size
    defaults: dict[str, Any] = {
        "symbol": "T",
        "frequency": Frequency.DAY_1,
        "timestamps": stamps(n),
        "open_": close.copy(),
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": np.full(n, 1e6),
        "policy": QualityPolicy(min_bars=1, check_sessions=False),
    }
    defaults.update(kwargs)
    return run_quality_checks(**defaults)


def test_serie_propre_sans_anomalie() -> None:
    report = check(np.linspace(100.0, 110.0, 50))
    assert report.is_clean
    assert not report.findings


def test_detection_de_nan() -> None:
    close = np.linspace(100.0, 110.0, 20)
    close[5] = np.nan
    report = check(close)
    assert report.has(FindingKind.NAN_VALUE)
    assert report.of_kind(FindingKind.NAN_VALUE)[0].severity is Severity.BLOCKING


def test_detection_de_prix_negatif() -> None:
    close = np.linspace(100.0, 110.0, 20)
    close[3] = -1.0
    report = check(close, low=close * 1.01, high=close * 0.99)
    assert report.has(FindingKind.NON_POSITIVE_PRICE)


def test_detection_dohlc_incoherent() -> None:
    close = np.full(20, 100.0)
    high = np.full(20, 100.0)
    high[7] = 90.0  # high < close
    report = check(close, high=high, low=np.full(20, 99.0))
    assert report.has(FindingKind.OHLC_INCOHERENT)


def test_detection_de_volume_nul_et_negatif() -> None:
    close = np.full(20, 100.0)
    volume = np.full(20, 1e6)
    volume[4] = 0.0
    volume[9] = -5.0
    report = check(close, volume=volume)
    assert report.of_kind(FindingKind.ZERO_VOLUME)[0].count == 1
    assert report.of_kind(FindingKind.NEGATIVE_VOLUME)[0].count == 1


def test_detection_de_timestamps_dupliques() -> None:
    close = np.full(5, 100.0)
    duplicated = stamps(5)
    duplicated = duplicated.copy()
    duplicated[3] = duplicated[2]
    report = check(close, timestamps=duplicated)
    assert report.has(FindingKind.DUPLICATE_TIMESTAMP)


def test_detection_de_barres_figees() -> None:
    close = np.concatenate([np.linspace(100.0, 105.0, 10), np.full(6, 105.0)])
    report = check(close, open_=close.copy(), high=close.copy(), low=close.copy())
    finding = report.of_kind(FindingKind.STALE_BAR)
    assert finding and finding[0].count >= 3


def test_split_declare_nest_pas_signale_comme_aberration() -> None:
    """Un split connu explique le saut : le signaler serait un faux positif
    quotidien qui finirait par etre ignore."""
    close = np.concatenate([np.full(10, 400.0), np.full(10, 100.0)])
    actions = CorporateActions(splits=(Split(datetime(2020, 1, 11, tzinfo=UTC), 4.0),))
    report = check(close, actions=actions)
    assert not report.has(FindingKind.SUSPECTED_UNDECLARED_SPLIT)
    assert not report.has(FindingKind.PRICE_JUMP_UNEXPLAINED)
    assert report.has(FindingKind.SPLIT_APPLIED)


def test_split_non_declare_est_bloquant() -> None:
    """Un cours passant de 400 a 100 sans operation declaree n'est pas un
    krach de -75 % : c'est une serie incoherente. Confondre les deux fabrique
    des rendements qui n'ont jamais existe."""
    close = np.concatenate([np.full(10, 400.0), np.full(10, 100.0)])
    report = check(close)
    finding = report.of_kind(FindingKind.SUSPECTED_UNDECLARED_SPLIT)
    assert finding and finding[0].severity is Severity.BLOCKING
    with pytest.raises(DataQualityError, match="bloquante"):
        report.raise_if_blocking()


def test_saut_inexplique_reste_un_avertissement() -> None:
    """Un -34 % (ratio hors des fractions simples) est plausible : krach,
    profit warning. On avertit sans bloquer."""
    close = np.concatenate([np.full(15, 100.0), np.full(15, 66.0)])
    report = check(close)
    assert report.has(FindingKind.PRICE_JUMP_UNEXPLAINED)
    assert not report.has(FindingKind.SUSPECTED_UNDECLARED_SPLIT)


@pytest.mark.parametrize(
    ("ratio", "attendu"),
    [
        (0.5, True), (0.25, True), (2.0, True), (0.333, True), (0.05, True),
        # Ratios ambigus : un 3-pour-2 est indiscernable d'une seance a -33 %.
        # Volontairement non reconnus, pour ne pas bloquer sur un krach reel.
        (0.66, False), (1.5, False),
        (0.9, False), (1.37, False),
    ],
)
def test_reconnaissance_dun_ratio_de_split(ratio: float, attendu: bool) -> None:
    assert (_looks_like_split(ratio) is not None) is attendu


def test_echantillon_court_signale() -> None:
    report = check(np.full(50, 100.0), policy=QualityPolicy(min_bars=252, check_sessions=False))
    finding = report.of_kind(FindingKind.SHORT_HISTORY)
    assert finding and "intervalle de confiance" in finding[0].detail


def test_seances_manquantes_detectees() -> None:
    spec = SyntheticSpec(missing_days=(date(2020, 7, 20), date(2020, 7, 21)))
    data = build_market_data(spec)
    assert data.quality is not None
    finding = data.quality.of_kind(FindingKind.MISSING_SESSION)
    assert finding and finding[0].count == 2


def test_trop_de_seances_manquantes_devient_bloquant() -> None:
    missing = tuple(
        day for day in XNYSCalendar().sessions(date(2020, 1, 1), date(2020, 12, 31))
    )[:60]
    spec = SyntheticSpec(missing_days=missing)
    data = build_market_data(spec)
    assert data.quality is not None
    finding = data.quality.of_kind(FindingKind.MISSING_SESSION)
    assert finding and finding[0].severity is Severity.BLOCKING


def test_seance_inattendue_detectee() -> None:
    """Une barre un jour ferie signale un calendrier inadapte ou une donnee
    fabriquee."""
    spec = SyntheticSpec(calendar="24/7")
    raw = build_raw(spec, start=datetime(2021, 1, 1, tzinfo=UTC),
                    end=datetime(2021, 3, 1, tzinfo=UTC))
    data = normalize(raw, NormalizationPolicy(calendar="XNYS", raise_on_blocking=False,
                                              quality=QualityPolicy(min_bars=1)))
    assert data.quality is not None
    assert data.quality.has(FindingKind.UNEXPECTED_SESSION)


def test_severite_surchargeable() -> None:
    policy = QualityPolicy(
        min_bars=1,
        check_sessions=False,
        severity_overrides={FindingKind.ZERO_VOLUME: Severity.BLOCKING},
    )
    volume = np.full(20, 1e6)
    volume[3] = 0.0
    report = check(np.full(20, 100.0), volume=volume, policy=policy)
    assert report.of_kind(FindingKind.ZERO_VOLUME)[0].severity is Severity.BLOCKING


def test_serialisation_du_rapport() -> None:
    report = check(np.full(20, 100.0), policy=QualityPolicy(min_bars=252, check_sessions=False))
    payload = report.to_dict()
    assert payload["symbol"] == "T"
    findings = payload["findings"]
    assert isinstance(findings, list)
    kinds = {item["kind"] for item in findings}
    assert FindingKind.SHORT_HISTORY.value in kinds


def test_resume_lisible() -> None:
    report = check(np.linspace(100.0, 110.0, 50))
    assert "aucune anomalie" in report.summary()


def test_fusion_de_constats() -> None:
    from quant_engine.data.quality import Finding

    report = check(np.linspace(100.0, 110.0, 50))
    extra = Finding(FindingKind.DROPPED_ROWS, Severity.WARNING, 2, "test")
    merged = report.merged_with([extra])
    assert merged.has(FindingKind.DROPPED_ROWS)
    assert not report.has(FindingKind.DROPPED_ROWS)  # immuable

"""Fixtures partagees.

Aucun test n'accede au reseau. Toutes les series proviennent du generateur
deterministe : un test qui depend de Yahoo Finance n'est pas un test, c'est un
detecteur de panne reseau.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pytest

from quant_engine.data import (
    CorporateActions,
    Frequency,
    MarketData,
    NormalizationPolicy,
    QualityPolicy,
    Split,
    SyntheticProvider,
    SyntheticSpec,
    normalize,
)
from quant_engine.data.providers.base import DataRequest
from quant_engine.data.types import UTC

if TYPE_CHECKING:
    from quant_engine.data import RawSeries

START = datetime(2019, 1, 1, tzinfo=UTC)
END = datetime(2023, 1, 1, tzinfo=UTC)

#: Politique permissive : les tests inspectent le rapport qualite au lieu de
#: laisser une exception masquer ce qu'on cherche a observer.
LENIENT = NormalizationPolicy(
    raise_on_blocking=False,
    quality=QualityPolicy(min_bars=1),
)


def build_raw(spec: SyntheticSpec | None = None, *, symbol: str = "SYNTH",
              start: datetime = START, end: datetime = END) -> RawSeries:
    return SyntheticProvider(spec or SyntheticSpec()).fetch(
        DataRequest(symbol=symbol, frequency=Frequency.DAY_1, start=start, end=end)
    )


def build_market_data(
    spec: SyntheticSpec | None = None,
    *,
    symbol: str = "SYNTH",
    start: datetime = START,
    end: datetime = END,
    policy: NormalizationPolicy | None = None,
) -> MarketData:
    return normalize(
        build_raw(spec, symbol=symbol, start=start, end=end),
        policy or LENIENT,
        now=end + timedelta(days=1),
    )


def make_data(
    closes: list[float],
    *,
    symbol: str = "TEST",
    actions: CorporateActions | None = None,
    start: datetime = datetime(2020, 1, 1, 21, tzinfo=UTC),
) -> MarketData:
    """Jeu de donnees minimal a partir d'une liste de closes.

    Une barre par jour calendaire ; le calendrier n'est pas verifie ici, ces
    series servent a tester la mecanique de vue, pas la couverture temporelle.
    """
    n = len(closes)
    timestamps = np.array(
        [int((start + timedelta(days=i)).timestamp() * 1_000_000_000) for i in range(n)],
        dtype=np.int64,
    )
    close = np.asarray(closes, dtype=np.float64)
    open_ = np.concatenate(([close[0]], close[:-1]))
    return MarketData(
        symbol=symbol,
        frequency=Frequency.DAY_1,
        timestamps=timestamps,
        open_=open_,
        high=np.maximum(open_, close) * 1.01,
        low=np.minimum(open_, close) * 0.99,
        close=close,
        volume=np.full(n, 1_000_000.0),
        actions=actions,
        provider="test",
    )


@pytest.fixture
def clean_data() -> MarketData:
    """Serie propre de 4 ans, sans anomalie injectee."""
    return build_market_data()


@pytest.fixture
def ramp_data() -> MarketData:
    """Serie strictement croissante 100, 101, ... 199."""
    return make_data([100.0 + i for i in range(100)])


@pytest.fixture
def split_data() -> MarketData:
    """Serie de 10 barres avec un split 2-pour-1 a l'index 5."""
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 52.5, 53.0, 53.5, 54.0, 54.5]
    ex_date = datetime(2020, 1, 6, 21, tzinfo=UTC)
    return make_data(closes, actions=CorporateActions(splits=(Split(ex_date, 2.0),)))


@pytest.fixture
def anomalous_spec() -> SyntheticSpec:
    """Specification chargee d'anomalies, une par detecteur."""
    return SyntheticSpec(
        splits=((date(2021, 6, 15), 4.0),),
        dividends=((date(2020, 3, 20), 0.5),),
        missing_days=(date(2020, 7, 20), date(2020, 7, 21), date(2020, 7, 22)),
        outlier_days=((date(2022, 5, 10), 1.45),),
        stale_days=(date(2022, 2, 15), date(2022, 2, 16), date(2022, 2, 17), date(2022, 2, 18)),
    )

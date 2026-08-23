"""Mecanique de MarketData, HistoryView et BarCursor."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from conftest import make_data

from quant_engine.data import AdjustmentPolicy, Field, Frequency, MarketData, Multipliers
from quant_engine.data.types import UTC
from quant_engine.errors import DataError


def raw_policy(data: MarketData) -> Multipliers:
    return data.multipliers(AdjustmentPolicy.RAW)


def test_construction_refuse_une_serie_vide() -> None:
    with pytest.raises(DataError, match="vide"):
        MarketData(
            symbol="X", frequency=Frequency.DAY_1,
            timestamps=np.array([], dtype=np.int64),
            open_=np.array([]), high=np.array([]), low=np.array([]),
            close=np.array([]), volume=np.array([]),
        )


def test_construction_refuse_des_longueurs_incoherentes() -> None:
    stamps = np.array([1_600_000_000_000_000_000, 1_600_086_400_000_000_000], dtype=np.int64)
    with pytest.raises(DataError, match="valeurs pour 2 timestamps"):
        MarketData(
            symbol="X", frequency=Frequency.DAY_1, timestamps=stamps,
            open_=np.array([1.0]), high=np.array([1.0, 2.0]), low=np.array([1.0, 2.0]),
            close=np.array([1.0, 2.0]), volume=np.array([1.0, 2.0]),
        )


def test_construction_refuse_un_index_non_monotone() -> None:
    base = int(datetime(2020, 1, 1, tzinfo=UTC).timestamp() * 1e9)
    day = 86_400_000_000_000
    stamps = np.array([base, base + 2 * day, base + day], dtype=np.int64)
    ones = np.ones(3)
    with pytest.raises(DataError, match="non strictement croissants"):
        MarketData(
            symbol="X", frequency=Frequency.DAY_1, timestamps=stamps,
            open_=ones, high=ones, low=ones, close=ones, volume=ones,
        )


def test_controle_despacement_detecte_une_erreur_dunite() -> None:
    """Serie journaliere dont les timestamps sont espaces d'une seconde."""
    base = int(datetime(2020, 1, 1, tzinfo=UTC).timestamp() * 1e9)
    stamps = np.array([base + i * 1_000_000_000 for i in range(10)], dtype=np.int64)
    ones = np.ones(10)
    with pytest.raises(DataError, match="unite temporelle"):
        MarketData(
            symbol="X", frequency=Frequency.DAY_1, timestamps=stamps,
            open_=ones, high=ones, low=ones, close=ones, volume=ones,
        )


def test_les_tableaux_internes_sont_geles(ramp_data: MarketData) -> None:
    with pytest.raises(ValueError, match="read-only"):
        ramp_data.raw(Field.CLOSE)[0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        ramp_data.timestamps[0] = 0


def test_bornes_temporelles(ramp_data: MarketData) -> None:
    assert ramp_data.start < ramp_data.end
    assert ramp_data.start.tzinfo is not None
    assert len(ramp_data) == 100


def test_index_at_or_before(ramp_data: MarketData) -> None:
    cible = ramp_data.start + timedelta(days=10, hours=3)
    assert ramp_data.index_at_or_before(cible) == 10
    assert ramp_data.index_at_or_before(ramp_data.start - timedelta(days=1)) == -1
    assert ramp_data.index_at_or_before(ramp_data.end) == 99


def test_bar_par_anciennete(ramp_data: MarketData) -> None:
    view = ramp_data.view_at(10, raw_policy(ramp_data))
    assert view.bar(0).close == 110.0
    assert view.bar(1).close == 109.0
    assert view.bar(10).close == 100.0
    assert view.last() == 110.0
    assert view.last(Field.VOLUME) == 1_000_000.0


def test_fenetre_lookback(ramp_data: MarketData) -> None:
    view = ramp_data.view_at(20, raw_policy(ramp_data))
    np.testing.assert_array_equal(view.close(3), [118.0, 119.0, 120.0])
    assert view.close().size == 21
    with pytest.raises(ValueError, match="strictement positif"):
        view.close(0)


def test_as_frame(ramp_data: MarketData) -> None:
    view = ramp_data.view_at(5, raw_policy(ramp_data))
    frame = view.as_frame()
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert len(frame) == 6
    assert str(pd.DatetimeIndex(frame.index).tz) == "UTC"
    assert frame.index.is_monotonic_increasing


def test_as_frame_avec_lookback(ramp_data: MarketData) -> None:
    view = ramp_data.view_at(50, raw_policy(ramp_data))
    assert len(view.as_frame(10)) == 10


def test_view_at_hors_bornes(ramp_data: MarketData) -> None:
    with pytest.raises(IndexError):
        ramp_data.view_at(100, raw_policy(ramp_data))
    with pytest.raises(ValueError, match="negatif"):
        ramp_data.view_at(-1, raw_policy(ramp_data))


def test_execution_bar_est_brute(split_data: MarketData) -> None:
    """Le moteur execute au prix cote, jamais au prix ajuste."""
    assert split_data.execution_bar(0).close == 100.0
    assert split_data.execution_bar(5).close == 52.5


def test_repr_lisible(ramp_data: MarketData) -> None:
    assert "TEST" in repr(ramp_data)
    view = ramp_data.view_at(3, raw_policy(ramp_data))
    assert "4 barres" in repr(view)


def test_to_frame_complet(ramp_data: MarketData) -> None:
    frame = ramp_data.to_frame()
    assert len(frame) == 100
    frame.loc[frame.index[0], "close"] = -1.0
    assert ramp_data.raw(Field.CLOSE)[0] == 100.0


def test_warmup_trop_grand(ramp_data: MarketData) -> None:
    from quant_engine.errors import InsufficientHistoryError

    with pytest.raises(InsufficientHistoryError):
        ramp_data.cursor(AdjustmentPolicy.RAW, warmup=100)


def test_curseur_avec_arret_anticipe(ramp_data: MarketData) -> None:
    points = list(ramp_data.cursor(AdjustmentPolicy.RAW, warmup=10, stop=20))
    assert [p.index for p in points] == list(range(10, 20))


def test_empoisonnement_bornes(ramp_data: MarketData) -> None:
    with pytest.raises(ValueError, match="hors"):
        ramp_data.with_future_poisoned(0)
    poisoned = ramp_data.with_future_poisoned(10)
    assert np.isfinite(poisoned.raw(Field.CLOSE)[:10]).all()
    assert np.isnan(poisoned.raw(Field.CLOSE)[10:]).all()
    assert "poisoned" in poisoned.provider


def test_les_vues_ajustees_restent_en_lecture_seule(split_data: MarketData) -> None:
    view = split_data.view_at(8, split_data.multipliers(AdjustmentPolicy.SPLIT_PIT))
    with pytest.raises(ValueError, match="read-only"):
        view.close()[0] = 0.0


def test_volume_ajuste_en_sens_inverse_du_prix(split_data: MarketData) -> None:
    """Un split 2-pour-1 divise le prix par deux et double le nombre de titres :
    le volume historique doit etre multiplie, pas divise."""
    view = split_data.view_at(9, split_data.multipliers(AdjustmentPolicy.SPLIT_PIT))
    assert view.volume()[0] == pytest.approx(2_000_000.0)
    assert view.volume()[-1] == pytest.approx(1_000_000.0)
    assert view.close()[0] == pytest.approx(50.0)


def test_timestamps_de_vue(ramp_data: MarketData) -> None:
    view = ramp_data.view_at(5, raw_policy(ramp_data))
    stamps = view.timestamps()
    assert stamps.size == 6
    assert str(stamps.dtype) == "datetime64[ns]"


def test_donnees_synthetiques_deterministes() -> None:
    from conftest import build_market_data

    first = build_market_data()
    second = build_market_data()
    np.testing.assert_array_equal(first.raw(Field.CLOSE), second.raw(Field.CLOSE))


def test_serie_a_rendement_constant_est_analytique() -> None:
    """Reference de non-regression : P_n = P_0 (1+r)^n, exactement."""
    from quant_engine.data.providers.synthetic import constant_return_series

    raw = constant_return_series(0.001, 100, start_price=100.0)
    closes = raw.frame["close"].to_numpy()
    for n in (0, 1, 10, 50, 99):
        assert closes[n] == pytest.approx(100.0 * (1.001**n), rel=1e-12)


def test_make_data_helper_coherent() -> None:
    data = make_data([10.0, 11.0, 12.0])
    assert len(data) == 3
    assert data.execution_bar(1).close == 11.0

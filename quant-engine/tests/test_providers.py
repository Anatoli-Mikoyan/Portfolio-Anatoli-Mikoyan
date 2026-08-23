"""Sources de donnees : contrat, determinisme, CSV."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest

from quant_engine.data import (
    BarLabel,
    CsvProvider,
    Frequency,
    SyntheticProvider,
    SyntheticSpec,
    YFinanceProvider,
)
from quant_engine.data.providers.base import DataRequest
from quant_engine.data.types import UTC
from quant_engine.errors import ProviderError

START = datetime(2021, 1, 1, tzinfo=UTC)
END = datetime(2021, 12, 31, tzinfo=UTC)


def test_requete_refuse_un_intervalle_inverse() -> None:
    with pytest.raises(ValueError, match="Intervalle vide ou inverse"):
        DataRequest(symbol="X", frequency=Frequency.DAY_1, start=END, end=START)


def test_requete_refuse_un_datetime_naif() -> None:
    with pytest.raises(ValueError, match="naif"):
        DataRequest(symbol="X", frequency=Frequency.DAY_1, start=datetime(2021, 1, 1), end=END)


def test_requete_refuse_un_symbole_vide() -> None:
    with pytest.raises(ValueError, match="Symbole vide"):
        DataRequest(symbol="  ", frequency=Frequency.DAY_1, start=START, end=END)


def test_synthetique_deterministe() -> None:
    request = DataRequest(symbol="S", frequency=Frequency.DAY_1, start=START, end=END)
    first = SyntheticProvider(SyntheticSpec(seed=7)).fetch(request)
    second = SyntheticProvider(SyntheticSpec(seed=7)).fetch(request)
    np.testing.assert_array_equal(first.frame.to_numpy(), second.frame.to_numpy())

    autre = SyntheticProvider(SyntheticSpec(seed=8)).fetch(request)
    assert not np.array_equal(first.frame.to_numpy(), autre.frame.to_numpy())


def test_synthetique_livre_des_prix_bruts() -> None:
    """La discontinuite de split doit etre presente : c'est ce qu'une source
    honnete renvoie, et c'est ce que le moteur doit savoir gerer."""
    request = DataRequest(symbol="S", frequency=Frequency.DAY_1, start=START, end=END)
    raw = SyntheticProvider(SyntheticSpec(splits=((date(2021, 6, 15), 4.0),))).fetch(request)
    closes = raw.frame["close"].to_numpy()
    ratios = closes[1:] / closes[:-1]
    assert ratios.min() < 0.3, "la discontinuite de split doit rester visible"
    assert not raw.is_preadjusted


def test_synthetique_respecte_le_calendrier() -> None:
    request = DataRequest(symbol="S", frequency=Frequency.DAY_1, start=START, end=END)
    raw = SyntheticProvider().fetch(request)
    assert len(raw.frame) == 252  # seances NYSE en 2021


def test_synthetique_refuse_lintraday() -> None:
    assert not SyntheticProvider().supports(Frequency.HOUR_1)


def test_csv_aller_retour(tmp_path: Path) -> None:
    csv = tmp_path / "AAA_1d.csv"
    csv.write_text(
        "Date,Open,High,Low,Close,Volume,Dividends,Stock Splits\n"
        "2021-01-04,100,101,99,100.5,1000000,0,0\n"
        "2021-01-05,100.5,102,100,101.5,1100000,0.25,0\n"
        "2021-01-06,101.5,103,101,102.5,1200000,0,2\n",
        encoding="utf-8",
    )
    provider = CsvProvider(tmp_path, timezone="America/New_York", bar_label=BarLabel.SESSION_DATE)
    raw = provider.fetch(
        DataRequest(symbol="AAA", frequency=Frequency.DAY_1, start=START, end=END)
    )
    assert len(raw.frame) == 3
    assert list(raw.frame.columns) == ["open", "high", "low", "close", "volume"]
    assert len(raw.actions.dividends) == 1
    assert raw.actions.dividends[0].amount == 0.25
    assert len(raw.actions.splits) == 1
    assert raw.actions.splits[0].ratio == 2.0


def test_csv_ignore_la_colonne_adj_close(tmp_path: Path) -> None:
    """La colonne ajustee d'un export est contaminee : on ne la lit pas."""
    csv = tmp_path / "BBB_1d.csv"
    csv.write_text(
        "Date,Open,High,Low,Close,Adj Close,Volume\n"
        "2021-01-04,100,101,99,100.5,24.1,1000000\n"
        "2021-01-05,100.5,102,100,101.5,24.3,1100000\n",
        encoding="utf-8",
    )
    provider = CsvProvider(tmp_path, timezone="UTC", bar_label=BarLabel.SESSION_DATE)
    raw = provider.fetch(
        DataRequest(symbol="BBB", frequency=Frequency.DAY_1, start=START, end=END)
    )
    assert "adj_close" not in raw.frame.columns
    assert raw.frame["close"].iloc[0] == 100.5


def test_csv_fichier_absent(tmp_path: Path) -> None:
    provider = CsvProvider(tmp_path, timezone="UTC", bar_label=BarLabel.CLOSE)
    with pytest.raises(ProviderError, match="introuvable"):
        provider.fetch(DataRequest(symbol="ZZZ", frequency=Frequency.DAY_1, start=START, end=END))


def test_csv_sans_colonne_temporelle(tmp_path: Path) -> None:
    (tmp_path / "CCC_1d.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    provider = CsvProvider(tmp_path, timezone="UTC", bar_label=BarLabel.CLOSE)
    with pytest.raises(ProviderError, match="colonne temporelle"):
        provider.fetch(DataRequest(symbol="CCC", frequency=Frequency.DAY_1, start=START, end=END))


def test_csv_hors_intervalle(tmp_path: Path) -> None:
    (tmp_path / "DDD_1d.csv").write_text(
        "Date,Open,High,Low,Close,Volume\n2019-01-04,1,1,1,1,1\n", encoding="utf-8"
    )
    provider = CsvProvider(tmp_path, timezone="UTC", bar_label=BarLabel.CLOSE)
    with pytest.raises(ProviderError, match="aucune ligne"):
        provider.fetch(DataRequest(symbol="DDD", frequency=Frequency.DAY_1, start=START, end=END))


def test_yfinance_declare_ses_frequences() -> None:
    provider = YFinanceProvider()
    assert provider.supports(Frequency.DAY_1)
    assert provider.supports(Frequency.MINUTE_5)
    assert provider.name == "yfinance"


@pytest.mark.network
def test_yfinance_reel() -> None:  # pragma: no cover - exclu du CI
    """Exclu par defaut : un test qui depend du reseau n'est pas un test."""
    provider = YFinanceProvider()
    raw = provider.fetch(
        DataRequest(symbol="AAPL", frequency=Frequency.DAY_1, start=START, end=END)
    )
    assert not raw.is_preadjusted
    assert raw.bar_label is BarLabel.SESSION_DATE


def test_aberration_injectee_visible() -> None:
    """Le generateur doit savoir fabriquer le defaut qu'on veut detecter."""
    request = DataRequest(symbol="S", frequency=Frequency.DAY_1, start=START, end=END)
    normal = SyntheticProvider(SyntheticSpec(seed=3)).fetch(request)
    abime = SyntheticProvider(
        SyntheticSpec(seed=3, outlier_days=((date(2021, 5, 10), 1.6),))
    ).fetch(request)
    ecart = (abime.frame["close"] / normal.frame["close"]).max()
    assert ecart == pytest.approx(1.6)


def test_serie_en_rampe_lineaire() -> None:
    """Reference analytique : la progression est exactement de `step` par barre."""
    from quant_engine.data.providers.synthetic import linear_ramp_series

    raw = linear_ramp_series(50, start_price=10.0, step=0.5)
    closes = raw.frame["close"].to_numpy()
    assert len(closes) == 50
    np.testing.assert_allclose(np.diff(closes), 0.5)
    assert closes[0] == 10.0
    assert closes[-1] == pytest.approx(10.0 + 0.5 * 49)


def test_generateur_refuse_un_intervalle_sans_seance() -> None:
    request = DataRequest(
        symbol="S", frequency=Frequency.DAY_1,
        start=datetime(2021, 12, 25, tzinfo=UTC), end=datetime(2021, 12, 26, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="Aucune seance"):
        SyntheticProvider().fetch(request)

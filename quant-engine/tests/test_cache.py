"""Cache Parquet : reproductibilite et invalidation."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from conftest import END, START, build_raw

from quant_engine.data import (
    CachedProvider,
    DataProvider,
    Frequency,
    ParquetCache,
    RawSeries,
    SyntheticProvider,
)
from quant_engine.data.cache import SCHEMA_VERSION, CacheEntry
from quant_engine.data.providers.base import DataRequest
from quant_engine.data.types import UTC, BarLabel
from quant_engine.errors import CacheError

NOW = datetime(2023, 1, 2, tzinfo=UTC)


@pytest.fixture
def cache(tmp_path: Path) -> ParquetCache:
    return ParquetCache(tmp_path / "market")


@pytest.fixture
def request_(pytestconfig: pytest.Config) -> DataRequest:
    return DataRequest(symbol="SYNTH", frequency=Frequency.DAY_1, start=START, end=END)


def test_aller_retour(cache: ParquetCache, request_: DataRequest) -> None:
    raw = build_raw()
    entry = cache.put(raw, now=NOW)
    assert entry.rows == len(raw.frame)
    assert entry.schema_version == SCHEMA_VERSION

    restored = cache.get("synthetic", request_, now=NOW)
    assert restored is not None
    assert len(restored.frame) == len(raw.frame)
    assert restored.bar_label is raw.bar_label
    assert restored.timezone == raw.timezone


def test_operations_sur_titre_conservees(cache: ParquetCache, request_: DataRequest) -> None:
    from datetime import date

    from quant_engine.data import SyntheticSpec

    raw = build_raw(SyntheticSpec(splits=((date(2021, 6, 15), 4.0),),
                                  dividends=((date(2020, 3, 20), 0.5),)))
    cache.put(raw, now=NOW)
    restored = cache.get("synthetic", request_, now=NOW)
    assert restored is not None
    assert len(restored.actions.splits) == 1
    assert restored.actions.splits[0].ratio == 4.0
    assert len(restored.actions.dividends) == 1


def test_absence_de_cache(cache: ParquetCache, request_: DataRequest) -> None:
    assert cache.get("synthetic", request_, now=NOW) is None


def test_invalidation_par_duree_de_vie(cache: ParquetCache, request_: DataRequest) -> None:
    cache.put(build_raw(), now=NOW)
    assert cache.get("synthetic", request_, now=NOW + timedelta(hours=2)) is not None
    assert cache.get("synthetic", request_, now=NOW + timedelta(days=3)) is None


def test_invalidation_par_couverture(cache: ParquetCache) -> None:
    """Une demande depassant la fenetre stockee doit repartir a la source."""
    cache.put(build_raw(start=START, end=datetime(2021, 1, 1, tzinfo=UTC)), now=NOW)
    trop_large = DataRequest(
        symbol="SYNTH", frequency=Frequency.DAY_1, start=START, end=datetime(2022, 6, 1, tzinfo=UTC)
    )
    assert cache.get("synthetic", trop_large, now=NOW) is None


def test_la_fenetre_recente_est_toujours_rechargee(cache: ParquetCache) -> None:
    """Les dernieres barres sont celles que les sources corrigent le plus.

    Les servir depuis un cache vieux de quelques jours revient a backtester une
    version des donnees que plus personne ne peut reproduire.
    """
    recent_end = NOW - timedelta(days=1)
    raw = build_raw(start=datetime(2022, 1, 1, tzinfo=UTC), end=recent_end)
    cache.put(raw, now=NOW - timedelta(days=10))
    demande = DataRequest(
        symbol="SYNTH", frequency=Frequency.DAY_1,
        start=datetime(2022, 1, 1, tzinfo=UTC), end=recent_end,
    )
    assert cache.get("synthetic", demande, now=NOW) is None


def test_version_de_schema_incompatible(cache: ParquetCache, request_: DataRequest) -> None:
    cache.put(build_raw(), now=NOW)
    _, manifest = cache._paths("synthetic", request_)
    manifest.write_text(manifest.read_text().replace(f'"schema_version": {SCHEMA_VERSION}',
                                                     '"schema_version": 1'))
    assert cache.get("synthetic", request_, now=NOW) is None


def test_falsification_detectee(cache: ParquetCache, request_: DataRequest) -> None:
    """L'empreinte SHA-256 rend une modification hors moteur impossible a ignorer.

    Sans elle, "j'ai obtenu un Sharpe de 1,3" n'est pas une affirmation
    verifiable : personne ne peut prouver sur quelles donnees.
    """
    cache.put(build_raw(), now=NOW)
    data_path, _ = cache._paths("synthetic", request_)
    data_path.write_bytes(data_path.read_bytes() + b"corruption")
    with pytest.raises(CacheError, match="Empreinte"):
        cache.get("synthetic", request_, now=NOW)


def test_empreinte_stable(cache: ParquetCache, request_: DataRequest) -> None:
    cache.put(build_raw(), now=NOW)
    first = cache.fingerprint("synthetic", request_)
    assert first is not None and len(first) == 64
    cache.put(build_raw(), now=NOW)
    assert cache.fingerprint("synthetic", request_) == first


def test_invalidation_manuelle(cache: ParquetCache, request_: DataRequest) -> None:
    cache.put(build_raw(), now=NOW)
    assert cache.invalidate("synthetic", request_)
    assert not cache.invalidate("synthetic", request_)
    assert cache.get("synthetic", request_, now=NOW) is None


def test_manifeste_illisible(cache: ParquetCache, request_: DataRequest) -> None:
    cache.put(build_raw(), now=NOW)
    _, manifest = cache._paths("synthetic", request_)
    manifest.write_text("{ pas du json")
    assert cache.get("synthetic", request_, now=NOW) is None


def test_provider_decore_ne_frappe_la_source_quune_fois(
    cache: ParquetCache, request_: DataRequest
) -> None:
    appels = 0

    class Comptant(DataProvider):
        """Enveloppe comptant les appels reels a la source."""

        name = "synthetic"

        def fetch(self, request: DataRequest) -> RawSeries:
            nonlocal appels
            appels += 1
            return SyntheticProvider().fetch(request)

    provider = CachedProvider(Comptant(), cache)
    provider.fetch(request_, now=NOW)
    provider.fetch(request_, now=NOW)
    assert appels == 1


def test_entree_couvre_avec_tolerance() -> None:
    """Une demande commencant un samedi ne peut pas produire de barre ce
    jour-la : exiger une correspondance exacte rechargerait tout, a chaque fois."""
    entry = CacheEntry(
        symbol="X", frequency=Frequency.DAY_1, provider="p",
        bar_label=BarLabel.CLOSE,
        timezone="UTC", is_preadjusted=False,
        covered_start=datetime(2020, 1, 6, tzinfo=UTC),
        covered_end=datetime(2020, 12, 31, tzinfo=UTC),
        rows=250, fetched_at=NOW, content_sha256="0" * 64,
    )
    demande = DataRequest(
        symbol="X", frequency=Frequency.DAY_1,
        start=datetime(2020, 1, 4, tzinfo=UTC), end=datetime(2020, 12, 31, tzinfo=UTC),
    )
    assert entry.covers(demande)

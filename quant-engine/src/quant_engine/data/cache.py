"""Cache Parquet des series brutes.

Trois objectifs, par ordre d'importance :

1. **Reproductibilite.** Les sources grand public revisent leur historique sans
   prevenir. Deux executions du meme backtest a six mois d'ecart peuvent donner
   des chiffres differents sans qu'une ligne de code ait bouge. Le cache fige ce
   qui a reellement servi et calcule une empreinte SHA-256 que le rapport de
   backtest peut citer : sans cette empreinte, "j'ai obtenu un Sharpe de 1,3"
   n'est pas une affirmation verifiable.
2. **Invalidation explicite.** Une entree n'est servie que si elle couvre
   l'intervalle demande et n'a pas depasse sa duree de vie. La queue recente est
   systematiquement rafraichie : les dernieres barres sont les plus souvent
   corrigees a posteriori.
3. **Vitesse.** Accessoire, mais un walk-forward relance des centaines de fois
   la meme serie.

Le cache stocke des donnees **brutes**, avant normalisation. Normaliser avant
de cacher figerait une version de la politique de nettoyage dans les fichiers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final, final

import pandas as pd

from ..errors import CacheError
from ..logging_setup import get_logger
from .corporate_actions import CorporateActions, Dividend, Split
from .providers.base import DataProvider, DataRequest, RawSeries
from .types import UTC, BarLabel, Frequency

__all__ = ["CacheEntry", "CachedProvider", "ParquetCache"]

_LOG = get_logger("data.cache")
SCHEMA_VERSION: Final = 2


@final
@dataclass(frozen=True, slots=True)
class CacheEntry:
    """Metadonnees d'une entree, telles qu'ecrites dans le manifeste."""

    symbol: str
    frequency: Frequency
    provider: str
    bar_label: BarLabel
    timezone: str
    is_preadjusted: bool
    covered_start: datetime
    covered_end: datetime
    rows: int
    fetched_at: datetime
    content_sha256: str
    schema_version: int = SCHEMA_VERSION

    def covers(self, request: DataRequest) -> bool:
        """L'entree couvre-t-elle l'intervalle demande ?

        Tolerance de 4 jours a chaque extremite : une demande commencant un
        samedi ne peut pas produire de barre ce jour-la, et exiger une
        correspondance exacte provoquerait un rechargement systematique.
        """
        slack = timedelta(days=4)
        return (
            self.covered_start <= request.start + slack
            and self.covered_end >= request.end - slack
        )

    def age(self, now: datetime) -> timedelta:
        return now - self.fetched_at

    def to_json(self, actions: CorporateActions) -> str:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "symbol": self.symbol,
            "frequency": self.frequency.value,
            "provider": self.provider,
            "bar_label": self.bar_label.value,
            "timezone": self.timezone,
            "is_preadjusted": self.is_preadjusted,
            "covered_start": self.covered_start.isoformat(),
            "covered_end": self.covered_end.isoformat(),
            "rows": self.rows,
            "fetched_at": self.fetched_at.isoformat(),
            "content_sha256": self.content_sha256,
            "actions": {
                "splits": [
                    {"ex_date": s.ex_date.isoformat(), "ratio": s.ratio} for s in actions.splits
                ],
                "dividends": [
                    {"ex_date": d.ex_date.isoformat(), "amount": d.amount}
                    for d in actions.dividends
                ],
            },
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    @staticmethod
    def from_json(text: str) -> tuple[CacheEntry, CorporateActions]:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CacheError(f"Manifeste illisible : {exc}") from exc
        version = int(payload.get("schema_version", 0))
        if version != SCHEMA_VERSION:
            raise CacheError(
                f"Manifeste en version {version}, attendu {SCHEMA_VERSION} : entree perimee"
            )
        entry = CacheEntry(
            symbol=str(payload["symbol"]),
            frequency=Frequency(payload["frequency"]),
            provider=str(payload["provider"]),
            bar_label=BarLabel(payload["bar_label"]),
            timezone=str(payload["timezone"]),
            is_preadjusted=bool(payload["is_preadjusted"]),
            covered_start=datetime.fromisoformat(payload["covered_start"]),
            covered_end=datetime.fromisoformat(payload["covered_end"]),
            rows=int(payload["rows"]),
            fetched_at=datetime.fromisoformat(payload["fetched_at"]),
            content_sha256=str(payload["content_sha256"]),
            schema_version=version,
        )
        raw_actions = payload.get("actions", {})
        actions = CorporateActions(
            splits=tuple(
                Split(datetime.fromisoformat(item["ex_date"]), float(item["ratio"]))
                for item in raw_actions.get("splits", [])
            ),
            dividends=tuple(
                Dividend(datetime.fromisoformat(item["ex_date"]), float(item["amount"]))
                for item in raw_actions.get("dividends", [])
            ),
        )
        return entry, actions


@final
class ParquetCache:
    """Stockage sur disque des series brutes."""

    def __init__(
        self,
        root: str | Path,
        *,
        ttl: timedelta = timedelta(days=1),
        refresh_tail: timedelta = timedelta(days=5),
    ) -> None:
        self.root = Path(root)
        self.ttl = ttl
        self.refresh_tail = refresh_tail
        """Fenetre recente toujours rechargee : les dernieres barres sont
        celles que les sources corrigent le plus souvent."""

    # -- chemins --------------------------------------------------------------
    def _dir(self, provider: str, request: DataRequest) -> Path:
        return self.root / provider / request.symbol.upper()

    def _paths(self, provider: str, request: DataRequest) -> tuple[Path, Path]:
        base = self._dir(provider, request)
        stem = request.frequency.value
        return base / f"{stem}.parquet", base / f"{stem}.manifest.json"

    # -- lecture --------------------------------------------------------------
    def get(
        self, provider: str, request: DataRequest, *, now: datetime | None = None
    ) -> RawSeries | None:
        """Retourne l'entree si elle est valide, ``None`` sinon (avec la raison en log)."""
        reference = now if now is not None else datetime.now(tz=UTC)
        data_path, manifest_path = self._paths(provider, request)
        if not data_path.is_file() or not manifest_path.is_file():
            return None
        try:
            entry, actions = CacheEntry.from_json(manifest_path.read_text(encoding="utf-8"))
        except CacheError as exc:
            _LOG.warning(
                "entree de cache invalidee",
                extra={"path": str(manifest_path), "reason": str(exc)},
            )
            return None

        reason = self._staleness_reason(entry, request, reference)
        if reason is not None:
            _LOG.info(
                "cache ignore",
                extra={"symbol": request.symbol, "provider": provider, "reason": reason},
            )
            return None

        payload = data_path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != entry.content_sha256:
            raise CacheError(
                f"Empreinte du cache incoherente pour {data_path} "
                f"(attendu {entry.content_sha256[:12]}, calcule {digest[:12]}). "
                "Le fichier a ete modifie hors du moteur : supprime-le."
            )

        frame = pd.read_parquet(data_path)
        window = frame.loc[
            (frame.index >= pd.Timestamp(request.start))
            & (frame.index <= pd.Timestamp(request.end))
        ]
        if window.empty:
            return None
        _LOG.info(
            "cache utilise",
            extra={
                "symbol": request.symbol,
                "rows": len(window),
                "sha256": digest[:12],
                "fetched_at": entry.fetched_at.isoformat(),
            },
        )
        return RawSeries(
            symbol=request.symbol,
            frequency=entry.frequency,
            frame=window,
            bar_label=entry.bar_label,
            timezone=entry.timezone,
            provider=entry.provider,
            actions=actions,
            is_preadjusted=entry.is_preadjusted,
        )

    def _staleness_reason(
        self, entry: CacheEntry, request: DataRequest, now: datetime
    ) -> str | None:
        if not entry.covers(request):
            return (
                f"couverture insuffisante ({entry.covered_start.date()} -> "
                f"{entry.covered_end.date()} pour {request.start.date()} -> {request.end.date()})"
            )
        if entry.age(now) > self.ttl:
            return f"entree agee de {entry.age(now)} (ttl {self.ttl})"
        if request.end > now - self.refresh_tail and entry.fetched_at < now - self.refresh_tail:
            return "l'intervalle demande touche la fenetre recente, sujette a revision"
        return None

    # -- ecriture -------------------------------------------------------------
    def put(self, raw: RawSeries, *, now: datetime | None = None) -> CacheEntry:
        reference = now if now is not None else datetime.now(tz=UTC)
        request = DataRequest(
            symbol=raw.symbol,
            frequency=raw.frequency,
            start=_index_min(raw.frame),
            end=_index_max(raw.frame) + timedelta(seconds=1),
        )
        data_path, manifest_path = self._paths(raw.provider, request)
        data_path.parent.mkdir(parents=True, exist_ok=True)

        frame = raw.frame.copy()
        frame.to_parquet(data_path, engine="pyarrow", compression="snappy", index=True)
        digest = hashlib.sha256(data_path.read_bytes()).hexdigest()

        entry = CacheEntry(
            symbol=raw.symbol,
            frequency=raw.frequency,
            provider=raw.provider,
            bar_label=raw.bar_label,
            timezone=raw.timezone,
            is_preadjusted=raw.is_preadjusted,
            covered_start=_index_min(raw.frame),
            covered_end=_index_max(raw.frame),
            rows=len(raw.frame),
            fetched_at=reference,
            content_sha256=digest,
        )
        manifest_path.write_text(entry.to_json(raw.actions), encoding="utf-8")
        _LOG.info(
            "serie mise en cache",
            extra={
                "symbol": raw.symbol,
                "provider": raw.provider,
                "rows": entry.rows,
                "sha256": digest[:12],
                "path": str(data_path),
            },
        )
        return entry

    def invalidate(self, provider: str, request: DataRequest) -> bool:
        data_path, manifest_path = self._paths(provider, request)
        removed = False
        for path in (data_path, manifest_path):
            if path.is_file():
                path.unlink()
                removed = True
        return removed

    def fingerprint(self, provider: str, request: DataRequest) -> str | None:
        """Empreinte SHA-256 de la serie servie, a citer dans un rapport."""
        _, manifest_path = self._paths(provider, request)
        if not manifest_path.is_file():
            return None
        try:
            entry, _ = CacheEntry.from_json(manifest_path.read_text(encoding="utf-8"))
        except CacheError:
            return None
        return entry.content_sha256


@final
class CachedProvider(DataProvider):
    """Decore une source d'un cache Parquet, sans en modifier le contrat."""

    def __init__(self, inner: DataProvider, cache: ParquetCache) -> None:
        self.inner = inner
        self.cache = cache
        self.name = inner.name

    def supports(self, frequency: Frequency) -> bool:
        return self.inner.supports(frequency)

    def fetch(self, request: DataRequest, *, now: datetime | None = None) -> RawSeries:
        cached = self.cache.get(self.inner.name, request, now=now)
        if cached is not None:
            return cached
        fresh = self.inner.fetch(request)
        self.cache.put(fresh, now=now)
        return fresh

    def __repr__(self) -> str:
        return f"CachedProvider({self.inner!r}, root={self.cache.root})"


def _index_min(frame: pd.DataFrame) -> datetime:
    stamp = pd.Timestamp(pd.DatetimeIndex(frame.index).min())
    return _as_utc(stamp)


def _index_max(frame: pd.DataFrame) -> datetime:
    stamp = pd.Timestamp(pd.DatetimeIndex(frame.index).max())
    return _as_utc(stamp)


def _as_utc(stamp: pd.Timestamp) -> datetime:
    moment = stamp.to_pydatetime()
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)

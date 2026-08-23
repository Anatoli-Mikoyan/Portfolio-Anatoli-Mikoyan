"""Assemblage de la couche donnees depuis la configuration.

Un seul point d'entree : ``DataLoader.from_config(...).load(symbol, start, end)``.
Le choix de la source, du calendrier, de la politique de nettoyage et de la
politique d'ajustement vit dans un YAML versionne, jamais dans le code d'une
strategie. Deux backtests ne peuvent pas diverger a cause d'un parametre
different code en dur quelque part.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Final, cast, final

from ..config import ConfigNode, load_config
from ..errors import ConfigError
from ..logging_setup import get_logger
from .adjustment import AdjustmentPolicy
from .cache import CachedProvider, ParquetCache
from .dataset import MarketData
from .normalize import NormalizationPolicy, OnDuplicate, OnInvalid, normalize
from .providers.base import DataProvider, DataRequest
from .providers.csv_provider import CsvProvider
from .providers.synthetic import SyntheticProvider
from .providers.yfinance_provider import YFinanceProvider
from .quality import QualityPolicy
from .types import BarLabel, Frequency, ensure_utc

__all__ = ["DataLoader"]

_LOG = get_logger("data.loader")

_DATA_KEYS: Final = (
    "provider", "calendar", "frequency", "adjustment",
    "cache", "normalization", "quality", "csv", "yfinance", "synthetic",
)


@final
class DataLoader:
    """Fabrique de ``MarketData`` configuree."""

    def __init__(
        self,
        provider: DataProvider,
        *,
        frequency: Frequency = Frequency.DAY_1,
        adjustment: AdjustmentPolicy = AdjustmentPolicy.SPLIT_PIT,
        normalization: NormalizationPolicy | None = None,
    ) -> None:
        self.provider = provider
        self.frequency = frequency
        self.adjustment = adjustment
        self.normalization = normalization or NormalizationPolicy()

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_config(cls, config: ConfigNode | str | Path) -> DataLoader:
        root = config if isinstance(config, ConfigNode) else load_config(config)
        node = root.section("data") if "data" in root else root
        node.reject_unknown(_DATA_KEYS)

        frequency = Frequency.parse(node.str_("frequency", "1d"))
        calendar = node.str_("calendar", "XNYS")
        adjustment = AdjustmentPolicy(
            node.enum_(
                "adjustment",
                [policy.value for policy in AdjustmentPolicy],
                AdjustmentPolicy.SPLIT_PIT.value,
            )
        )
        provider = _build_provider(node, frequency)

        cache_node = node.optional_section("cache")
        cache_node.reject_unknown(("enabled", "root", "ttl_hours", "refresh_tail_days"))
        if cache_node.bool_("enabled", True):
            provider = CachedProvider(
                provider,
                ParquetCache(
                    cache_node.str_("root", ".cache/market"),
                    ttl=timedelta(hours=cache_node.float_("ttl_hours", 24.0)),
                    refresh_tail=timedelta(days=cache_node.float_("refresh_tail_days", 5.0)),
                ),
            )

        quality_node = node.optional_section("quality")
        quality_node.reject_unknown(
            (
                "jump_sigma", "jump_floor", "stale_run_length", "min_bars",
                "max_missing_session_ratio", "check_sessions",
            )
        )
        quality = QualityPolicy(
            jump_sigma=quality_node.float_("jump_sigma", 8.0),
            jump_floor=quality_node.float_("jump_floor", 0.20),
            stale_run_length=quality_node.int_("stale_run_length", 3),
            min_bars=quality_node.int_("min_bars", 252),
            max_missing_session_ratio=quality_node.float_("max_missing_session_ratio", 0.02),
            check_sessions=quality_node.bool_("check_sessions", True),
        )

        norm_node = node.optional_section("normalization")
        norm_node.reject_unknown(
            (
                "on_duplicate", "on_nan", "on_incoherent_ohlc", "drop_zero_volume",
                "drop_incomplete_last_bar", "allow_preadjusted", "raise_on_blocking",
            )
        )
        normalization = NormalizationPolicy(
            calendar=calendar,
            on_duplicate=cast(
                OnDuplicate, norm_node.enum_("on_duplicate", ("last", "first", "fail"), "last")
            ),
            on_nan=cast(OnInvalid, norm_node.enum_("on_nan", ("drop", "fail", "ffill"), "drop")),
            on_incoherent_ohlc=cast(
                OnInvalid,
                norm_node.enum_("on_incoherent_ohlc", ("drop", "fail", "ffill"), "fail"),
            ),
            drop_zero_volume=norm_node.bool_("drop_zero_volume", False),
            drop_incomplete_last_bar=norm_node.bool_("drop_incomplete_last_bar", True),
            allow_preadjusted=norm_node.bool_("allow_preadjusted", False),
            raise_on_blocking=norm_node.bool_("raise_on_blocking", True),
            quality=quality,
        )
        return cls(
            provider,
            frequency=frequency,
            adjustment=adjustment,
            normalization=normalization,
        )

    # -- chargement -----------------------------------------------------------
    def load(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        now: datetime | None = None,
    ) -> MarketData:
        """Recupere, normalise et valide une serie."""
        request = DataRequest(
            symbol=symbol,
            frequency=self.frequency,
            start=ensure_utc(start, what="start"),
            end=ensure_utc(end, what="end"),
        )
        raw = self.provider.fetch(request)
        data = normalize(raw, self.normalization, now=now)
        _LOG.info(
            "jeu de donnees pret",
            extra={
                "symbol": data.symbol,
                "bars": len(data),
                "start": data.start.isoformat(),
                "end": data.end.isoformat(),
                "adjustment": self.adjustment.value,
            },
        )
        return data

    def load_many(
        self, symbols: list[str], start: datetime, end: datetime
    ) -> dict[str, MarketData]:
        return {symbol: self.load(symbol, start, end) for symbol in symbols}


def _parse_events(node: ConfigNode, key: str) -> tuple[tuple[date, float], ...]:
    """Lit une liste d'evenements au format ``"AAAA-MM-JJ:valeur"``."""
    if key not in node:
        return ()
    events: list[tuple[date, float]] = []
    for item in node.str_list(key):
        raw_date, _, raw_value = item.partition(":")
        if not raw_value:
            raise ConfigError(f"{node.path}.{key} : attendu 'AAAA-MM-JJ:valeur', recu {item!r}")
        try:
            events.append((date.fromisoformat(raw_date.strip()), float(raw_value)))
        except ValueError as exc:
            raise ConfigError(f"{node.path}.{key} : entree invalide {item!r} ({exc})") from exc
    return tuple(events)


def _build_provider(node: ConfigNode, frequency: Frequency) -> DataProvider:
    kind = node.enum_("provider", ("yfinance", "csv", "synthetic"), "yfinance")
    if kind == "yfinance":
        section = node.optional_section("yfinance")
        section.reject_unknown(("exchange_timezone",))
        return YFinanceProvider(
            exchange_timezone=section.str_("exchange_timezone", "America/New_York")
        )
    if kind == "csv":
        section = node.section("csv")
        section.reject_unknown(("root", "timezone", "bar_label", "filename_template"))
        label = section.enum_("bar_label", tuple(item.value for item in BarLabel))
        return CsvProvider(
            section.str_("root"),
            timezone=section.str_("timezone"),
            bar_label=BarLabel(label),
            filename_template=section.str_("filename_template", "{symbol}_{frequency}.csv"),
        )
    if kind == "synthetic":
        section = node.optional_section("synthetic")
        section.reject_unknown(
            ("seed", "start_price", "annual_drift", "annual_volatility", "splits", "dividends")
        )
        from .providers.synthetic import SyntheticSpec

        if frequency is not Frequency.DAY_1:
            raise ConfigError("La source synthetique ne genere que du journalier")
        return SyntheticProvider(
            SyntheticSpec(
                seed=section.int_("seed", 20240101),
                start_price=section.float_("start_price", 100.0),
                annual_drift=section.float_("annual_drift", 0.08),
                annual_volatility=section.float_("annual_volatility", 0.20),
                splits=_parse_events(section, "splits"),
                dividends=_parse_events(section, "dividends"),
            )
        )
    raise ConfigError(f"Source inconnue : {kind}")  # pragma: no cover - enum_ filtre deja

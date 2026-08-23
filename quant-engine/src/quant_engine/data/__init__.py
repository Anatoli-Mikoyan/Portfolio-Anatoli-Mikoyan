"""Couche donnees : ingestion, normalisation, cache, protection anti-look-ahead.

Regle d'usage a retenir : une strategie ne manipule que des ``HistoryView``.
Le type ``MarketData`` contient le futur et appartient au moteur seul.
"""

from __future__ import annotations

from .adjustment import AdjustmentPolicy, Multipliers, build_multipliers
from .cache import CachedProvider, CacheEntry, ParquetCache
from .calendar import AlwaysOpenCalendar, TradingCalendar, XNYSCalendar, get_calendar
from .corporate_actions import CorporateActions, Dividend, Split
from .dataset import BarCursor, DecisionPoint, HistoryView, MarketData
from .loader import DataLoader
from .normalize import NormalizationPolicy, normalize
from .providers import (
    CsvProvider,
    DataProvider,
    DataRequest,
    RawSeries,
    SyntheticProvider,
    SyntheticSpec,
    YFinanceProvider,
)
from .quality import (
    DataQualityReport,
    Finding,
    FindingKind,
    QualityPolicy,
    Severity,
    run_quality_checks,
)
from .types import OHLCV_FIELDS, UTC, Bar, BarLabel, Field, Frequency, ensure_utc

__all__ = [
    "OHLCV_FIELDS",
    "UTC",
    "AdjustmentPolicy",
    "AlwaysOpenCalendar",
    "Bar",
    "BarCursor",
    "BarLabel",
    "CacheEntry",
    "CachedProvider",
    "CorporateActions",
    "CsvProvider",
    "DataLoader",
    "DataProvider",
    "DataQualityReport",
    "DataRequest",
    "DecisionPoint",
    "Dividend",
    "Field",
    "Finding",
    "FindingKind",
    "Frequency",
    "HistoryView",
    "MarketData",
    "Multipliers",
    "NormalizationPolicy",
    "ParquetCache",
    "QualityPolicy",
    "RawSeries",
    "Severity",
    "Split",
    "SyntheticProvider",
    "SyntheticSpec",
    "TradingCalendar",
    "XNYSCalendar",
    "YFinanceProvider",
    "build_multipliers",
    "ensure_utc",
    "get_calendar",
    "normalize",
    "run_quality_checks",
]

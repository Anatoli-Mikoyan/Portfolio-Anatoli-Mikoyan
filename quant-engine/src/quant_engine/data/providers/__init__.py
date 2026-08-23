"""Sources de donnees interchangeables."""

from __future__ import annotations

from .base import DataProvider, DataRequest, RawSeries
from .csv_provider import CsvProvider
from .synthetic import SyntheticProvider, SyntheticSpec
from .yfinance_provider import YFinanceProvider

__all__ = [
    "CsvProvider",
    "DataProvider",
    "DataRequest",
    "RawSeries",
    "SyntheticProvider",
    "SyntheticSpec",
    "YFinanceProvider",
]

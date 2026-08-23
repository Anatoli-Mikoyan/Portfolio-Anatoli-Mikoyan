"""Adaptateur yfinance.

Trois precautions non negociables, appliquees ici et nulle part ailleurs :

``auto_adjust=False``
    Par defaut, yfinance renvoie des prix retro-ajustes avec toutes les
    operations sur titre connues *aujourd'hui*. Le close d'Apple au 2 janvier
    2015 y apparait vers 24 $ alors qu'il cotait 109 $ : la serie integre le
    split 4-pour-1 de 2020. C'est du look-ahead pur, et il est invisible.
    On demande les prix bruts et on ajuste au curseur (``adjustment.py``).

``actions=True``
    Splits et dividendes sont recuperes separement pour reconstruire
    l'ajustement point-in-time.

``BarLabel.SESSION_DATE``
    En journalier, l'index yfinance est la date de seance a minuit, pas un
    instant. Le normaliseur la projette sur la cloture reelle.

Limites que le code ne peut pas corriger
----------------------------------------
* **Biais du survivant.** yfinance n'expose que les titres encore cotes. Un
  backtest sur un panier choisi aujourd'hui exclut mecaniquement les faillites :
  le rendement en ressort surestime, souvent de plusieurs points par an.
* **Revisions silencieuses.** Les series sont regulierement corrigees a
  posteriori sans versionnage. Deux backtests identiques a six mois d'ecart
  peuvent differer. Le cache Parquet horodate est le seul garde-fou : il fige
  ce qui a reellement servi.
* **Pas de point-in-time sur les univers.** Reconstituer la composition
  historique d'un indice est impossible avec cette source.

Ces limites sont acceptables pour prototyper, jamais pour valider un capital.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, final

import pandas as pd

from ...errors import ProviderError
from ...logging_setup import get_logger
from ..corporate_actions import CorporateActions, Dividend, Split
from ..types import BarLabel, Frequency
from .base import DataProvider, DataRequest, RawSeries

__all__ = ["YFinanceProvider"]

_LOG = get_logger("data.providers.yfinance")

_INTERVAL = {
    Frequency.MINUTE_1: "1m",
    Frequency.MINUTE_5: "5m",
    Frequency.MINUTE_15: "15m",
    Frequency.MINUTE_30: "30m",
    Frequency.HOUR_1: "1h",
    Frequency.DAY_1: "1d",
    Frequency.WEEK_1: "1wk",
}


@final
class YFinanceProvider(DataProvider):
    """Source Yahoo Finance via ``yfinance``."""

    def __init__(self, *, exchange_timezone: str = "America/New_York") -> None:
        self.name = "yfinance"
        self.exchange_timezone = exchange_timezone

    def supports(self, frequency: Frequency) -> bool:
        return frequency in _INTERVAL

    def fetch(self, request: DataRequest) -> RawSeries:
        try:
            import yfinance
        except ImportError as exc:  # pragma: no cover - dependance optionnelle
            raise ProviderError(
                "yfinance n'est pas installe. `pip install 'quant-engine[yfinance]'`"
            ) from exc

        if not self.supports(request.frequency):
            raise ProviderError(f"Frequence {request.frequency.value} non supportee par yfinance")

        ticker = yfinance.Ticker(request.symbol)
        try:
            frame: pd.DataFrame = ticker.history(
                start=request.start.date().isoformat(),
                # yfinance exclut la borne de fin : on l'ouvre d'un jour.
                end=(request.end.date() + timedelta(days=1)).isoformat(),
                interval=_INTERVAL[request.frequency],
                auto_adjust=False,  # NE JAMAIS passer a True : cf. docstring du module
                back_adjust=False,
                actions=True,
                raise_errors=True,
            )
        except Exception as exc:
            raise ProviderError(f"Echec yfinance pour {request.symbol} : {exc}") from exc

        if frame is None or frame.empty:
            raise ProviderError(
                f"yfinance n'a renvoye aucune donnee pour {request.symbol} "
                f"entre {request.start.date()} et {request.end.date()}"
            )

        frame = frame.rename(columns={str(c): str(c).strip().lower() for c in frame.columns})
        actions = _extract_actions(frame)
        missing = [c for c in ("open", "high", "low", "close", "volume") if c not in frame.columns]
        if missing:
            raise ProviderError(f"Colonnes absentes du retour yfinance : {missing}")
        ohlcv = frame.loc[:, ["open", "high", "low", "close", "volume"]]

        bar_label = (
            BarLabel.SESSION_DATE
            if request.frequency in (Frequency.DAY_1, Frequency.WEEK_1)
            else BarLabel.OPEN
        )
        index = pd.DatetimeIndex(ohlcv.index)
        timezone = str(index.tz) if index.tz is not None else self.exchange_timezone

        _LOG.info(
            "serie recuperee",
            extra={
                "symbol": request.symbol,
                "rows": len(ohlcv),
                "splits": len(actions.splits),
                "dividends": len(actions.dividends),
                "bar_label": bar_label.value,
            },
        )
        return RawSeries(
            symbol=request.symbol,
            frequency=request.frequency,
            frame=ohlcv,
            bar_label=bar_label,
            timezone=timezone,
            provider=self.name,
            actions=actions,
            is_preadjusted=False,
        )


def _extract_actions(frame: pd.DataFrame) -> CorporateActions:
    """Reconstruit splits et dividendes depuis les colonnes d'actions."""
    splits: list[Split] = []
    dividends: list[Dividend] = []
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        index = index.tz_localize("UTC")

    def _column(name: str) -> Any:
        return frame[name] if name in frame.columns else None

    split_col = _column("stock splits")
    if split_col is not None:
        ratios = pd.to_numeric(split_col, errors="coerce").fillna(0.0)
        for stamp, ratio in zip(index, ratios, strict=True):
            value = float(ratio)
            if value > 0.0 and value != 1.0:
                splits.append(Split(stamp.to_pydatetime(), value))

    dividend_col = _column("dividends")
    if dividend_col is not None:
        amounts = pd.to_numeric(dividend_col, errors="coerce").fillna(0.0)
        for stamp, amount in zip(index, amounts, strict=True):
            value = float(amount)
            if value > 0.0:
                dividends.append(Dividend(stamp.to_pydatetime(), value))

    return CorporateActions(splits=tuple(splits), dividends=tuple(dividends))

"""Source CSV locale.

Utile pour les donnees achetees, les exports de broker et les jeux figes qui
servent de reference de non-regression. Le contrat est le meme que pour une
source distante : le fichier doit declarer -- via la configuration -- son
fuseau et sa convention de labellisation. Deviner ces deux informations est
exactement la facon dont un decalage d'une seance s'installe sans bruit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, final

import pandas as pd

from ...errors import ProviderError
from ..corporate_actions import CorporateActions, Dividend, Split
from ..types import BarLabel, Frequency, ensure_utc
from .base import DataProvider, DataRequest, RawSeries

__all__ = ["CsvProvider"]

_OHLCV: Final = ("open", "high", "low", "close", "volume")

_ALIASES: Final[dict[str, str]] = {
    "date": "timestamp",
    "datetime": "timestamp",
    "time": "timestamp",
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
    "vol": "volume",
    "adj close": "adj_close",
    "adjclose": "adj_close",
    "stock splits": "splits",
}


@final
class CsvProvider(DataProvider):
    """Lit ``{root}/{SYMBOLE}_{frequence}.csv``.

    Colonnes attendues (insensible a la casse) : timestamp, open, high, low,
    close, volume. Colonnes facultatives : dividends, splits.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        timezone: str,
        bar_label: BarLabel,
        filename_template: str = "{symbol}_{frequency}.csv",
    ) -> None:
        self.name = "csv"
        self.root = Path(root)
        self.timezone = timezone
        self.bar_label = bar_label
        self.filename_template = filename_template

    def _path(self, request: DataRequest) -> Path:
        return self.root / self.filename_template.format(
            symbol=request.symbol.upper(), frequency=request.frequency.value
        )

    def fetch(self, request: DataRequest) -> RawSeries:
        path = self._path(request)
        if not path.is_file():
            raise ProviderError(f"Fichier introuvable : {path}")
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            raise ProviderError(f"Lecture impossible de {path} : {exc}") from exc

        frame.columns = [
            _ALIASES.get(str(c).strip().lower(), str(c).strip().lower()) for c in frame.columns
        ]
        if "timestamp" not in frame.columns:
            raise ProviderError(
                f"{path} : aucune colonne temporelle reconnue. "
                f"Colonnes lues : {list(frame.columns)}"
            )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        if frame["timestamp"].isna().any():
            bad = int(frame["timestamp"].isna().sum())
            raise ProviderError(f"{path} : {bad} timestamps illisibles")
        frame = frame.set_index("timestamp").sort_index()

        actions = _extract_actions(frame, self.timezone)
        window = frame.loc[
            (frame.index >= _naive_or_aware(request.start, frame))
            & (frame.index <= _naive_or_aware(request.end, frame))
        ]
        if window.empty:
            raise ProviderError(
                f"{path} : aucune ligne entre {request.start.date()} et {request.end.date()}"
            )
        # Seules les colonnes canoniques sortent d'ici. En particulier,
        # "adj_close" est ecartee : c'est une serie retro-ajustee, contaminee
        # par des operations sur titre posterieures a chaque barre.
        missing = [name for name in _OHLCV if name not in window.columns]
        if missing:
            raise ProviderError(f"{path} : colonnes OHLCV manquantes {missing}")
        window = window.loc[:, list(_OHLCV)]

        return RawSeries(
            symbol=request.symbol,
            frequency=request.frequency,
            frame=window,
            bar_label=self.bar_label,
            timezone=self.timezone,
            provider=self.name,
            actions=actions,
            is_preadjusted=False,
        )

    def supports(self, frequency: Frequency) -> bool:  # noqa: ARG002
        """Un CSV peut contenir n'importe quel pas ; c'est au fichier de dire."""
        return True


def _naive_or_aware(moment: object, frame: pd.DataFrame) -> pd.Timestamp:
    """Aligne la borne de filtrage sur la nature (naive/aware) de l'index."""
    stamp = pd.Timestamp(ensure_utc(moment))  # type: ignore[arg-type]
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        return stamp.tz_localize(None)
    return stamp.tz_convert(index.tz)


def _extract_actions(frame: pd.DataFrame, timezone: str) -> CorporateActions:
    splits: list[Split] = []
    dividends: list[Dividend] = []
    index = pd.DatetimeIndex(frame.index)
    localized = index.tz_localize(timezone) if index.tz is None else index

    if "splits" in frame.columns:
        series = pd.to_numeric(frame["splits"], errors="coerce").fillna(0.0)
        for stamp, ratio in zip(localized, series.to_numpy(), strict=True):
            if float(ratio) > 0.0 and float(ratio) != 1.0:
                splits.append(Split(stamp.to_pydatetime(), float(ratio)))
    if "dividends" in frame.columns:
        series = pd.to_numeric(frame["dividends"], errors="coerce").fillna(0.0)
        for stamp, amount in zip(localized, series.to_numpy(), strict=True):
            if float(amount) > 0.0:
                dividends.append(Dividend(stamp.to_pydatetime(), float(amount)))
    return CorporateActions(splits=tuple(splits), dividends=tuple(dividends))

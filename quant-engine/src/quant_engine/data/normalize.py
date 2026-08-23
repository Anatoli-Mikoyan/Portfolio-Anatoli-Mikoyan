"""Normalisation : ``RawSeries`` (heterogene) -> ``MarketData`` (canonique).

Point de passage oblige entre le monde exterieur et le moteur. Tout ce qui
entre est ici recale, valide, et documente. Trois responsabilites :

1. **Recalage temporel.** Chaque barre recoit le timestamp UTC de sa cloture
   reelle. C'est la que se joue la difference entre un backtest juste et un
   backtest qui dispose d'une seance d'avance sans que personne ne s'en apercoive.
2. **Reparation explicite.** Doublons, NaN, OHLC incoherent : chaque traitement
   est pilote par une politique declaree et laisse une trace dans le rapport
   qualite. Aucune correction muette.
3. **Refus des series irrecuperables.** Une source qui livre des prix deja
   retro-ajustes est rejetee : l'information necessaire a l'ajustement
   point-in-time a ete detruite en amont et rien ne permet de la reconstruire.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Final, Literal, final

import numpy as np
import pandas as pd

from ..errors import SchemaError
from ..logging_setup import get_logger
from .calendar import TradingCalendar, get_calendar
from .dataset import MarketData
from .providers.base import RawSeries
from .quality import (
    Finding,
    FindingKind,
    QualityPolicy,
    Severity,
    run_quality_checks,
)
from .types import UTC, BarLabel, to_epoch_ns

__all__ = ["NormalizationPolicy", "normalize"]

_LOG = get_logger("data.normalize")
_COLUMNS: Final = ("open", "high", "low", "close", "volume")

OnDuplicate = Literal["last", "first", "fail"]
OnInvalid = Literal["drop", "fail", "ffill"]


@final
@dataclass(frozen=True, slots=True)
class NormalizationPolicy:
    """Comportement du normaliseur face aux donnees imparfaites."""

    calendar: str = "XNYS"
    on_duplicate: OnDuplicate = "last"
    on_nan: OnInvalid = "drop"
    on_incoherent_ohlc: OnInvalid = "fail"
    drop_zero_volume: bool = False
    drop_incomplete_last_bar: bool = True
    """Ecarte la derniere barre si sa cloture est posterieure a maintenant.
    Une barre du jour en cours est partielle : son high, son low et son close
    changeront encore. La backtester revient a connaitre la fin de la seance."""
    allow_preadjusted: bool = False
    raise_on_blocking: bool = True
    quality: QualityPolicy = field(default_factory=QualityPolicy)


def normalize(
    raw: RawSeries,
    policy: NormalizationPolicy | None = None,
    *,
    now: datetime | None = None,
) -> MarketData:
    """Convertit un payload de source en jeu de donnees canonique."""
    pol = policy if policy is not None else NormalizationPolicy()
    calendar = get_calendar(pol.calendar)
    repairs: list[Finding] = []

    if raw.is_preadjusted and not pol.allow_preadjusted:
        raise SchemaError(
            f"{raw.provider}/{raw.symbol} : la source livre des prix deja retro-ajustes. "
            "Les niveaux de prix historiques y sont recalcules avec des operations sur "
            "titre posterieures, ce qui constitue un look-ahead bias irrecuperable : "
            "les prix bruts ne peuvent pas etre reconstruits. Demande la serie non "
            "ajustee a la source (yfinance : auto_adjust=False)."
        )
    if raw.is_preadjusted:
        repairs.append(
            Finding(
                kind=FindingKind.PREADJUSTED_SOURCE,
                severity=Severity.WARNING,
                count=1,
                detail=(
                    "Serie deja retro-ajustee acceptee explicitement. Tout resultat "
                    "produit a partir de ces donnees est contamine par du look-ahead."
                ),
            )
        )

    frame = raw.frame.loc[:, list(_COLUMNS)].copy()
    frame = frame.astype(np.float64)
    n_input = len(frame)

    # -- 1. recalage temporel -------------------------------------------------
    frame.index = _canonical_index(raw, calendar)

    # -- 2. ordre et doublons -------------------------------------------------
    if not frame.index.is_monotonic_increasing:
        frame = frame.sort_index(kind="stable")
        _LOG.warning("index non trie, reordonne", extra={"symbol": raw.symbol})

    duplicated = frame.index.duplicated(keep=False)
    n_duplicated = int(duplicated.sum())
    if n_duplicated:
        if pol.on_duplicate == "fail":
            first = frame.index[duplicated][0]
            raise SchemaError(
                f"{raw.symbol} : {n_duplicated} timestamps dupliques (premier : {first})"
            )
        frame = frame[~frame.index.duplicated(keep=pol.on_duplicate)]
        repairs.append(
            Finding(
                kind=FindingKind.DEDUPLICATED,
                severity=Severity.WARNING,
                count=n_duplicated,
                detail=f"{n_duplicated} lignes dupliquees, resolution '{pol.on_duplicate}'",
            )
        )

    # -- 3. barre en cours ----------------------------------------------------
    if pol.drop_incomplete_last_bar and len(frame) > 0:
        reference = now if now is not None else datetime.now(tz=UTC)
        incomplete = frame.index > pd.Timestamp(reference)
        n_incomplete = int(incomplete.sum())
        if n_incomplete:
            frame = frame[~incomplete]
            repairs.append(
                Finding(
                    kind=FindingKind.TRIMMED_INCOMPLETE_BAR,
                    severity=Severity.INFO,
                    count=n_incomplete,
                    detail=(
                        f"{n_incomplete} barre(s) dont la cloture est future ecartee(s) : "
                        "une barre en cours de formation n'est pas backtestable."
                    ),
                )
            )

    # -- 4. valeurs invalides -------------------------------------------------
    frame, nan_finding = _handle_invalid(
        frame,
        mask=~np.isfinite(frame.to_numpy(dtype=np.float64)).all(axis=1)
        | (frame[["open", "high", "low", "close"]].to_numpy(dtype=np.float64) <= 0.0).any(axis=1),
        mode=pol.on_nan,
        kind=FindingKind.DROPPED_ROWS,
        label="valeur non finie ou prix nul/negatif",
        symbol=raw.symbol,
    )
    if nan_finding is not None:
        repairs.append(nan_finding)

    values = frame.to_numpy(dtype=np.float64)
    open_, high, low, close = values[:, 0], values[:, 1], values[:, 2], values[:, 3]
    incoherent = ~(
        (low <= np.minimum(open_, close)) & (np.maximum(open_, close) <= high) & (low <= high)
    )
    frame, ohlc_finding = _handle_invalid(
        frame,
        mask=incoherent,
        mode=pol.on_incoherent_ohlc,
        kind=FindingKind.DROPPED_ROWS,
        label="OHLC incoherent",
        symbol=raw.symbol,
    )
    if ohlc_finding is not None:
        repairs.append(ohlc_finding)

    if pol.drop_zero_volume:
        zero = frame["volume"].to_numpy(dtype=np.float64) == 0.0
        n_zero = int(zero.sum())
        if n_zero:
            frame = frame[~zero]
            repairs.append(
                Finding(
                    kind=FindingKind.DROPPED_ROWS,
                    severity=Severity.WARNING,
                    count=n_zero,
                    detail=f"{n_zero} barres a volume nul ecartees (inexecutables)",
                )
            )

    if frame.empty:
        raise SchemaError(
            f"{raw.symbol} : aucune barre exploitable apres normalisation "
            f"({n_input} lignes en entree). Verifie la source et la politique."
        )

    # -- 5. materialisation ---------------------------------------------------
    timestamps = to_epoch_ns(pd.DatetimeIndex(frame.index))
    columns = {name: frame[name].to_numpy(dtype=np.float64) for name in _COLUMNS}

    report = run_quality_checks(
        symbol=raw.symbol,
        frequency=raw.frequency,
        timestamps=timestamps,
        open_=columns["open"],
        high=columns["high"],
        low=columns["low"],
        close=columns["close"],
        volume=columns["volume"],
        actions=raw.actions,
        calendar=calendar,
        policy=pol.quality,
    ).merged_with(repairs)

    _LOG.info(
        "normalisation terminee",
        extra={
            "symbol": raw.symbol,
            "provider": raw.provider,
            "rows_in": n_input,
            "rows_out": len(frame),
            "blocking": len(report.blocking),
        },
    )
    if pol.raise_on_blocking:
        report.raise_if_blocking()

    return MarketData(
        symbol=raw.symbol,
        frequency=raw.frequency,
        timestamps=timestamps,
        open_=columns["open"],
        high=columns["high"],
        low=columns["low"],
        close=columns["close"],
        volume=columns["volume"],
        actions=raw.actions,
        quality=report,
        provider=raw.provider,
        calendar=pol.calendar,
    )


def _canonical_index(raw: RawSeries, calendar: TradingCalendar) -> pd.DatetimeIndex:
    """Projette l'index de la source sur des timestamps de cloture UTC.

    C'est la fonction la plus importante du module. Une barre journaliere
    yfinance arrive datee ``2024-03-15 00:00 America/New_York`` ; la laisser
    telle quelle signifie qu'a minuit, avant l'ouverture, le moteur connait deja
    le close du soir.
    """
    index = pd.DatetimeIndex(raw.frame.index)
    if index.tz is None:
        try:
            index = index.tz_localize(raw.timezone)
        except Exception as exc:  # pragma: no cover - depend de zoneinfo
            raise SchemaError(
                f"{raw.symbol} : localisation impossible dans {raw.timezone!r} ({exc}). "
                "Une source doit declarer son fuseau ; un index naif est ambigu "
                "deux fois par an au changement d'heure."
            ) from exc

    if raw.bar_label is BarLabel.SESSION_DATE:
        local = index.tz_convert(calendar.timezone)
        closes = [calendar.session_close_utc(stamp.date()) for stamp in local]
        return pd.DatetimeIndex(closes, name="timestamp").tz_convert("UTC").as_unit("ns")

    index = index.tz_convert("UTC")
    if raw.bar_label is BarLabel.OPEN:
        index = index + pd.Timedelta(raw.frequency.delta)
    return pd.DatetimeIndex(index, name="timestamp").as_unit("ns")


def _handle_invalid(
    frame: pd.DataFrame,
    *,
    mask: np.ndarray,
    mode: OnInvalid,
    kind: FindingKind,
    label: str,
    symbol: str,
) -> tuple[pd.DataFrame, Finding | None]:
    count = int(np.asarray(mask).sum())
    if count == 0:
        return frame, None
    if mode == "fail":
        first = frame.index[np.asarray(mask)][0]
        raise SchemaError(
            f"{symbol} : {count} barre(s) avec {label} (premiere : {first}). "
            "Choisis explicitement une politique de reparation si c'est attendu."
        )
    if mode == "ffill":
        repaired = frame.copy()
        repaired[np.asarray(mask)] = np.nan
        repaired = repaired.ffill()
        _LOG.warning(
            "forward-fill applique : cree des barres a rendement nul et sous-estime "
            "la volatilite realisee",
            extra={"symbol": symbol, "count": count, "reason": label},
        )
        return repaired.dropna(), Finding(
            kind=FindingKind.FORWARD_FILLED,
            severity=Severity.WARNING,
            count=count,
            detail=(
                f"{count} barre(s) comblees par forward-fill ({label}). Chaque barre "
                "comblee affiche un rendement nul : la volatilite mesuree est biaisee "
                "vers le bas et tout ratio de Sharpe calcule dessus est surestime."
            ),
        )
    kept = frame[~np.asarray(mask)]
    return kept, Finding(
        kind=kind,
        severity=Severity.WARNING,
        count=count,
        detail=f"{count} barre(s) ecartee(s) : {label}",
    )

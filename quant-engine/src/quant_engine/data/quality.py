"""Controle qualite des series OHLCV.

Position de principe : une anomalie de donnees n'est jamais corrigee en
silence. Chaque anomalie est detectee, classee, comptee, et le rapport voyage
avec le jeu de donnees jusque dans le rapport de backtest final. Un backtest
tourne sur une serie a 4 % de barres manquantes n'est pas faux -- il est
*invalide*, et le lecteur doit le savoir sans avoir a fouiller.

Sur le forward-fill : combler un trou de prix par report de la derniere valeur
cree une barre a rendement exactement nul. Sur une serie qui en compte
beaucoup, cela reduit mecaniquement la volatilite mesuree et gonfle le Sharpe
sans qu'aucun signal d'alerte ne se declenche. La politique par defaut est donc
``drop``, et ``ffill`` emet un avertissement bruyant.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from fractions import Fraction
from typing import TYPE_CHECKING, Final, final

import numpy as np
from numpy.typing import NDArray

from ..errors import DataQualityError
from ..logging_setup import get_logger
from .types import UTC, Frequency

if TYPE_CHECKING:
    from .calendar import TradingCalendar
    from .corporate_actions import CorporateActions

__all__ = [
    "DataQualityReport",
    "Finding",
    "FindingKind",
    "QualityPolicy",
    "Severity",
    "run_quality_checks",
]

_LOG = get_logger("data.quality")
_NS: Final = 1_000_000_000


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    BLOCKING = "blocking"


class FindingKind(Enum):
    """Taxonomie fermee des anomalies detectables."""

    DUPLICATE_TIMESTAMP = "duplicate_timestamp"
    UNSORTED_TIMESTAMPS = "unsorted_timestamps"
    NAN_VALUE = "nan_value"
    NON_POSITIVE_PRICE = "non_positive_price"
    NEGATIVE_VOLUME = "negative_volume"
    ZERO_VOLUME = "zero_volume"
    OHLC_INCOHERENT = "ohlc_incoherent"
    MISSING_SESSION = "missing_session"
    UNEXPECTED_SESSION = "unexpected_session"
    STALE_BAR = "stale_bar"
    PRICE_JUMP_UNEXPLAINED = "price_jump_unexplained"
    SUSPECTED_UNDECLARED_SPLIT = "suspected_undeclared_split"
    SPLIT_APPLIED = "split_applied"
    DIVIDEND_APPLIED = "dividend_applied"
    SHORT_HISTORY = "short_history"
    DROPPED_ROWS = "dropped_rows"
    DEDUPLICATED = "deduplicated"
    FORWARD_FILLED = "forward_filled"
    TRIMMED_INCOMPLETE_BAR = "trimmed_incomplete_bar"
    PREADJUSTED_SOURCE = "preadjusted_source"


_DEFAULT_SEVERITY: Final[Mapping[FindingKind, Severity]] = {
    FindingKind.DUPLICATE_TIMESTAMP: Severity.BLOCKING,
    FindingKind.UNSORTED_TIMESTAMPS: Severity.BLOCKING,
    FindingKind.NAN_VALUE: Severity.BLOCKING,
    FindingKind.NON_POSITIVE_PRICE: Severity.BLOCKING,
    FindingKind.NEGATIVE_VOLUME: Severity.BLOCKING,
    FindingKind.OHLC_INCOHERENT: Severity.BLOCKING,
    FindingKind.SUSPECTED_UNDECLARED_SPLIT: Severity.BLOCKING,
    FindingKind.MISSING_SESSION: Severity.WARNING,
    FindingKind.UNEXPECTED_SESSION: Severity.WARNING,
    FindingKind.ZERO_VOLUME: Severity.WARNING,
    FindingKind.STALE_BAR: Severity.WARNING,
    FindingKind.PRICE_JUMP_UNEXPLAINED: Severity.WARNING,
    FindingKind.SHORT_HISTORY: Severity.WARNING,
    FindingKind.SPLIT_APPLIED: Severity.INFO,
    FindingKind.DIVIDEND_APPLIED: Severity.INFO,
    FindingKind.DROPPED_ROWS: Severity.WARNING,
    FindingKind.DEDUPLICATED: Severity.WARNING,
    FindingKind.FORWARD_FILLED: Severity.WARNING,
    FindingKind.TRIMMED_INCOMPLETE_BAR: Severity.INFO,
    FindingKind.PREADJUSTED_SOURCE: Severity.BLOCKING,
}


@final
@dataclass(frozen=True, slots=True)
class Finding:
    """Une anomalie localisee."""

    kind: FindingKind
    severity: Severity
    count: int
    detail: str
    first_timestamp: datetime | None = None
    sample_indices: tuple[int, ...] = ()

    def __str__(self) -> str:
        where = f" a partir de {self.first_timestamp.date()}" if self.first_timestamp else ""
        return (
            f"[{self.severity.value.upper():8}] {self.kind.value} "
            f"x{self.count}{where} : {self.detail}"
        )


@final
@dataclass(frozen=True, slots=True)
class QualityPolicy:
    """Parametrage des seuils et de la severite des controles."""

    severity_overrides: Mapping[FindingKind, Severity] = field(default_factory=dict)
    jump_sigma: float = 8.0
    """Seuil de detection d'un saut de prix, en ecarts-types robustes (MAD)."""
    jump_floor: float = 0.20
    """Plancher absolu du seuil, en log-rendement. Evite les faux positifs sur
    les series a tres faible volatilite ou le MAD est quasi nul."""
    stale_run_length: int = 3
    """Nombre de barres OHLC strictement identiques consecutives declenchant
    une alerte de barre figee."""
    min_bars: int = 252
    """En dessous, l'echantillon est signale comme trop court."""
    max_missing_session_ratio: float = 0.02
    """Au dela, les seances manquantes deviennent bloquantes."""
    check_sessions: bool = True

    def severity_for(self, kind: FindingKind) -> Severity:
        return self.severity_overrides.get(kind, _DEFAULT_SEVERITY[kind])


@final
@dataclass(frozen=True, slots=True)
class DataQualityReport:
    """Verdict de qualite attache a un jeu de donnees."""

    symbol: str
    n_bars: int
    findings: tuple[Finding, ...] = ()

    @property
    def blocking(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.BLOCKING)

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.WARNING)

    @property
    def is_clean(self) -> bool:
        return not self.blocking and not self.warnings

    def merged_with(self, extra: Sequence[Finding]) -> DataQualityReport:
        """Nouveau rapport incluant des constats produits hors des detecteurs
        (typiquement les reparations appliquees par le normaliseur)."""
        return DataQualityReport(
            symbol=self.symbol,
            n_bars=self.n_bars,
            findings=tuple(extra) + self.findings,
        )

    def of_kind(self, kind: FindingKind) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.kind is kind)

    def has(self, kind: FindingKind) -> bool:
        return any(f.kind is kind for f in self.findings)

    def raise_if_blocking(self) -> None:
        blocking = self.blocking
        if blocking:
            lines = "\n  ".join(str(f) for f in blocking)
            raise DataQualityError(
                f"{self.symbol} : {len(blocking)} anomalie(s) bloquante(s).\n  {lines}"
            )

    def summary(self) -> str:
        if not self.findings:
            return f"{self.symbol} : {self.n_bars} barres, aucune anomalie detectee."
        lines = [f"{self.symbol} : {self.n_bars} barres"]
        lines.extend(f"  {finding}" for finding in self.findings)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """Serialisation pour le rapport HTML et les logs structures."""
        return {
            "symbol": self.symbol,
            "n_bars": self.n_bars,
            "findings": [
                {
                    "kind": f.kind.value,
                    "severity": f.severity.value,
                    "count": f.count,
                    "detail": f.detail,
                    "first_timestamp": (
                        f.first_timestamp.isoformat() if f.first_timestamp else None
                    ),
                }
                for f in self.findings
            ],
        }


# ---------------------------------------------------------------------------
# Detecteurs
# ---------------------------------------------------------------------------
def _first_ts(timestamps: NDArray[np.int64], indices: NDArray[np.int64]) -> datetime | None:
    if indices.size == 0:
        return None
    return datetime.fromtimestamp(int(timestamps[int(indices[0])]) / _NS, tz=UTC)


def _looks_like_split(ratio: float, *, tolerance: float = 0.02) -> Fraction | None:
    """Le rapport de prix correspond-il a un split a petits entiers ?

    Un cours qui passe de 500 a 125 du jour au lendemain est presque surement un
    split 4-pour-1 non declare, pas un krach de -75 %. Confondre les deux
    fabrique un rendement de -75 % qui n'a jamais eu lieu -- et c'est une des
    facons les plus courantes de "decouvrir" une strategie short miraculeuse.

    La liste ne retient que des ratios **non ambigus** : au moins un doublement
    ou une division par deux. Les splits 3-pour-2 ou 5-pour-4 existent bel et
    bien, mais un rapport de 0,67 est tout aussi compatible avec une seance a
    -33 % parfaitement reelle. Les inclure transformerait chaque krach en
    anomalie bloquante ; ces cas sont donc laisses en avertissement
    (``PRICE_JUMP_UNEXPLAINED``), a charge de l'operateur de verifier. Un
    arbitrage assume : on prefere rater un split mineur non declare -- que la
    source declare de toute facon presque toujours -- plutot que de bloquer sur
    un mouvement de marche authentique.
    """
    if not np.isfinite(ratio) or ratio <= 0.0:
        return None
    for candidate in (
        Fraction(1, 2), Fraction(1, 3), Fraction(1, 4), Fraction(1, 5),
        Fraction(1, 6), Fraction(1, 7), Fraction(1, 8), Fraction(1, 10),
        Fraction(1, 15), Fraction(1, 20), Fraction(1, 30), Fraction(1, 50),
        Fraction(2, 1), Fraction(3, 1), Fraction(4, 1), Fraction(5, 1),
        Fraction(6, 1), Fraction(8, 1), Fraction(10, 1), Fraction(20, 1),
    ):
        if abs(ratio - float(candidate)) <= tolerance * float(candidate):
            return candidate
    return None


def run_quality_checks(
    *,
    symbol: str,
    frequency: Frequency,
    timestamps: NDArray[np.int64],
    open_: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    actions: CorporateActions | None = None,
    calendar: TradingCalendar | None = None,
    policy: QualityPolicy | None = None,
) -> DataQualityReport:
    """Passe l'integralite des controles et retourne un rapport."""
    pol = policy if policy is not None else QualityPolicy()
    findings: list[Finding] = []
    n = int(timestamps.size)

    def add(
        kind: FindingKind,
        count: int,
        detail: str,
        indices: NDArray[np.int64] | Sequence[int] = (),
    ) -> None:
        if count <= 0:
            return
        idx = np.asarray(indices, dtype=np.int64)
        findings.append(
            Finding(
                kind=kind,
                severity=pol.severity_for(kind),
                count=count,
                detail=detail,
                first_timestamp=_first_ts(timestamps, idx),
                sample_indices=tuple(int(i) for i in idx[:10]),
            )
        )

    # -- integrite de l'index temporel --------------------------------------
    if n > 1:
        deltas = np.diff(timestamps)
        dup = np.flatnonzero(deltas == 0)
        add(FindingKind.DUPLICATE_TIMESTAMP, int(dup.size), "timestamps repetes", dup)
        unsorted_idx = np.flatnonzero(deltas < 0)
        add(
            FindingKind.UNSORTED_TIMESTAMPS,
            int(unsorted_idx.size),
            "index temporel non monotone",
            unsorted_idx,
        )

    # -- valeurs --------------------------------------------------------------
    prices = np.vstack([open_, high, low, close])
    nan_mask = ~np.isfinite(prices).all(axis=0) | ~np.isfinite(volume)
    nan_idx = np.flatnonzero(nan_mask)
    add(FindingKind.NAN_VALUE, int(nan_idx.size), "NaN ou infini dans OHLCV", nan_idx)

    finite = np.isfinite(prices).all(axis=0)
    nonpos_idx = np.flatnonzero(finite & (prices <= 0.0).any(axis=0))
    add(FindingKind.NON_POSITIVE_PRICE, int(nonpos_idx.size), "prix nul ou negatif", nonpos_idx)

    neg_vol_idx = np.flatnonzero(np.isfinite(volume) & (volume < 0.0))
    add(FindingKind.NEGATIVE_VOLUME, int(neg_vol_idx.size), "volume negatif", neg_vol_idx)

    zero_vol_idx = np.flatnonzero(np.isfinite(volume) & (volume == 0.0))
    add(
        FindingKind.ZERO_VOLUME,
        int(zero_vol_idx.size),
        "volume nul : barre probablement non negociee, ordre inexecutable",
        zero_vol_idx,
    )

    coherent = (
        (low <= np.minimum(open_, close))
        & (np.maximum(open_, close) <= high)
        & (low <= high)
    )
    incoherent_idx = np.flatnonzero(finite & ~coherent)
    add(
        FindingKind.OHLC_INCOHERENT,
        int(incoherent_idx.size),
        "violation de low <= min(O,C) <= max(O,C) <= high",
        incoherent_idx,
    )

    # -- barres figees --------------------------------------------------------
    if n > pol.stale_run_length:
        same = (
            (open_[1:] == open_[:-1])
            & (high[1:] == high[:-1])
            & (low[1:] == low[:-1])
            & (close[1:] == close[:-1])
        )
        run = 0
        stale: list[int] = []
        for i, identical in enumerate(same):
            run = run + 1 if identical else 0
            if run >= pol.stale_run_length:
                stale.append(i + 1)
        add(
            FindingKind.STALE_BAR,
            len(stale),
            f"OHLC identique sur >= {pol.stale_run_length} barres consecutives "
            "(cotation suspendue, ou trou comble par forward-fill en amont)",
            stale,
        )

    # -- sauts de prix --------------------------------------------------------
    if n > 3:
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ret = np.diff(np.log(np.where(close > 0.0, close, np.nan)))
        valid = np.isfinite(log_ret)
        if valid.sum() > 10:
            sample = log_ret[valid]
            mad = float(np.median(np.abs(sample - np.median(sample))))
            threshold = max(pol.jump_sigma * 1.4826 * mad, pol.jump_floor)
            jump_idx = np.flatnonzero(valid & (np.abs(log_ret) > threshold)) + 1

            known_split_indices = _known_split_indices(timestamps, actions)
            unexplained: list[int] = []
            suspected: list[int] = []
            for idx in jump_idx.tolist():
                if idx in known_split_indices:
                    continue  # saut explique par un split declare
                ratio = float(close[idx] / close[idx - 1])
                if _looks_like_split(ratio) is not None:
                    suspected.append(idx)
                else:
                    unexplained.append(idx)
            add(
                FindingKind.SUSPECTED_UNDECLARED_SPLIT,
                len(suspected),
                f"saut de prix proche d'un ratio entier simple sans split declare "
                f"(seuil {threshold:.1%}) : la serie est probablement incoherente",
                suspected,
            )
            add(
                FindingKind.PRICE_JUMP_UNEXPLAINED,
                len(unexplained),
                f"variation superieure a {threshold:.1%} en une barre, "
                "sans operation sur titre correspondante",
                unexplained,
            )

    # -- couverture du calendrier --------------------------------------------
    if calendar is not None and pol.check_sessions and n > 0 and frequency is Frequency.DAY_1:
        observed_days = {
            datetime.fromtimestamp(int(ts) / _NS, tz=UTC)
            .astimezone(calendar.timezone)
            .date()
            for ts in timestamps
        }
        first_day = min(observed_days)
        last_day = max(observed_days)
        expected_days = set(calendar.sessions(first_day, last_day))
        missing = sorted(expected_days - observed_days)
        unexpected = sorted(observed_days - expected_days)
        if missing:
            ratio = len(missing) / max(1, len(expected_days))
            severity = pol.severity_for(FindingKind.MISSING_SESSION)
            if ratio > pol.max_missing_session_ratio:
                severity = Severity.BLOCKING
            findings.append(
                Finding(
                    kind=FindingKind.MISSING_SESSION,
                    severity=severity,
                    count=len(missing),
                    detail=(
                        f"{len(missing)} seance(s) attendues absentes ({ratio:.2%} "
                        f"de {len(expected_days)}), calendrier {calendar.name}. "
                        f"Premieres : {[d.isoformat() for d in missing[:5]]}"
                    ),
                    first_timestamp=datetime.combine(missing[0], datetime.min.time(), tzinfo=UTC),
                )
            )
        if unexpected:
            findings.append(
                Finding(
                    kind=FindingKind.UNEXPECTED_SESSION,
                    severity=pol.severity_for(FindingKind.UNEXPECTED_SESSION),
                    count=len(unexpected),
                    detail=(
                        f"{len(unexpected)} barre(s) sur des jours non ouvres selon "
                        f"{calendar.name} : {[d.isoformat() for d in unexpected[:5]]}. "
                        "Calendrier inadapte au titre, ou donnee fabriquee."
                    ),
                    first_timestamp=datetime.combine(
                        unexpected[0], datetime.min.time(), tzinfo=UTC
                    ),
                )
            )

    # -- taille d'echantillon -------------------------------------------------
    if n < pol.min_bars:
        findings.append(
            Finding(
                kind=FindingKind.SHORT_HISTORY,
                severity=pol.severity_for(FindingKind.SHORT_HISTORY),
                count=n,
                detail=(
                    f"{n} barres seulement (< {pol.min_bars}). Toute metrique de "
                    "performance calculee sur cet echantillon aura un intervalle de "
                    "confiance trop large pour conclure."
                ),
            )
        )

    # -- operations sur titre (informatif) ------------------------------------
    if actions is not None and not actions.is_empty:
        if actions.splits:
            add(
                FindingKind.SPLIT_APPLIED,
                len(actions.splits),
                f"{len(actions.splits)} split(s) declare(s) par la source",
                _known_split_indices_list(timestamps, actions),
            )
        if actions.dividends:
            findings.append(
                Finding(
                    kind=FindingKind.DIVIDEND_APPLIED,
                    severity=Severity.INFO,
                    count=len(actions.dividends),
                    detail=f"{len(actions.dividends)} dividende(s) declare(s) par la source",
                )
            )

    report = DataQualityReport(symbol=symbol, n_bars=n, findings=tuple(findings))
    _LOG.info(
        "controle qualite termine",
        extra={
            "symbol": symbol,
            "n_bars": n,
            "blocking": len(report.blocking),
            "warnings": len(report.warnings),
        },
    )
    return report


def _known_split_indices(
    timestamps: NDArray[np.int64], actions: CorporateActions | None
) -> frozenset[int]:
    return frozenset(_known_split_indices_list(timestamps, actions))


def _known_split_indices_list(
    timestamps: NDArray[np.int64], actions: CorporateActions | None
) -> list[int]:
    if actions is None or not actions.splits:
        return []
    out: list[int] = []
    for split in actions.splits:
        target = np.int64(int(split.ex_date.timestamp() * _NS))
        idx = int(np.searchsorted(timestamps, target, side="left"))
        # Tolerance d'une barre : les ex-dates des sources sont parfois decalees
        # d'un jour selon le fuseau de publication.
        out.extend(i for i in (idx - 1, idx, idx + 1) if 0 <= i < timestamps.size)
    return sorted(set(out))

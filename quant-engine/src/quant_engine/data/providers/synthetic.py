"""Generateur de series synthetiques deterministes.

Deux usages, tous deux critiques :

* **Tests de non-regression analytiques.** Certaines series ont un resultat
  connu a la main (rampe lineaire, sinusoide, marche a rendement constant).
  Elles permettent de verifier que le moteur calcule ce qu'on croit, sans
  dependre d'un provider externe ni du reseau.
* **Injection controlee d'anomalies.** Splits, dividendes, trous, valeurs
  aberrantes, barres figees : on fabrique le defaut, on verifie qu'il est
  detecte. Attendre qu'il apparaisse dans une vraie serie n'est pas une
  strategie de test.

Aucun appel reseau, aucune dependance externe, reproductible au bit pres.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Final, final

import numpy as np
import pandas as pd

from ..calendar import TradingCalendar, get_calendar
from ..corporate_actions import CorporateActions, Dividend, Split
from ..types import UTC, BarLabel, Frequency
from .base import DataProvider, DataRequest, RawSeries

__all__ = ["SyntheticProvider", "SyntheticSpec", "constant_return_series", "linear_ramp_series"]

_TRADING_DAYS: Final = 252


@final
@dataclass(frozen=True, slots=True)
class SyntheticSpec:
    """Parametres du generateur."""

    start_price: float = 100.0
    annual_drift: float = 0.08
    annual_volatility: float = 0.20
    seed: int = 20240101
    intraday_range: float = 0.01
    """Amplitude high/low autour du close, en fraction du prix."""
    base_volume: float = 1_000_000.0
    calendar: str = "XNYS"
    splits: tuple[tuple[date, float], ...] = ()
    dividends: tuple[tuple[date, float], ...] = ()
    missing_days: tuple[date, ...] = ()
    """Seances a supprimer, pour fabriquer un trou de donnees."""
    outlier_days: tuple[tuple[date, float], ...] = ()
    """Jour -> facteur multiplicatif applique au close, pour fabriquer une aberration."""
    stale_days: tuple[date, ...] = ()
    """Jours dont l'OHLC recopie celui de la veille."""
    constant_daily_return: float | None = None
    """Si defini, remplace le processus stochastique par un rendement constant.
    Le resultat est analytiquement calculable : ``P_n = P_0 * (1+r)^n``."""
    _reserved: tuple[()] = field(default=(), repr=False)


@final
class SyntheticProvider(DataProvider):
    """Source deterministe, hors ligne."""

    def __init__(self, spec: SyntheticSpec | None = None) -> None:
        self.name = "synthetic"
        self.spec = spec if spec is not None else SyntheticSpec()

    def fetch(self, request: DataRequest) -> RawSeries:
        spec = self.spec
        calendar: TradingCalendar = get_calendar(spec.calendar)
        sessions = [
            day
            for day in calendar.sessions(request.start.date(), request.end.date())
            if day not in set(spec.missing_days)
        ]
        if not sessions:
            raise ValueError(
                f"Aucune seance entre {request.start.date()} et {request.end.date()} "
                f"pour le calendrier {spec.calendar}"
            )
        n = len(sessions)

        # -- trajectoire du close --------------------------------------------
        if spec.constant_daily_return is not None:
            growth = np.full(n, 1.0 + spec.constant_daily_return, dtype=np.float64)
            growth[0] = 1.0
            close = spec.start_price * np.cumprod(growth)
        else:
            rng = np.random.default_rng(spec.seed)
            dt = 1.0 / _TRADING_DAYS
            mu = spec.annual_drift
            sigma = spec.annual_volatility
            shocks = rng.standard_normal(n)
            log_steps = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
            log_steps[0] = 0.0
            close = spec.start_price * np.exp(np.cumsum(log_steps))

        # -- aberrations injectees --------------------------------------------
        session_index = {day: i for i, day in enumerate(sessions)}
        for day, factor in spec.outlier_days:
            if day in session_index:
                close[session_index[day]] *= factor

        # -- splits : la serie generee est "vraie", on la desajuste ------------
        # Un split divise le prix cote a partir de son ex-date. On reproduit
        # cette discontinuite pour que la source livre bien des prix BRUTS.
        split_objects: list[Split] = []
        for day, ratio in spec.splits:
            ex_date = datetime.combine(day, datetime.min.time(), tzinfo=UTC)
            split_objects.append(Split(ex_date, ratio))
            first = next((i for i, session in enumerate(sessions) if session >= day), None)
            if first is not None:
                close[first:] /= ratio

        dividend_objects = [
            Dividend(datetime.combine(day, datetime.min.time(), tzinfo=UTC), amount)
            for day, amount in spec.dividends
        ]

        # -- OHLC autour du close ---------------------------------------------
        half = spec.intraday_range / 2.0
        previous_close = np.concatenate(([close[0]], close[:-1]))
        open_ = previous_close * (1.0 + 0.0)
        high = np.maximum(open_, close) * (1.0 + half)
        low = np.minimum(open_, close) * (1.0 - half)
        volume = np.full(n, spec.base_volume, dtype=np.float64)

        for day in spec.stale_days:
            i = session_index.get(day)
            if i is not None and i > 0:
                open_[i] = open_[i - 1]
                high[i] = high[i - 1]
                low[i] = low[i - 1]
                close[i] = close[i - 1]
                volume[i] = 0.0

        index = pd.DatetimeIndex(
            [calendar.session_close_utc(day) for day in sessions], name="timestamp"
        )
        frame = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=index,
        )
        return RawSeries(
            symbol=request.symbol,
            frequency=request.frequency,
            frame=frame,
            bar_label=BarLabel.CLOSE,
            timezone="UTC",
            provider=self.name,
            actions=CorporateActions(
                splits=tuple(split_objects), dividends=tuple(dividend_objects)
            ),
            is_preadjusted=False,
        )

    def supports(self, frequency: Frequency) -> bool:
        return frequency is Frequency.DAY_1


# ---------------------------------------------------------------------------
# Raccourcis pour les tests analytiques
# ---------------------------------------------------------------------------
def constant_return_series(
    daily_return: float, n_sessions: int, *, start_price: float = 100.0, symbol: str = "CONST"
) -> RawSeries:
    """Serie a rendement journalier constant : ``P_n = P_0 (1+r)^n``.

    Reference analytique pour les tests de non-regression : le CAGR, le Sharpe
    (infini, volatilite nulle) et le drawdown (nul si r > 0) sont connus.
    """
    spec = SyntheticSpec(start_price=start_price, constant_daily_return=daily_return)
    provider = SyntheticProvider(spec)
    start = datetime(2015, 1, 1, tzinfo=UTC)
    return provider.fetch(
        DataRequest(
            symbol=symbol,
            frequency=Frequency.DAY_1,
            start=start,
            end=start + timedelta(days=int(n_sessions * 1.55) + 20),
        )
    )


def linear_ramp_series(
    n_sessions: int, *, start_price: float = 100.0, step: float = 1.0, symbol: str = "RAMP"
) -> RawSeries:
    """Serie strictement croissante de pas fixe. Buy & hold connu exactement."""
    calendar = get_calendar("XNYS")
    sessions = calendar.sessions(
        date(2015, 1, 1), date(2015, 1, 1) + timedelta(days=n_sessions * 2 + 30)
    )
    sessions = sessions[:n_sessions]
    close = start_price + step * np.arange(len(sessions), dtype=np.float64)
    open_ = np.concatenate(([close[0]], close[:-1]))
    frame = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close),
            "low": np.minimum(open_, close),
            "close": close,
            "volume": np.full(len(sessions), 1_000_000.0),
        },
        index=pd.DatetimeIndex(
            [calendar.session_close_utc(day) for day in sessions], name="timestamp"
        ),
    )
    return RawSeries(
        symbol=symbol,
        frequency=Frequency.DAY_1,
        frame=frame,
        bar_label=BarLabel.CLOSE,
        timezone="UTC",
        provider="synthetic",
    )

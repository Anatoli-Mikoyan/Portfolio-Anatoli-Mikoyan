"""Resultat d'un backtest : courbe d'equity, executions, diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..strategy.base import ParameterSet
from .orders import Fill, Order, Trade

__all__ = ["BacktestResult", "CostBreakdown"]


@final
@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Ou est passe l'argent. La ligne la plus regardee du rapport."""

    commissions: float
    market_friction: float
    """Spread et slippage cumules."""
    dividends_received: float

    @property
    def total_costs(self) -> float:
        return self.commissions + self.market_friction

    def as_pct_of(self, capital: float) -> dict[str, float]:
        if capital <= 0.0:
            return {}
        return {
            "commissions": self.commissions / capital,
            "friction_marche": self.market_friction / capital,
            "total": self.total_costs / capital,
            "dividendes": self.dividends_received / capital,
        }


@final
@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Tout ce que le backtest a produit, sans interpretation."""

    strategy_name: str
    symbol: str
    parameters: ParameterSet
    timestamps: NDArray[np.int64]
    equity: NDArray[np.float64]
    exposure: NDArray[np.float64]
    """Poids investi a la cloture de chaque barre."""
    benchmark_equity: NDArray[np.float64]
    """Valeur d'un buy & hold demarre le meme jour avec le meme capital."""
    fills: tuple[Fill, ...]
    trades: tuple[Trade, ...]
    unfilled_orders: tuple[Order, ...]
    costs: CostBreakdown
    initial_capital: float
    degrees_of_freedom: int
    search_space_size: int
    warnings: tuple[str, ...] = field(default=())

    # -- resultats bruts ------------------------------------------------------
    @property
    def final_equity(self) -> float:
        return float(self.equity[-1])

    @property
    def total_return(self) -> float:
        return self.final_equity / self.initial_capital - 1.0

    @property
    def benchmark_return(self) -> float:
        return float(self.benchmark_equity[-1]) / self.initial_capital - 1.0

    @property
    def excess_return(self) -> float:
        """Sur-performance face au buy & hold. Le seul chiffre qui compte."""
        return self.total_return - self.benchmark_return

    @property
    def beats_benchmark(self) -> bool:
        return self.excess_return > 0.0

    @property
    def n_bars(self) -> int:
        return int(self.equity.size)

    @property
    def n_trades(self) -> int:
        return sum(1 for trade in self.trades if trade.is_closed)

    @property
    def start(self) -> datetime:
        return _to_datetime(int(self.timestamps[0]))

    @property
    def end(self) -> datetime:
        return _to_datetime(int(self.timestamps[-1]))

    @property
    def years(self) -> float:
        return max((self.end - self.start).days / 365.25, 1e-9)

    @property
    def cost_drag_pct(self) -> float:
        """Couts totaux rapportes au capital initial."""
        return self.costs.total_costs / self.initial_capital

    @property
    def gross_return(self) -> float:
        """Performance qu'aurait affichee un backtest ignorant les couts.

        Approximation : on rajoute les couts au resultat final. Ce n'est pas
        exact -- sans couts la trajectoire aurait differe, donc les quantites
        aussi -- mais l'ordre de grandeur suffit a montrer ce que la friction
        a mange.
        """
        return (self.final_equity + self.costs.total_costs) / self.initial_capital - 1.0

    # -- exports --------------------------------------------------------------
    def equity_frame(self) -> pd.DataFrame:
        index = pd.DatetimeIndex(
            pd.to_datetime(self.timestamps, unit="ns", utc=True), name="timestamp"
        ).as_unit("ns")
        return pd.DataFrame(
            {
                "equity": self.equity,
                "benchmark": self.benchmark_equity,
                "exposure": self.exposure,
            },
            index=index,
        )

    def fills_frame(self) -> pd.DataFrame:
        if not self.fills:
            return pd.DataFrame(
                columns=["timestamp", "side", "quantity", "price", "reference_price",
                         "commission", "slippage_bps", "reason"]
            )
        return pd.DataFrame(
            [
                {
                    "timestamp": fill.timestamp,
                    "side": fill.side.value,
                    "quantity": fill.quantity,
                    "price": fill.price,
                    "reference_price": fill.reference_price,
                    "commission": fill.commission,
                    "slippage_bps": fill.slippage_bps,
                    "reason": fill.reason,
                }
                for fill in self.fills
            ]
        )

    def summary(self) -> str:
        verdict = (
            "BAT le buy & hold" if self.beats_benchmark else "NE BAT PAS le buy & hold"
        )
        lines = [
            f"{self.strategy_name} sur {self.symbol}",
            f"  periode           : {self.start.date()} -> {self.end.date()} "
            f"({self.years:.1f} ans, {self.n_bars} barres)",
            f"  capital initial   : {self.initial_capital:,.2f}",
            f"  capital final     : {self.final_equity:,.2f}",
            f"  performance       : {self.total_return:+.2%}",
            f"  buy & hold        : {self.benchmark_return:+.2%}",
            f"  ecart             : {self.excess_return:+.2%}  -> {verdict}",
            f"  couts totaux      : {self.costs.total_costs:,.2f} "
            f"({self.cost_drag_pct:.2%} du capital)",
            f"    commissions     : {self.costs.commissions:,.2f}",
            f"    spread+slippage : {self.costs.market_friction:,.2f}",
            f"  dividendes recus  : {self.costs.dividends_received:,.2f}",
            f"  allers-retours    : {self.n_trades}",
            f"  degres de liberte : {self.degrees_of_freedom} "
            f"(espace de recherche : {self.search_space_size:,} configurations)",
        ]
        if self.unfilled_orders:
            lines.append(f"  ordres non executes : {len(self.unfilled_orders)}")
        for warning in self.warnings:
            lines.append(f"  [!] {warning}")
        return "\n".join(lines)


def _to_datetime(epoch_ns: int) -> datetime:
    from ..data.types import UTC

    return datetime.fromtimestamp(epoch_ns / 1_000_000_000, tz=UTC)

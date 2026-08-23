"""Moteur de backtest evenementiel et modelisation des frictions."""

from __future__ import annotations

from .costs import (
    CommissionModel,
    CostModel,
    FillContext,
    FixedSpread,
    LinearSlippage,
    SlippageModel,
    SpreadModel,
    SquareRootSlippage,
    VolatilitySpread,
)
from .engine import BacktestEngine, ExecutionConfig
from .orders import Fill, Order, OrderStatus, Side, Trade
from .portfolio import Portfolio, PortfolioError
from .result import BacktestResult, CostBreakdown

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "CommissionModel",
    "CostBreakdown",
    "CostModel",
    "ExecutionConfig",
    "Fill",
    "FillContext",
    "FixedSpread",
    "LinearSlippage",
    "Order",
    "OrderStatus",
    "Portfolio",
    "PortfolioError",
    "Side",
    "SlippageModel",
    "SpreadModel",
    "SquareRootSlippage",
    "Trade",
    "VolatilitySpread",
]

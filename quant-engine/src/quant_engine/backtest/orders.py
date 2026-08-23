"""Ordres, executions et trades.

Chaque execution conserve la trace complete de sa formation : prix theorique
vise, prix reellement obtenu, decomposition du cout. C'est ce qui permettra,
a l'etape 7, de comparer le simule au reel -- la seule metrique qui dise
vraiment si le moteur ne se raconte pas d'histoires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import final

__all__ = ["Fill", "Order", "OrderStatus", "Side", "Trade"]


class Side(Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> float:
        return 1.0 if self is Side.BUY else -1.0


class OrderStatus(Enum):
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    EXPIRED = "expired"
    REJECTED = "rejected"


@final
@dataclass(slots=True)
class Order:
    """Ordre en attente d'execution.

    ``decision_index`` et ``eligible_index`` sont distincts par construction :
    un ordre decide a la cloture de la barre t ne peut pas s'executer avant la
    barre t+1. C'est la latence minimale, et elle n'est pas parametrable a zero.
    """

    order_id: int
    decision_index: int
    eligible_index: int
    decision_time: datetime
    side: Side
    quantity: float
    reason: str = ""
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    rejection: str = ""

    def __post_init__(self) -> None:
        if self.quantity <= 0.0:
            raise ValueError(f"Quantite d'ordre invalide : {self.quantity}")
        if self.eligible_index <= self.decision_index:
            raise ValueError(
                f"Ordre executable a l'index {self.eligible_index} alors qu'il est decide "
                f"a {self.decision_index} : un ordre ne peut pas s'executer avant "
                "d'avoir ete emis."
            )

    @property
    def remaining(self) -> float:
        return max(0.0, self.quantity - self.filled_quantity)

    @property
    def is_open(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED)


@final
@dataclass(frozen=True, slots=True)
class Fill:
    """Execution effective, avec sa decomposition de cout."""

    order_id: int
    index: int
    timestamp: datetime
    side: Side
    quantity: float
    price: float
    """Prix reellement obtenu, frictions de marche incluses."""
    reference_price: float
    """Prix theorique vise, avant frictions."""
    commission: float
    reason: str = ""

    @property
    def notional(self) -> float:
        return self.quantity * self.price

    @property
    def market_friction(self) -> float:
        """Cout du spread et du slippage, en monnaie."""
        return abs(self.price - self.reference_price) * self.quantity

    @property
    def total_cost(self) -> float:
        return self.market_friction + self.commission

    @property
    def slippage_bps(self) -> float:
        """Ecart entre prix obtenu et prix vise, en points de base."""
        if self.reference_price <= 0.0:
            return 0.0
        return abs(self.price - self.reference_price) / self.reference_price * 10_000.0

    @property
    def cash_delta(self) -> float:
        """Variation de tresorerie : negative a l'achat, positive a la vente."""
        return -self.side.sign * self.notional - self.commission


@final
@dataclass(slots=True)
class Trade:
    """Aller-retour complet : de l'ouverture d'une position a sa fermeture."""

    entry_time: datetime
    entry_price: float
    quantity: float
    exit_time: datetime | None = None
    exit_price: float | None = None
    costs: float = 0.0
    fills: list[int] = field(default_factory=list)

    @property
    def is_closed(self) -> bool:
        return self.exit_time is not None

    @property
    def gross_pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) * self.quantity

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.costs

    @property
    def return_pct(self) -> float:
        base = abs(self.entry_price * self.quantity)
        return self.net_pnl / base if base > 0.0 else 0.0

    @property
    def is_winner(self) -> bool:
        return self.net_pnl > 0.0

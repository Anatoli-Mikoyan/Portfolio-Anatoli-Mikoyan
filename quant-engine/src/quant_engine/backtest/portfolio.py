"""Comptabilite du portefeuille.

Deux details que la plupart des moteurs amateurs ignorent, et qui faussent
silencieusement le resultat :

**Les splits.** Le moteur execute aux prix bruts, donc reellement cotes. Quand
un split 4-pour-1 survient, le cours est divise par quatre du jour au lendemain.
Si le nombre de titres detenus n'est pas multiplie par quatre au meme instant,
le portefeuille enregistre une perte de 75 % qui n'a jamais eu lieu.

**Les dividendes.** Un dividende fait mecaniquement decrocher le cours de son
montant a l'ex-date. Si le detenteur ne recoit pas le montant en tresorerie, le
moteur compte une perte fantome a chaque detachement -- de l'ordre de 2 a 4 %
par an sur des actions de rendement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import final

from ..errors import QuantEngineError
from .orders import Fill, Trade

__all__ = ["Portfolio", "PortfolioError"]


class PortfolioError(QuantEngineError):
    """Incoherence comptable : tresorerie negative interdite, quantite absurde."""


@final
@dataclass(slots=True)
class Portfolio:
    """Etat comptable d'un compte mono-actif."""

    initial_cash: float
    cash: float = field(init=False)
    units: float = 0.0
    """Quantite detenue, en titres bruts. Negative pour une position vendeuse."""
    allow_negative_cash: bool = False
    """Autoriser le levier. Faux par defaut : un backtest qui s'endette sans le
    declarer surestime sa performance et masque son vrai risque."""

    fills: list[Fill] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    total_commission: float = 0.0
    total_friction: float = 0.0
    total_dividends: float = 0.0
    _open_trade: Trade | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.initial_cash <= 0.0:
            raise PortfolioError(f"Capital initial invalide : {self.initial_cash}")
        self.cash = self.initial_cash

    # -- valorisation ---------------------------------------------------------
    def position_value(self, price: float) -> float:
        return self.units * price

    def equity(self, price: float) -> float:
        return self.cash + self.position_value(price)

    def weight(self, price: float) -> float:
        """Fraction de l'equity exposee a l'actif."""
        total = self.equity(price)
        if total <= 0.0:
            return 0.0
        return self.position_value(price) / total

    @property
    def is_flat(self) -> bool:
        return abs(self.units) < 1e-12

    # -- evenements de marche -------------------------------------------------
    def apply_split(self, ratio: float) -> None:
        """Ajuste la quantite detenue lors d'un split.

        Le cours est divise par ``ratio`` a l'ex-date ; la quantite doit etre
        multipliee d'autant pour que la valeur de la position soit conservee.
        """
        if ratio <= 0.0:
            raise PortfolioError(f"Ratio de split invalide : {ratio}")
        if self.is_flat:
            return
        self.units *= ratio
        if self._open_trade is not None:
            self._open_trade.entry_price /= ratio
            self._open_trade.quantity *= ratio

    def apply_dividend(self, amount_per_unit: float, timestamp: datetime) -> float:  # noqa: ARG002
        """Credite le dividende en tresorerie. Le debite si la position est vendeuse.

        ``timestamp`` figure dans la signature pour que la couche d'execution
        reelle (etape 7) puisse journaliser la date de detachement sans avoir a
        changer le contrat.
        """
        if self.is_flat or amount_per_unit <= 0.0:
            return 0.0
        cash_flow = self.units * amount_per_unit
        self.cash += cash_flow
        self.total_dividends += cash_flow
        return cash_flow

    # -- executions -----------------------------------------------------------
    def apply_fill(self, fill: Fill) -> None:
        """Applique une execution a la comptabilite."""
        new_cash = self.cash + fill.cash_delta
        if new_cash < -1e-9 and not self.allow_negative_cash:
            raise PortfolioError(
                f"Tresorerie negative apres execution ({new_cash:.2f}). Le levier "
                "implicite est interdit : declare-le explicitement via "
                "allow_negative_cash si c'est voulu."
            )
        signed_quantity = fill.side.sign * fill.quantity
        previous_units = self.units

        self.cash = new_cash
        self.units += signed_quantity
        self.fills.append(fill)
        self.total_commission += fill.commission
        self.total_friction += fill.market_friction

        self._track_trade(fill, previous_units, signed_quantity)

    def _track_trade(self, fill: Fill, previous_units: float, signed_quantity: float) -> None:
        """Maintient le journal des allers-retours.

        Une position ouverte puis renforcee compte comme un seul trade, avec un
        prix d'entree moyen pondere : compter chaque execution comme un trade
        distinct gonflerait artificiellement le nombre d'observations et
        retrecirait a tort les intervalles de confiance.
        """
        opening = abs(self.units) > abs(previous_units) or (
            previous_units == 0.0 and signed_quantity != 0.0
        )
        if opening:
            if self._open_trade is None:
                self._open_trade = Trade(
                    entry_time=fill.timestamp,
                    entry_price=fill.price,
                    quantity=signed_quantity,
                    costs=fill.total_cost,
                    fills=[fill.order_id],
                )
                self.trades.append(self._open_trade)
            else:
                trade = self._open_trade
                total = trade.quantity + signed_quantity
                if abs(total) > 1e-12:
                    trade.entry_price = (
                        trade.entry_price * trade.quantity + fill.price * signed_quantity
                    ) / total
                trade.quantity = total
                trade.costs += fill.total_cost
                trade.fills.append(fill.order_id)
            return

        if self._open_trade is not None:
            trade = self._open_trade
            trade.costs += fill.total_cost
            trade.fills.append(fill.order_id)
            if self.is_flat:
                trade.exit_time = fill.timestamp
                trade.exit_price = fill.price
                self._open_trade = None

    def close_open_trade(self, price: float, timestamp: datetime) -> None:
        """Solde comptablement le trade encore ouvert en fin de backtest.

        Sans ca, un backtest dont la derniere position est gagnante mais non
        soldee affiche un profit latent parmi ses trades realises.
        """
        if self._open_trade is not None and not self._open_trade.is_closed:
            self._open_trade.exit_time = timestamp
            self._open_trade.exit_price = price
            self._open_trade = None

    @property
    def closed_trades(self) -> list[Trade]:
        return [trade for trade in self.trades if trade.is_closed]

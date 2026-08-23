"""Moteur de backtest evenementiel.

Architecture evenementielle, pas vectorisee. C'est plus lent d'un ordre de
grandeur, et c'est le prix a payer pour que la sequence des evenements soit
celle de la realite : on decide avec l'information de la cloture, on execute a
l'ouverture suivante, on paie les frais, on subit l'execution partielle.

Une implementation vectorisee calcule tous les signaux d'un coup sur la serie
entiere, puis multiplie par les rendements. C'est cent fois plus rapide, et ca
rend le look-ahead presque inevitable : le moindre decalage d'indice oublie
devient invisible.

Sequence a l'interieur d'une barre t
------------------------------------
1. **Operations sur titre** dont l'ex-date tombe sur cette barre. Elles
   s'appliquent avant tout le reste : le cours d'ouverture est deja
   post-operation.
2. **Execution des ordres eligibles**, au prix d'ouverture majore des
   frictions. Un ordre decide a la cloture de t-1 devient eligible ici.
3. **Valorisation** a la cloture de t, enregistree dans la courbe d'equity.
4. **Decision de la strategie**, a partir d'une vue bornee a la cloture de t.
5. **Emission d'un ordre** eligible au plus tot a t+1.

Les etapes 3 et 4 sont volontairement apres l'etape 2 : la strategie voit donc
l'effet de ses propres executions, comme dans la realite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final, Literal, final

import numpy as np
from numpy.typing import NDArray

from ..data.adjustment import AdjustmentPolicy
from ..data.dataset import MarketData
from ..data.types import Field as DataField
from ..errors import ConfigError, QuantEngineError
from ..logging_setup import get_logger
from ..strategy.base import Signal, Strategy, StrategyContext
from ..strategy.reference import BuyAndHold
from .costs import CostModel, FillContext
from .orders import Fill, Order, OrderStatus, Side
from .portfolio import Portfolio
from .result import BacktestResult, CostBreakdown

__all__ = ["BacktestEngine", "ExecutionConfig"]

_LOG = get_logger("backtest.engine")
_NS: Final = 1_000_000_000

FillPrice = Literal["next_open", "next_close"]


@final
@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Regles de passage d'un signal a une execution."""

    latency_bars: int = 1
    """Barres entre la decision et la premiere execution possible. Minimum 1 :
    decider a la cloture et executer a cette meme cloture, c'est se donner le
    prix de fermeture apres l'avoir vu."""

    fill_at: FillPrice = "next_open"
    """Ouverture de la barre d'execution par defaut. ``next_close`` suppose
    qu'on tient jusqu'a la cloture, ce qui est plus favorable et moins realiste
    pour un ordre emis sur un signal de veille."""

    max_participation: float = 0.05
    """Part maximale du volume d'une barre qu'un ordre peut consommer.
    Au-dela, l'ordre est execute partiellement et le reste attend."""

    max_order_age_bars: int = 5
    """Age au-dela duquel un ordre non execute est annule."""

    allow_fractional_units: bool = False
    """Actions fractionnees. Faux par defaut : la plupart des courtiers ne les
    proposent pas, et l'arrondi a l'entier est ce qui rend les tres petits
    comptes structurellement inoperants."""

    min_order_notional: float = 0.0
    """Montant en dessous duquel on n'emet pas d'ordre."""

    rebalance_tolerance: float = 0.02
    """Ecart de poids en dessous duquel on ne rebalance pas. Sans ce seuil, la
    seule derive du prix declencherait un ordre a chaque barre."""

    volatility_window: int = 20
    volume_window: int = 20

    def __post_init__(self) -> None:
        if self.latency_bars < 1:
            raise ConfigError(
                f"latency_bars={self.latency_bars} : une latence nulle revient a "
                "executer au prix de la barre qui a produit le signal, donc a "
                "connaitre ce prix avant de decider. Minimum 1."
            )
        if not 0.0 < self.max_participation <= 1.0:
            raise ConfigError("max_participation doit etre dans ]0, 1]")
        if self.max_order_age_bars < 1:
            raise ConfigError("max_order_age_bars doit valoir au moins 1")
        if self.rebalance_tolerance < 0.0:
            raise ConfigError("rebalance_tolerance ne peut pas etre negatif")


@dataclass(slots=True)
class _RunOutput:
    equity: list[float] = field(default_factory=list)
    exposure: list[float] = field(default_factory=list)
    timestamps: list[int] = field(default_factory=list)
    portfolio: Portfolio | None = None
    unfilled: list[Order] = field(default_factory=list)


@final
class BacktestEngine:
    """Execute une strategie sur un jeu de donnees historique."""

    def __init__(
        self,
        costs: CostModel,
        *,
        initial_capital: float,
        execution: ExecutionConfig | None = None,
        adjustment: AdjustmentPolicy = AdjustmentPolicy.SPLIT_PIT,
        acknowledge_frictionless: bool = False,
    ) -> None:
        """``costs`` et ``initial_capital`` sont obligatoires et sans valeur par defaut.

        Un moteur qui accepterait de demarrer sans couts configures produirait
        des resultats faux avec l'apparence du serieux. La seule facon d'obtenir
        un backtest sans frictions est de le demander explicitement, via
        ``CostModel.frictionless()`` et ``acknowledge_frictionless=True`` --
        combinaison reservee aux tests analytiques du moteur.
        """
        if initial_capital <= 0.0:
            raise ConfigError(f"Capital initial invalide : {initial_capital}")
        probe = costs.round_trip_cost_pct(max(initial_capital, 1.0))
        if probe <= 0.0 and not acknowledge_frictionless:
            raise ConfigError(
                "Le modele de couts configure produit un aller-retour gratuit. "
                "Un backtest sans frictions n'est pas optimiste, il est faux : "
                "commissions, spread et slippage sont ce qui separe la plupart "
                "des strategies rentables sur le papier des comptes qui se vident. "
                "Configure des couts reels, ou passe acknowledge_frictionless=True "
                "si tu testes le moteur lui-meme."
            )
        self.costs = costs
        self.initial_capital = initial_capital
        self.execution = execution if execution is not None else ExecutionConfig()
        self.adjustment = adjustment

    # -- API ------------------------------------------------------------------
    def run(
        self,
        strategy: Strategy,
        data: MarketData,
        *,
        start_index: int | None = None,
    ) -> BacktestResult:
        """Execute la strategie et son buy & hold de reference sur la meme periode."""
        total = len(data)
        start = self._resolve_start(strategy, data, start_index)

        strategy.reset()
        run = self._run_single(strategy, data, start)
        benchmark = self._run_single(BuyAndHold(), data, start)

        portfolio = run.portfolio
        assert portfolio is not None
        breakdown = CostBreakdown(
            commissions=portfolio.total_commission,
            market_friction=portfolio.total_friction,
            dividends_received=portfolio.total_dividends,
        )
        result = BacktestResult(
            strategy_name=strategy.name,
            symbol=data.symbol,
            parameters=strategy.params,
            timestamps=np.asarray(run.timestamps, dtype=np.int64),
            equity=np.asarray(run.equity, dtype=np.float64),
            exposure=np.asarray(run.exposure, dtype=np.float64),
            benchmark_equity=np.asarray(benchmark.equity, dtype=np.float64),
            fills=tuple(portfolio.fills),
            trades=tuple(portfolio.trades),
            unfilled_orders=tuple(run.unfilled),
            costs=breakdown,
            initial_capital=self.initial_capital,
            degrees_of_freedom=strategy.degrees_of_freedom,
            search_space_size=strategy.search_space_size,
            warnings=self._diagnose(strategy, data, run, start, total),
        )
        _LOG.info(
            "backtest termine",
            extra={
                "strategy": strategy.name,
                "symbol": data.symbol,
                "return": round(result.total_return, 6),
                "benchmark": round(result.benchmark_return, 6),
                "trades": result.n_trades,
                "cost_drag": round(result.cost_drag_pct, 6),
            },
        )
        return result

    def _resolve_start(
        self, strategy: Strategy, data: MarketData, start_index: int | None
    ) -> int:
        total = len(data)
        warmup = max(1, strategy.warmup_bars)
        start = warmup - 1 if start_index is None else start_index
        if start < warmup - 1:
            raise ConfigError(
                f"start_index={start} inferieur au warmup de la strategie "
                f"({warmup} barres). Le moteur refuse d'appeler une strategie avant "
                "qu'elle dispose de son historique : elle produirait des signaux "
                "precoces qui n'auraient jamais existe."
            )
        if start >= total - 1:
            raise ConfigError(
                f"{total} barres pour un warmup de {warmup} : il ne reste aucune "
                "barre exploitable. Charge un historique plus long."
            )
        return start

    # -- boucle principale ----------------------------------------------------
    def _run_single(self, strategy: Strategy, data: MarketData, start: int) -> _RunOutput:
        strategy.reset()
        total = len(data)
        config = self.execution
        portfolio = Portfolio(initial_cash=self.initial_capital)
        output = _RunOutput(portfolio=portfolio)

        multipliers = data.multipliers(self.adjustment)
        actions_by_index = _index_corporate_actions(data)
        volatility = _trailing_volatility(data, config.volatility_window)
        average_volume = _trailing_average_volume(data, config.volume_window)

        pending: list[Order] = []
        next_order_id = 1
        started = False

        for index in range(start, total):
            bar = data.execution_bar(index)

            # 1. operations sur titre : le cours d'ouverture est deja post-operation
            for split_ratio, dividend in actions_by_index.get(index, ()):
                if split_ratio is not None:
                    portfolio.apply_split(split_ratio)
                if dividend is not None:
                    portfolio.apply_dividend(dividend, bar.timestamp)

            # 2. execution des ordres eligibles
            reference = bar.open if config.fill_at == "next_open" else bar.close
            pending = self._execute_pending(
                pending=pending,
                portfolio=portfolio,
                index=index,
                timestamp=bar.timestamp,
                reference_price=reference,
                bar_volume=bar.volume,
                average_volume=float(average_volume[index]),
                volatility=float(volatility[index]),
                unfilled=output.unfilled,
            )

            # 3. valorisation a la cloture
            equity = portfolio.equity(bar.close)
            output.equity.append(equity)
            output.exposure.append(portfolio.weight(bar.close))
            output.timestamps.append(int(data.timestamps[index]))

            if index == total - 1:
                break  # plus aucune barre pour executer : inutile de decider

            # 4. decision de la strategie, sur une vue bornee
            context = StrategyContext(
                history=data.view_at(index, multipliers),
                as_of=bar.timestamp,
                position_units=portfolio.units,
                position_weight=portfolio.weight(bar.close),
                cash=portfolio.cash,
                equity=equity,
                bar_index=index,
            )
            if not started:
                strategy.on_start(context)
                started = True
            signal = strategy.on_bar(context)
            if signal is None:
                continue

            # 5. traduction en ordre, eligible au plus tot a index + latence
            order = self._build_order(
                signal=signal,
                context=context,
                price=bar.close,
                order_id=next_order_id,
                index=index,
                total=total,
            )
            if order is not None:
                pending.append(order)
                next_order_id += 1

        portfolio.close_open_trade(data.execution_bar(total - 1).close, data.end)
        for order in pending:
            order.status = OrderStatus.EXPIRED
            output.unfilled.append(order)
        return output

    # -- execution ------------------------------------------------------------
    def _execute_pending(
        self,
        *,
        pending: list[Order],
        portfolio: Portfolio,
        index: int,
        timestamp: datetime,
        reference_price: float,
        bar_volume: float,
        average_volume: float,
        volatility: float,
        unfilled: list[Order],
    ) -> list[Order]:
        config = self.execution
        still_open: list[Order] = []

        for order in pending:
            if order.eligible_index > index:
                still_open.append(order)
                continue
            if index - order.decision_index > config.max_order_age_bars:
                order.status = OrderStatus.EXPIRED
                order.rejection = "non execute dans le delai maximal"
                unfilled.append(order)
                continue

            # Execution partielle : un ordre ne peut consommer qu'une fraction du
            # volume de la barre. Utiliser le volume realise de la barre est une
            # simplification, mais elle ne joue jamais en faveur du backtest --
            # elle ne fait que reduire la quantite executee.
            capacity = config.max_participation * max(bar_volume, 0.0)
            quantity = order.remaining
            if capacity <= 0.0:
                still_open.append(order)
                continue
            filled = min(quantity, capacity)
            if not config.allow_fractional_units:
                filled = math.floor(filled)
            if filled <= 0.0:
                still_open.append(order)
                continue

            signed = order.side.sign * filled
            fill_context = FillContext(
                reference_price=reference_price,
                quantity=signed,
                bar_volume=bar_volume,
                average_volume=average_volume,
                volatility=volatility,
            )
            price = self.costs.execution_price(fill_context)
            commission = self.costs.commission_for(fill_context)

            if order.side is Side.BUY:
                affordable = portfolio.cash - commission
                if affordable < filled * price:
                    filled = math.floor(max(0.0, affordable) / price) if price > 0 else 0.0
                    if not config.allow_fractional_units:
                        filled = math.floor(filled)
                    if filled <= 0.0:
                        order.status = OrderStatus.REJECTED
                        order.rejection = "tresorerie insuffisante"
                        unfilled.append(order)
                        continue
                    fill_context = FillContext(
                        reference_price=reference_price,
                        quantity=filled,
                        bar_volume=bar_volume,
                        average_volume=average_volume,
                        volatility=volatility,
                    )
                    price = self.costs.execution_price(fill_context)
                    commission = self.costs.commission_for(fill_context)

            fill = Fill(
                order_id=order.order_id,
                index=index,
                timestamp=timestamp,
                side=order.side,
                quantity=filled,
                price=price,
                reference_price=reference_price,
                commission=commission,
                reason=order.reason,
            )
            portfolio.apply_fill(fill)
            order.filled_quantity += filled
            if order.remaining <= 1e-9:
                order.status = OrderStatus.FILLED
            else:
                order.status = OrderStatus.PARTIALLY_FILLED
                still_open.append(order)

        return still_open

    def _build_order(
        self,
        *,
        signal: Signal,
        context: StrategyContext,
        price: float,
        order_id: int,
        index: int,
        total: int,
    ) -> Order | None:
        config = self.execution
        if price <= 0.0:
            return None
        weight_gap = signal.target_weight - context.position_weight
        if abs(weight_gap) < config.rebalance_tolerance:
            return None

        target_units = signal.target_weight * context.equity / price
        delta_units = target_units - context.position_units
        if not config.allow_fractional_units:
            delta_units = math.trunc(delta_units)
        if abs(delta_units) < 1e-9:
            return None
        if abs(delta_units) * price < config.min_order_notional:
            return None

        eligible = index + config.latency_bars
        if eligible >= total:
            return None
        return Order(
            order_id=order_id,
            decision_index=index,
            eligible_index=eligible,
            decision_time=context.as_of,
            side=Side.BUY if delta_units > 0 else Side.SELL,
            quantity=abs(delta_units),
            reason=signal.reason,
        )

    # -- diagnostics ----------------------------------------------------------
    def _diagnose(
        self, strategy: Strategy, data: MarketData, run: _RunOutput, start: int, total: int
    ) -> tuple[str, ...]:
        """Signaux d'alerte detectes automatiquement.

        Ils figureront tels quels dans le rapport HTML de l'etape 3. L'idee est
        qu'un lecteur presse ne puisse pas passer a cote de ce qui invalide le
        resultat qu'il a sous les yeux.
        """
        warnings: list[str] = []
        portfolio = run.portfolio
        assert portfolio is not None
        n_trades = len(portfolio.closed_trades)

        if n_trades < 30:
            warnings.append(
                f"Echantillon insuffisant : {n_trades} allers-retours. En dessous "
                "d'une trentaine, aucune metrique de performance n'est distinguable "
                "du hasard."
            )
        dof = strategy.degrees_of_freedom
        if dof > 0:
            bars = total - start
            per_dof = bars / dof
            if per_dof < 250:
                warnings.append(
                    f"{dof} degres de liberte pour {bars} barres ({per_dof:.0f} par "
                    "parametre). Compter en centaines d'observations par parametre "
                    "est le minimum pour que le calibrage signifie quelque chose."
                )
        if strategy.search_space_size > 100:
            warnings.append(
                f"Espace de recherche de {strategy.search_space_size:,} configurations. "
                "Si ce jeu de parametres a ete choisi en explorant la grille, le "
                "resultat est le maximum d'autant de tirages et doit etre corrige."
            )
        turnover = strategy.expected_annual_turnover
        if turnover > 0.0:
            round_trip = self.costs.round_trip_cost_pct(self.initial_capital)
            drag = turnover * round_trip
            if drag > 0.10:
                warnings.append(
                    f"Friction annuelle estimee a {drag:.1%} du capital "
                    f"({turnover:.0f} allers-retours par an a {round_trip:.2%} piece). "
                    "La strategie doit produire au moins autant en brut pour ne rien "
                    "perdre."
                )
        if run.unfilled:
            warnings.append(
                f"{len(run.unfilled)} ordre(s) non execute(s) : liquidite insuffisante "
                "ou tresorerie manquante. La strategie n'a pas pu faire ce qu'elle voulait."
            )
        if data.quality is not None and data.quality.warnings:
            warnings.append(
                f"{len(data.quality.warnings)} avertissement(s) de qualite sur les "
                "donnees sous-jacentes."
            )
        return tuple(warnings)


# ---------------------------------------------------------------------------
# Precalculs
# ---------------------------------------------------------------------------
def _index_corporate_actions(
    data: MarketData,
) -> dict[int, list[tuple[float | None, float | None]]]:
    """Associe chaque barre aux operations dont l'ex-date y tombe."""
    mapping: dict[int, list[tuple[float | None, float | None]]] = {}
    stamps = data.timestamps
    for split in data.actions.splits:
        target = np.int64(int(split.ex_date.timestamp() * _NS))
        index = int(np.searchsorted(stamps, target, side="left"))
        if 0 <= index < stamps.size:
            mapping.setdefault(index, []).append((split.ratio, None))
    for dividend in data.actions.dividends:
        target = np.int64(int(dividend.ex_date.timestamp() * _NS))
        index = int(np.searchsorted(stamps, target, side="left"))
        if 0 <= index < stamps.size:
            mapping.setdefault(index, []).append((None, dividend.amount))
    return mapping


def _trailing_volatility(data: MarketData, window: int) -> NDArray[np.float64]:
    """Volatilite des rendements sur les barres **anterieures** a chaque index.

    Le decalage d'une barre est essentiel : estimer la volatilite du jour avec
    le rendement du jour reviendrait a connaitre l'amplitude de la seance avant
    de passer l'ordre, donc a sous-estimer le cout d'execution exactement les
    jours agites.
    """
    close = data.raw(DataField.CLOSE)
    n = close.size
    out = np.full(n, 0.01, dtype=np.float64)
    if n < 3:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.diff(np.log(np.where(close > 0.0, close, np.nan)))
    for index in range(2, n):
        lookback = returns[max(0, index - 1 - window) : index - 1]
        finite = lookback[np.isfinite(lookback)]
        if finite.size >= 2:
            out[index] = max(float(np.std(finite, ddof=1)), 1e-6)
    return out


def _trailing_average_volume(data: MarketData, window: int) -> NDArray[np.float64]:
    """Volume moyen des barres anterieures, meme logique de decalage."""
    volume = data.raw(DataField.VOLUME)
    n = volume.size
    out = np.full(n, max(float(volume[0]), 1.0), dtype=np.float64)
    for index in range(1, n):
        lookback = volume[max(0, index - window) : index]
        if lookback.size:
            out[index] = max(float(lookback.mean()), 1.0)
    return out


class BacktestError(QuantEngineError):
    """Echec pendant l'execution d'un backtest."""

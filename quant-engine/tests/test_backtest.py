"""Moteur de backtest : comptabilite, frictions, latence, garde-fous.

Le fichier contient les tests de non-regression analytiques : des series dont
le resultat se calcule a la main. Ce sont eux qui detectent une derive du
moteur avant qu'elle ne contamine une vraie evaluation.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
from conftest import make_data

from quant_engine.backtest import (
    BacktestEngine,
    CommissionModel,
    CostModel,
    ExecutionConfig,
    FillContext,
    FixedSpread,
    Portfolio,
    PortfolioError,
    Side,
    SquareRootSlippage,
    VolatilitySpread,
)
from quant_engine.backtest.orders import Fill, Order
from quant_engine.data import (
    CorporateActions,
    Dividend,
    Field,
    MarketData,
    NormalizationPolicy,
    QualityPolicy,
    Split,
    normalize,
)
from quant_engine.data.providers.synthetic import constant_return_series
from quant_engine.data.types import UTC
from quant_engine.errors import ConfigError
from quant_engine.strategy import BuyAndHold, MovingAverageCrossover, Strategy
from quant_engine.strategy.base import ParameterSpec, Signal, StrategyContext

FRICTIONLESS = CostModel.frictionless()
FRACTIONAL = ExecutionConfig(allow_fractional_units=True)


def engine(costs: CostModel = FRICTIONLESS, *, capital: float = 100_000.0,
           execution: ExecutionConfig | None = None) -> BacktestEngine:
    return BacktestEngine(
        costs, initial_capital=capital,
        execution=execution if execution is not None else FRACTIONAL,
        acknowledge_frictionless=True,
    )


# ---------------------------------------------------------------------------
# Garde-fous de configuration
# ---------------------------------------------------------------------------
def test_le_moteur_refuse_de_demarrer_sans_couts() -> None:
    """Exigence centrale : pas de backtest gratuit par defaut.

    Un moteur qui accepte des couts nuls produit des resultats faux avec
    l'apparence du serieux.
    """
    with pytest.raises(ConfigError, match="sans frictions n'est pas optimiste"):
        BacktestEngine(CostModel.frictionless(), initial_capital=10_000.0)


def test_couts_nuls_autorises_seulement_sur_declaration_explicite() -> None:
    BacktestEngine(
        CostModel.frictionless(), initial_capital=10_000.0, acknowledge_frictionless=True
    )


def test_latence_nulle_refusee() -> None:
    """Executer a la cloture qui a produit le signal, c'est connaitre ce prix
    avant d'avoir decide."""
    with pytest.raises(ConfigError, match="latence nulle"):
        ExecutionConfig(latency_bars=0)


def test_capital_initial_invalide() -> None:
    with pytest.raises(ConfigError, match="Capital initial invalide"):
        BacktestEngine(CostModel.interactive_brokers_us_equity(), initial_capital=0.0)


def test_start_index_avant_le_warmup_refuse() -> None:
    data = make_data([100.0 + i for i in range(300)])
    with pytest.raises(ConfigError, match="inferieur au warmup"):
        engine().run(MovingAverageCrossover(slow=200), data, start_index=10)


def test_historique_trop_court() -> None:
    """30 barres pour un warmup de 30 : il ne reste aucune barre pour decider."""
    data = make_data([100.0 + i for i in range(30)])
    with pytest.raises(ConfigError, match="aucune barre exploitable"):
        engine().run(MovingAverageCrossover(fast=10, slow=30), data)


# ---------------------------------------------------------------------------
# Non-regression analytique
# ---------------------------------------------------------------------------
def test_buy_and_hold_reproduit_le_resultat_analytique() -> None:
    """Serie a rendement constant : P_n = P_0 (1+r)^n, resultat connu a la main.

    Le test de derive du moteur. Toute modification qui casse la comptabilite
    -- signe d'un flux, double comptage d'une commission, decalage d'indice --
    fait echouer cette egalite exacte.
    """
    rate = 0.0005
    raw = constant_return_series(rate, 500, start_price=100.0)
    data = normalize(
        raw, NormalizationPolicy(raise_on_blocking=False, quality=QualityPolicy(min_bars=1)),
        now=datetime(2030, 1, 1, tzinfo=UTC),
    )
    result = engine().run(BuyAndHold(), data, start_index=0)

    # Achat decide a la cloture de la barre 0, execute a l'ouverture de la barre 1.
    entry_price = data.execution_bar(1).open
    final_price = float(data.raw(Field.CLOSE)[-1])
    expected = 100_000.0 / entry_price * final_price
    assert result.final_equity == pytest.approx(expected, rel=1e-12)


def test_buy_and_hold_est_son_propre_benchmark() -> None:
    data = make_data([100.0 + i for i in range(200)])
    result = engine().run(BuyAndHold(), data, start_index=0)
    np.testing.assert_allclose(result.equity, result.benchmark_equity)
    assert result.excess_return == pytest.approx(0.0, abs=1e-12)


def test_equity_sans_position_reste_constante() -> None:
    """Une strategie qui ne fait rien ne doit rien gagner ni rien perdre."""

    class Inerte(Strategy):
        name = "inerte"

        @classmethod
        def specs(cls) -> tuple[ParameterSpec, ...]:
            return ()

        @property
        def warmup_bars(self) -> int:
            return 1

        def on_bar(self, context: StrategyContext) -> Signal | None:
            return None

    data = make_data([100.0 + i for i in range(50)])
    result = engine().run(Inerte(), data, start_index=0)
    np.testing.assert_allclose(result.equity, 100_000.0)
    assert result.costs.total_costs == 0.0


# ---------------------------------------------------------------------------
# Latence et sequencement
# ---------------------------------------------------------------------------
def test_lordre_sexecute_a_louverture_de_la_barre_suivante() -> None:
    """La latence minimale d'une barre n'est pas negociable."""
    data = make_data([100.0, 110.0, 120.0, 130.0, 140.0, 150.0])
    result = engine().run(BuyAndHold(), data, start_index=0)
    assert len(result.fills) == 1
    fill = result.fills[0]
    assert fill.index == 1
    # L'ouverture de la barre 1 vaut le close de la barre 0 dans make_data.
    assert fill.reference_price == pytest.approx(data.execution_bar(1).open)
    assert fill.timestamp == data.execution_bar(1).timestamp


def test_latence_configurable_au_dela_dune_barre() -> None:
    data = make_data([100.0 + i for i in range(20)])
    slow = BacktestEngine(
        FRICTIONLESS, initial_capital=100_000.0, acknowledge_frictionless=True,
        execution=ExecutionConfig(allow_fractional_units=True, latency_bars=3),
    )
    result = slow.run(BuyAndHold(), data, start_index=0)
    assert result.fills[0].index == 3


def test_la_strategie_ne_recoit_jamais_le_jeu_de_donnees_complet() -> None:
    """Garde anti-look-ahead au niveau du moteur.

    On enregistre ce que la strategie recoit reellement a chaque appel : la vue
    doit s'arreter exactement a la barre courante, jamais au-dela.
    """
    observed: list[tuple[int, int]] = []

    class Espion(Strategy):
        name = "espion"

        @classmethod
        def specs(cls) -> tuple[ParameterSpec, ...]:
            return ()

        @property
        def warmup_bars(self) -> int:
            return 1

        def on_bar(self, context: StrategyContext) -> Signal | None:
            assert type(context.history).__name__ == "HistoryView"
            observed.append((context.bar_index, context.history.n_bars))
            return None

    data = make_data([100.0 + i for i in range(40)])
    engine().run(Espion(), data, start_index=0)
    assert observed
    for bar_index, visible in observed:
        assert visible == bar_index + 1, "la vue depasse la barre de decision"


def test_empoisonnement_du_futur_ne_change_pas_le_resultat() -> None:
    """Le futur remplace par des NaN apres la moitie de la serie ne doit pas
    modifier la trajectoire d'equity sur la premiere moitie."""
    closes = [100.0 + 0.5 * i for i in range(120)]
    clean = make_data(closes)
    cut = 60
    reference = engine().run(BuyAndHold(), clean, start_index=0)
    poisoned = engine().run(BuyAndHold(), clean.with_future_poisoned(cut), start_index=0)
    np.testing.assert_allclose(reference.equity[:cut], poisoned.equity[:cut])


# ---------------------------------------------------------------------------
# Operations sur titre
# ---------------------------------------------------------------------------
def test_le_split_ne_detruit_pas_de_valeur() -> None:
    """Le moteur execute aux prix bruts : sans ajustement du nombre de titres,
    un split 4-pour-1 enregistrerait une perte de 75 % qui n'a jamais eu lieu."""
    closes = [100.0] * 10 + [25.0] * 10
    ex_date = datetime(2020, 1, 11, 21, tzinfo=UTC)
    data = make_data(closes, actions=CorporateActions(splits=(Split(ex_date, 4.0),)))
    result = engine().run(BuyAndHold(), data, start_index=0)
    # Prix constant hors split : l'equity doit rester plate de bout en bout.
    np.testing.assert_allclose(result.equity, result.equity[1], rtol=1e-12)
    assert result.total_return == pytest.approx(0.0, abs=1e-12)


def test_le_dividende_est_credite_en_tresorerie() -> None:
    """Sans credit, chaque detachement compterait comme une perte fantome."""
    closes = [100.0] * 10 + [98.0] * 10
    ex_date = datetime(2020, 1, 11, 21, tzinfo=UTC)
    data = make_data(closes, actions=CorporateActions(dividends=(Dividend(ex_date, 2.0),)))
    result = engine().run(BuyAndHold(), data, start_index=0)
    assert result.costs.dividends_received == pytest.approx(2.0 * 1000.0, rel=1e-9)
    # Le decrochage du cours est exactement compense par le coupon.
    assert result.total_return == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Frictions
# ---------------------------------------------------------------------------
def test_le_slippage_joue_toujours_contre_lordre() -> None:
    costs = CostModel(
        commission=CommissionModel.zero(),
        spread=FixedSpread(20.0),
        slippage=SquareRootSlippage(coefficient=0.0),
    )
    achat = FillContext(100.0, 10.0, 1e6, 1e6, 0.01)
    vente = FillContext(100.0, -10.0, 1e6, 1e6, 0.01)
    assert costs.execution_price(achat) > 100.0
    assert costs.execution_price(vente) < 100.0


def test_commission_ib_plafonnee_a_un_pourcent() -> None:
    """Sur un ordre de 100 EUR, la commission vaut 1 EUR : 1 % du montant.

    C'est ce plafond -- qui est aussi le plancher a ce niveau de montant -- qui
    rend les tres petits comptes structurellement inoperants.
    """
    model = CommissionModel()
    petit = FillContext(reference_price=100.0, quantity=1.0, bar_volume=1e6,
                        average_volume=1e6, volatility=0.01)
    assert model.charge(petit) == pytest.approx(1.0)

    gros = FillContext(reference_price=100.0, quantity=10_000.0, bar_volume=1e8,
                       average_volume=1e8, volatility=0.01)
    assert model.charge(gros) == pytest.approx(50.0)  # 0,005 x 10 000


def test_cout_aller_retour_estime() -> None:
    costs = CostModel.interactive_brokers_us_equity()
    petit = costs.round_trip_cost_pct(100.0)
    gros = costs.round_trip_cost_pct(1_000_000.0)
    assert petit > 0.019, "un aller-retour sur 100 EUR doit couter environ 2 %"
    assert gros < petit / 10.0, "l'effet de taille doit ecraser le cout relatif"


def test_spread_dynamique_suit_la_volatilite() -> None:
    spread = VolatilitySpread(coefficient=0.10, floor_bps=1.0)
    calme = FillContext(100.0, 1.0, 1e6, 1e6, 0.005)
    agite = FillContext(100.0, 1.0, 1e6, 1e6, 0.05)
    assert spread.half_spread_bps(agite) > spread.half_spread_bps(calme)


def test_les_couts_reduisent_la_performance() -> None:
    data = make_data([100.0 + i for i in range(300)])
    sans = engine().run(BuyAndHold(), data, start_index=0)
    avec = BacktestEngine(
        CostModel.interactive_brokers_us_equity(), initial_capital=100_000.0,
        execution=FRACTIONAL,
    ).run(BuyAndHold(), data, start_index=0)
    assert avec.total_return < sans.total_return
    assert avec.costs.total_costs > 0.0


# ---------------------------------------------------------------------------
# Contraintes d'execution
# ---------------------------------------------------------------------------
def test_actions_entieres_bloquent_les_tres_petits_comptes() -> None:
    """Avec 100 EUR et un titre a 150 EUR, on ne peut acheter aucune action.

    Fait elementaire, mais absent de la quasi-totalite des backtests amateurs,
    qui supposent implicitement des fractions d'action.
    """
    data = make_data([150.0] * 20)
    result = BacktestEngine(
        CostModel.interactive_brokers_us_equity(), initial_capital=100.0,
        execution=ExecutionConfig(allow_fractional_units=False),
    ).run(BuyAndHold(), data, start_index=0)
    assert result.n_trades == 0
    assert result.final_equity == pytest.approx(100.0)


def test_execution_partielle_si_lordre_depasse_le_volume() -> None:
    n = 40
    closes = [100.0] * n
    data = MarketData(
        symbol="THIN", frequency=make_data([1.0, 2.0]).frequency,
        timestamps=make_data(closes).timestamps,
        open_=np.full(n, 100.0), high=np.full(n, 101.0), low=np.full(n, 99.0),
        close=np.asarray(closes), volume=np.full(n, 100.0),  # volume tres faible
    )
    result = BacktestEngine(
        FRICTIONLESS, initial_capital=1_000_000.0, acknowledge_frictionless=True,
        execution=ExecutionConfig(max_participation=0.05, max_order_age_bars=3),
    ).run(BuyAndHold(), data, start_index=0)
    # 5 % de 100 titres = 5 titres par barre : l'ordre ne peut pas etre solde.
    assert all(fill.quantity <= 5.0 for fill in result.fills)
    assert len(result.fills) > 1
    assert result.unfilled_orders


def test_tresorerie_insuffisante_rejette_lordre() -> None:
    portfolio = Portfolio(initial_cash=100.0)
    fill = Fill(
        order_id=1, index=1, timestamp=datetime(2020, 1, 2, tzinfo=UTC),
        side=Side.BUY, quantity=100.0, price=50.0, reference_price=50.0, commission=1.0,
    )
    with pytest.raises(PortfolioError, match="Tresorerie negative"):
        portfolio.apply_fill(fill)


def test_ordre_avec_execution_anterieure_a_la_decision_refuse() -> None:
    with pytest.raises(ValueError, match="avant d'avoir ete emis"):
        Order(
            order_id=1, decision_index=10, eligible_index=10,
            decision_time=datetime(2020, 1, 1, tzinfo=UTC), side=Side.BUY, quantity=1.0,
        )


def test_seuil_de_rebalancement_evite_le_churn() -> None:
    """Sans seuil, la seule derive du prix declencherait un ordre par barre."""

    class Cible(Strategy):
        name = "cible_fixe"

        @classmethod
        def specs(cls) -> tuple[ParameterSpec, ...]:
            return ()

        @property
        def warmup_bars(self) -> int:
            return 1

        def on_bar(self, context: StrategyContext) -> Signal | None:
            return Signal(1.0)

    data = make_data([100.0 + 0.3 * i for i in range(200)])
    result = engine(execution=ExecutionConfig(
        allow_fractional_units=True, rebalance_tolerance=0.02
    )).run(Cible(), data, start_index=0)
    assert len(result.fills) < 20, f"{len(result.fills)} executions : churn excessif"


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def test_alerte_echantillon_insuffisant() -> None:
    data = make_data([100.0 + i for i in range(200)])
    result = engine().run(BuyAndHold(), data, start_index=0)
    assert any("Echantillon insuffisant" in w for w in result.warnings)


def test_alerte_espace_de_recherche() -> None:
    data = make_data([100.0 + np.sin(i / 5.0) * 5 for i in range(600)])
    result = engine().run(MovingAverageCrossover(fast=10, slow=50), data)
    assert any("Espace de recherche" in w for w in result.warnings)


def test_alerte_friction_sur_petit_compte() -> None:
    """Le moteur previent AVANT que le resultat ne soit lu."""
    # 100 EUR : la commission plancher de 1 EUR represente 1 % par ordre, donc
    # 2 % par aller-retour. Avec ~10 allers-retours par an, la friction annuelle
    # atteint 20 % du capital -- avant meme de savoir si la strategie a raison.
    data = make_data([100.0 + np.sin(i / 5.0) * 3 for i in range(400)])
    result = BacktestEngine(
        CostModel.interactive_brokers_us_equity(), initial_capital=100.0,
        execution=ExecutionConfig(allow_fractional_units=False),
    ).run(MovingAverageCrossover(fast=10, slow=50), data)
    assert any("Friction annuelle estimee" in w for w in result.warnings)


def test_resume_et_exports() -> None:
    data = make_data([100.0 + i for i in range(200)])
    result = engine().run(BuyAndHold(), data, start_index=0)
    assert "buy_and_hold" in result.summary()
    frame = result.equity_frame()
    assert list(frame.columns) == ["equity", "benchmark", "exposure"]
    assert len(frame) == result.n_bars
    assert len(result.fills_frame()) == len(result.fills)


def test_ventilation_des_couts() -> None:
    data = make_data([100.0 + i for i in range(300)])
    result = BacktestEngine(
        CostModel.interactive_brokers_us_equity(), initial_capital=100_000.0,
        execution=FRACTIONAL,
    ).run(BuyAndHold(), data, start_index=0)
    breakdown = result.costs
    assert breakdown.total_costs == pytest.approx(
        breakdown.commissions + breakdown.market_friction
    )
    assert result.gross_return > result.total_return
    parts = breakdown.as_pct_of(100_000.0)
    assert parts["total"] == pytest.approx(breakdown.total_costs / 100_000.0)


def test_le_trade_final_est_solde() -> None:
    """Sans ca, une derniere position gagnante non soldee compterait comme un
    profit realise."""
    data = make_data([100.0 + i for i in range(50)])
    result = engine().run(BuyAndHold(), data, start_index=0)
    assert result.n_trades == 1
    assert all(trade.is_closed for trade in result.trades)

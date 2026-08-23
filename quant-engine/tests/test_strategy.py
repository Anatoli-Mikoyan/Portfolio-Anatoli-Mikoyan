"""Contrat de strategie : parametres declares, degres de liberte, signaux."""

from __future__ import annotations

from datetime import datetime

import pytest
from conftest import make_data

from quant_engine.data import AdjustmentPolicy, MarketData
from quant_engine.data.types import UTC
from quant_engine.errors import ConfigError
from quant_engine.strategy import (
    BollingerMeanReversion,
    BuyAndHold,
    MovingAverageCrossover,
    ParameterSet,
    ParameterSpec,
    Signal,
    StrategyContext,
)


def context_at(data: MarketData, index: int, *, units: float = 0.0,
               weight: float = 0.0, equity: float = 10_000.0) -> StrategyContext:
    view = data.view_at(index, data.multipliers(AdjustmentPolicy.RAW))
    return StrategyContext(
        history=view, as_of=view.as_of, position_units=units,
        position_weight=weight, cash=equity - weight * equity,
        equity=equity, bar_index=index,
    )


# ---------------------------------------------------------------------------
# Parametres
# ---------------------------------------------------------------------------
def test_parametre_reglable_exige_un_domaine() -> None:
    with pytest.raises(ConfigError, match="sans domaine"):
        ParameterSpec(name="x", default=5, description="", kind="int")


def test_parametre_non_reglable_dispense_de_domaine() -> None:
    spec = ParameterSpec(name="x", default=5, description="", kind="int", tunable=False)
    assert spec.candidates() == (5,)


def test_validation_des_bornes() -> None:
    spec = ParameterSpec(name="n", default=10, description="", kind="int", low=1, high=20)
    assert spec.validate(15) == 15
    with pytest.raises(ConfigError, match="borne haute"):
        spec.validate(21)
    with pytest.raises(ConfigError, match="borne basse"):
        spec.validate(0)
    with pytest.raises(ConfigError, match="entier attendu"):
        spec.validate(10.5)


def test_parametre_inconnu_refuse() -> None:
    """Une faute de frappe laisserait tourner la valeur par defaut en silence."""
    with pytest.raises(ConfigError, match="Parametres inconnus"):
        MovingAverageCrossover(fastt=20)


def test_degres_de_liberte_comptes() -> None:
    assert BuyAndHold().degrees_of_freedom == 0
    assert MovingAverageCrossover().degrees_of_freedom == 3
    assert BollingerMeanReversion().degrees_of_freedom == 3


def test_buy_and_hold_a_zero_degre_de_liberte() -> None:
    """C'est ce qui en fait une baseline honnete : rien a optimiser, donc sa
    performance mesuree est une estimation non biaisee de sa performance future."""
    strategy = BuyAndHold()
    assert strategy.degrees_of_freedom == 0
    assert strategy.search_space_size == 1


def test_taille_de_lespace_de_recherche() -> None:
    """Le nombre d'essais implicites d'une recherche en grille.

    Retenir le meilleur de N configurations gonfle le resultat meme si aucune
    n'a d'edge : c'est le maximum d'un echantillon de N tirages.
    """
    strategy = MovingAverageCrossover()
    fast = len(range(5, 101, 5))
    slow = len(range(20, 301, 10))
    assert strategy.search_space_size == fast * slow * 2
    assert strategy.search_space_size > 1000


def test_parametres_surchargeables() -> None:
    strategy = MovingAverageCrossover(fast=10, slow=60)
    assert strategy.params.int_("fast") == 10
    assert strategy.warmup_bars == 60


def test_croisement_incoherent_refuse() -> None:
    with pytest.raises(ValueError, match="strictement inferieur"):
        MovingAverageCrossover(fast=100, slow=50)


def test_jeu_de_parametres_immuable() -> None:
    base = ParameterSet.build(MovingAverageCrossover.specs())
    modified = base.with_values(fast=20)
    assert base.int_("fast") == 50
    assert modified.int_("fast") == 20


# ---------------------------------------------------------------------------
# Signaux
# ---------------------------------------------------------------------------
def test_poids_cible_aberrant_refuse() -> None:
    with pytest.raises(ValueError, match="hors de toute plage sensee"):
        Signal(50.0)


def test_buy_and_hold_achete_puis_se_tait() -> None:
    data = make_data([100.0 + i for i in range(30)])
    strategy = BuyAndHold()
    assert strategy.on_bar(context_at(data, 5)) == Signal(1.0, "entree initiale")
    assert strategy.on_bar(context_at(data, 6, units=10.0, weight=1.0)) is None


def test_ma_crossover_suit_le_croisement() -> None:
    """Serie montante puis descendante : le signal doit basculer une fois."""
    montant = [100.0 + i for i in range(60)]
    descendant = [160.0 - 2.0 * i for i in range(60)]
    data = make_data(montant + descendant)
    strategy = MovingAverageCrossover(fast=10, slow=30)

    signals: list[tuple[int, float]] = []
    weight = 0.0
    for index in range(29, len(data)):
        signal = strategy.on_bar(context_at(data, index, weight=weight))
        if signal is not None:
            signals.append((index, signal.target_weight))
            weight = signal.target_weight
    assert signals, "aucun signal produit"
    assert signals[0][1] == 1.0
    assert signals[-1][1] == 0.0


def test_ma_crossover_nemet_pas_de_signal_redondant() -> None:
    """Sans etat interne, la derive du prix declencherait un ordre par barre --
    donc des frais par barre, pour une strategie censee traiter 2 fois par an."""
    data = make_data([100.0 + i for i in range(120)])
    strategy = MovingAverageCrossover(fast=10, slow=30)
    emitted = 0
    for index in range(29, len(data)):
        # Le poids constate derive volontairement a chaque barre.
        drift = 1.0 + 0.01 * (index % 7)
        if strategy.on_bar(context_at(data, index, weight=drift)) is not None:
            emitted += 1
    assert emitted == 1, f"{emitted} signaux emis pour une seule bascule reelle"


def test_ma_crossover_reset_efface_letat() -> None:
    """Indispensable au walk-forward : sans reset, l'etat d'une fenetre fuit
    dans la suivante et le hors-echantillon est contamine."""
    data = make_data([100.0 + i for i in range(60)])
    strategy = MovingAverageCrossover(fast=10, slow=30)
    assert strategy.on_bar(context_at(data, 40)) is not None
    assert strategy.on_bar(context_at(data, 41)) is None
    strategy.reset()
    assert strategy.on_bar(context_at(data, 41)) is not None


def test_bollinger_achete_sous_la_bande() -> None:
    closes = [100.0] * 25 + [80.0]
    data = make_data(closes)
    strategy = BollingerMeanReversion(window=20, entry_sigma=2.0)
    signal = strategy.on_bar(context_at(data, len(data) - 1))
    assert signal is not None
    assert signal.target_weight == 1.0


def test_bollinger_sort_au_retour_a_la_moyenne() -> None:
    data = make_data([100.0 + (i % 5) for i in range(40)])
    strategy = BollingerMeanReversion(window=20, exit_sigma=0.0)
    signal = strategy.on_bar(context_at(data, 39, units=10.0, weight=1.0))
    assert signal is None or signal.target_weight == 0.0


def test_strategie_silencieuse_avant_le_warmup() -> None:
    data = make_data([100.0 + i for i in range(30)])
    strategy = MovingAverageCrossover(fast=10, slow=30)
    assert strategy.on_bar(context_at(data, 5)) is None


def test_rotation_declaree() -> None:
    """Sert a estimer la friction AVANT de lancer le backtest."""
    assert BuyAndHold().expected_annual_turnover == 0.0
    assert MovingAverageCrossover().expected_annual_turnover > 0.0
    assert BollingerMeanReversion().expected_annual_turnover > (
        MovingAverageCrossover().expected_annual_turnover
    )


def test_repr_lisible() -> None:
    assert "fast = 50" in repr(MovingAverageCrossover())


def test_le_contexte_ne_donne_acces_qua_une_vue() -> None:
    """Garantie structurelle : une strategie ne recoit jamais de MarketData."""
    data = make_data([100.0 + i for i in range(30)])
    context = context_at(data, 10)
    # mypy prouve statiquement qu'un HistoryView ne peut pas etre un MarketData
    # (bases disjointes) : la verification a l'execution serait du code mort.
    # On verifie donc l'absence des accesseurs reserves au moteur.
    assert type(context.history).__name__ == "HistoryView"
    for engine_only in ("execution_bar", "to_frame", "with_future_poisoned", "cursor"):
        assert not hasattr(context.history, engine_only), (
            f"la vue expose {engine_only}, reserve au moteur"
        )
    assert context.history.n_bars == 11
    assert isinstance(context.as_of, datetime)
    assert context.as_of.tzinfo is UTC or context.as_of.utcoffset() is not None

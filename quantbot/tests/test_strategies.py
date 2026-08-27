"""Tests du Strategy Engine (cahier des charges §8).

L'exigence centrale du cahier est qu'aucune stratégie ne soit *supposée* rentable. Les
tests l'appliquent au banc de criblage lui-même : il doit rejeter la totalité des
hypothèses sur un marché sans edge, et n'en retenir que sur un marché où l'edge existe
par construction. Un banc qui ne sait pas dire non n'a aucune valeur.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qbot.config import CostConfig, EnvConfig, ValidationConfig
from qbot.data.synthetic import RegimeSwitchingGBM, generate_synthetic_ohlcv
from qbot.strategies import (
    STRATEGY_CLASSES, DonchianBreakout, MeanReversion, TimeSeriesMomentum,
    TrendFollowing, VolatilitySqueeze, all_strategies, default_strategies,
    family_pbo, screen_family, screen_fixed, screening_table,
)

CC = CostConfig(spread_bps=0.6, commission_bps=0.1, slippage_coef=0.05, min_trade_size=0.05)
EC = EnvConfig(vol_target=None)
VC = ValidationConfig(train_bars=6_000, test_bars=2_000, embargo_pct=0.005)
BPY = 6240.0


@pytest.fixture(scope="module")
def trending() -> pd.DataFrame:
    model = RegimeSwitchingGBM(mu=(0.10, -0.08, 0.0), sigma=(0.06, 0.18, 0.10),
                               persistence=0.997, autocorr=0.25, t_df=5.0)
    return generate_synthetic_ohlcv(n=25_000, seed=12, model=model, spread_bps=0.6).drop(
        columns=["regime"])


@pytest.fixture(scope="module")
def random_walk() -> pd.DataFrame:
    model = RegimeSwitchingGBM(mu=(0.0,), sigma=(0.10,), persistence=1.0, autocorr=0.0)
    return generate_synthetic_ohlcv(n=25_000, seed=11, model=model, spread_bps=0.6).drop(
        columns=["regime"])


# ---------------------------------------------------------------------------------------
# Contrat des stratégies
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("strategy", default_strategies(), ids=lambda s: type(s).__name__)
def test_signal_contract(strategy, trending):
    signal = strategy.signal(trending)
    assert len(signal) == len(trending)
    assert np.isfinite(signal.to_numpy()).all()
    assert signal.between(-1.0, 1.0).all()
    assert (signal.iloc[: strategy.warmup] == 0.0).all(), "signal non nul pendant le warm-up"


@pytest.mark.parametrize("strategy", default_strategies(), ids=lambda s: type(s).__name__)
def test_signal_is_causal(strategy, trending):
    """Perturber le futur ne doit rien changer au signal passé."""
    cut = 15_000
    corrupted = trending.copy()
    cols = ["open", "high", "low", "close"]
    corrupted.iloc[cut:, [corrupted.columns.get_loc(c) for c in cols]] *= 1.5

    a = strategy.signal(trending).iloc[:cut]
    b = strategy.signal(corrupted).iloc[:cut]
    assert np.abs(a.to_numpy() - b.to_numpy()).max() < 1e-10, (
        f"{type(strategy).__name__} regarde le futur")


@pytest.mark.parametrize("cls", STRATEGY_CLASSES, ids=lambda c: c.__name__)
def test_strategy_declares_falsifiable_hypothesis(cls):
    """Le cahier des charges exige que chaque stratégie soit une hypothèse testable.

    Exiger la déclaration au niveau du type, et non d'un commentaire, garantit qu'on ne
    peut pas ajouter une stratégie sans avoir formulé ce qui la ferait échouer.
    """
    proto = cls.enumerate()[0]
    assert len(proto.hypothesis) > 40
    assert len(proto.fails_when) > 20
    assert cls.n_trials() == len(cls.enumerate())
    assert cls.n_trials() <= 12, (
        "grille trop large : chaque combinaison est un essai qui dégrade le Deflated Sharpe")


def test_trial_count_is_honest():
    assert len(all_strategies()) == sum(c.n_trials() for c in STRATEGY_CLASSES)


# ---------------------------------------------------------------------------------------
# Le banc de criblage
# ---------------------------------------------------------------------------------------
def test_screening_rejects_everything_on_random_walk(random_walk):
    """LE test qui compte : sur une marche aléatoire, aucune hypothèse ne doit survivre."""
    results = [screen_fixed(s, random_walk, CC, EC, VC, BPY) for s in default_strategies()]
    retained = [r for r in results if r.verdict == "RETENUE"]
    assert not retained, f"faux positifs sur du bruit pur : {[r.name for r in retained]}"

    family = family_pbo(results, n_partitions=8)
    assert family is not None
    assert family["reality_check_p"] > 0.05, (
        "le Reality Check déclare un edge là où il n'y en a aucun")


def test_screening_finds_edge_when_it_exists(trending):
    """Contrôle inverse : un banc qui rejette tout est aussi inutile qu'un banc naïf."""
    results = [screen_fixed(s, trending, CC, EC, VC, BPY) for s in default_strategies()]
    retained = [r for r in results if r.verdict == "RETENUE"]
    assert retained, "aucune hypothèse retenue alors que le momentum existe par construction"

    family = family_pbo(results, n_partitions=8)
    assert family["reality_check_p"] < 0.05
    assert family["pbo"] < 0.35


def test_mean_reversion_fails_exactly_where_it_says(trending):
    """La déclaration `fails_when` doit correspondre au comportement mesuré.

    MeanReversion annonce échouer en tendance persistante ; le marché de test est
    précisément tendanciel. Si elle réussissait quand même, c'est que sa déclaration
    est fausse — ou que le backtest est faux."""
    res = screen_fixed(MeanReversion(window=20, entry_z=2.0), trending, CC, EC, VC, BPY)
    assert "Tendance" in res.fails_when
    assert res.report.sharpe < 0, "la réversion devrait perdre sur un marché tendanciel"


def test_deflated_sharpe_penalises_parameter_search(trending):
    """À performance comparable, le protocole optimisé doit être jugé plus sévèrement."""
    fixed = screen_fixed(DonchianBreakout(channel=55, exit_channel=20, atr_mult=0.5),
                         trending, CC, EC, VC, BPY)
    family = screen_family(DonchianBreakout, trending, CC, EC, VC, BPY)
    assert family.n_trials == DonchianBreakout.n_trials() > 1
    assert fixed.n_trials == 1
    # Le DSR du protocole optimisé intègre le coût de la sélection.
    assert family.report.deflated_sharpe <= fixed.report.deflated_sharpe + 1e-9


def test_screening_table_is_sorted_and_complete(trending):
    results = [screen_fixed(s, trending, CC, EC, VC, BPY) for s in default_strategies()[:3]]
    table = screening_table(results)
    assert len(table) == 3
    assert list(table["sharpe_oos"]) == sorted(table["sharpe_oos"], reverse=True)
    assert table["verdict"].notna().all()


def test_screening_refuses_too_short_series():
    short = generate_synthetic_ohlcv(n=2_000, seed=1).drop(columns=["regime"])
    with pytest.raises(ValueError, match="fold"):
        screen_fixed(TrendFollowing(), short, CC, EC, VC, BPY)

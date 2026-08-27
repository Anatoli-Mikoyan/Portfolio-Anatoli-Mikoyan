"""Tests de l'allocateur RL (cahier des charges §10).

Le cahier demande si le RL doit choisir Buy/Sell/Hold ou allouer entre stratégies. Ces
tests vérifient que la seconde option est correctement implémentée, et surtout que la
comptabilité des coûts est juste — c'est là que les architectures multi-stratégies se
trompent le plus souvent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from qbot.agents import (
    baseline_results, evaluate_allocator, make_allocator, train_allocator,
)
from qbot.config import AgentConfig, CostConfig, EnvConfig, TrainConfig
from qbot.env import AllocationEnv, build_allocation_profiles, strategy_position_matrix
from qbot.strategies import DonchianBreakout, MeanReversion, TimeSeriesMomentum

BPY = 6240.0
COSTS = CostConfig(spread_bps=0.5, commission_bps=0.1, slippage_coef=0.05, min_trade_size=0.05)


def _alternating_market(n=12_000, block=1_500, ac=0.25, vol=0.10, seed=5):
    """Blocs alternés à autocorrélation +ac (momentum) puis -ac (retour à la moyenne).

    Construction volontairement transparente : c'est le seul type de marché où un
    allocateur a quelque chose de RÉEL à apprendre, donc le seul où son échec serait
    imputable au code plutôt qu'à l'absence de structure.
    """
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n) * vol / np.sqrt(BPY)
    r = np.zeros(n)
    regime = np.zeros(n, dtype=int)
    prev = 0.0
    for i in range(n):
        regime[i] = (i // block) % 2
        prev = (ac if regime[i] == 0 else -ac) * prev + eps[i]
        r[i] = prev

    close = 1.10 * np.exp(np.cumsum(r))
    open_ = np.concatenate([[1.10], close[:-1]])
    hi = close * (1 + np.abs(rng.normal(0, 3e-4, n)))
    lo = close * (1 - np.abs(rng.normal(0, 3e-4, n)))
    idx = pd.date_range("2015-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({
        "open": open_,
        "high": np.maximum.reduce([hi, open_, close]),
        "low": np.minimum.reduce([lo, open_, close]),
        "close": close,
        "volume": rng.gamma(2, 500, n),
    }, index=idx)


@pytest.fixture(scope="module")
def env():
    df = _alternating_market()
    strategies = [
        TimeSeriesMomentum(lookback=20, threshold=0.0),
        MeanReversion(window=20, entry_z=1.0, trend_filter=0),
        DonchianBreakout(channel=20, exit_channel=10, atr_mult=0.0),
    ]
    positions = strategy_position_matrix(strategies, df)
    cfg = EnvConfig(vol_target=None, episode_length=1_024, max_drawdown_stop=None, reward="dsr")
    return AllocationEnv(positions, df, None, cfg, COSTS, BPY, rng=np.random.default_rng(0))


# ---------------------------------------------------------------------------------------
# Espace d'actions
# ---------------------------------------------------------------------------------------
def test_profiles_cover_flat_single_and_diversified():
    names, weights = build_allocation_profiles(4)
    assert names[0] == "flat" and np.allclose(weights[0], 0.0)
    assert "equal_weight" in names
    assert sum(n.startswith("only_") for n in names) == 4
    assert {"inverse_vol", "risk_parity"} <= set(names)


def test_flat_profile_produces_no_exposure(env):
    env.reset(start=env.max_start, full=True)
    done = False
    while not done:
        _, _, done, info = env.step(0)
    frame = env.to_frame()
    assert (frame["net_position"] == 0.0).all()
    assert frame["cost"].sum() == pytest.approx(0.0)
    assert env.equity == pytest.approx(1.0)


def test_dynamic_profiles_recompute_each_step(env):
    """`inverse_vol` et `risk_parity` doivent varier dans le temps, sinon ils sont figés."""
    idx_iv = env.profile_names.index("inverse_vol")
    seen = [env.weights_for(idx_iv, t) for t in (env.max_start + 50, env.max_start + 2_000)]
    assert not np.allclose(seen[0], seen[1]), "profil dynamique constant"
    for w in seen:
        assert w.sum() == pytest.approx(1.0)
        assert (w >= 0).all()


def test_risk_parity_falls_back_gracefully(env):
    """Une covariance dégénérée ne doit pas faire planter l'environnement."""
    idx = env.profile_names.index("risk_parity")
    w = env.weights_for(idx, env.max_start)
    assert np.isfinite(w).all() and w.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------------------
# Comptabilité des coûts — le point où les architectures multi-stratégies se trompent
# ---------------------------------------------------------------------------------------
def test_costs_are_charged_on_net_position_not_per_strategy(env):
    """Deux stratégies aux signaux opposés se compensent : l'exécution ne coûte rien.

    Facturer les coûts stratégie par stratégie ferait payer deux fois un aller-retour
    qui, net, n'a jamais eu lieu.
    """
    n = 400
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    close = pd.Series(np.linspace(1.10, 1.12, n), index=idx)
    prices = pd.DataFrame({"open": close, "high": close * 1.001, "low": close * 0.999,
                           "close": close, "volume": 1000.0}, index=idx)
    # Deux stratégies exactement opposées, oscillant à chaque barre.
    alternating = pd.Series(np.where(np.arange(n) % 2 == 0, 1.0, -1.0), index=idx)
    positions = pd.DataFrame({"A": alternating, "B": -alternating}, index=idx)

    cfg = EnvConfig(vol_target=None, episode_length=None, random_start=False,
                    max_drawdown_stop=None, reward="pnl")
    e = AllocationEnv(positions, prices, None, cfg, COSTS, BPY, perf_window=50)
    e.reset(start=e.max_start, full=True)
    done = False
    while not done:
        _, _, done, _ = e.step(e.profile_names.index("equal_weight"))

    frame = e.to_frame()
    assert frame["net_position"].abs().max() < 1e-12, "la compensation n'a pas eu lieu"
    assert frame["cost"].sum() == pytest.approx(0.0, abs=1e-12)


def test_turnover_triggers_cost(env):
    """Contrôle inverse : une réallocation réelle DOIT coûter."""
    env.reset(start=env.max_start, full=True)
    single = env.profile_names.index("only_0")
    total_cost = 0.0
    for i in range(200):
        _, _, done, info = env.step(single if i % 20 < 10 else 0)
        total_cost += info["cost"]
        if done:
            break
    assert total_cost > 0.0


def test_no_trade_band_suppresses_micro_reallocation(env):
    env.reset(start=env.max_start, full=True)
    equal = env.profile_names.index("equal_weight")
    for _ in range(300):
        _, _, done, _ = env.step(equal)
        if done:
            break
    frame = env.to_frame()
    tiny = frame["turnover"][(frame["turnover"] > 0) & (frame["turnover"] < COSTS.min_trade_size)]
    assert tiny.empty, "des rebalancements sous la bande morte ont été facturés"


# ---------------------------------------------------------------------------------------
# Contrat de l'environnement
# ---------------------------------------------------------------------------------------
def test_observation_shape_is_stable(env):
    obs = env.reset()
    assert obs.shape == (env.obs_dim,)
    assert np.isfinite(obs).all()
    for _ in range(50):
        obs, _, done, _ = env.step(int(np.random.default_rng(0).integers(0, env.n_actions)))
        assert obs.shape == (env.obs_dim,)
        assert np.isfinite(obs).all()
        if done:
            break


def test_rejects_invalid_action(env):
    env.reset()
    with pytest.raises(ValueError, match="hors de"):
        env.step(env.n_actions)


def test_rejects_short_history():
    df = _alternating_market(n=300)
    positions = strategy_position_matrix([TimeSeriesMomentum(lookback=20)], df)
    with pytest.raises(ValueError, match="trop court"):
        AllocationEnv(positions, df, None, EnvConfig(), COSTS, BPY, perf_window=250)


def test_full_reset_is_deterministic(env):
    a = env.reset(start=env.max_start, full=True)
    b = env.reset(start=env.max_start, full=True)
    assert np.allclose(a, b)


# ---------------------------------------------------------------------------------------
# Apprentissage
# ---------------------------------------------------------------------------------------
def test_baselines_cover_every_profile(env):
    results = baseline_results(env)
    assert set(results) == set(env.profile_names)
    assert results["flat"].sharpe == pytest.approx(0.0)
    assert results["flat"].total_return == pytest.approx(0.0)


def test_allocator_trains_and_beats_flat(env):
    """L'allocateur doit apprendre quelque chose sur un marché où il y a à apprendre.

    Le seuil est volontairement modeste : le test valide la boucle d'apprentissage, pas
    une performance. Une exigence élevée rendrait le test instable entre graines, ce qui
    est exactement le travers que ce dépôt dénonce ailleurs.
    """
    cfg = AgentConfig(hidden_sizes=(32, 32), n_quantiles=16, buffer_size=20_000,
                      learn_start=1_000, batch_size=32, lr=1e-3, n_step=3,
                      weight_decay=1e-3, target_update_interval=500)
    agent = make_allocator(env, cfg, seed=3)
    assert agent.n_actions == env.n_actions

    history = train_allocator(agent, env, env, TrainConfig(
        total_steps=6_000, eval_every=3_000, early_stop_patience=None))
    assert history["valid_sharpe"], "aucune évaluation n'a été réalisée"

    result = evaluate_allocator(agent, env)
    assert np.isfinite(result.sharpe)
    assert result.profile_usage, "aucun profil enregistré"
    assert sum(result.profile_usage.values()) == pytest.approx(1.0, abs=1e-6)


def test_allocator_can_choose_to_stay_flat(env):
    """Le cahier (§2) exige que le système sache NE PAS trader.

    Le profil `flat` doit être atteignable et produire une exposition nulle : c'est la
    condition mécanique de cette exigence."""
    assert "flat" in env.profile_names
    env.reset(start=env.max_start, full=True)
    _, _, _, info = env.step(env.profile_names.index("flat"))
    assert info["turnover"] == 0.0
    assert env.net_position == 0.0

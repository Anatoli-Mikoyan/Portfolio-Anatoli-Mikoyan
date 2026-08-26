"""Cohérence entre l'environnement RL et le moteur de backtest, et exactitude des coûts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qbot.backtest import run_backtest
from qbot.config import CostConfig, EnvConfig
from qbot.env import CostModel, make_env_from_frames
from qbot.features import FeaturePipeline, align_features_prices


@pytest.fixture(scope="module")
def aligned(ohlcv, feature_cfg):
    pipe = FeaturePipeline(feature_cfg)
    x = pipe.fit_transform(ohlcv)
    return align_features_prices(x, ohlcv)


def test_buy_and_hold_replication(aligned, env_cfg, zero_cost):
    """Position longue constante sans coûts == acheter et conserver, au flottant près."""
    xa, pa = aligned
    cfg = EnvConfig(**{**env_cfg.__dict__, "positions": (0.0, 1.0)})
    env = make_env_from_frames(xa, pa, cfg, zero_cost)
    env.reset(full=True)
    done = False
    while not done:
        _, _, done, _ = env.step(1)

    hist = env.to_frame()
    expected = pa["close"].iloc[hist["t"].iloc[-1] + 1] / pa["close"].iloc[hist["t"].iloc[0]] - 1.0
    assert abs((env.equity - 1.0) - expected) < 1e-10


def test_engine_matches_env(aligned, env_cfg):
    """Le moteur de backtest et l'environnement doivent produire la MÊME comptabilité.

    Sans cette garantie, un agent pourrait « gagner » en backtest simplement parce que
    l'évaluation est plus clémente que l'entraînement.
    """
    xa, pa = aligned
    costs = CostConfig()
    env = make_env_from_frames(xa, pa, env_cfg, costs)
    rng = np.random.default_rng(11)
    actions = rng.integers(0, env.n_actions, env.n_bars)

    env.reset(full=True)
    done = False
    while not done:
        _, _, done, _ = env.step(int(actions[env.t]))

    hist = env.to_frame()
    positions = np.zeros(len(pa))
    positions[hist["t"].to_numpy()] = hist["position"].to_numpy()
    res = run_backtest(positions, pa, costs, env_cfg, 6240.0)
    sub = res.frame.iloc[hist["t"].iloc[0]: hist["t"].iloc[-1] + 1]

    assert abs(env.equity - float(np.prod(1.0 + sub["net_return"].to_numpy()))) < 1e-9


def test_costs_are_actually_charged(aligned, env_cfg):
    """Doubler le spread doit dégrader strictement une stratégie qui trade."""
    xa, pa = aligned
    rng = np.random.default_rng(3)
    positions = rng.choice([-1.0, 0.0, 1.0], size=len(pa))

    cheap = run_backtest(positions, pa, CostConfig(spread_bps=1.0), env_cfg, 6240.0)
    dear = run_backtest(positions, pa, CostConfig(spread_bps=10.0), env_cfg, 6240.0)
    assert dear.report.total_return < cheap.report.total_return
    assert dear.frame["cost"].sum() > cheap.frame["cost"].sum() * 2.0


def test_no_trade_band_suppresses_micro_rebalancing(aligned, env_cfg):
    xa, pa = aligned
    positions = np.full(len(pa), 0.5)
    positions[::2] = 0.52                                   # oscillation de 0.02
    wide = run_backtest(positions, pa, CostConfig(min_trade_size=0.1), env_cfg, 6240.0)
    narrow = run_backtest(positions, pa, CostConfig(min_trade_size=0.0), env_cfg, 6240.0)
    assert wide.frame["turnover"].sum() < narrow.frame["turnover"].sum() * 0.1


def test_cost_model_components():
    cm = CostModel(CostConfig(spread_bps=2.0, commission_bps=0.5, slippage_model="sqrt",
                              slippage_coef=0.1, financing_bps_per_bar=0.01))
    assert cm.half_spread() == pytest.approx(1e-4)
    assert cm.commission(2.0) == pytest.approx(1e-4)
    # Impact en racine : quadrupler la taille ne double que le coût unitaire.
    assert cm.slippage(4.0, 0.01) == pytest.approx(2.0 * cm.slippage(1.0, 0.01))
    assert cm.financing(0.5) == pytest.approx(0.5e-6)
    assert cm.breakeven_move_bps() == pytest.approx(3.0)


def test_drawdown_circuit_breaker_stops_episode(aligned, zero_cost):
    """Un agent qui perd doit voir son épisode coupé au seuil de drawdown."""
    xa, pa = aligned
    cfg = EnvConfig(window=16, positions=(-1.0, 0.0, 1.0), vol_target=None,
                    episode_length=None, random_start=False, max_drawdown_stop=0.02)
    env = make_env_from_frames(xa, pa, cfg, CostConfig(spread_bps=200.0))
    env.reset(full=True)
    done, blown = False, False
    while not done:
        _, _, done, info = env.step(0)
        blown = info["blown_up"]
    assert blown, "le coupe-circuit de drawdown ne s'est pas déclenché"
    assert env.drawdown <= -0.02


def test_vol_targeting_reduces_dispersion(aligned, zero_cost):
    """Le vol targeting doit rapprocher la volatilité réalisée de la cible."""
    xa, pa = aligned
    positions = np.ones(len(pa))
    plain = run_backtest(positions, pa, zero_cost, EnvConfig(vol_target=None), 6240.0)
    scaled = run_backtest(positions, pa, zero_cost,
                          EnvConfig(vol_target=0.10, max_leverage=3.0), 6240.0)
    assert abs(scaled.report.ann_volatility - 0.10) < abs(plain.report.ann_volatility - 0.10)

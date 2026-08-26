"""Moteur de backtest vectorisé.

Séparé de `TradingEnv` à dessein : l'environnement sert à ENTRAÎNER (une transition à la
fois, avec récompense), le moteur sert à ÉVALUER n'importe quelle série de positions —
agent RL, règle simple, benchmark — avec exactement la même comptabilité de coûts.

Les deux implémentations partagent `CostModel` et sont vérifiées identiques par les tests :
c'est la garantie qu'un agent ne « gagne » pas en backtest grâce à une comptabilité plus
clémente que celle sous laquelle il a appris.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..config import CostConfig, EnvConfig
from ..env.costs import CostModel
from .metrics import PerformanceReport, compute_report


@dataclass
class BacktestResult:
    """Résultat complet d'un backtest, auditable barre par barre."""
    frame: pd.DataFrame
    report: PerformanceReport
    bars_per_year: float

    @property
    def returns(self) -> np.ndarray:
        return self.frame["net_return"].to_numpy()

    @property
    def equity(self) -> pd.Series:
        return self.frame["equity"]

    def __str__(self) -> str:  # pragma: no cover - affichage
        return str(self.report)


def run_backtest(
    positions: np.ndarray | pd.Series,
    prices: pd.DataFrame,
    cost_cfg: Optional[CostConfig] = None,
    env_cfg: Optional[EnvConfig] = None,
    bars_per_year: float = 6240.0,
    n_trials: int = 1,
    sharpe_std: Optional[float] = None,
) -> BacktestResult:
    """Simule une série de positions cibles.

    `positions[t]` est la position décidée avec l'information disponible à la CLÔTURE de
    la barre t. Le rendement encaissé est celui de la barre t+1 — jamais celui de t.
    Le décalage est appliqué ici, une fois pour toutes : c'est le seul endroit du dépôt
    où cette convention est matérialisée, ce qui évite les doubles décalages.
    """
    cost_cfg = cost_cfg or CostConfig()
    env_cfg = env_cfg or EnvConfig()
    cost_model = CostModel(cost_cfg)

    pos = (positions.to_numpy(dtype=float) if isinstance(positions, pd.Series)
           else np.asarray(positions, dtype=float))
    if len(pos) != len(prices):
        raise ValueError(f"Désalignement positions/prix : {len(pos)} vs {len(prices)}")

    close = prices["close"].to_numpy(dtype=float)
    open_ = prices["open"].to_numpy(dtype=float) if "open" in prices else close
    n = len(close)

    # --- rendement forward, selon la convention d'exécution ----------------------------
    fwd = np.zeros(n, dtype=float)
    if env_cfg.execution == "close":
        fwd[:-1] = close[1:] / close[:-1] - 1.0
        valid_end = n - 1
    elif env_cfg.execution == "next_open":
        fwd[:-2] = open_[2:] / open_[1:-1] - 1.0
        valid_end = n - 2
    else:
        raise ValueError(f"execution inconnue : {env_cfg.execution}")

    bar_ret = np.zeros(n, dtype=float)
    bar_ret[1:] = close[1:] / close[:-1] - 1.0
    w = max(int(env_cfg.vol_target_window), 2)
    bar_vol = pd.Series(bar_ret).rolling(w, min_periods=2).std(ddof=0).bfill().fillna(1e-4).to_numpy()

    # --- vol targeting (causal, borné) --------------------------------------------------
    if env_cfg.vol_target:
        ann_vol = bar_vol * np.sqrt(bars_per_year)
        vol_scalar = np.clip(env_cfg.vol_target / np.maximum(ann_vol, 1e-6), 0.0, env_cfg.max_leverage)
    else:
        vol_scalar = np.ones(n, dtype=float)

    target = np.clip(pos * vol_scalar, -env_cfg.max_leverage, env_cfg.max_leverage)

    spread_bps = None
    if cost_cfg.spread_col and cost_cfg.spread_col in prices.columns:
        spread_bps = (prices[cost_cfg.spread_col] / prices["close"] * 1e4).to_numpy(dtype=float)

    # --- boucle de rebalancement --------------------------------------------------------
    # Non vectorisable : la bande de non-négociation rend la position à t dépendante de
    # la position effectivement retenue à t-1, pas de la position théorique.
    held = np.zeros(n, dtype=float)
    turnover = np.zeros(n, dtype=float)
    costs = np.zeros(n, dtype=float)
    prev = 0.0
    n_trades = 0

    for t in range(valid_end):
        want = target[t]
        to = abs(want - prev)
        if to < cost_cfg.min_trade_size:
            want, to = prev, 0.0
        elif to > 0:
            n_trades += 1
        sp = float(spread_bps[t]) if spread_bps is not None else None
        costs[t] = cost_model.total(to, abs(want), float(bar_vol[t]), spread_bps=sp)
        held[t], turnover[t] = want, to
        prev = want

    gross = held * fwd
    net = gross - costs
    equity = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(equity)

    frame = pd.DataFrame(
        {
            "position": held, "raw_position": pos, "vol_scalar": vol_scalar,
            "turnover": turnover, "cost": costs,
            "gross_return": gross, "net_return": net,
            "equity": equity, "drawdown": equity / peak - 1.0,
            "close": close,
        },
        index=prices.index,
    ).iloc[:valid_end]

    report = compute_report(
        frame["net_return"].to_numpy(), bars_per_year=bars_per_year, n_trials=n_trials,
        turnover=frame["turnover"].to_numpy(), costs=frame["cost"].to_numpy(),
        positions=frame["position"].to_numpy(), n_trades=n_trades, sharpe_std=sharpe_std,
    )
    return BacktestResult(frame=frame, report=report, bars_per_year=bars_per_year)


# =======================================================================================
# Benchmarks — toute stratégie doit être comparée à ces références
# =======================================================================================
def buy_and_hold_positions(n: int) -> np.ndarray:
    return np.ones(n, dtype=float)


def random_positions(n: int, seed: int = 0, choices: tuple = (-1.0, 0.0, 1.0)) -> np.ndarray:
    """Benchmark indispensable : une stratégie aléatoire montre le coût pur du trading.

    Si votre modèle ne bat pas franchement l'aléatoire NET DE COÛTS, il n'apporte rien.
    """
    rng = np.random.default_rng(seed)
    return rng.choice(np.asarray(choices, dtype=float), size=n)


def momentum_positions(close: pd.Series, lookback: int = 20) -> np.ndarray:
    """Momentum trivial : signe du rendement passé. Référence naïve mais redoutable
    en tendance — beaucoup de modèles ML complexes ne la battent pas."""
    sig = np.sign(np.log(close).diff(lookback)).fillna(0.0)
    return sig.to_numpy(dtype=float)


def mean_reversion_positions(close: pd.Series, window: int = 20, z_entry: float = 1.0) -> np.ndarray:
    """Retour à la moyenne : short si le z-score dépasse +z_entry, long en dessous de -z_entry."""
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std(ddof=0)
    z = ((close - ma) / sd.replace(0.0, np.nan)).fillna(0.0)
    return np.clip(-z / z_entry, -1.0, 1.0).to_numpy(dtype=float)

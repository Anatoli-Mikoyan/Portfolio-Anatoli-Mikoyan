"""Environnement d'ALLOCATION entre stratégies (cahier des charges §10).

Le cahier pose la bonne question : « Le RL doit-il choisir Buy/Sell/Hold ou allouer le
capital entre stratégies ? » — et penche pour l'allocation. C'est le bon choix, pour
trois raisons mesurables :

1. **L'espace d'états est infiniment plus petit.** Prédire la direction demande
   d'apprendre la dynamique du prix à partir de centaines de features bruitées. Allouer
   demande d'apprendre quelle stratégie marche dans quel régime : une poignée de
   variables, et un signal beaucoup plus fort.
2. **Les stratégies portent déjà l'hypothèse économique.** Le RL n'a pas à redécouvrir
   le momentum ; il arbitre entre des hypothèses déjà validées individuellement.
3. **L'échec est gracieux.** Un allocateur qui n'apprend rien converge vers l'équipondéré
   ou vers le plat, deux comportements raisonnables. Un agent directionnel qui n'apprend
   rien trade du bruit et paie le spread.

**Prérequis non négociable** : n'allouer qu'entre des stratégies dont l'edge individuel a
survécu au criblage du §8. Répartir du capital entre cinq hypothèses non validées revient
à répartir du bruit, et l'allocateur apprendra le bruit.

Point de comptabilité important. Les coûts sont facturés sur la position NETTE du
portefeuille, pas stratégie par stratégie. Sommer des rendements déjà nets de coûts
double-compterait les frais et, surtout, ignorerait la compensation : deux stratégies aux
signaux opposés s'annulent et ne coûtent rien à exécuter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import CostConfig, EnvConfig
from ..utils.logging import get_logger
from .costs import CostModel
from .rewards import build_reward

log = get_logger("env.allocation")

N_PORTFOLIO_FEATURES = 4


# =======================================================================================
def build_allocation_profiles(n_strategies: int) -> Tuple[List[str], np.ndarray]:
    """Bibliothèque de profils d'allocation — l'espace d'actions discret.

    Pourquoi discret plutôt que des poids continus : le Rainbow (et le Q-learning en
    général) exige un espace d'actions fini, et un simplexe continu demanderait un
    algorithme acteur-critique. Surtout, une poignée de profils lisibles est plus robuste
    qu'un simplexe continu sur des données à faible rapport signal/bruit — chaque degré de
    liberté supplémentaire est une occasion de sur-apprendre.

    Les profils `inverse_vol` et `risk_parity` sont marqués comme dynamiques : leurs poids
    sont recalculés à chaque pas à partir de la covariance récente.
    """
    names = ["flat", "equal_weight"]
    profiles = [np.zeros(n_strategies), np.full(n_strategies, 1.0 / n_strategies)]
    for k in range(n_strategies):
        w = np.zeros(n_strategies)
        w[k] = 1.0
        names.append(f"only_{k}")
        profiles.append(w)
    names += ["inverse_vol", "risk_parity"]
    profiles += [np.full(n_strategies, np.nan), np.full(n_strategies, np.nan)]
    return names, np.asarray(profiles, dtype=float)


@dataclass
class AllocationStep:
    t: int
    timestamp: pd.Timestamp
    action: int
    profile: str
    weights: np.ndarray
    net_position: float
    turnover: float
    cost: float
    gross_return: float
    net_return: float
    equity: float
    drawdown: float


# =======================================================================================
class AllocationEnv:
    """Environnement RL dont l'action est une répartition du capital entre stratégies."""

    def __init__(
        self,
        strategy_positions: pd.DataFrame,      # (T, K) position cible de chaque stratégie
        prices: pd.DataFrame,
        regime_features: Optional[pd.DataFrame] = None,
        env_cfg: Optional[EnvConfig] = None,
        cost_cfg: Optional[CostConfig] = None,
        bars_per_year: float = 6240.0,
        perf_window: int = 250,
        rng: Optional[np.random.Generator] = None,
    ):
        self.cfg = env_cfg or EnvConfig()
        self.cost_model = CostModel(cost_cfg or CostConfig())
        self.bars_per_year = float(bars_per_year)
        self.perf_window = int(perf_window)
        self.rng = rng or np.random.default_rng(0)

        idx = strategy_positions.index.intersection(prices.index)
        if regime_features is not None:
            idx = idx.intersection(regime_features.index)
        if len(idx) < perf_window + 100:
            raise ValueError(f"Historique commun trop court : {len(idx)} barres.")

        self.strategy_names = list(strategy_positions.columns)
        self.n_strategies = len(self.strategy_names)
        self.positions = strategy_positions.loc[idx].to_numpy(dtype=np.float64)
        self.prices = prices.loc[idx]
        self.index = idx
        self.regime = (regime_features.loc[idx].to_numpy(dtype=np.float32)
                       if regime_features is not None
                       else np.zeros((len(idx), 0), dtype=np.float32))

        self.profile_names, self.profile_weights = build_allocation_profiles(self.n_strategies)
        self.n_actions = len(self.profile_names)

        self._precompute()
        self.reward_fn = build_reward(self.cfg)
        self.n_regime_features = self.regime.shape[1]
        self.obs_dim = (self.n_regime_features + 2 * self.n_strategies
                        + self.n_strategies + N_PORTFOLIO_FEATURES)
        self.reset()

    # ---------------------------------------------------------------------------------
    def _precompute(self) -> None:
        close = self.prices["close"].to_numpy(dtype=np.float64)
        n = len(close)

        fwd = np.zeros(n, dtype=np.float64)
        fwd[:-1] = close[1:] / close[:-1] - 1.0
        self.fwd_ret = fwd
        self._last_decision_idx = n - 2

        bar_ret = np.zeros(n, dtype=np.float64)
        bar_ret[1:] = close[1:] / close[:-1] - 1.0
        w = max(int(self.cfg.vol_target_window), 2)
        self.bar_vol = (pd.Series(bar_ret).rolling(w, min_periods=2).std(ddof=0)
                        .bfill().fillna(1e-4).to_numpy())

        # Rendement BRUT de chaque stratégie à exposition unitaire — la brique de base.
        # Les coûts ne sont PAS inclus ici : ils sont facturés une seule fois, sur la
        # position nette du portefeuille (voir docstring du module).
        self.strategy_gross = self.positions * fwd[:, None]

        # Sharpe glissant causal par stratégie : ce que l'allocateur observe de leur forme.
        gross = pd.DataFrame(self.strategy_gross)
        roll_mean = gross.rolling(self.perf_window, min_periods=self.perf_window // 4).mean()
        roll_std = gross.rolling(self.perf_window, min_periods=self.perf_window // 4).std(ddof=0)
        sharpe = (roll_mean / roll_std.replace(0.0, np.nan)) * np.sqrt(self.bars_per_year)
        self.rolling_sharpe = sharpe.fillna(0.0).clip(-5.0, 5.0).to_numpy(dtype=np.float32)
        self.rolling_vol = (roll_std.fillna(0.0) * np.sqrt(self.bars_per_year)).to_numpy(dtype=np.float32)

        if self.cfg.vol_target:
            ann_vol = self.bar_vol * np.sqrt(self.bars_per_year)
            self.vol_scalar = np.clip(self.cfg.vol_target / np.maximum(ann_vol, 1e-6),
                                      0.0, self.cfg.max_leverage)
        else:
            self.vol_scalar = np.ones(n, dtype=np.float64)

    # ---------------------------------------------------------------------------------
    def _dynamic_weights(self, profile: str, t: int) -> np.ndarray:
        """Poids recalculés à chaque pas pour les profils dynamiques."""
        lo = max(t - self.perf_window, 0)
        window = self.strategy_gross[lo: t + 1]
        if window.shape[0] < 20:
            return np.full(self.n_strategies, 1.0 / self.n_strategies)

        if profile == "inverse_vol":
            vol = window.std(axis=0)
            inv = 1.0 / np.maximum(vol, 1e-8)
            return inv / inv.sum()

        # risk_parity : contributions au risque égalisées sur la covariance récente.
        from ..risk import risk_parity_weights

        cov = np.cov(window, rowvar=False)
        cov = np.atleast_2d(cov)
        if cov.shape[0] != self.n_strategies or not np.isfinite(cov).all():
            return np.full(self.n_strategies, 1.0 / self.n_strategies)
        cov = cov + np.eye(self.n_strategies) * 1e-12
        try:
            return risk_parity_weights(cov)
        except Exception:                                     # pragma: no cover
            return np.full(self.n_strategies, 1.0 / self.n_strategies)

    def weights_for(self, action: int, t: int) -> np.ndarray:
        profile = self.profile_names[action]
        w = self.profile_weights[action]
        return self._dynamic_weights(profile, t) if np.isnan(w).any() else w.copy()

    # ---------------------------------------------------------------------------------
    @property
    def max_start(self) -> int:
        return max(self.perf_window, 2)

    def reset(self, start: Optional[int] = None, length: Optional[int] = None,
              full: bool = False) -> np.ndarray:
        ep_len = None if full else (length if length is not None else self.cfg.episode_length)
        lo = self.max_start if start is None else max(int(start), self.max_start)
        hi = self._last_decision_idx

        if ep_len is None or ep_len >= (hi - lo):
            self.t0, self.t_end = lo, hi
        elif full or not self.cfg.random_start:
            self.t0, self.t_end = lo, min(lo + ep_len, hi)
        else:
            self.t0 = int(self.rng.integers(lo, max(hi - ep_len, lo + 1)))
            self.t_end = min(self.t0 + ep_len, hi)

        self.t = self.t0
        self.weights = np.zeros(self.n_strategies, dtype=np.float64)
        self.net_position = 0.0
        self.equity = 1.0
        self.peak_equity = 1.0
        self.drawdown = 0.0
        self.cum_turnover = 0.0
        self.n_reallocations = 0
        self.history: List[AllocationStep] = []
        self.reward_fn.reset()
        return self._observation()

    # ---------------------------------------------------------------------------------
    def _observation(self) -> np.ndarray:
        t = self.t
        parts = [self.regime[t]] if self.n_regime_features else []
        parts.append(self.rolling_sharpe[t])
        parts.append(np.clip(self.rolling_vol[t], 0.0, 2.0))
        parts.append(self.weights.astype(np.float32))
        parts.append(np.array([
            self.net_position,
            self.drawdown,
            np.tanh(self.cum_turnover / max(t - self.t0 + 1, 1) * 10.0),
            float(self.positions[t].mean()),          # consensus directionnel des stratégies
        ], dtype=np.float32))
        return np.concatenate(parts).astype(np.float32, copy=False)

    # ---------------------------------------------------------------------------------
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if not 0 <= action < self.n_actions:
            raise ValueError(f"Action {action} hors de [0, {self.n_actions})")

        t = self.t
        weights = self.weights_for(action, t)

        # Position NETTE : c'est elle qui est exécutée, et elle bénéficie de la
        # compensation entre stratégies aux signaux opposés.
        raw_net = float(weights @ self.positions[t])
        target = float(np.clip(raw_net * self.vol_scalar[t],
                               -self.cfg.max_leverage, self.cfg.max_leverage))

        turnover = abs(target - self.net_position)
        if turnover < self.cost_model.cfg.min_trade_size:
            target, turnover = self.net_position, 0.0
        elif turnover > 0:
            self.n_reallocations += 1

        cost = self.cost_model.total(turnover, abs(target), float(self.bar_vol[t]))
        gross = target * float(self.fwd_ret[t])
        net = gross - cost
        if self.cfg.turnover_penalty:
            net -= self.cfg.turnover_penalty * turnover

        self.weights = weights
        self.net_position = target
        self.cum_turnover += turnover
        self.equity *= (1.0 + net)
        self.peak_equity = max(self.peak_equity, self.equity)
        self.drawdown = self.equity / self.peak_equity - 1.0

        reward = self.reward_fn(net, position=target, turnover=turnover,
                                drawdown=self.drawdown)

        self.history.append(AllocationStep(
            t=t, timestamp=self.index[t], action=action, profile=self.profile_names[action],
            weights=weights.copy(), net_position=target, turnover=turnover, cost=cost,
            gross_return=gross, net_return=net, equity=self.equity, drawdown=self.drawdown,
        ))

        self.t += 1
        truncated = self.t >= self.t_end
        blown_up = (self.cfg.max_drawdown_stop is not None
                    and self.drawdown <= -abs(self.cfg.max_drawdown_stop))
        done = bool(truncated or blown_up or self.equity <= 1e-6)

        info = {"net_return": net, "cost": cost, "turnover": turnover,
                "equity": self.equity, "drawdown": self.drawdown,
                "profile": self.profile_names[action], "blown_up": blown_up}
        return self._observation(), float(reward), done, info

    # ---------------------------------------------------------------------------------
    def to_frame(self) -> pd.DataFrame:
        if not self.history:
            return pd.DataFrame()
        rows = []
        for h in self.history:
            row = {k: v for k, v in h.__dict__.items() if k != "weights"}
            for name, w in zip(self.strategy_names, h.weights):
                row[f"w_{name}"] = w
            rows.append(row)
        return pd.DataFrame(rows).set_index("timestamp")

    def summary(self) -> Dict[str, float]:
        df = self.to_frame()
        if df.empty:
            return {}
        rets = df["net_return"].to_numpy()
        sd = float(rets.std(ddof=0))
        usage = df["profile"].value_counts(normalize=True)
        return {
            "n_steps": int(len(df)),
            "total_return": float(self.equity - 1.0),
            "sharpe": float(rets.mean() / sd * np.sqrt(self.bars_per_year)) if sd > 1e-12 else 0.0,
            "max_drawdown": float(df["drawdown"].min()),
            "turnover_per_bar": float(df["turnover"].mean()),
            "n_reallocations": int(self.n_reallocations),
            "profil_dominant": str(usage.index[0]),
            "part_profil_dominant": float(usage.iloc[0]),
            "part_flat": float(usage.get("flat", 0.0)),
        }


# =======================================================================================
def strategy_position_matrix(strategies: Sequence, prices: pd.DataFrame) -> pd.DataFrame:
    """Assemble la matrice (T, K) des positions cibles de chaque stratégie."""
    return pd.DataFrame(
        {type(s).__name__: s.signal(prices) for s in strategies}, index=prices.index
    )

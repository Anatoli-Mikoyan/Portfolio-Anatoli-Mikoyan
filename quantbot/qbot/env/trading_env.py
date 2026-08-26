"""Environnement de trading pour apprentissage par renforcement.

API compatible Gymnasium (`reset` / `step`) mais sans dépendance à gym : le contrat est
suffisamment simple pour ne pas justifier une dépendance externe supplémentaire.

Conventions temporelles — c'est ici que se jouent la plupart des fuites de données :

    t                  : l'agent observe les features calculées avec l'information ≤ clôture de t
    action a_t         : position CIBLE à détenir sur la barre suivante
    exécution          : "close"     -> au cours de clôture de t (ordre MOC / fin de barre)
                         "next_open" -> à l'ouverture de t+1 (plus conservateur)
    rendement encaissé : celui de la barre t+1, jamais celui de la barre t

L'agent ne peut donc JAMAIS agir sur une information qu'il n'avait pas au moment de décider.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import CostConfig, EnvConfig
from ..utils.logging import get_logger
from .costs import CostModel
from .rewards import build_reward

log = get_logger("env.trading")

N_PORTFOLIO_FEATURES = 6


@dataclass
class StepInfo:
    """Détail complet d'un pas — indispensable pour auditer un backtest ligne à ligne."""
    t: int
    timestamp: pd.Timestamp
    position: float
    target_position: float
    turnover: float
    gross_return: float
    cost: float
    net_return: float
    equity: float
    drawdown: float
    vol_scalar: float
    price: float


class TradingEnv:
    """Environnement à espace d'actions discret sur une série d'actifs unique."""

    def __init__(
        self,
        features: np.ndarray,
        prices: pd.DataFrame,
        env_cfg: Optional[EnvConfig] = None,
        cost_cfg: Optional[CostConfig] = None,
        bars_per_year: float = 6240.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self.cfg = env_cfg or EnvConfig()
        self.cost_model = CostModel(cost_cfg or CostConfig())
        self.bars_per_year = float(bars_per_year)
        self.rng = rng or np.random.default_rng(0)

        self.features = np.asarray(features, dtype=np.float32)
        if self.features.ndim != 2:
            raise ValueError(f"features doit être 2D (T, F), reçu {self.features.shape}")
        if len(prices) != self.features.shape[0]:
            raise ValueError(
                f"Désalignement features/prix : {self.features.shape[0]} vs {len(prices)}. "
                "Utiliser align_features_prices() avant de construire l'environnement."
            )

        self.prices = prices
        self.index = prices.index
        self.close = prices["close"].to_numpy(dtype=np.float64)
        self.open = prices["open"].to_numpy(dtype=np.float64) if "open" in prices else self.close
        self.spread_bps = (
            (prices[self.cost_model.cfg.spread_col] / prices["close"] * 1e4).to_numpy(dtype=np.float64)
            if self.cost_model.cfg.spread_col and self.cost_model.cfg.spread_col in prices.columns
            else None
        )

        self.n_bars, self.n_features = self.features.shape
        self.window = int(self.cfg.window)
        self.positions = np.asarray(self.cfg.positions, dtype=np.float64)
        self.n_actions = self.positions.shape[0]

        self._precompute()
        self.reward_fn = build_reward(self.cfg)

        self.obs_dim = self.window * self.n_features + (
            N_PORTFOLIO_FEATURES if self.cfg.include_position_in_obs else 0
        )
        self._validate()
        self.reset()

    # ---------------------------------------------------------------------------------
    def _precompute(self) -> None:
        """Pré-calcule les rendements forward et les volatilités — le pas doit rester O(1)."""
        n = self.n_bars
        fwd = np.zeros(n, dtype=np.float64)
        if self.cfg.execution == "close":
            # Décision à la clôture de t, exécutée à la clôture de t, rendement de t -> t+1.
            fwd[:-1] = self.close[1:] / self.close[:-1] - 1.0
            self._last_decision_idx = n - 2
        elif self.cfg.execution == "next_open":
            # Décision à la clôture de t, exécutée à l'ouverture de t+1, rendement open->open.
            fwd[:-2] = self.open[2:] / self.open[1:-1] - 1.0
            self._last_decision_idx = n - 3
        else:
            raise ValueError(f"execution inconnue : {self.cfg.execution}")
        self.fwd_ret = fwd

        bar_ret = np.zeros(n, dtype=np.float64)
        bar_ret[1:] = self.close[1:] / self.close[:-1] - 1.0
        self.bar_ret = bar_ret

        # Volatilité réalisée causale, utilisée pour l'impact et le vol targeting.
        w = max(int(self.cfg.vol_target_window), 2)
        s = pd.Series(bar_ret)
        self.bar_vol = s.rolling(w, min_periods=2).std(ddof=0).bfill().fillna(1e-4).to_numpy()
        self.ann_vol = self.bar_vol * np.sqrt(self.bars_per_year)

        # Scalaire de vol targeting : borné, et STRICTEMENT causal (vol estimée jusqu'à t).
        if self.cfg.vol_target:
            scalar = self.cfg.vol_target / np.maximum(self.ann_vol, 1e-6)
            self.vol_scalar = np.clip(scalar, 0.0, self.cfg.max_leverage)
        else:
            self.vol_scalar = np.ones(n, dtype=np.float64)

    def _validate(self) -> None:
        if self.n_bars < self.window + 10:
            raise ValueError(f"Série trop courte : {self.n_bars} barres pour une fenêtre de {self.window}.")
        if not np.isfinite(self.features).all():
            raise ValueError("features contient des NaN/inf — nettoyer en amont du pipeline.")

    # ---------------------------------------------------------------------------------
    @property
    def max_start(self) -> int:
        return max(self.window - 1, 0)

    def reset(self, start: Optional[int] = None, length: Optional[int] = None,
              full: bool = False) -> np.ndarray:
        """Démarre un épisode.

        - `full=True` : un unique passage déterministe sur TOUT le segment. C'est le mode
          d'ÉVALUATION. Sans ce drapeau, `episode_length` s'appliquerait aussi à
          l'évaluation, qui ne porterait alors que sur une fenêtre aléatoire — la
          sélection de checkpoint se ferait sur du bruit d'échantillonnage.
        - sinon : épisode de `episode_length` barres, à début aléatoire si configuré.
          Le début aléatoire décorrèle les trajectoires d'entraînement et empêche l'agent
          de mémoriser une unique séquence historique.
        """
        ep_len = None if full else (length if length is not None else self.cfg.episode_length)
        lo = self.max_start if start is None else max(int(start), self.max_start)
        hi = self._last_decision_idx

        if ep_len is None or ep_len >= (hi - lo):
            self.t0, self.t_end = lo, hi
        elif full or not self.cfg.random_start:
            self.t0, self.t_end = lo, min(lo + ep_len, hi)
        elif self.cfg.random_start:
            self.t0 = int(self.rng.integers(lo, max(hi - ep_len, lo + 1)))
            self.t_end = min(self.t0 + ep_len, hi)

        self.t = self.t0
        self.position = 0.0
        self.equity = 1.0
        self.peak_equity = 1.0
        self.drawdown = 0.0
        self.bars_in_position = 0
        self.entry_price = self.close[self.t]
        self.cum_turnover = 0.0
        self.cum_cost = 0.0
        self.n_trades = 0
        self._ret_history: list[float] = []
        self.history: list[StepInfo] = []
        self.reward_fn.reset()
        return self._observation()

    # ---------------------------------------------------------------------------------
    def _observation(self) -> np.ndarray:
        lo = self.t - self.window + 1
        obs = self.features[lo: self.t + 1].ravel()
        if not self.cfg.include_position_in_obs:
            return obs.astype(np.float32, copy=False)

        unrealized = (self.close[self.t] / self.entry_price - 1.0) * np.sign(self.position)
        vol = max(self.bar_vol[self.t], 1e-8)
        recent = np.asarray(self._ret_history[-60:], dtype=np.float64)
        strat_vol = float(recent.std()) if recent.size >= 10 else 0.0

        portfolio = np.array(
            [
                self.position,                                          # exposition courante
                self.drawdown,                                          # drawdown ∈ [-1, 0]
                np.tanh(self.bars_in_position / 50.0),                  # ancienneté, bornée
                np.clip(unrealized / (vol * 10.0), -3.0, 3.0),          # P&L latent en unités de σ
                np.clip(strat_vol / vol, 0.0, 5.0) - 1.0,               # vol stratégie / vol marché
                np.tanh(self.cum_turnover / max(self.t - self.t0 + 1, 1) * 10.0),  # intensité de trading
            ],
            dtype=np.float32,
        )
        return np.concatenate([obs, portfolio]).astype(np.float32, copy=False)

    # ---------------------------------------------------------------------------------
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if not 0 <= action < self.n_actions:
            raise ValueError(f"Action {action} hors de [0, {self.n_actions})")

        t = self.t
        raw_target = float(self.positions[action])

        # --- Couche de risque intégrée à l'environnement ------------------------------
        # Le vol targeting est appliqué PENDANT l'entraînement, pas seulement au déploiement :
        # sinon l'agent apprend une politique pour une échelle de risque et en exécute
        # une autre en production (mismatch classique et coûteux).
        vol_scalar = float(self.vol_scalar[t])
        target = float(np.clip(raw_target * vol_scalar, -self.cfg.max_leverage, self.cfg.max_leverage))

        turnover = abs(target - self.position)
        # Bande de non-négociation : évite de payer le spread pour un ajustement marginal.
        if turnover < self.cost_model.cfg.min_trade_size:
            target, turnover = self.position, 0.0

        spread = float(self.spread_bps[t]) if self.spread_bps is not None else None
        cost = self.cost_model.total(turnover, abs(target), float(self.bar_vol[t]), spread_bps=spread)

        gross = target * float(self.fwd_ret[t])
        net = gross - cost
        if self.cfg.turnover_penalty:
            net -= self.cfg.turnover_penalty * turnover
        if self.cfg.holding_penalty:
            net -= self.cfg.holding_penalty * abs(target)

        # --- Mise à jour de l'état ------------------------------------------------------
        if turnover > 0:
            self.n_trades += 1
            if np.sign(target) != np.sign(self.position) or self.position == 0.0:
                self.entry_price = self.close[t]
                self.bars_in_position = 0
        self.position = target
        self.bars_in_position += 1
        self.cum_turnover += turnover
        self.cum_cost += cost

        self.equity *= (1.0 + net)
        self.peak_equity = max(self.peak_equity, self.equity)
        self.drawdown = self.equity / self.peak_equity - 1.0
        self._ret_history.append(net)

        reward = self.reward_fn(net, position=target, turnover=turnover, drawdown=self.drawdown)

        self.history.append(
            StepInfo(
                t=t, timestamp=self.index[t], position=target, target_position=raw_target,
                turnover=turnover, gross_return=gross, cost=cost, net_return=net,
                equity=self.equity, drawdown=self.drawdown, vol_scalar=vol_scalar,
                price=float(self.close[t]),
            )
        )

        self.t += 1
        truncated = self.t >= self.t_end
        blown_up = (
            self.cfg.max_drawdown_stop is not None
            and self.drawdown <= -abs(self.cfg.max_drawdown_stop)
        )
        done = bool(truncated or blown_up or self.equity <= 1e-6)

        info = {
            "net_return": net, "gross_return": gross, "cost": cost, "turnover": turnover,
            "equity": self.equity, "drawdown": self.drawdown, "position": target,
            "blown_up": blown_up, "timestamp": self.index[t],
        }
        obs = self._observation() if not done else self._terminal_observation()
        return obs, float(reward), done, info

    def _terminal_observation(self) -> np.ndarray:
        self.t = min(self.t, self.n_bars - 1)
        return self._observation()

    # ---------------------------------------------------------------------------------
    def to_frame(self) -> pd.DataFrame:
        """Historique de l'épisode sous forme tabulaire, prêt pour le calcul de métriques."""
        if not self.history:
            return pd.DataFrame()
        rows = [h.__dict__ for h in self.history]
        out = pd.DataFrame(rows).set_index("timestamp")
        return out

    def summary(self) -> Dict[str, float]:
        df = self.to_frame()
        if df.empty:
            return {}
        rets = df["net_return"].to_numpy()
        ann = np.sqrt(self.bars_per_year)
        sd = float(rets.std(ddof=0))
        return {
            "n_steps": int(len(df)),
            "total_return": float(self.equity - 1.0),
            "sharpe": float(rets.mean() / sd * ann) if sd > 1e-12 else 0.0,
            "max_drawdown": float(df["drawdown"].min()),
            "turnover_per_bar": float(df["turnover"].mean()),
            "cost_drag_annual": float(df["cost"].mean() * self.bars_per_year),
            "exposure": float(df["position"].abs().mean()),
            "n_trades": int(self.n_trades),
        }


def make_env_from_frames(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    env_cfg: Optional[EnvConfig] = None,
    cost_cfg: Optional[CostConfig] = None,
    bars_per_year: float = 6240.0,
    rng: Optional[np.random.Generator] = None,
) -> TradingEnv:
    """Construit un environnement en garantissant l'alignement strict des index."""
    idx = features.index.intersection(prices.index)
    f = features.loc[idx]
    p = prices.loc[idx]
    return TradingEnv(f.to_numpy(dtype=np.float32), p, env_cfg, cost_cfg, bars_per_year, rng)

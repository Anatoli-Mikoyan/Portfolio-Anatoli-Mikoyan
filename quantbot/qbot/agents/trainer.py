"""Boucle d'entraînement avec sélection de modèle sur validation.

Le point critique n'est pas la boucle elle-même mais la DISCIPLINE de sélection :

  * l'entraînement se fait sur le segment `train`,
  * le choix du meilleur checkpoint se fait sur `valid` (jamais vu à l'entraînement),
  * le segment `test` n'est touché qu'une seule fois, à la toute fin.

Sélectionner le checkpoint sur les données d'entraînement — ou pire, réutiliser le test
pour choisir les hyperparamètres — produit mécaniquement un Sharpe de backtest élevé et
sans aucune valeur prédictive. C'est la cause n°1 d'échec des fonds ML (López de Prado).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from ..config import AgentConfig, TrainConfig
from ..env import TradingEnv, N_PORTFOLIO_FEATURES
from ..utils.logging import get_logger
from .rainbow import RainbowAgent, resolve_device
from .replay import NStepAccumulator, Transition

log = get_logger("agents.trainer")


@dataclass
class TrainHistory:
    steps: List[int] = field(default_factory=list)
    train_reward: List[float] = field(default_factory=list)
    valid_sharpe: List[float] = field(default_factory=list)
    valid_return: List[float] = field(default_factory=list)
    valid_maxdd: List[float] = field(default_factory=list)
    loss: List[float] = field(default_factory=list)

    def best_index(self, metric: str = "sharpe") -> int:
        arr = {"sharpe": self.valid_sharpe, "return": self.valid_return}.get(metric, self.valid_sharpe)
        return int(np.argmax(arr)) if arr else -1


@dataclass
class EvalResult:
    sharpe: float
    total_return: float
    max_drawdown: float
    n_trades: int
    turnover: float
    exposure: float
    equity_curve: np.ndarray


def evaluate(agent: RainbowAgent, env: TradingEnv, cvar_alpha: Optional[float] = None) -> EvalResult:
    """Évaluation déterministe sur le segment COMPLET (politique greedy, bruit désactivé).

    `full=True` est indispensable : sans lui, `episode_length` tronquerait l'évaluation à
    une fenêtre aléatoire et le Sharpe de validation serait dominé par le bruit
    d'échantillonnage — on sélectionnerait alors le checkpoint le plus chanceux, pas le
    meilleur.
    """
    agent.bind_features(env.features)
    obs = env.reset(start=env.max_start, full=True)
    done = False
    while not done:
        action = agent.act(obs, greedy=True, cvar_alpha=cvar_alpha)
        obs, _, done, _ = env.step(action)

    df = env.to_frame()
    if df.empty:
        return EvalResult(0.0, 0.0, 0.0, 0, 0.0, 0.0, np.array([1.0]))
    rets = df["net_return"].to_numpy()
    sd = float(rets.std(ddof=0))
    ann = np.sqrt(env.bars_per_year)
    return EvalResult(
        sharpe=float(rets.mean() / sd * ann) if sd > 1e-12 else 0.0,
        total_return=float(env.equity - 1.0),
        max_drawdown=float(df["drawdown"].min()),
        n_trades=int(env.n_trades),
        turnover=float(df["turnover"].mean()),
        exposure=float(df["position"].abs().mean()),
        equity_curve=df["equity"].to_numpy(),
    )


class Trainer:
    """Orchestre l'interaction agent/environnement et la sélection de modèle."""

    def __init__(
        self,
        agent: RainbowAgent,
        train_env: TradingEnv,
        valid_env: Optional[TradingEnv] = None,
        cfg: Optional[TrainConfig] = None,
        checkpoint_path: Optional[str | Path] = None,
    ):
        self.agent = agent
        self.train_env = train_env
        self.valid_env = valid_env
        self.cfg = cfg or TrainConfig()
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.history = TrainHistory()
        self.best_metric = -np.inf
        self.best_state: Optional[dict] = None

    # ---------------------------------------------------------------------------------
    def fit(self, total_steps: Optional[int] = None, progress: Optional[Callable] = None) -> TrainHistory:
        steps = int(total_steps or self.cfg.total_steps)
        agent, env = self.agent, self.train_env
        agent.bind_features(env.features)

        acc = NStepAccumulator(agent.n_step, agent.gamma)
        obs = env.reset()
        ep_reward, ep_count, patience = 0.0, 0, 0
        loss_window: List[float] = []
        t_start = time.time()

        for step in range(1, steps + 1):
            t_idx = env.t
            portfolio = obs[-N_PORTFOLIO_FEATURES:].copy() if agent.n_portfolio else np.zeros(0, np.float32)

            action = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            agent.env_steps += 1
            ep_reward += reward

            next_portfolio = (next_obs[-N_PORTFOLIO_FEATURES:].copy()
                              if agent.n_portfolio else np.zeros(0, np.float32))
            collapsed = acc.push(Transition(
                t=t_idx, portfolio=portfolio, action=action, reward=reward,
                next_t=min(env.t, env.n_bars - 1), next_portfolio=next_portfolio, done=done,
            ))
            if collapsed is not None:
                agent.buffer.add(collapsed)

            if done:
                # Vider l'accumulateur : sans cela, les n-1 dernières transitions de chaque
                # épisode seraient perdues — un biais systématique contre les fins d'épisode,
                # c'est-à-dire précisément contre les scénarios de drawdown maximal.
                for tr in acc.flush():
                    agent.buffer.add(tr)
                acc.reset()
                ep_count += 1
                obs = env.reset()
                ep_reward = 0.0
            else:
                obs = next_obs

            if step % max(self.cfg_train_freq, 1) == 0:
                metrics = agent.learn()
                if metrics:
                    loss_window.append(metrics["loss"])

            if step % max(self.cfg.log_every, 1) == 0:
                rate = step / max(time.time() - t_start, 1e-9)
                log.info(
                    "step %7d/%d | eps=%d | loss=%.4f | buffer=%d | %.0f pas/s",
                    step, steps, ep_count,
                    float(np.mean(loss_window[-200:])) if loss_window else float("nan"),
                    len(agent.buffer), rate,
                )

            if self.valid_env is not None and step % max(self.cfg.eval_every, 1) == 0:
                stop = self._run_eval(step, loss_window)
                obs = env.reset()          # l'évaluation a modifié l'état de train_env ? non,
                acc.reset()                # mais on repart proprement pour éviter tout mélange
                agent.bind_features(env.features)
                if stop:
                    log.info("Arrêt anticipé au pas %d (patience épuisée).", step)
                    break

            if progress is not None:
                progress(step, steps)

        if self.best_state is not None:
            self.agent.online.load_state_dict(self.best_state)
            self.agent.target.load_state_dict(self.best_state)
            log.info("Meilleur checkpoint restauré (%s validation = %.3f).",
                     self.cfg.early_stop_metric, self.best_metric)
        return self.history

    @property
    def cfg_train_freq(self) -> int:
        return getattr(self.agent.cfg, "train_freq", 1)

    # ---------------------------------------------------------------------------------
    def _run_eval(self, step: int, loss_window: List[float]) -> bool:
        res = evaluate(self.agent, self.valid_env)
        self.history.steps.append(step)
        self.history.valid_sharpe.append(res.sharpe)
        self.history.valid_return.append(res.total_return)
        self.history.valid_maxdd.append(res.max_drawdown)
        self.history.loss.append(float(np.mean(loss_window[-200:])) if loss_window else float("nan"))

        metric = {"sharpe": res.sharpe, "return": res.total_return,
                  "calmar": res.total_return / abs(min(res.max_drawdown, -1e-9))
                  }.get(self.cfg.early_stop_metric, res.sharpe)

        log.info(
            "  eval @%d | sharpe=%6.3f | ret=%7.2f%% | maxDD=%6.2f%% | trades=%4d | expo=%.2f",
            step, res.sharpe, 100 * res.total_return, 100 * res.max_drawdown,
            res.n_trades, res.exposure,
        )

        improved = metric > self.best_metric
        if improved:
            self.best_metric = metric
            self.best_state = {k: v.detach().cpu().clone()
                               for k, v in self.agent.online.state_dict().items()}
            if self.checkpoint_path:
                self.agent.save(self.checkpoint_path)
            self._since_improve = 0
        else:
            self._since_improve = getattr(self, "_since_improve", 0) + 1

        if self.cfg.early_stop_patience is None:
            return False
        return self._since_improve >= self.cfg.early_stop_patience

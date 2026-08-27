"""Entraînement de l'allocateur RL (cahier des charges §10).

L'agent est le même Rainbow que pour la décision directionnelle — c'est tout l'intérêt
d'avoir séparé l'algorithme de l'environnement. Seule change la nature de l'action :
au lieu de choisir une position, il choisit une RÉPARTITION du capital entre stratégies.

Détail d'implémentation. Le tampon de rejeu du Rainbow ne stocke pas les observations en
clair : il conserve (indice temporel, état de portefeuille) et reconstruit la fenêtre à
la volée, ce qui divise la mémoire par cent. Cette astuce suppose que l'observation soit
entièrement déterminée par l'instant t — vrai pour l'environnement directionnel, FAUX
ici, puisque l'observation contient les poids courants, qui dépendent des décisions
passées de l'agent.

L'observation de l'allocateur étant petite (quelques dizaines de flottants), on la stocke
donc en clair, via un reconstructeur identité obtenu en liant une matrice de features à
zéro colonne. C'est explicite plutôt qu'astucieux : `bind_direct_observations` porte ce
choix et son motif.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..config import AgentConfig, TrainConfig
from ..env.allocation_env import AllocationEnv
from ..utils.logging import get_logger
from .rainbow import RainbowAgent
from .replay import NStepAccumulator, Transition

log = get_logger("agents.allocator")


def bind_direct_observations(agent: RainbowAgent) -> None:
    """Configure le tampon pour stocker les observations en clair.

    Une matrice de features à zéro colonne fait renvoyer au reconstructeur exactement le
    vecteur stocké, sans fenêtre ni recollage. C'est la bonne solution ici parce que
    l'observation de l'allocateur dépend de l'historique de ses propres décisions et
    n'est donc pas reconstructible depuis le seul indice temporel.
    """
    agent.bind_features(np.zeros((1, 0), dtype=np.float32))


def make_allocator(env: AllocationEnv, cfg: Optional[AgentConfig] = None,
                   seed: int = 0) -> RainbowAgent:
    """Instancie un Rainbow dimensionné pour l'espace d'actions de l'allocation."""
    cfg = cfg or AgentConfig()
    if cfg.encoder != "mlp":
        # Les encodeurs séquentiels attendent une fenêtre de features ; l'observation de
        # l'allocateur est un vecteur d'état, pas une séquence.
        log.warning("encoder=%s ignoré pour l'allocateur : seul 'mlp' a du sens ici.", cfg.encoder)
        cfg = AgentConfig(**{**cfg.__dict__, "encoder": "mlp"})

    agent = RainbowAgent(
        obs_dim=env.obs_dim, n_actions=env.n_actions,
        n_features=0, window=1, n_portfolio=env.obs_dim,
        cfg=cfg, seed=seed,
    )
    bind_direct_observations(agent)
    return agent


# =======================================================================================
@dataclass
class AllocatorResult:
    sharpe: float
    total_return: float
    max_drawdown: float
    turnover: float
    profile_usage: Dict[str, float] = field(default_factory=dict)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([1.0]), repr=False)


def evaluate_allocator(agent: RainbowAgent, env: AllocationEnv,
                       cvar_alpha: Optional[float] = None) -> AllocatorResult:
    """Passage déterministe sur tout le segment (politique greedy, bruit désactivé)."""
    bind_direct_observations(agent)
    obs = env.reset(start=env.max_start, full=True)
    done = False
    while not done:
        obs, _, done, _ = env.step(agent.act(obs, greedy=True, cvar_alpha=cvar_alpha))

    df = env.to_frame()
    if df.empty:
        return AllocatorResult(0.0, 0.0, 0.0, 0.0)
    rets = df["net_return"].to_numpy()
    sd = float(rets.std(ddof=0))
    return AllocatorResult(
        sharpe=float(rets.mean() / sd * np.sqrt(env.bars_per_year)) if sd > 1e-12 else 0.0,
        total_return=float(env.equity - 1.0),
        max_drawdown=float(df["drawdown"].min()),
        turnover=float(df["turnover"].mean()),
        profile_usage=df["profile"].value_counts(normalize=True).to_dict(),
        equity_curve=df["equity"].to_numpy(),
    )


def train_allocator(
    agent: RainbowAgent,
    train_env: AllocationEnv,
    valid_env: Optional[AllocationEnv] = None,
    cfg: Optional[TrainConfig] = None,
    total_steps: Optional[int] = None,
) -> Dict[str, List[float]]:
    """Boucle d'entraînement, avec sélection du checkpoint sur validation."""
    cfg = cfg or TrainConfig()
    steps = int(total_steps or cfg.total_steps)
    bind_direct_observations(agent)

    acc = NStepAccumulator(agent.n_step, agent.gamma)
    obs = train_env.reset()
    history: Dict[str, List[float]] = {"step": [], "valid_sharpe": [], "loss": []}
    losses: List[float] = []
    best_metric, best_state, since_improve = -np.inf, None, 0
    t_start = time.time()

    for step in range(1, steps + 1):
        action = agent.act(obs)
        next_obs, reward, done, _ = train_env.step(action)
        agent.env_steps += 1

        collapsed = acc.push(Transition(
            t=0, portfolio=obs.copy(), action=action, reward=reward,
            next_t=0, next_portfolio=next_obs.copy(), done=done,
        ))
        if collapsed is not None:
            agent.buffer.add(collapsed)

        if done:
            for tr in acc.flush():
                agent.buffer.add(tr)
            acc.reset()
            obs = train_env.reset()
        else:
            obs = next_obs

        metrics = agent.learn()
        if metrics:
            losses.append(metrics["loss"])

        if valid_env is not None and step % max(cfg.eval_every, 1) == 0:
            res = evaluate_allocator(agent, valid_env)
            history["step"].append(step)
            history["valid_sharpe"].append(res.sharpe)
            history["loss"].append(float(np.mean(losses[-200:])) if losses else float("nan"))
            log.info("  eval @%d | sharpe=%+6.3f | ret=%+7.2f%% | maxDD=%6.2f%% | "
                     "profil dominant=%s | %.0f pas/s",
                     step, res.sharpe, 100 * res.total_return, 100 * res.max_drawdown,
                     max(res.profile_usage, key=res.profile_usage.get) if res.profile_usage else "-",
                     step / max(time.time() - t_start, 1e-9))

            if res.sharpe > best_metric:
                best_metric = res.sharpe
                best_state = {k: v.detach().cpu().clone()
                              for k, v in agent.online.state_dict().items()}
                since_improve = 0
            else:
                since_improve += 1
                if cfg.early_stop_patience and since_improve >= cfg.early_stop_patience:
                    log.info("Arrêt anticipé au pas %d.", step)
                    break

            obs = train_env.reset()
            acc.reset()
            bind_direct_observations(agent)

    if best_state is not None:
        agent.online.load_state_dict(best_state)
        agent.target.load_state_dict(best_state)
        log.info("Meilleur checkpoint restauré (sharpe validation = %.3f).", best_metric)
    return history


# =======================================================================================
def baseline_results(env: AllocationEnv) -> Dict[str, AllocatorResult]:
    """Références obligatoires : chaque profil fixe, joué en permanence.

    Un allocateur RL qui ne bat pas l'équipondéré constant n'apporte rien — et coûte
    beaucoup plus cher en complexité et en risque de sur-apprentissage.
    """
    out: Dict[str, AllocatorResult] = {}
    for action, name in enumerate(env.profile_names):
        env.reset(start=env.max_start, full=True)
        done = False
        while not done:
            _, _, done, _ = env.step(action)
        df = env.to_frame()
        if df.empty:
            continue
        rets = df["net_return"].to_numpy()
        sd = float(rets.std(ddof=0))
        out[name] = AllocatorResult(
            sharpe=float(rets.mean() / sd * np.sqrt(env.bars_per_year)) if sd > 1e-12 else 0.0,
            total_return=float(env.equity - 1.0),
            max_drawdown=float(df["drawdown"].min()),
            turnover=float(df["turnover"].mean()),
            profile_usage={name: 1.0},
        )
    return out

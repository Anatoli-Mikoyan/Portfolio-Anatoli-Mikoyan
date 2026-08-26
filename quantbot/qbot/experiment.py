"""Orchestration de bout en bout : données -> features -> agent -> backtest -> validation.

Ce module contient la logique métier ; les scripts de `scripts/` ne font que l'appeler.
Objectif : tout ce qui est exécuté en ligne de commande est aussi appelable et testable
depuis Python, sans dupliquer une seule ligne.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .backtest import BacktestResult, run_backtest
from .config import Config
from .data import generate_synthetic_ohlcv, load_ohlcv
from .data.bars import build_bars
from .features import FeaturePipeline, align_features_prices
from .utils.logging import get_logger
from .utils.seeding import seed_everything, spawn_seeds
from .utils.timeutils import infer_bars_per_year

log = get_logger("experiment")


# =======================================================================================
# Données
# =======================================================================================
def load_dataset(cfg: Config) -> pd.DataFrame:
    """Charge les données réelles, ou génère un marché synthétique si aucun CSV n'est fourni."""
    if cfg.data.csv_path:
        df = load_ohlcv(cfg.data.csv_path, start=cfg.data.start, end=cfg.data.end)
    else:
        log.warning(
            "Aucun csv_path : génération d'un marché SYNTHÉTIQUE. Les résultats obtenus "
            "ne valident que la mécanique du pipeline, jamais une performance réelle."
        )
        df = generate_synthetic_ohlcv(n=40_000, seed=cfg.seed).drop(columns=["regime"], errors="ignore")

    if cfg.data.bar_type != "time":
        before = len(df)
        df = build_bars(df, cfg.data.bar_type, cfg.data.bar_threshold)
        log.info("Barres %s : %d -> %d", cfg.data.bar_type, before, len(df))
    return df


def bars_per_year(df: pd.DataFrame, cfg: Config) -> float:
    return infer_bars_per_year(df.index, cfg.data.asset_class)


# =======================================================================================
# Entraînement
# =======================================================================================
@dataclass
class TrainedModel:
    agent: Any
    pipeline: FeaturePipeline
    config: Config
    valid_sharpe: float
    history: Any = None
    seeds: Tuple[int, ...] = ()

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.config.save(directory / "config.json")
        self.pipeline.save(directory / "pipeline.json")
        if hasattr(self.agent, "agents"):
            self.agent.save(directory)
        else:
            self.agent.save(directory / "agent.pt")
        (directory / "metadata.json").write_text(
            json.dumps({"valid_sharpe": self.valid_sharpe, "seeds": list(self.seeds)}, indent=2),
            encoding="utf-8",
        )
        log.info("Modèle exporté dans %s", directory)
        return directory


def train_model(
    cfg: Config,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    bpy: float,
    total_steps: Optional[int] = None,
    quiet: bool = False,
) -> TrainedModel:
    """Entraîne un agent (ou un ensemble multi-graines) sur `train_df`.

    La pipeline de features est ajustée UNIQUEMENT sur `train_df`. `valid_df` sert
    exclusivement à choisir le checkpoint et à déclencher l'arrêt anticipé.
    """
    from .agents import EnsembleAgent, RainbowAgent, Trainer, evaluate
    from .env import N_PORTFOLIO_FEATURES, make_env_from_frames

    pipeline = FeaturePipeline(cfg.features)
    x_train = pipeline.fit_transform(train_df)
    xa_train, pa_train = align_features_prices(x_train, train_df)

    # Le segment de validation a besoin de l'historique du train pour amorcer ses fenêtres
    # glissantes ; on concatène puis on ne conserve que la partie validation.
    combined = pd.concat([train_df, valid_df]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    x_valid_full = pipeline.transform(combined)
    x_valid = x_valid_full.loc[x_valid_full.index >= valid_df.index[0]]
    xa_valid, pa_valid = align_features_prices(x_valid, valid_df)

    if len(xa_train) < cfg.env.window + 100:
        raise ValueError(f"Segment d'entraînement trop court après features : {len(xa_train)} barres.")
    if len(xa_valid) < cfg.env.window + 10:
        raise ValueError(f"Segment de validation trop court après features : {len(xa_valid)} barres.")

    seeds = tuple(cfg.train.seeds) if cfg.train.seeds else (cfg.seed,)
    if len(seeds) == 1 and seeds[0] == 0:
        seeds = (cfg.seed,)
    agents, sharpes, histories = [], [], []

    for i, seed in enumerate(spawn_seeds(cfg.seed, len(seeds)) if len(seeds) > 1 else seeds):
        seed_everything(int(seed))
        rng = np.random.default_rng(int(seed))
        train_env = make_env_from_frames(xa_train, pa_train, cfg.env, cfg.costs, bpy, rng)
        valid_env = make_env_from_frames(xa_valid, pa_valid, cfg.env, cfg.costs, bpy, rng)

        agent = RainbowAgent(
            obs_dim=train_env.obs_dim, n_actions=train_env.n_actions,
            n_features=train_env.n_features, window=train_env.window,
            n_portfolio=N_PORTFOLIO_FEATURES if cfg.env.include_position_in_obs else 0,
            cfg=cfg.agent, seed=int(seed),
        )
        if i == 0:
            log.info("Agent : %s/%s, %d paramètres, obs_dim=%d, %d actions",
                     cfg.agent.algo, cfg.agent.distributional, agent.n_parameters(),
                     train_env.obs_dim, train_env.n_actions)

        trainer = Trainer(agent, train_env, valid_env, cfg.train)
        history = trainer.fit(total_steps=total_steps)
        res = evaluate(agent, valid_env)

        agents.append(agent)
        sharpes.append(res.sharpe)
        histories.append(history)
        if not quiet:
            log.info("graine %d -> sharpe validation = %.3f", seed, res.sharpe)

    if len(agents) > 1:
        # Le seuil d'accord force l'agent à rester plat quand les graines divergent :
        # le désaccord inter-graines est la mesure la plus honnête de l'incertitude.
        final_agent: Any = EnsembleAgent(agents, agreement_threshold=0.6)
        log.info("Ensemble de %d agents | sharpes validation : %s",
                 len(agents), np.round(sharpes, 3).tolist())
    else:
        final_agent = agents[0]

    return TrainedModel(
        agent=final_agent, pipeline=pipeline, config=cfg,
        valid_sharpe=float(np.mean(sharpes)), history=histories[0], seeds=tuple(int(s) for s in seeds),
    )


# =======================================================================================
# Inférence en lot / backtest
# =======================================================================================
def run_agent_on_segment(
    model: TrainedModel,
    df: pd.DataFrame,
    context_df: Optional[pd.DataFrame] = None,
    bpy: float = 6240.0,
    cvar_alpha: Optional[float] = None,
) -> Tuple[pd.Series, Any]:
    """Déroule l'agent SÉQUENTIELLEMENT sur `df` et retourne (positions, environnement).

    Pourquoi séquentiellement et non en lot : l'observation contient l'état de portefeuille
    (position courante, drawdown, P&L latent, ancienneté). Cet état dépend des décisions
    passées de l'agent — il n'est donc pas calculable à l'avance. Une inférence en lot
    devrait le remplir de zéros, ce qui reviendrait à interroger le modèle sur des états
    qu'il n'a jamais rencontrés à l'entraînement : un écart entraînement/service qui
    dégrade silencieusement la performance et rend le backtest non représentatif.

    `context_df` fournit l'historique antérieur nécessaire pour amorcer les fenêtres
    glissantes des features ET la fenêtre d'observation de l'agent.
    """
    from .env import make_env_from_frames

    source = df if context_df is None else pd.concat([context_df, df]).sort_index()
    source = source[~source.index.duplicated(keep="last")]

    feats_full = model.pipeline.transform(source)
    if feats_full.empty:
        raise ValueError("Aucune feature valide (historique de contexte insuffisant ?).")

    xa, pa = align_features_prices(feats_full, source)
    window = int(model.config.env.window)

    seg_start = pa.index.searchsorted(df.index[0])
    if seg_start < window - 1:
        raise ValueError(
            f"Contexte insuffisant : {seg_start + 1} barres exploitables avant le segment, "
            f"{window} requises par la fenêtre d'observation. Fournir un context_df plus long."
        )

    # `random_start=False` et `episode_length=None` : un seul passage déterministe,
    # du début à la fin du segment, sans coupure d'épisode ni redémarrage aléatoire.
    env_cfg = replace(model.config.env, random_start=False, episode_length=None,
                      max_drawdown_stop=None)
    env = make_env_from_frames(xa, pa, env_cfg, model.config.costs, bpy)

    obs = env.reset(start=int(seg_start), length=None)
    agent = model.agent
    agent.bind_features(env.features)

    done = False
    while not done:
        action = agent.act(obs, greedy=True, cvar_alpha=cvar_alpha)
        obs, _, done, _ = env.step(int(action))

    hist = env.to_frame()
    positions = pd.Series(hist["position"].to_numpy(), index=hist.index, name="position")
    return positions.reindex(df.index).ffill().fillna(0.0), env


def predict_positions(
    model: TrainedModel,
    df: pd.DataFrame,
    context_df: Optional[pd.DataFrame] = None,
    cvar_alpha: Optional[float] = None,
    bpy: float = 6240.0,
) -> pd.Series:
    """Positions cibles de l'agent sur `df` (voir `run_agent_on_segment`)."""
    positions, _ = run_agent_on_segment(model, df, context_df, bpy, cvar_alpha)
    return positions


def backtest_model(
    model: TrainedModel,
    df: pd.DataFrame,
    context_df: Optional[pd.DataFrame] = None,
    bpy: float = 6240.0,
    n_trials: int = 1,
    cvar_alpha: Optional[float] = None,
) -> Tuple[BacktestResult, pd.Series]:
    """Backteste l'agent sur `df`. Les positions proviennent d'un déroulé séquentiel,
    la comptabilité du `run_backtest` — les deux sont vérifiées identiques par les tests."""
    positions, _ = run_agent_on_segment(model, df, context_df, bpy, cvar_alpha)
    result = run_backtest(positions, df, model.config.costs, model.config.env, bpy, n_trials)
    return result, positions

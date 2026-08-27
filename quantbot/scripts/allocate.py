#!/usr/bin/env python3
"""Allocateur RL entre stratégies — cahier des charges §10.

Répond à la question du cahier « le RL doit-il choisir Buy/Sell/Hold ou allouer le
capital entre stratégies ? » en implémentant la seconde option, qui est la bonne :

  * l'espace d'états est bien plus petit — apprendre quelle stratégie marche dans quel
    régime, au lieu d'apprendre la dynamique du prix ;
  * les stratégies portent déjà l'hypothèse économique, le RL ne fait qu'arbitrer ;
  * l'échec est gracieux — un allocateur qui n'apprend rien converge vers l'équipondéré
    ou vers le plat, deux comportements raisonnables. Un agent directionnel qui n'apprend
    rien trade du bruit et paie le spread.

PRÉREQUIS : n'allouer qu'entre des stratégies RETENUES par scripts/screen.py. Répartir du
capital entre des hypothèses non validées revient à répartir du bruit.

    python scripts/allocate.py --config configs/eurusd_h1.yaml --csv data/EURUSD_H1.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from qbot.agents import (
    baseline_results, evaluate_allocator, make_allocator, train_allocator,
)
from qbot.config import Config
from qbot.env import AllocationEnv, strategy_position_matrix
from qbot.experiment import bars_per_year, load_dataset
from qbot.regime import build_regime_matrix
from qbot.strategies import default_strategies
from qbot.utils.logging import configure_logging, get_logger
from qbot.utils.seeding import seed_everything

log = get_logger("scripts.allocate")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--steps", type=int, default=40_000)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--regime-window", type=int, default=4_000,
                   help="Fenêtre de normalisation des features de régime (doit dépasser "
                        "largement la durée d'un régime)")
    p.add_argument("--out", type=str, default="runs/allocator")
    args = p.parse_args()

    configure_logging(logging.INFO)
    cfg = Config.load(args.config) if args.config else Config()
    if args.csv:
        cfg.data.csv_path = args.csv
    seed_everything(cfg.seed)

    df = load_dataset(cfg)
    bpy = bars_per_year(df, cfg)

    strategies = default_strategies()
    positions = strategy_position_matrix(strategies, df)
    regime = build_regime_matrix(df)

    common = positions.index.intersection(regime.index)
    positions, regime, prices = positions.loc[common], regime.loc[common], df.loc[common]

    # Normalisation sur fenêtre LONGUE : une fenêtre courte effacerait le niveau, qui
    # est l'information de régime (voir qbot/regime/features.py).
    w = args.regime_window
    mu = regime.rolling(w, min_periods=w // 4).mean()
    sd = regime.rolling(w, min_periods=w // 4).std(ddof=0).replace(0.0, np.nan)
    regime_n = ((regime - mu) / sd).fillna(0.0).clip(-5.0, 5.0)

    cut = int(len(positions) * args.train_frac)
    train_env = AllocationEnv(positions.iloc[:cut], prices.iloc[:cut], regime_n.iloc[:cut],
                              cfg.env, cfg.costs, bpy, rng=np.random.default_rng(cfg.seed))
    test_env = AllocationEnv(positions.iloc[cut:], prices.iloc[cut:], regime_n.iloc[cut:],
                             cfg.env, cfg.costs, bpy)

    print(f"\n{len(strategies)} stratégies | {train_env.n_actions} profils d'allocation :")
    print(f"  {train_env.profile_names}")
    print(f"observation = {train_env.obs_dim} variables "
          f"({train_env.n_regime_features} de régime + performances + poids + portefeuille)\n")

    # --- références obligatoires --------------------------------------------------------
    print("=" * 78)
    print("  RÉFÉRENCES — chaque profil joué en permanence, sur le segment de TEST")
    print("=" * 78)
    baselines = baseline_results(test_env)
    for name, res in sorted(baselines.items(), key=lambda kv: -kv[1].sharpe):
        print(f"  {name:<14} sharpe={res.sharpe:+7.3f} | CAGR={100 * res.total_return:+7.2f}% "
              f"| maxDD={100 * res.max_drawdown:6.2f}%")
    best = max(baselines.values(), key=lambda r: r.sharpe)
    best_name = max(baselines, key=lambda k: baselines[k].sharpe)

    # --- entraînement --------------------------------------------------------------------
    print(f"\nEntraînement de l'allocateur ({args.steps} pas)...")
    agent = make_allocator(train_env, cfg.agent, seed=cfg.seed)
    history = train_allocator(agent, train_env, test_env, cfg.train, total_steps=args.steps)
    result = evaluate_allocator(agent, test_env)

    print("\n" + "=" * 78)
    print("  ALLOCATEUR RL sur le segment de TEST")
    print("=" * 78)
    print(f"  sharpe={result.sharpe:+7.3f} | CAGR={100 * result.total_return:+7.2f}% "
          f"| maxDD={100 * result.max_drawdown:6.2f}% | turnover={result.turnover:.4f}")
    usage = sorted(result.profile_usage.items(), key=lambda kv: -kv[1])
    print("  profils utilisés : " + ", ".join(f"{k} {v:.0%}" for k, v in usage[:5]))
    flat_share = result.profile_usage.get("flat", 0.0)
    print(f"  part du temps HORS marché : {flat_share:.0%}"
          + ("  <- le système sait ne pas trader (§2)" if flat_share > 0.05 else ""))

    delta = result.sharpe - best.sharpe
    print(f"\n  meilleure référence ({best_name}) = {best.sharpe:+.3f}")
    print(f"  allocateur                       = {result.sharpe:+.3f}")
    print(f"  écart                            = {delta:+.3f}")
    if delta <= 0:
        print("\n  L'allocateur NE BAT PAS la meilleure référence fixe : sa complexité")
        print("  n'est pas justifiée ici. Conserver le profil fixe. C'est un résultat, pas un échec.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    agent.save(out_dir / "allocator.pt")
    test_env.to_frame().to_csv(out_dir / "allocation_test.csv")
    (out_dir / "allocator.json").write_text(json.dumps({
        "strategies": [type(s).__name__ for s in strategies],
        "profiles": train_env.profile_names,
        "allocator": {"sharpe": result.sharpe, "total_return": result.total_return,
                      "max_drawdown": result.max_drawdown, "turnover": result.turnover,
                      "profile_usage": result.profile_usage},
        "baselines": {k: v.sharpe for k, v in baselines.items()},
        "history": history,
    }, indent=2, default=float), encoding="utf-8")
    log.info("Résultats dans %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

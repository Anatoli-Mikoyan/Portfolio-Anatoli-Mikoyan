#!/usr/bin/env python3
"""Walk-forward complet avec ré-entraînement de l'agent à chaque fold.

C'est le protocole de référence du dépôt : chaque barre de la courbe d'équité produite
a été générée par un modèle qui ne l'avait jamais vue. Le Sharpe qui en sort est le seul
chiffre de ce projet qu'il soit raisonnable de citer.

    python scripts/walkforward.py --config configs/eurusd_h1.yaml --folds 6
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

from qbot.config import Config
from qbot.experiment import bars_per_year, load_dataset, run_agent_on_segment, train_model
from qbot.utils.logging import configure_logging, get_logger
from qbot.validation import (
    compute_pbo, monte_carlo_drawdown, run_walkforward, bootstrap_metric,
)
from qbot.backtest import run_backtest, sharpe_ratio

log = get_logger("scripts.walkforward")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--out", type=str, default="runs/walkforward")
    p.add_argument("--train-bars", type=int, default=None)
    p.add_argument("--test-bars", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--anchored", action="store_true", help="Fenêtre d'entraînement ancrée")
    args = p.parse_args()

    configure_logging(logging.INFO)
    cfg = Config.load(args.config) if args.config else Config()
    if args.csv:
        cfg.data.csv_path = args.csv
    if args.train_bars:
        cfg.validation.train_bars = args.train_bars
    if args.test_bars:
        cfg.validation.test_bars = args.test_bars
    if args.steps:
        cfg.train.total_steps = args.steps

    df = load_dataset(cfg)
    bpy = bars_per_year(df, cfg)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_returns: list[np.ndarray] = []

    def fit_predict(train_df, valid_df, test_df, fold):
        model = train_model(cfg, train_df, valid_df, bpy, total_steps=cfg.train.total_steps, quiet=True)
        model.save(out_dir / f"fold_{fold.idx}")
        # Contexte = fin du segment de validation : indispensable pour amorcer les fenêtres.
        ctx = pd.concat([train_df, valid_df]).sort_index()
        positions, _ = run_agent_on_segment(model, test_df, context_df=ctx, bpy=bpy)
        res = run_backtest(positions, test_df, cfg.costs, cfg.env, bpy)
        return res.frame["net_return"], res.frame["position"]

    result = run_walkforward(
        df, fit_predict,
        train_bars=cfg.validation.train_bars,
        test_bars=cfg.validation.test_bars,
        anchored=args.anchored or cfg.validation.anchored,
        embargo_bars=int(len(df) * cfg.validation.embargo_pct),
        bars_per_year=bpy,
        n_trials=cfg.validation.n_trials_for_dsr,
    )

    print()
    print(result.report)
    print()
    print("┌─ RÉGULARITÉ INTER-FOLDS ─────────────────────────────────────┐")
    cons = result.consistency()
    for k, v in cons.items():
        print(f"│ {k:<24} {v:>12.4f}" + " " * 25 + "│")
    print("└──────────────────────────────────────────────────────────────┘")

    oos = result.oos_returns.to_numpy()

    print()
    boot = bootstrap_metric(oos, lambda x: sharpe_ratio(x, bpy), cfg.validation.bootstrap_samples,
                            cfg.validation.block_size, cfg.seed, "sharpe")
    print("Bootstrap par blocs :", boot)

    mc = monte_carlo_drawdown(oos, cfg.validation.bootstrap_samples, cfg.validation.block_size, cfg.seed)
    print(f"Drawdown : observé {mc['observed']:.2%} | médiane MC {mc['median']:.2%} | "
          f"pire 5% {mc['p95_worst']:.2%} | pire 1% {mc['p99_worst']:.2%}")

    # PBO : nécessite les rendements de TOUTES les configurations essayées, alignés.
    # Ici on utilise les folds comme configurations — indicatif, à compléter par un
    # vrai balayage d'hyperparamètres via scripts/sweep.py.
    matrix = _align_fold_matrix(result)
    if matrix is not None and matrix.shape[1] >= 4:
        pbo = compute_pbo(matrix, n_partitions=8, max_combinations=200)
        print()
        print(pbo)

    result.oos_returns.to_frame("net_return").to_csv(out_dir / "oos_returns.csv")
    (out_dir / "walkforward_report.json").write_text(
        json.dumps({"report": result.report.to_dict(), "consistency": cons,
                    "bootstrap_sharpe_ci": [boot.ci_low, boot.ci_high],
                    "monte_carlo_drawdown": mc}, indent=2, default=float),
        encoding="utf-8",
    )
    log.info("Résultats écrits dans %s", out_dir)
    return 0


def _align_fold_matrix(result) -> np.ndarray | None:
    """Empile les rendements des folds en matrice (T, n_folds) pour la PBO."""
    series = [f.returns.reset_index(drop=True) for f in result.folds]
    if not series:
        return None
    n = min(len(s) for s in series)
    if n < 100:
        return None
    return np.column_stack([s.iloc[:n].to_numpy() for s in series])


if __name__ == "__main__":
    raise SystemExit(main())

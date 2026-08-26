#!/usr/bin/env python3
"""Backteste un modèle déjà entraîné sur un segment de données.

    python scripts/backtest.py --model runs/v1 --csv data/EURUSD_H1.csv --start 2024-01-01
    python scripts/backtest.py --model runs/v1 --cvar 0.1     # politique averse au risque
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from qbot.backtest import (
    mean_reversion_positions, momentum_positions, random_positions, run_backtest,
)
from qbot.config import Config
from qbot.experiment import TrainedModel, bars_per_year, load_dataset
from qbot.features import FeaturePipeline
from qbot.utils.logging import configure_logging, get_logger

log = get_logger("scripts.backtest")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--trials", type=int, default=1, help="Configurations réellement essayées")
    p.add_argument("--cvar", type=float, default=None, help="Politique CVaR (ex. 0.1)")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    configure_logging(logging.INFO)
    model_dir = Path(args.model)
    cfg = Config.load(model_dir / "config.json")
    if args.csv:
        cfg.data.csv_path = args.csv

    df = load_dataset(cfg)
    bpy = bars_per_year(df, cfg)

    # Le contexte antérieur est indispensable pour amorcer les fenêtres glissantes.
    pipeline = FeaturePipeline.load(model_dir / "pipeline.json")
    need = pipeline.min_history + cfg.env.window + 10
    segment = df
    context = None
    if args.start:
        start_pos = int(df.index.searchsorted(pd.Timestamp(args.start, tz=df.index.tz)))
        if start_pos < need:
            raise SystemExit(f"Historique insuffisant avant {args.start} : {start_pos} barres, {need} requises.")
        context = df.iloc[:start_pos]
        segment = df.iloc[start_pos:]
    if args.end:
        segment = segment[segment.index <= pd.Timestamp(args.end, tz=df.index.tz)]

    from qbot.agents import EnsembleAgent, RainbowAgent

    agent = (EnsembleAgent.load(model_dir, agreement_threshold=0.6)
             if sorted(model_dir.glob("agent_*.pt")) else RainbowAgent.load(model_dir / "agent.pt"))
    model = TrainedModel(agent=agent, pipeline=pipeline, config=cfg, valid_sharpe=float("nan"))

    from qbot.experiment import backtest_model

    result, positions = backtest_model(model, segment, context_df=context, bpy=bpy,
                                       n_trials=args.trials, cvar_alpha=args.cvar)
    print(result.report)

    print("\n  Références sur le même segment :")
    for name, pos in [
        ("buy & hold", np.ones(len(segment))),
        ("aléatoire", random_positions(len(segment), cfg.seed)),
        ("momentum 20", momentum_positions(segment["close"], 20)),
        ("mean-reversion 20", mean_reversion_positions(segment["close"], 20)),
    ]:
        r = run_backtest(pos, segment, cfg.costs, cfg.env, bpy)
        print(f"    {name:<20} sharpe {r.report.sharpe:+7.3f} | CAGR {100 * r.report.cagr:+8.2f}%")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        result.frame.to_csv(out)
        log.info("Détail barre par barre écrit dans %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

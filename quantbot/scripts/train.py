#!/usr/bin/env python3
"""Entraîne un agent et exporte un modèle prêt pour le live.

Protocole appliqué :
    train (60 %)  -> apprentissage des poids
    valid (20 %)  -> sélection du checkpoint et arrêt anticipé
    test  (20 %)  -> touché UNE SEULE FOIS, à la fin, et jamais pour décider quoi que ce soit

    python scripts/train.py --config configs/eurusd_h1.yaml --out runs/eurusd_v1
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from qbot.backtest import momentum_positions, random_positions, run_backtest
from qbot.config import Config
from qbot.experiment import backtest_model, bars_per_year, load_dataset, train_model
from qbot.utils.logging import configure_logging, get_logger
from qbot.validation import train_valid_test_split

log = get_logger("scripts.train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default=None, help="Fichier YAML/JSON de configuration")
    p.add_argument("--csv", type=str, default=None, help="CSV OHLCV (surcharge la config)")
    p.add_argument("--out", type=str, default="runs/latest", help="Répertoire d'export du modèle")
    p.add_argument("--steps", type=int, default=None, help="Nombre de pas d'entraînement")
    p.add_argument("--seeds", type=int, default=None, help="Nombre de graines (>1 => ensemble)")
    p.add_argument("--seed", type=int, default=None, help="Graine maîtresse")
    p.add_argument("--no-test", action="store_true", help="Ne pas évaluer sur le test (le préserver)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(logging.DEBUG if args.verbose else logging.INFO)

    cfg = Config.load(args.config) if args.config else Config()
    if args.csv:
        cfg.data.csv_path = args.csv
    if args.seed is not None:
        cfg.seed = args.seed
    if args.seeds:
        cfg.train.seeds = tuple(range(args.seeds))
    if args.steps:
        cfg.train.total_steps = args.steps

    df = load_dataset(cfg)
    bpy = bars_per_year(df, cfg)
    log.info("Jeu de données : %d barres, %s -> %s (%.0f barres/an)",
             len(df), df.index[0].date(), df.index[-1].date(), bpy)

    tr, va, te = train_valid_test_split(df.index, 0.6, 0.2, cfg.validation.embargo_pct)
    train_df, valid_df, test_df = df.iloc[tr], df.iloc[va], df.iloc[te]
    log.info("Découpage : train=%d | valid=%d | test=%d (embargo %.1f%%)",
             len(train_df), len(valid_df), len(test_df), 100 * cfg.validation.embargo_pct)

    model = train_model(cfg, train_df, valid_df, bpy)
    out_dir = Path(args.out)
    model.save(out_dir)

    summary = {"valid_sharpe": model.valid_sharpe, "n_train_bars": len(train_df)}

    if not args.no_test:
        log.info("=" * 66)
        log.info("ÉVALUATION FINALE SUR LE TEST — ne doit servir à AUCUNE décision.")
        log.info("=" * 66)
        ctx = df.loc[: test_df.index[0]].iloc[:-1]
        result, positions = backtest_model(model, test_df, context_df=ctx, bpy=bpy,
                                           n_trials=cfg.validation.n_trials_for_dsr)
        print(result.report)
        summary["test"] = result.report.to_dict()

        # Références obligatoires : un modèle qui ne bat pas ces baselines n'apporte rien.
        for name, pos in [
            ("buy & hold", np.ones(len(test_df))),
            ("aléatoire", random_positions(len(test_df), cfg.seed)),
            ("momentum 20", momentum_positions(test_df["close"], 20)),
        ]:
            r = run_backtest(pos, test_df, cfg.costs, cfg.env, bpy)
            log.info("  référence %-14s sharpe=%+6.3f | CAGR=%+7.2f%%",
                     name, r.report.sharpe, 100 * r.report.cagr)
            summary.setdefault("baselines", {})[name] = r.report.sharpe

        positions.to_frame().to_csv(out_dir / "test_positions.csv")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    log.info("Terminé. Modèle et résumé dans %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

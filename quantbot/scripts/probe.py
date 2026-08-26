#!/usr/bin/env python3
"""Sonde de signal — à lancer AVANT tout entraînement RL.

Répond en quelques secondes à deux questions que la courbe de Sharpe d'un run RL ne
permet jamais de distinguer :

  1. Les features contiennent-elles un signal exploitable ?  (sonde linéaire)
  2. L'architecture choisie peut-elle l'extraire ?           (sonde réseau)

Si la sonde réseau fait moins bien que la sonde linéaire, le problème est la
REPRÉSENTATION (fenêtre trop large, réseau trop grand, régularisation absente) et
lancer un entraînement RL est une perte de temps.

    python scripts/probe.py --config configs/eurusd_h1.yaml --csv data/EURUSD_H1.csv
    python scripts/probe.py --windows 1 4 8 16 32 --encoder tcn
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qbot.config import Config
from qbot.diagnostics import signal_report
from qbot.experiment import load_dataset
from qbot.features import FeaturePipeline
from qbot.utils.logging import configure_logging, get_logger

log = get_logger("scripts.probe")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--windows", type=int, nargs="+", default=[1, 4, 16])
    p.add_argument("--encoder", type=str, default="mlp", choices=["mlp", "gru", "tcn"])
    p.add_argument("--steps", type=int, default=8_000)
    p.add_argument("--horizon", type=int, default=1, help="Horizon de prédiction, en barres")
    args = p.parse_args()

    configure_logging(logging.INFO)
    cfg = Config.load(args.config) if args.config else Config()
    if args.csv:
        cfg.data.csv_path = args.csv

    df = load_dataset(cfg)
    pipeline = FeaturePipeline(cfg.features)
    features = pipeline.fit_transform(df)
    log.info("%d features sur %d barres", features.shape[1], features.shape[0])

    signal_report(features, df.loc[features.index], windows=tuple(args.windows),
                  encoder=args.encoder, steps=args.steps, horizon=args.horizon)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

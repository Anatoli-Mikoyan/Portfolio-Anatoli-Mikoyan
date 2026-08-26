#!/usr/bin/env python3
"""Balayage d'hyperparamètres avec comptabilisation HONNÊTE des essais.

Ce script existe pour une raison précise : le Deflated Sharpe et la PBO n'ont de sens
que si l'on déclare **toutes** les configurations testées, y compris celles abandonnées
en cours de route. Faire le balayage à la main puis rapporter `n_trials=1` est la façon
la plus courante de produire un chiffre invalide sans le savoir.

Ici, chaque configuration essayée est enregistrée, et la matrice de rendements complète
est exportée pour `scripts/validate.py --matrix`.

    python scripts/sweep.py --config configs/eurusd_h1.yaml --grid lr=1e-4,3e-4 n_step=3,5
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from qbot.backtest import sharpe_ratio
from qbot.config import Config
from qbot.experiment import backtest_model, bars_per_year, load_dataset, train_model
from qbot.utils.logging import configure_logging, get_logger
from qbot.validation import compute_pbo, train_valid_test_split, whites_reality_check

log = get_logger("scripts.sweep")


def _parse_grid(items: list[str]) -> dict:
    """`lr=1e-4,3e-4` -> {"agent.lr": [1e-4, 3e-4]} (préfixe déduit si omis)."""
    prefixes = {"lr": "agent", "n_step": "agent", "gamma": "agent", "batch_size": "agent",
                "hidden_sizes": "agent", "distributional": "agent", "encoder": "agent",
                "window": "env", "reward": "env", "episode_length": "env",
                "vol_target": "env", "spread_bps": "costs", "min_trade_size": "costs"}
    grid: dict[str, list] = {}
    for item in items:
        key, _, raw = item.partition("=")
        key = key.strip()
        full = key if "." in key else f"{prefixes.get(key, 'agent')}.{key}"
        values = []
        for token in raw.split(","):
            token = token.strip()
            try:
                values.append(int(token) if token.isdigit() else float(token))
            except ValueError:
                values.append({"true": True, "false": False}.get(token.lower(), token))
        grid[full] = values
    return grid


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--grid", nargs="+", required=True, help='ex. lr=1e-4,3e-4 n_step=3,5')
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--out", type=str, default="runs/sweep")
    p.add_argument("--max-configs", type=int, default=64)
    args = p.parse_args()

    configure_logging(logging.INFO)
    base = Config.load(args.config) if args.config else Config()
    if args.csv:
        base.data.csv_path = args.csv
    if args.steps:
        base.train.total_steps = args.steps

    grid = _parse_grid(args.grid)
    combos = list(itertools.product(*grid.values()))
    if len(combos) > args.max_configs:
        raise SystemExit(
            f"{len(combos)} configurations demandées (> {args.max_configs}). "
            "Un balayage large ne rend pas la stratégie meilleure, il rend le Deflated "
            "Sharpe plus sévère. Restreindre la grille ou relever --max-configs en connaissance de cause."
        )

    df = load_dataset(base)
    bpy = bars_per_year(df, base)
    tr, va, te = train_valid_test_split(df.index, 0.6, 0.2, base.validation.embargo_pct)
    train_df, valid_df, test_df = df.iloc[tr], df.iloc[va], df.iloc[te]
    ctx = pd.concat([train_df, valid_df]).sort_index()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Balayage de %d configurations sur %s", len(combos), list(grid))
    rows, returns_by_config = [], {}

    for i, values in enumerate(combos):
        overrides = dict(zip(grid.keys(), values))
        cfg = base.merge(overrides)
        label = " ".join(f"{k.split('.')[-1]}={v}" for k, v in overrides.items())
        log.info("[%d/%d] %s", i + 1, len(combos), label)

        try:
            model = train_model(cfg, train_df, valid_df, bpy, quiet=True)
            # Évaluation sur la VALIDATION : le test reste intouché pendant le balayage.
            result, _ = backtest_model(model, valid_df, context_df=train_df, bpy=bpy)
        except Exception as exc:                                # pragma: no cover
            log.warning("   configuration échouée : %s", exc)
            continue

        returns_by_config[f"cfg_{i}"] = result.frame["net_return"].reset_index(drop=True)
        rows.append({"config": label, **overrides,
                     "valid_sharpe": result.report.sharpe,
                     "valid_return": result.report.total_return,
                     "valid_maxdd": result.report.max_drawdown})
        log.info("   -> sharpe validation = %+.3f", result.report.sharpe)

    if not rows:
        raise SystemExit("Aucune configuration n'a abouti.")

    table = pd.DataFrame(rows).sort_values("valid_sharpe", ascending=False)
    table.to_csv(out_dir / "sweep_results.csv", index=False)
    print()
    print(table.to_string(index=False))

    # Matrice alignée des rendements : c'est ELLE qui rend la PBO honnête.
    n = min(len(s) for s in returns_by_config.values())
    matrix = pd.DataFrame({k: v.iloc[:n].to_numpy() for k, v in returns_by_config.items()})
    matrix.to_csv(out_dir / "sweep_returns.csv")

    n_trials = len(rows)
    print()
    print(f"Configurations réellement testées : {n_trials}")
    print("-> reporter n_trials_for_dsr = %d, et NON 1." % n_trials)

    if n_trials >= 4:
        pbo = compute_pbo(matrix.to_numpy(), n_partitions=8, max_combinations=200)
        print()
        print(pbo)
        wrc = whites_reality_check(matrix.to_numpy(), n_samples=1000)
        print(f"\nWhite Reality Check : p-value = {wrc['p_value']:.4f} "
              f"(la meilleure config bat-elle le hasard, {n_trials} essais pris en compte ?)")

    (out_dir / "sweep_meta.json").write_text(
        json.dumps({"n_trials": n_trials, "grid": {k: list(map(str, v)) for k, v in grid.items()}},
                   indent=2),
        encoding="utf-8",
    )
    log.info("Résultats dans %s — utiliser --matrix %s/sweep_returns.csv pour valider.",
             out_dir, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

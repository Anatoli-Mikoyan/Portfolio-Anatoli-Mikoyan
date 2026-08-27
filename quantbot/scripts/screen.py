#!/usr/bin/env python3
"""Crible les stratégies candidates — cahier des charges §8 et §13.

Chaque famille est traitée comme une hypothèse à réfuter. Le script produit un verdict
par stratégie, puis une correction au niveau de la FAMILLE (PBO + Reality Check de
White), parce que la vraie question n'est pas « cette stratégie est-elle bonne ? » mais
« le fait d'avoir choisi la meilleure d'entre elles a-t-il une valeur prédictive ? ».

    python scripts/screen.py --config configs/eurusd_h1.yaml --csv data/EURUSD_H1.csv
    python scripts/screen.py --mode family        # optimise les paramètres à chaque fold

Deux modes :
  fixed  (défaut) — paramètres figés : mesure l'edge de l'hypothèse elle-même
  family          — sélection in-sample à chaque fold : mesure ce qu'un praticien obtient
                    réellement, coût du data-snooping inclus
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from qbot.config import Config
from qbot.experiment import bars_per_year, load_dataset
from qbot.strategies import (
    STRATEGY_CLASSES, default_strategies, family_pbo, print_screening,
    screen_family, screen_fixed, screening_table,
)
from qbot.utils.logging import configure_logging, get_logger

log = get_logger("scripts.screen")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--mode", choices=["fixed", "family", "both"], default="fixed")
    p.add_argument("--train-bars", type=int, default=None)
    p.add_argument("--test-bars", type=int, default=None)
    p.add_argument("--out", type=str, default="runs/screening")
    args = p.parse_args()

    configure_logging(logging.INFO)
    cfg = Config.load(args.config) if args.config else Config()
    if args.csv:
        cfg.data.csv_path = args.csv
    if args.train_bars:
        cfg.validation.train_bars = args.train_bars
    if args.test_bars:
        cfg.validation.test_bars = args.test_bars

    df = load_dataset(cfg)
    bpy = bars_per_year(df, cfg)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nHypothèses testées :")
    for s in default_strategies():
        print(f"\n{s.describe()}")

    payload = {}

    if args.mode in ("fixed", "both"):
        log.info("Criblage à paramètres figés (1 essai par hypothèse)...")
        results = [screen_fixed(s, df, cfg.costs, cfg.env, cfg.validation, bpy)
                   for s in default_strategies()]
        print("\n" + "=" * 78)
        print("  PARAMÈTRES FIGÉS — mesure l'edge de l'hypothèse")
        print("=" * 78)
        print_screening(results, family_pbo(results, n_partitions=10))
        screening_table(results).to_csv(out_dir / "screening_fixed.csv", index=False)
        _dump_returns(results, out_dir / "returns_fixed.csv")
        payload["fixed"] = _summary(results)

    if args.mode in ("family", "both"):
        log.info("Criblage avec sélection de paramètres (mesure le data-snooping)...")
        results = []
        for cls in STRATEGY_CLASSES:
            try:
                results.append(screen_family(cls, df, cfg.costs, cfg.env, cfg.validation, bpy))
            except ValueError as exc:
                log.warning("%s ignorée : %s", cls.__name__, exc)
        print("\n" + "=" * 78)
        print("  SÉLECTION DE PARAMÈTRES — mesure ce qu'un praticien obtient réellement")
        print("=" * 78)
        print_screening(results, family_pbo(results, n_partitions=10))
        screening_table(results).to_csv(out_dir / "screening_family.csv", index=False)
        _dump_returns(results, out_dir / "returns_family.csv")
        payload["family"] = _summary(results)

    if args.mode == "both" and payload.get("fixed") and payload.get("family"):
        print("\n┌─ COÛT DU DATA-SNOOPING ──────────────────────────────────────┐")
        for name in payload["fixed"]:
            if name.split("(")[0] in "".join(payload["family"]):
                pass
        fixed_best = max(v["sharpe"] for v in payload["fixed"].values())
        family_best = max(v["sharpe"] for v in payload["family"].values())
        print(f"│ Meilleur Sharpe, paramètres figés   {fixed_best:>10.3f}               │")
        print(f"│ Meilleur Sharpe, après sélection    {family_best:>10.3f}               │")
        print(f"│ Écart                               {family_best - fixed_best:>+10.3f}               │")
        print("└──────────────────────────────────────────────────────────────┘")

    (out_dir / "screening.json").write_text(json.dumps(payload, indent=2, default=float),
                                            encoding="utf-8")
    log.info("Résultats dans %s", out_dir)
    print("\nÉtape suivante : les stratégies RETENUES alimentent le méta-modèle "
          "(scripts/meta.py) puis l'allocateur.")
    return 0


def _summary(results) -> dict:
    return {r.name: {"sharpe": r.report.sharpe, "dsr": r.report.deflated_sharpe,
                     "verdict": r.verdict} for r in results}


def _dump_returns(results, path: Path) -> None:
    """Exporte la matrice des rendements OOS — l'entrée de scripts/validate.py --matrix."""
    frame = pd.DataFrame({r.name: r.oos_returns for r in results})
    frame.to_csv(path)


if __name__ == "__main__":
    raise SystemExit(main())

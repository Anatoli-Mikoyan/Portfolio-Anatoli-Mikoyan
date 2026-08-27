#!/usr/bin/env python3
"""Méta-modèle ML sur une stratégie primaire — cahier des charges §6 et §9.

Répond à deux questions du cahier :

  §9 « Compare des modèles simples et complexes, et explique pourquoi un modèle plus
      complexe serait réellement justifié. »
      -> comparaison du zoo par complexité croissante, jugée sur le GAIN ÉCONOMIQUE par
         trade et non sur l'AUC.

  §6 « Comment sélectionner les features, mesurer leur stabilité, éviter les redondances ? »
      -> MDA par permutation sur validation croisée purgée, avec regroupement des
         features corrélées pour neutraliser les effets de substitution.

    python scripts/meta.py --config configs/eurusd_h1.yaml --csv data/EURUSD_H1.csv \
                           --strategy TimeSeriesMomentum
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
from qbot.features import FeaturePipeline, align_features_prices
from qbot.ml import (
    MetaModel, build_meta_dataset, cluster_features, compare_models, cross_validate_meta,
    clustered_mda, justify_complexity, select_features,
)
from qbot.strategies import STRATEGY_CLASSES, default_strategies
from qbot.utils.logging import configure_logging, get_logger

log = get_logger("scripts.meta")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--strategy", type=str, default="TimeSeriesMomentum",
                   choices=[c.__name__ for c in STRATEGY_CLASSES])
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--importance", action="store_true", help="Calcule aussi la MDA par clusters")
    p.add_argument("--out", type=str, default="runs/meta")
    args = p.parse_args()

    configure_logging(logging.INFO)
    cfg = Config.load(args.config) if args.config else Config()
    if args.csv:
        cfg.data.csv_path = args.csv

    df = load_dataset(cfg)
    features = FeaturePipeline(cfg.features).fit_transform(df)
    x, prices = align_features_prices(features, df)

    strategy = next(s for s in default_strategies() if type(s).__name__ == args.strategy)
    log.info("Stratégie primaire : %s", strategy.name)
    print(f"\n{strategy.describe()}\n")

    dataset = build_meta_dataset(strategy, x, prices)
    print(f"Jeu de méta-labels : {len(dataset)} événements, "
          f"taux de base {dataset.base_rate:.1%}, {dataset.X.shape[1]} features")
    print("Le taux de base est la référence : un méta-modèle dont la précision ne le "
          "dépasse pas n'apporte rien.\n")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- §9 : simple contre complexe ---------------------------------------------------
    pd.set_option("display.width", 200)
    table = compare_models(dataset, n_splits=args.splits, embargo_pct=cfg.validation.embargo_pct)
    print("=" * 78)
    print("  §9 — DU PLUS SIMPLE AU PLUS COMPLEXE")
    print("=" * 78)
    print(table.to_string(index=False))
    verdict = justify_complexity(table)
    print(f"\nVERDICT : {verdict}\n")
    table.to_csv(out_dir / "model_comparison.csv", index=False)

    payload = {"strategy": strategy.name, "base_rate": dataset.base_rate,
               "n_events": len(dataset), "verdict": verdict,
               "comparison": table.to_dict(orient="records")}

    # --- §6 : sélection et stabilité des features --------------------------------------
    if args.importance:
        print("=" * 78)
        print("  §6 — IMPORTANCE DES FEATURES (MDA par clusters, CV purgée)")
        print("=" * 78)
        groups = cluster_features(dataset.X, threshold=0.5)
        multi = {k: v for k, v in groups.items() if len(v) > 1}
        print(f"{dataset.X.shape[1]} features regroupées en {len(groups)} clusters "
              f"({len(multi)} contiennent plusieurs features corrélées).\n")
        imp = clustered_mda(dataset, threshold=0.5, model_name="forest",
                            n_splits=max(args.splits - 1, 3), n_repeats=2)
        print(imp.head(15).to_string(index=False))
        kept = select_features(imp, min_t=2.0)
        print(f"\nClusters à importance STABLE entre folds (t >= 2) : {len(kept)}/{len(imp)}")
        print("Le critère porte sur la stabilité, pas sur l'importance moyenne : une "
              "feature\nimportante sur un seul fold décrit une période, pas une régularité.\n")
        imp.to_csv(out_dir / "feature_importance.csv", index=False)
        payload["stable_features"] = kept

    # --- modèle retenu -----------------------------------------------------------------
    best_row = table[table["verdict"] == "UTILE"]
    if not best_row.empty:
        best = best_row.loc[best_row["gain_par_trade"].idxmax()]
        ev = cross_validate_meta(dataset, str(best["modèle"]), n_splits=args.splits)
        meta = MetaModel(str(best["modèle"]), threshold=ev.threshold).fit(dataset)
        print(f"Modèle retenu : {best['modèle']} (seuil {ev.threshold:.2f})")
        print(f"  profit factor {ev.base_profit_factor:.3f} -> {ev.filtered_profit_factor:.3f}")
        print(f"  {ev.trade_retention:.0%} des signaux conservés")
        payload["selected"] = {"model": str(best["modèle"]), "threshold": ev.threshold}
    else:
        print("Aucun modèle n'améliore le rendement par trade : garder la stratégie "
              "primaire telle quelle.\nC'est un résultat, pas un échec.")

    (out_dir / "meta.json").write_text(json.dumps(payload, indent=2, default=float),
                                       encoding="utf-8")
    log.info("Résultats dans %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

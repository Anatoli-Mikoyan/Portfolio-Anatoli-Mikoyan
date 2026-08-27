#!/usr/bin/env python3
"""Compare les approches de détection de régime — cahier des charges §7.

Le cahier demande de « proposer plusieurs approches et de les comparer », puis
d'« expliquer comment cette couche peut déterminer quelles stratégies sont pertinentes ».
Ce script fait les deux, avec un critère unique et mesurable :

    un détecteur de régime n'a de valeur que si la performance des stratégies DIFFÈRE
    réellement d'un régime à l'autre.

La significativité est établie par permutation PAR BLOCS des étiquettes de régime, qui
préserve leur persistance tout en détruisant leur alignement avec la performance.

    python scripts/regime.py --config configs/eurusd_h1.yaml --csv data/EURUSD_H1.csv
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

from qbot.backtest import run_backtest
from qbot.config import Config
from qbot.experiment import bars_per_year, load_dataset
from qbot.regime import (
    build_detector, build_regime_matrix, compare_detectors, conditional_performance,
    lookahead_gain, strategy_regime_map,
)
from qbot.strategies import default_strategies
from qbot.utils.logging import configure_logging, get_logger

log = get_logger("scripts.regime")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--states", type=int, default=3)
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--samples", type=int, default=300, help="Tirages du test de permutation")
    p.add_argument("--out", type=str, default="runs/regime")
    args = p.parse_args()

    configure_logging(logging.INFO)
    cfg = Config.load(args.config) if args.config else Config()
    if args.csv:
        cfg.data.csv_path = args.csv

    df = load_dataset(cfg)
    bpy = bars_per_year(df, cfg)

    # Features de NIVEAU, jamais z-scorées sur fenêtre courte : voir qbot/regime/features.py
    X = build_regime_matrix(df)
    prices = df.loc[X.index]
    cut = int(len(X) * args.train_frac)
    X_train, X_test = X.iloc[:cut], X.iloc[cut:]
    print(f"\nMatrice de régime : {X.shape[0]} barres x {X.shape[1]} features de niveau")
    print(f"Ajustement sur {len(X_train)} barres, évaluation sur {len(X_test)}.\n")

    # Rendements par stratégie, sur le segment d'évaluation
    strategy_returns = {}
    for strat in default_strategies():
        sig = strat.signal(prices)
        res = run_backtest(sig, prices, cfg.costs, cfg.env, bpy)
        strategy_returns[type(strat).__name__] = res.frame["net_return"].reindex(
            res.frame.index.intersection(X_test.index)).dropna()

    # --- §7 : comparaison des approches -------------------------------------------------
    outputs = {}
    print("=" * 78)
    print("  §7 — TROIS APPROCHES, DE LA PLUS SIMPLE À LA PLUS COMPLEXE")
    print("=" * 78)
    for kind in ("rules", "kmeans", "gmm", "hmm"):
        try:
            det = build_detector(kind, **({} if kind == "rules" else {"n_states": args.states}))
            det.fit(X_train)
            outputs[kind] = det.filter(X_test)
            extra = ""
            if kind == "hmm":
                extra = f" | persistance apprise {np.round(det.persistence, 3)}"
            print(f"  {kind:<10} {outputs[kind].states.nunique()} régimes actifs | "
                  f"taux de transition {outputs[kind].transition_rate():.4f}{extra}")
        except Exception as exc:
            log.warning("%s indisponible : %s", kind, exc)

    pd.set_option("display.width", 220)
    table = compare_detectors(outputs, strategy_returns, bpy, n_samples=args.samples)
    print()
    print(table.to_string(index=False))
    print("\n`exploitable` = les performances diffèrent significativement entre régimes")
    print("(p < 0.05 au test de permutation par blocs) ET l'écart est économiquement utile.")

    # --- coût du lissage ----------------------------------------------------------------
    if "hmm" in outputs:
        det = build_detector("hmm", n_states=args.states)
        det.fit(X_train)
        ref = next(iter(strategy_returns.values()))
        gain = lookahead_gain(det, X_test, ref, bpy)
        print("\n┌─ CE QUE COÛTERAIT DE REGARDER LE FUTUR ──────────────────────┐")
        print(f"│ Dispersion du Sharpe, filtrage causal   {gain['dispersion_causale']:>8.3f}             │")
        print(f"│ Dispersion du Sharpe, lissage           {gain['dispersion_lissee']:>8.3f}             │")
        print(f"│ Séparation illusoire                    {gain['separation_illusoire']:>+8.3f}             │")
        print(f"│ Désaccord filtrage / lissage            {gain['taux_desaccord']:>8.2%}             │")
        print("└──────────────────────────────────────────────────────────────┘")
        print("Le lissage trompe le PLUS quand la détection est difficile — donc "
              "précisément\ndans le cas réaliste.")

    # --- sortie exploitable --------------------------------------------------------------
    best = table[table["exploitable"]]
    chosen = (best.groupby("détecteur")["dispersion_sharpe"].mean().idxmax()
              if not best.empty else "rules")
    mapping = strategy_regime_map(outputs[chosen], strategy_returns, bpy, min_sharpe=0.3)

    print(f"\n┌─ CARTE RÉGIME -> STRATÉGIES (détecteur retenu : {chosen}) ─────")
    for regime, strategies in mapping.items():
        listed = ", ".join(strategies) if strategies else "aucune (rester à plat)"
        print(f"│ {regime:<28} -> {listed}")
    print("└" + "─" * 60)
    print("\nCette carte alimente directement l'allocateur : réduire l'espace de recherche")
    print("d'un agent RL est le levier le plus efficace pour l'empêcher de sur-apprendre.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "detector_comparison.csv", index=False)
    for name, reg in outputs.items():
        conditional_performance(reg, next(iter(strategy_returns.values())), bpy).to_csv(
            out_dir / f"conditional_{name}.csv", index=False)
    (out_dir / "regime_map.json").write_text(
        json.dumps({"detector": chosen, "map": mapping}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    log.info("Résultats dans %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

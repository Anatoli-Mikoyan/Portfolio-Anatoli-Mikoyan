#!/usr/bin/env python3
"""Batterie de validation statistique sur une série de rendements out-of-sample.

    python scripts/validate.py --returns runs/walkforward/oos_returns.csv --trials 40

`--trials` est le nombre de configurations que vous avez RÉELLEMENT essayées avant
d'arriver à ce résultat. Le sous-déclarer rend le Deflated Sharpe faussement rassurant :
c'est le paramètre le plus facile à tricher et le plus coûteux à ignorer.
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

from qbot.backtest import compute_report, sharpe_ratio, max_drawdown
from qbot.utils.logging import configure_logging, get_logger
from qbot.validation import (
    bootstrap_metric, compute_pbo, monte_carlo_drawdown, shuffle_trades_test,
    whites_reality_check,
)

log = get_logger("scripts.validate")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--returns", type=str, required=True, help="CSV avec une colonne net_return")
    p.add_argument("--matrix", type=str, default=None,
                   help="CSV (T x N) des rendements de TOUTES les configs testées, pour la PBO")
    p.add_argument("--trials", type=int, default=1, help="Nombre de configurations essayées")
    p.add_argument("--bpy", type=float, default=6240.0, help="Barres par an")
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--block", type=int, default=20)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    configure_logging(logging.INFO)

    df = pd.read_csv(args.returns, index_col=0)
    col = "net_return" if "net_return" in df.columns else df.columns[0]
    r = df[col].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    log.info("%d observations chargées depuis %s", r.size, args.returns)

    report = compute_report(r, args.bpy, n_trials=args.trials)
    print(report)

    print()
    print("┌─ ROBUSTESSE PAR RÉÉCHANTILLONNAGE ───────────────────────────┐")
    boot = bootstrap_metric(r, lambda x: sharpe_ratio(x, args.bpy), args.boot, args.block, 0, "sharpe")
    print(f"│ Sharpe bootstrap  IC95% = [{boot.ci_low:+.3f}, {boot.ci_high:+.3f}]"
          f"   P(>0) = {boot.p_value_positive:.3f}   │")
    mc = monte_carlo_drawdown(r, args.boot, args.block, 0)
    print(f"│ Drawdown observé {mc['observed']:>8.2%} | médiane {mc['median']:>8.2%} "
          f"| pire 5% {mc['p95_worst']:>8.2%} │")
    sh = shuffle_trades_test(r[r != 0.0], min(args.boot, 3000), 0)
    print(f"│ Permutation des trades : DD observé {sh['observed_dd']:>7.2%} vs médiane "
          f"{sh['median_shuffled_dd']:>7.2%}  │")
    print(f"│   -> percentile de chance : {sh['luck_percentile']:.3f} "
          f"({'ordonnancement chanceux' if sh['luck_percentile'] < 0.2 else 'ordonnancement normal'})"
          + " " * 6 + "│")
    print("└──────────────────────────────────────────────────────────────┘")

    payload = {"report": report.to_dict(),
               "bootstrap_sharpe": {"ci_low": boot.ci_low, "ci_high": boot.ci_high,
                                    "p_positive": boot.p_value_positive},
               "monte_carlo_drawdown": mc, "shuffle_test": sh}

    if args.matrix:
        matrix = pd.read_csv(args.matrix, index_col=0).to_numpy(dtype=float)
        log.info("Matrice de configurations : %d obs x %d configs", *matrix.shape)
        pbo = compute_pbo(matrix, n_partitions=16, max_combinations=2000)
        print()
        print(pbo)
        wrc = whites_reality_check(matrix, n_samples=args.boot, block_size=args.block)
        print(f"\nWhite Reality Check : p-value = {wrc['p_value']:.4f} "
              f"(meilleure config = #{wrc['best_strategy']} sur {wrc['n_strategies']})")
        payload["pbo"] = {"pbo": pbo.pbo, "degradation_slope": pbo.degradation_slope,
                          "prob_oos_loss": pbo.prob_oos_loss}
        payload["reality_check"] = wrc

    _verdict(report, boot, payload.get("pbo"))

    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
        log.info("Rapport écrit dans %s", args.out)
    return 0


def _verdict(report, boot, pbo) -> None:
    """Synthèse binaire. Volontairement sévère : le coût d'un faux positif (déployer une
    stratégie perdante) est bien supérieur à celui d'un faux négatif (jeter une bonne)."""
    checks = [
        ("Sharpe out-of-sample > 0.5", report.sharpe > 0.5),
        ("Deflated Sharpe > 0.95", report.deflated_sharpe > 0.95),
        ("Borne basse IC95% du Sharpe > 0", boot.ci_low > 0),
        ("Track record suffisant", report.n_obs >= report.min_track_record),
        ("Profit factor > 1.1", report.profit_factor > 1.1),
        ("Calmar > 0.5", report.calmar > 0.5),
    ]
    if pbo is not None:
        checks.append(("PBO < 0.35", pbo["pbo"] < 0.35))

    print()
    print("┌─ VERDICT ────────────────────────────────────────────────────┐")
    for name, ok in checks:
        print(f"│ [{'OK ' if ok else 'NON'}] {name:<54} │")
    passed = sum(ok for _, ok in checks)
    print("├──────────────────────────────────────────────────────────────┤")
    if passed == len(checks):
        msg = "Tous les tests passent : candidat pour un compte de DÉMONSTRATION."
    elif passed >= len(checks) - 1:
        msg = "Un test échoue : ne pas déployer, comprendre pourquoi d'abord."
    else:
        msg = "Plusieurs tests échouent : cette stratégie n'est pas exploitable."
    print(f"│ {passed}/{len(checks)} — {msg:<52}│")
    print("└──────────────────────────────────────────────────────────────┘")


if __name__ == "__main__":
    raise SystemExit(main())

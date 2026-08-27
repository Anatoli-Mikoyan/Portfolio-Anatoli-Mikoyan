#!/usr/bin/env python3
"""Tenue de marché : gagner sans prédire — et pourquoi c'est hors de portée en retail.

    python scripts/market_making.py

Répond à trois questions, par la mesure et non par l'affirmation :

  1. Quelle politique de cotation gère le mieux l'inventaire ?
  2. Que devient EXACTEMENT la même stratégie sous chaque structure de coûts ?
  3. À partir de quelle toxicité du flux le métier devient-il perdant ?

Options :
    --steps 30000     durée de la session, en secondes de marché
    --seeds 10        nombre de tirages (les politiques partagent les mêmes)
    --html rapport.html
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from qbot.microstructure import (FEE_PROFILES, AvellanedaStoikov, FlowParams,
                                 GueantLehalleFT, LinearSkew, NaiveSymmetric,
                                 compare_fee_profiles, compare_policies, simulate_session)
from qbot.utils.logging import configure_logging
from qbot.utils.text import render_box


def _fmt(df: pd.DataFrame, nd: int = 4) -> str:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].map(lambda v: f"{v:,.{nd}f}")
    return out.to_string(index=False)


def titre(t: str) -> None:
    print(f"\n{'=' * 100}\n  {t}\n{'=' * 100}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=30_000)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--informed", type=float, default=0.15)
    p.add_argument("--json", default=None)
    args = p.parse_args()
    configure_logging(logging.WARNING)

    flow = FlowParams(informed_ratio=args.informed)
    N, S = args.steps, args.seeds
    resultats = {}

    print(render_box("TENUE DE MARCHÉ — GAGNER SANS PRÉDIRE", [
        (None, [("Session simulée", f"{N * flow.dt / 3600:.1f} heures de marché"),
                ("Tirages indépendants", str(S)),
                ("Flux informé", f"{args.informed:.0%}")]),
        ("LE PRINCIPE", [
            ("Une stratégie directionnelle", "PAIE la fourchette pour entrer"),
            ("Un teneur de marché", "l'ENCAISSE en cotant des deux côtés"),
            ("Il ne prédit rien", "il vend un service : la liquidité immédiate")]),
    ], width=100))

    # -- cadences ------------------------------------------------------------------------
    titre("1. LE MODÈLE DE FLUX — combien d'exécutions selon la distance de cotation")
    print(f"{'distance au prix moyen':>26}{'exécutions/min':>18}{'une toutes les':>20}")
    print("-" * 64)
    for d in (0.1, 0.25, 0.5, 1.0, 2.0):
        lam = float(flow.intensity(d * 1e-4))
        delai = f"{1 / lam:,.0f} s" if lam > 1e-9 else "jamais"
        print(f"{d:>22.2f} pip{lam * 60:>18.2f}{delai:>20}")

    # -- politiques ----------------------------------------------------------------------
    titre("2. LES POLITIQUES DE COTATION — même flux, régime teneur professionnel")
    pols = [NaiveSymmetric(), LinearSkew(), AvellanedaStoikov(), GueantLehalleFT()]
    for pol in pols:
        print("\n" + pol.describe(flow))
    t1 = compare_policies(pols, flow, FEE_PROFILES["hft_maker"], n_steps=N, n_seeds=S)
    t1["P&L / risque"] = t1["P&L médian"] / t1["écart-type"].replace(0, np.nan)
    print("\n" + _fmt(t1))
    print("\n→ La naïve gagne autant en moyenne — avec un inventaire dix fois plus gros et")
    print("  une variance dix fois plus forte. L'inventaire EST le risque du métier.")
    resultats["politiques"] = t1.to_dict("records")

    # -- structures de coûts ---------------------------------------------------------------
    titre("3. LA MÊME POLITIQUE SOUS CHAQUE STRUCTURE DE COÛTS")
    t2 = compare_fee_profiles(GueantLehalleFT(), flow, FEE_PROFILES, n_steps=N, n_seeds=S)
    print(_fmt(t2))
    print("\n→ Pas une ligne de code ne change entre ces quatre lignes. Seuls les frais")
    print("  et l'accès à la cotation passive changent — et le signe du résultat avec.")
    print("  La colonne « accès passif » est la frontière : sans elle, on ne tient pas")
    print("  un marché, on le traverse. On devient le client, pas le teneur.")
    resultats["frais"] = t2.to_dict("records")

    # -- toxicité ---------------------------------------------------------------------------
    titre("4. LA SÉLECTION ADVERSE — jusqu'où le flux peut-il être toxique ?")
    print(f"{'impact informé':>16}{'P&L médian':>13}{'fourchette':>13}{'inventaire':>13}{'gagnantes':>12}")
    print("-" * 67)
    tox = []
    for imp in (0.0, 0.3, 1.0, 2.0, 4.0, 8.0):
        f2 = FlowParams(informed_ratio=0.30, informed_impact=imp * 1e-4)
        res = [simulate_session(GueantLehalleFT(), f2, FEE_PROFILES["hft_maker"], N,
                                seed=s, keep_paths=False) for s in range(S)]
        pnl = np.array([r.pnl_total for r in res])
        tox.append({"impact_pip": imp, "pnl": float(np.median(pnl)),
                    "gagnantes": float(np.mean(pnl > 0))})
        print(f"{imp:>14.1f} pip{np.median(pnl):>13.4f}"
              f"{np.mean([r.pnl_spread for r in res]):>13.4f}"
              f"{np.mean([r.pnl_inventory for r in res]):>13.4f}"
              f"{np.mean(pnl > 0):>12.0%}")
    print("\n→ La fourchette encaissée ne bouge pas d'un centième : c'est l'inventaire qui")
    print("  bascule. Contre un flux qui sait, la fourchette ne suffit plus à payer le")
    print("  mouvement qui suit — et aucun réglage de cotation ne rattrape cela.")
    resultats["toxicite"] = tox

    # -- conclusion ---------------------------------------------------------------------------
    titre("CE QUE CETTE SIMULATION ÉTABLIT")
    print("""
  Le trading haute fréquence ne consiste pas à prédire plus vite. Il consiste à
  fournir de la liquidité et à être payé pour — un métier de service, pas de pari.

  Trois conditions le rendent possible, et aucune ne relève de l'algorithme :

    1. ACCÈS À LA COTATION PASSIVE. Sans lui, on paie la fourchette au lieu de
       l'encaisser. C'est la ligne « accès passif : NON » du tableau 3, et elle
       fait basculer le résultat du positif au franchement négatif.

    2. FRAIS NÉGATIFS. Les places versent des rebates aux apporteurs de liquidité.
       Un compte de détail paie ; un teneur agréé encaisse. L'écart entre les deux
       premières lignes du tableau 3 ne tient qu'à cela.

    3. VITESSE. Non pour prédire, mais pour ANNULER : retirer sa cotation avant
       qu'un flux informé ne la frappe. C'est la seule défense réelle contre la
       sélection adverse du tableau 4, et elle se joue en microsecondes — contre
       30 à 200 millisecondes depuis un terminal de détail.

  L'algorithme est la partie la moins chère et la moins différenciante du métier.
""")

    if args.json:
        Path(args.json).write_text(json.dumps(resultats, indent=2, default=str),
                                   encoding="utf-8")
        print(f"  Résultats détaillés : {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

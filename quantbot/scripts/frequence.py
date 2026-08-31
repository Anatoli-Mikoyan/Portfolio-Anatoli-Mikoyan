#!/usr/bin/env python3
"""À quelle fréquence ce modèle devrait-il trader ? Protocole sans biais de sélection.

    python scripts/frequence.py

Le bot perd de l'argent en tradant chaque heure : 69 % du capital par an en frais.
Sans frais il finissait positif. La question n'est donc pas « le modèle prévoit-il
mieux ? » mais « à quel rythme faut-il l'écouter pour que ses frais cessent de tout
manger ? ».

Protocole — c'est ici que tout se joue
--------------------------------------
Une première tentative avait été faite en balayant les fréquences et en gardant la
meilleure. Elle s'est auto-réfutée : le meilleur réglage (+3,6 %) avait un Deflated
Sharpe de 0,163 sur 6 essais. Choisir le maximum d'un balayage PUIS l'évaluer sur les
mêmes données produit un chiffre flatteur et faux, mécaniquement.

Ici, séparation stricte :

  1. les fréquences sont comparées sur la période de SÉLECTION uniquement ;
  2. la meilleure est GELÉE — plus aucun retour en arrière ;
  3. elle est évaluée UNE SEULE FOIS sur la période de TEST, jamais vue.

Le chiffre final porte donc sur un seul essai. Le nombre de candidats reste déclaré
au Deflated Sharpe, parce que la sélection a bien consommé de l'information — le
taire reviendrait à refaire l'erreur qu'on corrige.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BPY = 6240.0
# Une décision toutes les N barres H1. 1 = le comportement actuel.
FREQUENCES = [1, 2, 4, 8, 24, 48, 120, 240]
NOMS = {1: "chaque heure", 2: "2 heures", 4: "4 heures", 8: "8 heures",
        24: "chaque jour", 48: "2 jours", 120: "chaque semaine", 240: "2 semaines"}


def espacer(positions: pd.Series, pas: int) -> pd.Series:
    """Ne laisse le bot changer d'avis qu'une barre sur `pas`.

    La position décidée est CONSERVÉE entre deux points de décision : c'est bien un
    ralentissement du rythme, pas une mise à plat entre les décisions. Rien n'est
    recalculé — on écoute le même modèle, moins souvent.
    """
    if pas <= 1:
        return positions
    garde = positions.copy()
    masque = np.zeros(len(positions), dtype=bool)
    masque[::pas] = True
    garde[~masque] = np.nan
    return garde.ffill().fillna(0.0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modele", type=str, default=str(ROOT / "runs" / "start" / "modele"))
    p.add_argument("--csv", type=str, default=str(ROOT / "data" / "EURUSD_H1.csv"))
    p.add_argument("--capital", type=float, default=1000.0)
    args = p.parse_args()

    import logging

    from qbot.backtest import run_backtest
    from qbot.backtest.metrics import deflated_sharpe_ratio
    from qbot.data import load_ohlcv
    from qbot.experiment import TrainedModel, run_agent_on_segment
    from qbot.live.engine import load_bundle
    from qbot.utils.logging import configure_logging
    from qbot.validation.monte_carlo import stationary_bootstrap_indices

    configure_logging(logging.WARNING)

    bundle = load_bundle(Path(args.modele))
    df = load_ohlcv(args.csv)
    n = len(df)
    train = df.iloc[: int(n * 0.75)]
    valid = df.iloc[int(n * 0.75): int(n * 0.9)]
    test = df.iloc[int(n * 0.9):]

    modele = TrainedModel(agent=bundle.agent, pipeline=bundle.pipeline,
                          config=bundle.config, valid_sharpe=0.0)

    print("=" * 74)
    print("  À QUELLE FRÉQUENCE FAUT-IL ÉCOUTER CE MODÈLE ?")
    print("=" * 74)
    print(f"\n  sélection : {valid.index[0]:%d/%m/%Y} → {valid.index[-1]:%d/%m/%Y} "
          f"({len(valid):,} barres)")
    print(f"  test      : {test.index[0]:%d/%m/%Y} → {test.index[-1]:%d/%m/%Y} "
          f"({len(test):,} barres)  — INTOUCHÉE jusqu'à l'étape 2")

    # ---- 1. Sélection, sur la période de sélection UNIQUEMENT -------------------------
    print("\n  ÉTAPE 1 — comparaison sur la période de sélection\n")
    pos_valid, _ = run_agent_on_segment(modele, valid, context_df=train, bpy=BPY)

    print(f"  {'rythme':<16}{'Sharpe':>9}{'rendement':>12}{'frais/an':>11}{'trades':>9}")
    print("  " + "-" * 60)
    resultats = {}
    for pas in FREQUENCES:
        bt = run_backtest(espacer(pos_valid, pas), valid, bundle.config.costs,
                          bundle.config.env, BPY)
        r = bt.report
        resultats[pas] = r.sharpe
        print(f"  {NOMS[pas]:<16}{r.sharpe:>9.2f}{r.total_return:>11.2%}"
              f"{r.cost_drag_annual:>11.2%}{r.n_trades:>9,}")

    meilleur = max(resultats, key=resultats.get)
    print(f"\n  → retenu : « {NOMS[meilleur]} » (Sharpe {resultats[meilleur]:+.2f} "
          f"sur la sélection)")
    print("    Ce choix est GELÉ. La période de test n'a pas encore été touchée.")

    # ---- 2. Évaluation, une seule fois ----------------------------------------------
    print("\n  ÉTAPE 2 — évaluation unique sur la période de test\n")
    pos_test, _ = run_agent_on_segment(modele, test, context_df=valid, bpy=BPY)

    bt_ref = run_backtest(pos_test, test, bundle.config.costs, bundle.config.env, BPY)
    bt_new = run_backtest(espacer(pos_test, meilleur), test, bundle.config.costs,
                          bundle.config.env, BPY)

    for etiquette, bt in (("chaque heure (actuel)", bt_ref),
                          (f"{NOMS[meilleur]} (retenu)", bt_new)):
        final = args.capital * float(np.prod(1.0 + bt.returns))
        r = bt.report
        print(f"  {etiquette:<26}{final:>11,.2f} €  ({final / args.capital - 1:+.2%})"
              f"   Sharpe {r.sharpe:+.2f}   frais {r.cost_drag_annual:.2%}/an")

    # ---- 3. Ce résultat tient-il ? ----------------------------------------------------
    r = bt_new.returns
    rng = np.random.default_rng(0)
    tirages = np.array([
        float(np.prod(1.0 + r[stationary_bootstrap_indices(len(r), 24, rng)]))
        for _ in range(2000)])
    bas, haut = (args.capital * q for q in np.percentile(tirages, [5, 95]))
    dsr = deflated_sharpe_ratio(r, n_trials=len(FREQUENCES), bars_per_year=BPY)

    print(f"\n  Intervalle à 90 % (2 000 rééchantillonnages par blocs) :")
    print(f"    {bas:,.2f} €  →  {haut:,.2f} €")
    print(f"  Deflated Sharpe ({len(FREQUENCES)} candidats déclarés) : {dsr:.3f}")

    print("\n" + "=" * 74)
    if bas > args.capital:
        verdict = "L'intervalle est ENTIÈREMENT POSITIF."
    elif haut < args.capital:
        verdict = "L'intervalle est ENTIÈREMENT NÉGATIF : ce n'est pas de la malchance."
    else:
        verdict = "L'intervalle CONTIENT le capital de départ : rien n'est établi."
    print(f"  {verdict}")
    if dsr < 0.95:
        print(f"  Deflated Sharpe {dsr:.3f} < 0.95 : le résultat ne survit pas à la")
        print(f"  correction pour les {len(FREQUENCES)} fréquences essayées.")
    else:
        print(f"  Deflated Sharpe {dsr:.3f} : le résultat survit à la correction.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

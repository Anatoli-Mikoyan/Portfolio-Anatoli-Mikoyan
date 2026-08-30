#!/usr/bin/env python3
"""Regarder le bot trader, barre par barre, sans MetaTrader et sans marché ouvert.

    python scripts/regarder.py

Rejoue la période de TEST — celle que le modèle n'a jamais vue pendant son
entraînement — en affichant chaque décision au fur et à mesure : la position visée,
les transactions à leur ouverture et à leur fermeture, et le capital qui évolue.

C'est exactement ce que MetaTrader affichera, en accéléré. Onze mois de marché
défilent en deux minutes au lieu de onze mois.

    python scripts/regarder.py --vitesse 40     # plus rapide (barres par seconde)
    python scripts/regarder.py --capital 5000
    python scripts/regarder.py --tout           # sans attendre : tout d'un coup

Ce qui est simulé et ce qui ne l'est pas
---------------------------------------
Les décisions viennent du VRAI modèle entraîné, chargé depuis runs/start/modele.
La comptabilité — frais, spread, commission — est celle du moteur de backtest
audité par les tests, pas un calcul refait pour l'occasion. Ce que vous voyez
défiler est donc le comportement réel du bot sur des prix réels.

Ce n'est pas de la prédiction : cette période est passée. C'est une observation
de ce que le bot AURAIT fait, sur des données qu'il n'avait jamais vues.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODELE_DEFAUT = ROOT / "runs" / "start" / "modele"
DONNEES_DEFAUT = ROOT / "data" / "EURUSD_H1.csv"
BPY = 6240.0
LARGEUR = 74


def titre(texte: str) -> None:
    print(f"\n{'─' * LARGEUR}\n\033[1m  {texte}\033[0m\n{'─' * LARGEUR}")


def _sens(position: float) -> str:
    if position > 1e-9:
        return "ACHAT"
    if position < -1e-9:
        return "VENTE"
    return "PLAT"


def _couleur(valeur: float) -> str:
    return "\033[32m" if valeur > 0 else ("\033[31m" if valeur < 0 else "")


def _barre_position(position: float, largeur: int = 11) -> str:
    """Une jauge visuelle de l'exposition, de -100 % à +100 %."""
    milieu = largeur // 2
    cases = ["·"] * largeur
    cases[milieu] = "|"
    n = int(round(min(abs(position), 1.0) * milieu))
    for k in range(1, n + 1):
        idx = milieu + k if position > 0 else milieu - k
        if 0 <= idx < largeur:
            cases[idx] = "█"
    return "".join(cases)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modele", type=str, default=str(MODELE_DEFAUT))
    p.add_argument("--csv", type=str, default=str(DONNEES_DEFAUT))
    p.add_argument("--capital", type=float, default=1000.0)
    p.add_argument("--vitesse", type=float, default=25.0,
                   help="barres affichées par seconde (défaut 25)")
    p.add_argument("--tout", action="store_true", help="tout afficher sans attendre")
    args = p.parse_args()

    modele_dir = Path(args.modele)
    if not modele_dir.exists():
        print(f"  [X] Aucun modèle entraîné dans {modele_dir}")
        print("      Lancez d'abord :  python scripts/start.py")
        return 1
    csv = Path(args.csv)
    if not csv.exists():
        print(f"  [X] Données introuvables : {csv}")
        print("      Lancez d'abord :  python scripts/start.py")
        return 1

    titre("PRÉPARATION")
    print("  Chargement du modèle et des prix…")

    import logging

    from qbot.backtest import run_backtest
    from qbot.data import load_ohlcv
    from qbot.experiment import TrainedModel, run_agent_on_segment
    from qbot.live.engine import load_bundle
    from qbot.utils.logging import configure_logging

    configure_logging(logging.WARNING)

    bundle = load_bundle(modele_dir)
    # Le découpage doit être IDENTIQUE à celui de start.py, sinon la période
    # « jamais vue » n'en serait plus une et tout ce qui suit serait faux.
    df = load_ohlcv(csv)
    n = len(df)
    valid = df.iloc[int(n * 0.75): int(n * 0.9)]
    test = df.iloc[int(n * 0.9):]

    modele = TrainedModel(agent=bundle.agent, pipeline=bundle.pipeline,
                          config=bundle.config, valid_sharpe=0.0)

    print(f"  Période rejouée : du {test.index[0]:%d/%m/%Y} au {test.index[-1]:%d/%m/%Y} "
          f"({len(test):,} heures)")
    print("  Le modèle décide… (une trentaine de secondes)")

    positions, _ = run_agent_on_segment(modele, test, context_df=valid, bpy=BPY)
    resultat = run_backtest(positions, test, bundle.config.costs, bundle.config.env, BPY)
    frame = resultat.frame

    titre(f"LE BOT TRADE — {args.capital:,.0f} € de départ")
    print("  date       heure    prix     position                capital")
    print(f"  {'─' * (LARGEUR - 4)}")

    delai = 0.0 if args.tout else 1.0 / max(args.vitesse, 0.1)
    capital = args.capital
    precedente = 0.0
    n_trades = 0
    prix_entree = 0.0
    date_entree = None
    pic = args.capital

    try:
        for horodatage, ligne in frame.iterrows():
            position = float(ligne["position"])
            capital *= 1.0 + float(ligne["net_return"])
            pic = max(pic, capital)
            prix = float(test.loc[horodatage, "close"])

            change = abs(position - precedente) > 1e-9
            ouvre = change and abs(precedente) < 1e-9 and abs(position) > 1e-9
            ferme = change and abs(precedente) > 1e-9 and abs(position) < 1e-9

            if ouvre:
                n_trades += 1
                prix_entree, date_entree = prix, horodatage

            marque = " "
            if ouvre:
                marque = "\033[36m>\033[0m"
            elif ferme:
                marque = "\033[35m<\033[0m"
            elif change:
                marque = "\033[33m~\033[0m"

            c = _couleur(capital - args.capital)
            print(f"  {horodatage:%d/%m/%y}  {horodatage:%H:%M}  {prix:.5f}  "
                  f"{marque} {_barre_position(position)} {_sens(position):<5} "
                  f"{c}{capital:>10,.2f} €\033[0m", flush=True)

            if ferme and date_entree is not None:
                gain = (prix / prix_entree - 1.0) * (1.0 if precedente > 0 else -1.0)
                heures = (horodatage - date_entree).total_seconds() / 3600.0
                cg = _couleur(gain)
                print(f"           └─ transaction #{n_trades} fermée après "
                      f"{heures:.0f} h : {cg}{gain:+.2%}\033[0m sur le prix", flush=True)

            precedente = position
            if delai:
                time.sleep(delai)
    except KeyboardInterrupt:
        print("\n  (interrompu)")

    rep = resultat.report
    bh = args.capital * float(test["close"].iloc[-1] / test["close"].iloc[0])
    dd = capital / pic - 1.0 if pic > 0 else 0.0

    titre("CE QUI S'EST PASSÉ")
    print(f"  {'Le bot (net de frais)':<30}{capital:>12,.2f} €   "
          f"({capital / args.capital - 1:+.2%})")
    print(f"  {'Acheter et conserver':<30}{bh:>12,.2f} €   ({bh / args.capital - 1:+.2%})")
    print(f"  {'Ne rien faire':<30}{args.capital:>12,.2f} €   ( +0.00%)")
    print(f"\n  {rep.n_trades:,} transactions  |  Sharpe {rep.sharpe:+.2f}  |  "
          f"pire recul {rep.max_drawdown:.2%}  |  frais {rep.cost_drag_annual:.2%}/an")
    print(f"\n  Voilà exactement ce que MetaTrader affichera, à la vitesse du marché :")
    print("  une décision par heure, au lieu de vingt-cinq par seconde.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

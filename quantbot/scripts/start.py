#!/usr/bin/env python3
"""Démarrage en une commande : de rien du tout à un tableau de bord ouvert.

    python scripts/start.py

Fait tout, dans l'ordre, sans rien demander :

  1. vérifie que les dépendances sont là
  2. récupère des données EURUSD horaires si vous n'en avez pas
  3. mesure s'il y a un signal exploitable (la sonde)
  4. crible les stratégies classiques, frais compris
  5. entraîne l'agent et l'évalue sur une période jamais vue
  6. écrit un rapport HTML et l'ouvre dans votre navigateur

Compter 10 à 20 minutes. Chaque étape affiche son verdict au fur et à mesure : si la
première dit qu'il n'y a pas de signal, vous le saurez en trente secondes.

    python scripts/start.py --csv mes_donnees.csv   # vos propres données
    python scripts/start.py --rapide                # entraînement écourté (~5 min)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_URL = ("https://raw.githubusercontent.com/ejtraderLabs/historical-data/"
            "main/EURUSD/EURUSDh1.csv")
BPY = 6240.0
CAPITAL = 1000.0


# =======================================================================================
def dire(titre: str, corps: str = "") -> None:
    print(f"\n\033[1m{titre}\033[0m")
    if corps:
        print(corps)


def etape(n: int, total: int, titre: str) -> None:
    print(f"\n{'─' * 74}\n\033[1m  ÉTAPE {n}/{total} — {titre}\033[0m\n{'─' * 74}")


def verifier_dependances() -> bool:
    """Signale ce qui manque, avec la commande exacte pour l'installer."""
    manquants = []
    for module, paquet in [("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"),
                           ("sklearn", "scikit-learn"), ("torch", "torch")]:
        try:
            __import__(module)
        except ImportError:
            manquants.append(paquet)
    if manquants:
        dire("Il manque des bibliothèques.",
             f"Lancez :  pip install {' '.join(manquants)}\n"
             f"Ou d'un coup :  pip install -r {ROOT / 'requirements.txt'}")
        return False
    return True


def obtenir_donnees(csv: str | None) -> Path:
    """Retourne le chemin d'un CSV OHLCV utilisable, en le téléchargeant au besoin."""
    import pandas as pd

    if csv:
        chemin = Path(csv)
        if not chemin.exists():
            raise SystemExit(f"Fichier introuvable : {chemin}")
        return chemin

    cible = ROOT / "data" / "EURUSD_H1.csv"
    if cible.exists():
        print(f"Données déjà présentes : {cible}")
        return cible

    print("Aucune donnée fournie — téléchargement d'un historique EURUSD horaire public…")
    cible.parent.mkdir(parents=True, exist_ok=True)
    try:
        brut = pd.read_csv(DATA_URL)
    except Exception as exc:
        raise SystemExit(
            f"Téléchargement impossible ({exc}).\n\n"
            "Exportez vos propres données depuis MetaTrader 5 :\n"
            "  Outils -> Centre d'historique -> choisir le symbole et H1 -> Exporter\n"
            "puis relancez :  python scripts/start.py --csv chemin/vers/fichier.csv")

    # Ce jeu de données stocke les prix multipliés par 100 000.
    df = pd.DataFrame({
        "time": pd.to_datetime(brut["Date"], utc=True),
        "open": brut["open"] / 1e5, "high": brut["high"] / 1e5,
        "low": brut["low"] / 1e5, "close": brut["close"] / 1e5,
        "volume": brut["tick_volume"].astype(float),
    }).set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df["spread"] = df["close"] * 1.0e-4          # ~1 pip, valeur retail ECN typique
    df.to_csv(cible)
    print(f"{len(df):,} barres écrites dans {cible}")
    return cible


# =======================================================================================
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=None, help="vos données (sinon : téléchargement)")
    p.add_argument("--rapide", action="store_true", help="entraînement écourté")
    p.add_argument("--capital", type=float, default=CAPITAL, help="capital simulé, en euros")
    p.add_argument("--sortie", default=str(ROOT / "runs" / "start"), help="dossier de sortie")
    p.add_argument("--pas-de-navigateur", action="store_true")
    args = p.parse_args()

    if not verifier_dependances():
        return 1

    import numpy as np
    import pandas as pd

    from qbot.backtest import run_backtest
    from qbot.config import (AgentConfig, Config, CostConfig, EnvConfig, FeatureConfig,
                             RiskConfig, TrainConfig)
    from qbot.data.loader import load_ohlcv
    from qbot.diagnostics import signal_report
    from qbot.experiment import backtest_model, train_model
    from qbot.features import FeaturePipeline
    from qbot.strategies import default_strategies
    from qbot.strategies.screening import screen_fixed, screening_table
    from qbot.utils.logging import configure_logging
    from qbot.validation import bootstrap_metric

    configure_logging(logging.WARNING)
    t0 = time.time()
    sortie = Path(args.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    resultats: dict = {"capital": args.capital}

    print("\n" + "=" * 74)
    print("  QBOT — DE RIEN DU TOUT À UN VERDICT CHIFFRÉ")
    print("=" * 74)

    # -- 1. données --------------------------------------------------------------------
    etape(1, 6, "LES DONNÉES")
    chemin = obtenir_donnees(args.csv)
    df = load_ohlcv(chemin)
    n = len(df)
    train = df.iloc[: int(n * 0.75)]
    valid = df.iloc[int(n * 0.75): int(n * 0.9)]
    test = df.iloc[int(n * 0.9):]
    print(f"{n:,} barres, du {df.index[0]:%d/%m/%Y} au {df.index[-1]:%d/%m/%Y}")
    print(f"  entraînement {len(train):,}  |  sélection {len(valid):,}  |  "
          f"TEST {len(test):,} (jamais vu par le modèle)")
    resultats["periode"] = f"{df.index[0]:%d/%m/%Y} → {df.index[-1]:%d/%m/%Y}"
    resultats["n_barres"] = n

    # -- 2. le mur des coûts -------------------------------------------------------------
    etape(2, 6, "LE MUR DES COÛTS")
    spread_bps, comm_bps = 0.91, 0.20
    cout = spread_bps + comm_bps
    vol = float(np.log(test["close"]).diff().std() * np.sqrt(BPY))
    print(f"Spread 1 pip + commission ECN = {cout:.2f} bps par unité de turnover.")
    print(f"Volatilité annualisée du marché sur la période de test : {vol:.2%}\n")
    print(f"  {'Si le bot trade':<20}{'Frais/an':>12}{'Sharpe à produire pour les couvrir':>38}")
    for label, tpb in [("à chaque barre", 1.0), ("une fois par jour", 1 / 24),
                       ("une fois par semaine", 1 / 120)]:
        drag = tpb * cout / 1e4 * BPY
        print(f"  {label:<20}{drag:>11.2%}{drag / vol:>37.2f}")
    print("\n→ Un bot qui trade à chaque heure doit produire un Sharpe supérieur à celui")
    print("  des meilleurs fonds du monde AVANT de gagner le premier euro.")
    resultats["cout_bps"] = cout
    resultats["vol_marche"] = vol

    # -- 3. la sonde ---------------------------------------------------------------------
    etape(3, 6, "Y A-T-IL SEULEMENT UN SIGNAL ?")
    fcfg = FeatureConfig(returns_windows=(1, 5, 20, 60), vol_windows=(10, 20, 60),
                         ema_windows=(10, 30, 100), rsi_windows=(14,),
                         use_microstructure=True, use_calendar=True, scaler_window=500)
    pipe = FeaturePipeline(fcfg)
    X = pipe.fit_transform(df)
    lineaire, reseaux = signal_report(X, df, windows=(1, 16), steps=2_000)
    meilleur = max(reseaux, key=lambda r: r.ic) if reseaux else lineaire
    resultats["sonde"] = (
        f"régression linéaire   IC = {lineaire.ic:+.4f}  "
        f"(train {lineaire.ic_train:+.4f}, écart {lineaire.ic_train - lineaire.ic:+.4f})\n"
        f"meilleur réseau       IC = {meilleur.ic:+.4f}  "
        f"(train {meilleur.ic_train:+.4f}, écart {meilleur.ic_train - meilleur.ic:+.4f})\n\n"
        + ("Le réseau fait MOINS BIEN que la régression linéaire : il sur-apprend.\n"
           "Réduire la capacité du modèle plutôt que l'entraîner plus longtemps."
           if meilleur.ic < lineaire.ic else
           "Le réseau extrait davantage que la régression linéaire : il y a du\n"
           "non-linéaire exploitable, l'entraînement RL a une chance d'aboutir."))
    resultats["ic_lineaire"] = float(lineaire.ic)
    resultats["ic_reseau"] = float(meilleur.ic)

    # -- 4. le criblage ------------------------------------------------------------------
    etape(4, 6, "LES STRATÉGIES CLASSIQUES SURVIVENT-ELLES AUX FRAIS ?")
    ccfg = CostConfig(spread_bps=spread_bps, commission_bps=comm_bps, slippage_model="sqrt")
    ecfg = EnvConfig(window=16, positions=(-1.0, 0.0, 1.0), vol_target=0.10)
    crible = [screen_fixed(s, df, ccfg, ecfg, bpy=BPY) for s in default_strategies()]
    table = screening_table(crible)
    print(table.to_string(index=False))
    survivantes = int(sum(1 for r in crible if "REJET" not in r.verdict))
    print(f"\n→ {survivantes}/{len(crible)} hypothèses survivent.")
    resultats["survivantes"] = f"{survivantes}/{len(crible)}"

    # -- 5. l'agent ----------------------------------------------------------------------
    etape(5, 6, "L'AGENT PAR RENFORCEMENT")
    pas = 6_000 if args.rapide else 15_000
    print(f"Entraînement sur {len(train):,} barres ({pas:,} pas)… "
          f"{'~4' if args.rapide else '~10'} minutes.")
    cfg = Config(
        features=fcfg, costs=ccfg,
        env=EnvConfig(window=16, positions=(-1.0, 0.0, 1.0), vol_target=0.10,
                      episode_length=2048, random_start=True, max_drawdown_stop=0.25),
        agent=AgentConfig(hidden_sizes=(64, 64), weight_decay=1e-3, encoder="mlp",
                          distributional="qr", n_quantiles=51),
        train=TrainConfig(total_steps=pas, eval_every=max(pas // 6, 1_000),
                          log_every=10 ** 9, early_stop_patience=6),
        risk=RiskConfig(max_daily_loss=0.03, max_drawdown_stop=0.20),
        run_name="start", seed=42,
    )
    modele = train_model(cfg, train, valid, BPY)
    modele.save(sortie / "modele")
    print(f"Entraîné. Sharpe sur la période de sélection : {modele.valid_sharpe:+.3f}")

    # -- 6. le verdict -------------------------------------------------------------------
    etape(6, 6, "LE VERDICT SUR ARGENT SIMULÉ")
    bt, pos = backtest_model(modele, test, context_df=valid, bpy=BPY)
    r, rep = bt.returns, bt.report
    final = args.capital * float(np.prod(1.0 + r))
    bh = args.capital * float(test["close"].iloc[-1] / test["close"].iloc[0])

    print(f"\n  {args.capital:,.0f} € placés du {test.index[0]:%d/%m/%Y} "
          f"au {test.index[-1]:%d/%m/%Y}\n")
    print(f"  {'Le bot (net de frais)':<32}{final:>12,.2f} €   ({final / args.capital - 1:+.2%})")
    print(f"  {'Acheter et conserver':<32}{bh:>12,.2f} €   ({bh / args.capital - 1:+.2%})")
    print(f"  {'Ne rien faire':<32}{args.capital:>12,.2f} €   ( +0.00%)")
    print(f"\n  Sharpe {rep.sharpe:+.2f}  |  drawdown max {rep.max_drawdown:.2%}  |  "
          f"{rep.n_trades:,} transactions  |  frais {rep.cost_drag_annual:.2%}/an")

    # L'intervalle compte plus que le chiffre : un résultat unique ne prouve rien.
    bs = bootstrap_metric(r, lambda x: float(np.prod(1.0 + x) - 1.0),
                          n_samples=2000, block_size=24, seed=1)
    q = np.quantile(bs.samples, [0.05, 0.5, 0.95])
    print(f"\n  Et si cette année s'était déroulée un peu autrement ? "
          f"(2 000 rééchantillonnages)")
    print(f"    5e centile {args.capital * (1 + q[0]):>10,.2f} €   "
          f"médiane {args.capital * (1 + q[1]):>10,.2f} €   "
          f"95e centile {args.capital * (1 + q[2]):>10,.2f} €")
    if q[2] < 0:
        print("    → l'intervalle ENTIER est négatif : ce n'est pas de la malchance.")
    elif q[0] > 0:
        print("    → l'intervalle entier est positif : résultat inhabituellement solide.")
    else:
        print("    → l'intervalle contient zéro : ce résultat ne prouve rien, "
              "dans un sens ni dans l'autre.")

    # Ce qui reste sans les frais : le signal existe-t-il, indépendamment du coût ?
    sans_frais = run_backtest(pos, test, CostConfig(spread_bps=0.0, commission_bps=0.0,
                                                    slippage_model="none", min_trade_size=0.0),
                              cfg.env, BPY, 1)
    brut = args.capital * float(np.prod(1.0 + sans_frais.returns))
    print(f"\n  Mêmes décisions, frais mis à zéro (impossible en vrai) : {brut:,.2f} € "
          f"({brut / args.capital - 1:+.2%})")
    if brut > args.capital and final < args.capital:
        print("    → le modèle a un peu de flair, mais les frais le mangent entièrement.")

    resultats.update({
        "final": final, "buy_hold": bh, "brut_sans_frais": brut,
        "sharpe": rep.sharpe, "drawdown": rep.max_drawdown, "n_trades": rep.n_trades,
        "frais_annuels": rep.cost_drag_annual, "ci_bas": float(q[0]),
        "ci_median": float(q[1]), "ci_haut": float(q[2]),
        "valid_sharpe": modele.valid_sharpe, "psr": rep.psr,
        "deflated_sharpe": rep.deflated_sharpe, "hit_rate": rep.hit_rate,
        "test_debut": f"{test.index[0]:%d/%m/%Y}", "test_fin": f"{test.index[-1]:%d/%m/%Y}",
        "equity": (args.capital * np.cumprod(1.0 + r)).tolist(),
        "duree_s": time.time() - t0,
    })
    (sortie / "resultat.json").write_text(json.dumps(resultats, indent=2, default=str),
                                          encoding="utf-8")

    from qbot.rapport import ecrire_rapport
    page = ecrire_rapport(resultats, sortie / "rapport.html")

    print("\n" + "=" * 74)
    print(f"  TERMINÉ en {(time.time() - t0) / 60:.1f} minutes")
    print("=" * 74)
    print(f"\n  Rapport : {page}")
    if not args.pas_de_navigateur:
        try:
            webbrowser.open(page.resolve().as_uri())
            print("  (ouvert dans votre navigateur)")
        except Exception:
            print("  Ouvrez ce fichier dans votre navigateur.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

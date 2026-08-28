#!/usr/bin/env python3
"""Connecter le bot à MetaTrader 5, en trois commandes.

    python scripts/mt5.py installer    # copie l'EA dans MetaTrader et explique la suite
    python scripts/mt5.py tester       # prouve que la chaîne marche, SANS MetaTrader
    python scripts/mt5.py demarrer     # lance le serveur auquel MetaTrader se connecte

Comment ça marche
-----------------
Le bot ne « pilote » pas MetaTrader de l'extérieur. C'est l'inverse : un petit
programme installé DANS MetaTrader (un Expert Advisor, ou EA) appelle le bot à
chaque nouvelle bougie et lui demande quoi faire.

    MetaTrader 5                          Votre PC, fenêtre noire
    ┌────────────────────┐                ┌──────────────────────┐
    │  QBotBridge (EA)   │ ──1200 barres─▶│  serveur d'inférence │
    │  passe les ordres  │ ◀──exposition──│  le modèle entraîné  │
    └────────────────────┘                └──────────────────────┘
              │                    TCP 127.0.0.1:8912
              ▼
        votre courtier

Les deux doivent tourner en même temps, sur la même machine. Si vous fermez la
fenêtre noire, l'EA ne reçoit plus de réponse et reste à plat : il ne prend pas
de décision tout seul, il n'en est pas capable — il ne contient aucune stratégie.

Sécurité : deux verrous indépendants, tous les deux fermés par défaut.
  - Côté MetaTrader : InpDryRun = true  → l'EA n'envoie aucun ordre.
  - Côté Python     : pas de --reel     → le serveur refuse toute ouverture.
Il faut lever les DEUX pour qu'un ordre parte. Ce n'est pas une précaution
décorative : voir le verdict des tests avant d'y toucher.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import sys
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PORT_DEFAUT = 8912
MODELE_DEFAUT = ROOT / "runs" / "start" / "modele"
EA_SOURCE = ROOT / "mql5" / "QBotBridge.mq5"
LARGEUR = 74


# =======================================================================================
# Affichage
# =======================================================================================
def titre(texte: str) -> None:
    print(f"\n{'─' * LARGEUR}\n\033[1m  {texte}\033[0m\n{'─' * LARGEUR}")


def ok(texte: str) -> None:
    print(f"  [ok]  {texte}")


def attention(texte: str) -> None:
    print(f"  [!]   {texte}")


def echec(texte: str) -> None:
    print(f"  [X]   {texte}")


# =======================================================================================
# Trouver MetaTrader 5
# =======================================================================================
def dossiers_metatrader() -> List[Path]:
    """Repère les dossiers de données MetaTrader 5 installés sur la machine.

    MetaTrader ne range pas ses fichiers là où il est installé, mais dans un dossier
    par terminal, nommé d'après une empreinte de 32 caractères :

        %APPDATA%\\MetaQuotes\\Terminal\\<32 caractères>\\MQL5\\Experts

    D'où l'exploration : on ne peut pas deviner l'empreinte, seulement la trouver.
    Le dossier « Common » est partagé entre terminaux et n'accueille pas d'EA
    compilable — on l'écarte.
    """
    racines: List[Path] = []
    if platform.system() == "Windows":
        appdata = os.environ.get("APPDATA")
        if appdata:
            racines.append(Path(appdata) / "MetaQuotes" / "Terminal")
    else:
        # MetaTrader sous Linux/macOS tourne via Wine : même arborescence, préfixée.
        maison = Path.home()
        for prefixe in (maison / ".wine", maison / ".mt5", maison / ".PlayOnLinux"):
            racines.extend(prefixe.glob("**/AppData/Roaming/MetaQuotes/Terminal"))

    trouves: List[Path] = []
    for racine in racines:
        if not racine.is_dir():
            continue
        for candidat in sorted(racine.iterdir()):
            if not candidat.is_dir() or candidat.name.lower() == "common":
                continue
            if (candidat / "MQL5" / "Experts").is_dir():
                trouves.append(candidat)
    return trouves


def installer_ea(destination: Optional[str] = None) -> int:
    titre("INSTALLATION DE L'EA DANS METATRADER 5")

    if not EA_SOURCE.exists():
        echec(f"Fichier introuvable : {EA_SOURCE}")
        return 1

    if destination:
        cibles = [Path(destination)]
    else:
        cibles = dossiers_metatrader()

    if not cibles:
        echec("Aucune installation de MetaTrader 5 détectée.")
        print()
        print("  Deux causes possibles :")
        print("    1. MetaTrader 5 n'a jamais été lancé sur cette machine.")
        print("       Lancez-le une fois, connectez-vous à votre compte, puis relancez.")
        print("    2. Il est installé ailleurs. Dans MetaTrader :")
        print("       Fichier > Ouvrir le dossier de données, copiez le chemin, puis :")
        print("         python scripts/mt5.py installer --dossier \"C:\\chemin\\colle\"")
        return 1

    if len(cibles) > 1:
        attention(f"{len(cibles)} terminaux MetaTrader détectés — l'EA est copié dans chacun.")

    for cible in cibles:
        experts = cible / "MQL5" / "Experts"
        experts.mkdir(parents=True, exist_ok=True)
        shutil.copy2(EA_SOURCE, experts / EA_SOURCE.name)
        ok(f"QBotBridge.mq5 copié dans {experts}")

    print()
    print("  Il reste 3 manipulations À FAIRE DANS METATRADER — elles ne peuvent pas")
    print("  être automatisées depuis l'extérieur, MetaTrader ne le permet pas.")
    print()
    print("  1. COMPILER")
    print("     Dans MetaTrader : Ctrl+N (Navigateur) > clic droit sur « Expert Advisors »")
    print("     > Actualiser. QBotBridge apparaît. Double-cliquez dessus : MetaEditor")
    print("     s'ouvre. Appuyez sur F7. En bas doit s'afficher « 0 error(s) ».")
    print()
    print("  2. AUTORISER LA CONNEXION  (sans ça : erreur 5273, rien ne marche)")
    print("     Outils > Options > Expert Advisors")
    print("     > cocher « Autoriser WebRequest pour les URL listées »")
    print("     > ajouter dans la liste :  127.0.0.1")
    print()
    print("  3. LANCER LE SERVEUR PUIS POSER L'EA")
    print("     D'abord ici :        python scripts/mt5.py demarrer")
    print("     Puis dans MetaTrader : glissez QBotBridge sur un graphique EURUSD H1.")
    print("     Onglet « Expert » en bas : les échanges s'affichent en direct.")
    print()
    attention("InpDryRun reste sur true : l'EA calcule tout mais n'envoie AUCUN ordre.")
    print("        C'est volontaire. Laissez-le ainsi plusieurs semaines.")
    return 0


# =======================================================================================
# Test sans MetaTrader
# =======================================================================================
def tester(modele: Path, port: int) -> int:
    """Rejoue le dialogue exact de l'EA, depuis Python, sans MetaTrader.

    Intérêt : quand ça ne marche pas avec MetaTrader, ce test répond à la seule
    question qui compte — « le problème vient-il du bot ou de MetaTrader ? ». S'il
    passe, le bot est hors de cause et il faut chercher du côté des autorisations
    de l'EA.
    """
    titre("TEST DE LA CHAÎNE COMPLÈTE (sans MetaTrader)")

    if not modele.exists():
        echec(f"Aucun modèle entraîné dans {modele}")
        print()
        print("  Lancez d'abord l'analyse, qui entraîne et enregistre le modèle :")
        print("      python scripts/start.py")
        return 1
    ok(f"Modèle trouvé : {modele}")

    import numpy as np

    from qbot.config import LiveConfig
    from qbot.data import generate_synthetic_ohlcv
    from qbot.live import serve
    from qbot.live.server import SimpleClient
    from qbot.utils.logging import configure_logging
    import logging

    configure_logging(logging.WARNING)

    print("  Démarrage du serveur d'inférence…")
    serveur = serve(modele, LiveConfig(host="127.0.0.1", port=port, dry_run=True),
                    block=False, replay=True)
    temps_depart = time.time()
    try:
        with SimpleClient("127.0.0.1", port) as client:
            rep = client.request({"type": "ping"})
            assert rep.get("ok"), rep
            ok(f"Le serveur répond (ping/pong, protocole v{rep.get('version')})")

            info = client.request({"type": "info"})
            besoin = int(info.get("min_bars", 1200))
            ok(f"Le modèle réclame {besoin} barres par requête")

            # On fabrique un historique de la même forme que celui envoyé par l'EA.
            df = generate_synthetic_ohlcv(n=besoin + 200, freq="1h", seed=7)
            df = df.tail(besoin + 50)
            barres = [
                [int(t.timestamp()), float(r.open), float(r.high), float(r.low),
                 float(r.close), float(r.volume), float(getattr(r, "spread", 0.0))]
                for t, r in df.iterrows()
            ]
            print(f"  Envoi de {len(barres)} barres, comme le ferait l'EA…")

            t0 = time.time()
            rep = client.request({
                "type": "predict", "symbol": "EURUSD", "timeframe": "H1",
                "bars": barres, "equity": 1000.0, "balance": 1000.0,
                "current_exposure": 0.0, "magic": 770011,
            })
            ms = (time.time() - t0) * 1000.0

            if not rep.get("ok") and rep.get("status") == "error":
                echec(f"Le serveur a refusé la requête : {rep.get('error')}")
                return 1

            ok(f"Réponse reçue en {ms:.0f} ms")
            print()
            print(f"      exposition cible : {rep.get('target_exposure', 0.0):+.3f}"
                  "   (fraction du capital ; négatif = vente)")
            print(f"      état             : {rep.get('status')}")
            print(f"      confiance        : {rep.get('confidence', 0.0):.3f}")
            raisons = rep.get("reasons") or []
            if raisons:
                print(f"      motifs           : {', '.join(str(r) for r in raisons)}")
            print()
            if abs(float(rep.get("target_exposure", 0.0))) < 1e-9:
                print("      Une exposition nulle n'est PAS une panne : c'est une décision.")
                print("      Le modèle a évalué la situation et choisi de rester à plat.")

            # Deuxième requête, en se déclarant déjà exposé : elle montre le verrou
            # dry-run refuser un renforcement là où la première n'avait rien à refuser.
            rep2 = client.request({
                "type": "predict", "symbol": "EURUSD", "timeframe": "H1",
                "bars": barres, "equity": 1000.0, "balance": 1000.0,
                "current_exposure": 0.5, "bars_in_position": 3, "magic": 770011,
            })
            exp2 = float(rep2.get("target_exposure", 0.0))
            ok(f"Second appel, en se déclarant exposé à 50 % : cible {exp2:+.3f}")
            if exp2 <= 0.5 + 1e-9:
                ok("Le verrou dry-run tient : aucun renforcement autorisé.")
            else:
                echec("ANOMALIE : le dry-run a laissé passer un renforcement.")
                return 1

            statut = client.request({"type": "status"})
            if statut.get("ok", True):
                ok("La supervision répond (métriques, dérive, alertes)")
    except (ConnectionRefusedError, OSError) as exc:
        echec(f"Impossible de joindre le serveur : {exc}")
        return 1
    finally:
        serveur.shutdown()
        serveur.server_close()

    print()
    print(f"  Chaîne validée en {time.time() - temps_depart:.1f} s.")
    print("  Tout ce que MetaTrader aura à faire, Python vient de le faire.")
    print("  Si l'EA échoue malgré ça, le problème est dans MetaTrader :")
    print("  autorisations (étape 2 de l'installation) ou compilation.")
    return 0


# =======================================================================================
# Serveur
# =======================================================================================
def demarrer(modele: Path, port: int, reel: bool, rejeu: bool) -> int:
    titre("SERVEUR D'INFÉRENCE")

    if not modele.exists():
        echec(f"Aucun modèle entraîné dans {modele}")
        print("\n  Lancez d'abord :  python scripts/start.py")
        return 1

    # Un port déjà pris signifie presque toujours un serveur oublié dans une autre
    # fenêtre. Deux serveurs sur le même modèle, c'est deux décisions concurrentes
    # pour un seul compte.
    sonde = socket.socket()
    try:
        sonde.bind(("127.0.0.1", port))
    except OSError:
        echec(f"Le port {port} est déjà utilisé.")
        print("\n  Un serveur tourne probablement déjà dans une autre fenêtre.")
        print("  Fermez-la, ou choisissez un autre port :")
        print(f"      python scripts/mt5.py demarrer --port {port + 1}")
        print("  (dans ce cas, réglez aussi InpPort sur cette valeur dans l'EA)")
        return 1
    finally:
        sonde.close()

    import logging

    from qbot.config import LiveConfig
    from qbot.live import serve
    from qbot.utils.logging import configure_logging

    configure_logging(logging.INFO)

    if reel:
        print()
        attention("MODE RÉEL : le serveur autorisera de VRAIES ouvertures de position.")
        attention("Rappel des mesures faites sur ce modèle : 1 000 € sur un an ont donné")
        attention("795,98 € sur données EURUSD réelles, intervalle de confiance")
        attention("entièrement négatif. Ce n'est pas de la malchance, c'est le résultat.")
        print()
        try:
            reponse = input("  Tapez exactement OUI JE CONFIRME pour continuer : ").strip()
        except EOFError:
            # Pas de terminal : script lancé par double-clic, tâche planifiée, tuyau.
            # Le seul défaut acceptable ici est de NE PAS armer le trading réel.
            reponse = ""
            print("\n  (pas d'entrée interactive disponible)")
        if reponse != "OUI JE CONFIRME":
            print("  Annulé. Le serveur reste en dry-run.")
            reel = False

    mode = "*** RÉEL ***" if reel else "DRY-RUN (aucune ouverture autorisée)"
    print()
    ok(f"Modèle    : {modele}")
    ok(f"Écoute    : 127.0.0.1:{port}")
    ok(f"Mode      : {mode}")
    print()
    print("  Laissez cette fenêtre OUVERTE. Chaque bougie, MetaTrader viendra")
    print("  demander quoi faire et la réponse s'affichera ici.")
    print("  Pour arrêter : Ctrl+C (l'EA repassera à plat tout seul).")
    print()

    cfg = LiveConfig(host="127.0.0.1", port=port, model_path=str(modele), dry_run=not reel)
    serve(modele, cfg, block=True, replay=rejeu)
    return 0


# =======================================================================================
def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["installer", "tester", "demarrer"],
                   help="installer : copie l'EA | tester : vérifie sans MetaTrader | "
                        "demarrer : lance le serveur")
    p.add_argument("--modele", type=str, default=str(MODELE_DEFAUT))
    p.add_argument("--port", type=int, default=PORT_DEFAUT)
    p.add_argument("--dossier", type=str, default=None,
                   help="Chemin du dossier de données MetaTrader, si la détection échoue")
    p.add_argument("--reel", action="store_true",
                   help="Désactive le dry-run côté serveur (TRADING RÉEL)")
    p.add_argument("--rejeu", action="store_true",
                   help="Neutralise le contrôle de fraîcheur, pour tester sur barres passées")
    args = p.parse_args()

    modele = Path(args.modele)
    if args.action == "installer":
        return installer_ea(args.dossier)
    if args.action == "tester":
        return tester(modele, args.port)
    return demarrer(modele, args.port, args.reel, args.rejeu)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Connecter le bot à MetaTrader 5, en trois commandes.

    python scripts/mt5.py installer    # copie l'EA dans MetaTrader et explique la suite
    python scripts/mt5.py tester       # prouve que la chaîne marche, SANS MetaTrader
    python scripts/mt5.py demarrer     # lance le serveur auquel MetaTrader se connecte
    python scripts/mt5.py bilan        # où en est le bot, en direct
    python scripts/mt5.py verdict --rapport histo.html   # que valent vos résultats ?

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

Sécurité : trois verrous, tous fermés par défaut.
  - Côté MetaTrader : InpDryRun = true   → l'EA n'envoie aucun ordre.
  - Côté Python     : pas de --ordres    → le serveur refuse toute ouverture.
  - Côté compte     : pas de --argent-reel → même ordres armés, un compte RÉEL
                      reste bloqué ; l'EA transmet la nature du compte et le
                      serveur la vérifie à chaque décision.

Les deux premiers se lèvent ensemble pour faire tourner une DÉMO : c'est le
seul moyen de voir la chaîne produire de vraies écritures, avec de l'argent
fictif. Le troisième est ce qui sépare la démo de votre argent.
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


def _couleur(valeur: float) -> str:
    """Vert au-dessus de zéro, rouge en dessous — la seule lecture qui compte d'un coup d'œil."""
    return "\033[32m" if valeur > 0 else ("\033[31m" if valeur < 0 else "")


def _sens(position: float) -> str:
    if position > 1e-9:
        return "ACHAT"
    if position < -1e-9:
        return "VENTE"
    return "AUCUNE"


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


def dossier_experts(chemin: Path) -> Path:
    """Ramène n'importe quel chemin plausible au dossier MQL5/Experts.

    L'utilisateur arrive ici après « Fichier > Ouvrir le dossier de données », et
    colle ce que l'explorateur lui montre. Selon l'endroit où il s'est arrêté de
    naviguer, ce sera la racine du terminal, le dossier MQL5, ou déjà Experts.
    Exiger la racine reviendrait à créer MQL5/MQL5/Experts sans un mot d'erreur —
    l'EA serait copié, l'installeur dirait « ok », et MetaEditor ne verrait rien.
    """
    parties = [p.lower() for p in chemin.parts]
    if parties and parties[-1] == "experts":
        return chemin
    if parties and parties[-1] == "mql5":
        return chemin / "Experts"
    return chemin / "MQL5" / "Experts"


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
        experts = dossier_experts(cible)
        experts.mkdir(parents=True, exist_ok=True)
        shutil.copy2(EA_SOURCE, experts / EA_SOURCE.name)
        ok(f"QBotBridge.mq5 copié dans {experts}")

    print()
    print("  Il reste 3 manipulations À FAIRE DANS METATRADER — elles ne peuvent pas")
    print("  être automatisées depuis l'extérieur, MetaTrader ne le permet pas.")
    print()
    print("  1. COMPILER  — par MetaEditor, PAS par le Navigateur")
    print("     Le Navigateur de MetaTrader ne liste que les robots DÉJÀ compilés.")
    print("     QBotBridge est encore du code source : il n'y apparaît pas, et c'est")
    print("     normal. Il faut le compiler d'abord.")
    print()
    print("     Dans MetaTrader, cliquez sur le bouton « IDE » de la barre d'outils")
    print("     (ou touche F4) : MetaEditor s'ouvre.")
    print("     Dans SON navigateur à gauche : dossier Experts > QBotBridge.mq5.")
    print("     Double-cliquez, puis F7. En bas : « 0 error(s), 0 warning(s) ».")
    print("     QBotBridge apparaît alors dans le Navigateur de MetaTrader.")
    print()
    print("  2. AUTORISER LA CONNEXION  (sans ça : erreur 5273, rien ne marche)")
    print("     Outils > Options > Expert Advisors")
    print("     > cocher « Autoriser WebRequest pour les URL listées »")
    print("     > ajouter dans la liste :  127.0.0.1")
    print()
    print("  3. LANCER LE SERVEUR PUIS POSER L'EA")
    print("     D'abord ici :        python scripts/mt5.py demarrer --ordres")
    print("     Puis dans MetaTrader : glissez QBotBridge sur un graphique EURUSD H1.")
    print("     (le dossier s'appelle « Expert Consultants », « Conseillers Experts »")
    print("      ou « Expert Advisors » selon la version)")
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
def demarrer(modele: Path, port: int, ordres: bool, argent_reel: bool, rejeu: bool) -> int:
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

    # --argent-reel n'a aucun sens sans --ordres : le signaler plutôt que de laisser
    # croire que le trading est armé.
    if argent_reel and not ordres:
        attention("--argent-reel ignoré : il faut aussi --ordres pour armer le trading.")
        argent_reel = False

    if argent_reel:
        print()
        attention("ARGENT RÉEL AUTORISÉ : si le terminal au bout du fil est un compte")
        attention("réel, le serveur le laissera ouvrir des positions.")
        attention("Rappel des mesures faites sur ce modèle : 1 000 € sur un an ont donné")
        attention("795,98 € sur données EURUSD réelles, intervalle de confiance")
        attention("entièrement négatif. Ce n'est pas de la malchance, c'est le résultat.")
        print()
        try:
            reponse = input("  Tapez exactement OUI JE CONFIRME pour continuer : ").strip()
        except EOFError:
            # Pas de terminal : script lancé par double-clic, tâche planifiée, tuyau.
            # Le seul défaut acceptable ici est de NE PAS autoriser l'argent réel.
            reponse = ""
            print("\n  (pas d'entrée interactive disponible)")
        if reponse != "OUI JE CONFIRME":
            print("  Annulé. Les comptes réels resteront bloqués.")
            argent_reel = False

    if not ordres:
        mode = "OBSERVATION — aucun ordre, même en démo"
    elif argent_reel:
        mode = "ORDRES ARMÉS — comptes démo ET réels"
    else:
        mode = "ORDRES ARMÉS — démo uniquement, comptes réels bloqués"

    print()
    ok(f"Modèle    : {modele}")
    ok(f"Écoute    : 127.0.0.1:{port}")
    ok(f"Mode      : {mode}")
    if not ordres:
        print()
        print("      Le bot calcule tout et affiche ses décisions, mais MetaTrader")
        print("      n'ouvrira rien : votre historique restera vide.")
        print("      Pour que la démo passe de vrais ordres fictifs :")
        print("        1. dans l'EA, mettre InpDryRun sur false ;")
        print("        2. relancer ici avec  --ordres")
    print()
    print("  Laissez cette fenêtre OUVERTE. Chaque bougie, MetaTrader viendra")
    print("  demander quoi faire et la réponse s'affichera ici.")
    print("  Pour arrêter : Ctrl+C (l'EA repassera à plat tout seul).")
    print()

    cfg = LiveConfig(host="127.0.0.1", port=port, model_path=str(modele),
                     dry_run=not ordres)
    serve(modele, cfg, block=True, replay=rejeu, allow_real_account=argent_reel)
    return 0


# =======================================================================================
# Verdict sur une période de démo
# =======================================================================================
def verdict(rapport: str, capital: float) -> int:
    """Lit l'historique exporté depuis MetaTrader et dit ce qu'il vaut vraiment."""
    titre("VERDICT SUR VOTRE PÉRIODE DE DÉMO")

    from qbot.live.rapport_mt5 import juger, lire_rapport

    try:
        hist = lire_rapport(rapport, capital_initial=capital)
    except (FileNotFoundError, ValueError) as exc:
        echec(str(exc))
        return 1

    v = juger(hist)
    ok(f"{v.n} transactions lues dans {Path(rapport).name}")
    print()
    print(f"      résultat total        : {v.total:+.2f}")
    print(f"      par transaction       : {v.moyenne:+.2f} en moyenne")
    print(f"      transactions gagnantes: {v.taux_reussite:.1%}")
    print()

    couleur = {"SIGNIFICATIF": "\033[32m", "PERDANT": "\033[31m"}.get(v.conclusion, "\033[33m")
    print(f"  {couleur}\033[1m>>> {v.conclusion}\033[0m")
    print()
    for ligne in _plier(v.explication, 68):
        print(f"      {ligne}")
    print()
    print("      Et si ces mêmes transactions s'étaient présentées dans un autre")
    print("      ordre, ou avec un tirage un peu différent ?")
    print(f"        5e centile {v.ic_bas:+.2f}      95e centile {v.ic_haut:+.2f}")
    if v.ic_bas < 0 < v.ic_haut:
        print("        L'intervalle contient zéro : le gain n'est pas établi.")
    elif v.ic_bas > 0:
        print("        L'intervalle est entièrement positif.")
    else:
        print("        L'intervalle est entièrement négatif.")

    print()
    print("  " + "─" * (LARGEUR - 2))
    if v.conclusion == "SIGNIFICATIF":
        print("  Ce résultat tient statistiquement sur CET échantillon. C'est une")
        print("  condition nécessaire pour envisager du réel — pas une garantie :")
        print("  le marché de la période suivante n'est pas tenu de ressembler")
        print("  à celle-ci.")
    else:
        print("  Passer au réel sur cette base, c'est parier sur un chiffre que le")
        print("  test ne soutient pas. Le manque n'est pas de la patience : c'est")
        print("  du nombre de transactions.")
    return 0


def _plier(texte: str, largeur: int) -> List[str]:
    """Coupe un paragraphe en lignes sans casser les mots."""
    mots, lignes, courante = texte.split(), [], ""
    for mot in mots:
        if len(courante) + len(mot) + 1 > largeur:
            lignes.append(courante)
            courante = mot
        else:
            courante = f"{courante} {mot}".strip()
    if courante:
        lignes.append(courante)
    return lignes


# =======================================================================================
# Bilan en direct
# =======================================================================================
def bilan(port: int) -> int:
    """Interroge le serveur EN COURS D'EXÉCUTION et affiche où en est le bot.

    Le serveur tient déjà tout : équité rapportée par l'EA à chaque barre, drawdown,
    nombre de décisions, transactions, latence, alertes. Rien n'est recalculé ici — on
    demande et on met en forme. C'est une seconde fenêtre à ouvrir quand on veut, sans
    toucher à celle qui fait tourner le bot.
    """
    titre("OÙ EN EST LE BOT")

    from qbot.live.server import SimpleClient

    try:
        with SimpleClient("127.0.0.1", port, timeout=10.0) as client:
            snap = client.request({"type": "status"})
    except (ConnectionRefusedError, OSError):
        echec(f"Aucun serveur ne répond sur 127.0.0.1:{port}.")
        print("\n  Le bot n'est pas en train de tourner. Dans une autre fenêtre :")
        print("      python scripts/mt5.py demarrer --ordres")
        return 1

    if not snap.get("ok", False):
        echec(snap.get("error", "le serveur n'a pas de supervision active"))
        return 1

    n = int(snap.get("n_bars", 0) or 0)
    if n == 0:
        attention("Le serveur tourne, mais n'a encore reçu aucune barre.")
        print("\n  MetaTrader ne s'est pas connecté, ou aucune bougie n'a été")
        print("  clôturée depuis le démarrage. En H1, comptez jusqu'à une heure.")
        return 0

    equite = float(snap.get("equity", 0.0) or 0.0)
    rendement = float(snap.get("total_return", 0.0) or 0.0)
    dd = float(snap.get("drawdown", 0.0) or 0.0)
    n_trades = int(snap.get("n_trades", 0) or 0)
    expo = float(snap.get("net_exposure", 0.0) or 0.0)
    plat = float(snap.get("flat_rate", 0.0) or 0.0)
    conf = float(snap.get("mean_confidence", float("nan")) or 0.0)
    p99 = float(snap.get("p99_latency_ms", 0.0) or 0.0)

    depart = equite / (1.0 + rendement) if rendement > -1.0 else equite
    gain = equite - depart
    c = _couleur(gain)

    print(f"  Observé depuis      : {snap.get('first_ts', '?')}")
    print(f"  Dernière décision   : {snap.get('last_ts', '?')}")
    print(f"  Décisions prises    : {n:,}  ({n / 24.0:.1f} jours de marché)")
    print()
    print(f"  Capital de départ   : {depart:>14,.2f}")
    print(f"  Capital actuel      : {c}{equite:>14,.2f}\033[0m")
    print(f"  Résultat            : {c}{gain:>+14,.2f}   ({rendement:+.2%})\033[0m")
    print(f"  Pire recul          : {dd:>14.2%}")
    print()
    print(f"  Transactions        : {n_trades}")
    print(f"  Position actuelle   : {_sens(expo)} {abs(expo):.1%} du capital")
    print(f"  Temps passé à plat  : {plat:.0%} des barres")
    print(f"  Confiance moyenne   : {conf:.2f}")
    print(f"  Latence p99         : {p99:.0f} ms")

    alertes = snap.get("alerts") or {}
    n_alertes = int(alertes.get("count", 0) or 0)
    if n_alertes:
        pire = alertes.get("worst", "?")
        print(f"\n  Alertes             : {n_alertes} — la plus grave : {pire}")

    print()
    print("  " + "─" * (LARGEUR - 4))
    # Le nombre de transactions decide de ce qu'on peut lire dans ces chiffres. Le
    # rappeler ICI, a cote du resultat, evite de laisser un gain de trois jours passer
    # pour une preuve.
    if n_trades < 30:
        print(f"  {n_trades} transaction(s) : bien trop peu pour conclure quoi que ce soit.")
        print("  À ce stade le résultat, bon ou mauvais, est du bruit. Le chiffre")
        print("  ci-dessus dit que la mécanique tourne, rien de plus.")
    else:
        print(f"  {n_trades} transactions. Pour savoir si ce résultat veut dire quelque")
        print("  chose, exportez l'historique depuis MetaTrader et lancez :")
        print("      python scripts/mt5.py verdict --rapport chemin\\du\\rapport.html")
    return 0


# =======================================================================================
def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action",
                   choices=["installer", "tester", "demarrer", "bilan", "verdict"],
                   help="installer : copie l'EA | tester : vérifie sans MetaTrader | "
                        "demarrer : lance le serveur | bilan : où en est le bot, "
                        "en direct | verdict : juge une période de démo")
    p.add_argument("--rapport", type=str, default=None,
                   help="Rapport d'historique exporté depuis MetaTrader (HTML ou CSV)")
    p.add_argument("--capital", type=float, default=0.0,
                   help="Capital de départ du compte démo")
    p.add_argument("--modele", type=str, default=str(MODELE_DEFAUT))
    p.add_argument("--port", type=int, default=PORT_DEFAUT)
    p.add_argument("--dossier", type=str, default=None,
                   help="Chemin du dossier de données MetaTrader, si la détection échoue")
    p.add_argument("--ordres", "--reel", action="store_true", dest="ordres",
                   help="Autorise le passage d'ordres. Sur un compte démo c'est de "
                        "l'argent fictif ; les comptes réels restent bloqués sans "
                        "--argent-reel.")
    p.add_argument("--argent-reel", action="store_true", dest="argent_reel",
                   help="Lève le blocage des comptes RÉELS. Demande une confirmation.")
    p.add_argument("--rejeu", action="store_true",
                   help="Neutralise le contrôle de fraîcheur, pour tester sur barres passées")
    args = p.parse_args()

    modele = Path(args.modele)
    if args.action == "installer":
        return installer_ea(args.dossier)
    if args.action == "tester":
        return tester(modele, args.port)
    if args.action == "bilan":
        return bilan(args.port)
    if args.action == "verdict":
        if not args.rapport:
            p.error("verdict exige --rapport (dans MetaTrader : Historique > clic droit "
                    "> Rapport > HTML)")
        return verdict(args.rapport, args.capital)
    return demarrer(modele, args.port, args.ordres, args.argent_reel, args.rejeu)


if __name__ == "__main__":
    raise SystemExit(main())

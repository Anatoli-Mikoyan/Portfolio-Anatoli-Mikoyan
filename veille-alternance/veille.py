#!/usr/bin/env python3
"""Veille quotidienne d'offres d'alternance, calee sur le CV d'Anatoli Mikoyan.

    python3 veille.py                 # collecte reelle + e-mail + tableau de bord
    python3 veille.py --demo          # jeu d'essai hors ligne, pour voir le rendu
    python3 veille.py --sonde         # teste juste l'acces aux API et aux cles
    python3 veille.py --jours 3 --max 15
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

RACINE = Path(__file__).resolve().parent
sys.path.insert(0, str(RACINE))

import historique  # noqa: E402
import rendu  # noqa: E402
import sources  # noqa: E402
from modele import Offre, dedoublonner, depuis_dict  # noqa: E402
from scoring import charger_profil, classer  # noqa: E402

journal = logging.getLogger("veille")


def _configurer_journal(verbeux: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbeux else logging.INFO,
        format="%(levelname)-7s %(name)s — %(message)s",
        stream=sys.stdout,
    )


def _offres_de_demonstration() -> list[Offre]:
    """Jeu d'essai : dates relatives pour rester realiste a toute date."""
    brut = (RACINE / "fixtures" / "offres-demo.json").read_text(encoding="utf-8")
    brut = re.sub(
        r"__H(\d+)__",
        lambda m: (datetime.now(timezone.utc) - timedelta(hours=int(m.group(1)))).isoformat(),
        brut,
    )
    return [depuis_dict(o) for o in json.loads(brut)]


def sonder() -> int:
    """Verifie ce qui est joignable et configure. A lancer avant de suspecter un bug."""
    import os

    import requests

    print("\n--- Cles d'API presentes ---")
    attendues = {
        "FT_CLIENT_ID": "France Travail",
        "FT_CLIENT_SECRET": "France Travail",
        "LBA_API_KEY": "La Bonne Alternance",
        "ADZUNA_APP_ID": "Adzuna",
        "ADZUNA_APP_KEY": "Adzuna",
        "SMTP_USER": "envoi e-mail",
        "SMTP_PASS": "envoi e-mail",
    }
    for variable, usage in attendues.items():
        etat = "definie" if os.getenv(variable, "").strip() else "ABSENTE"
        print(f"  {variable:20s} {etat:9s} ({usage})")

    print("\n--- Joignabilite reseau ---")
    cibles = {
        "France Travail (auth)": sources.URL_TOKEN_FT,
        "France Travail (API)": sources.URL_RECHERCHE_FT,
        "La Bonne Alternance": sources.URL_LBA,
        "Adzuna": sources.URL_ADZUNA.format(page=1),
    }
    joignables = 0
    for nom, url in cibles.items():
        try:
            reponse = requests.get(url, timeout=20, headers=sources.ENTETE_UA)
            # 401/403/400 = le serveur repond, donc le reseau passe : c'est ce qu'on teste.
            print(f"  {nom:24s} HTTP {reponse.status_code} — joignable")
            joignables += 1
        except requests.RequestException as erreur:
            print(f"  {nom:24s} INJOIGNABLE — {type(erreur).__name__}: {str(erreur)[:90]}")
    print(f"\n{joignables}/{len(cibles)} endpoints joignables.\n")
    return 0 if joignables else 1


def executer(arguments: argparse.Namespace) -> int:
    profil = charger_profil(arguments.profil)
    jours = arguments.jours or profil["recherche"]["jours_recence_max"]
    maximum = arguments.max or profil["recherche"]["nb_offres_digest"]
    dossier = Path(arguments.sortie) if arguments.sortie else RACINE

    # 1. Collecte
    if arguments.demo:
        journal.info("Mode demonstration : jeu d'essai local, aucun appel reseau")
        brutes = _offres_de_demonstration()
        sources_actives = 1
        rapport_sources = ["demo : jeu d'essai"]
    else:
        resultats = sources.collecter(profil, jours, arguments.sources)
        brutes = [offre for r in resultats for offre in r.offres]
        sources_actives = sum(1 for r in resultats if r.offres)
        rapport_sources = []
        for r in resultats:
            if not r.configuree:
                rapport_sources.append(f"{r.nom} : non configuree ({r.erreurs[0]})")
            elif r.erreurs:
                rapport_sources.append(f"{r.nom} : {len(r.offres)} offres, {r.erreurs[0]}")
            else:
                rapport_sources.append(f"{r.nom} : {len(r.offres)} offres")
        for ligne in rapport_sources:
            journal.info("  %s", ligne)

        if not any(r.configuree for r in resultats):
            journal.error(
                "Aucune source configuree. Renseigne au moins FT_CLIENT_ID / FT_CLIENT_SECRET "
                "(voir README.md), ou lance avec --demo pour voir le rendu."
            )
            return 2

    total_collecte = len(brutes)
    journal.info("%d annonces collectees", total_collecte)

    # 2. Normalisation : recence, doublons, pertinence
    recentes = sources.filtrer_recentes(brutes, jours)
    uniques = dedoublonner(recentes)
    pertinentes = classer(uniques, profil)
    journal.info(
        "%d recentes -> %d uniques -> %d pertinentes", len(recentes), len(uniques), len(pertinentes)
    )

    # 3. Deduplication inter-jours : le coeur d'une veille utile
    vues = {} if arguments.ignorer_historique else historique.charger()
    inedites = historique.nouvelles(pertinentes, vues)
    retenues = inedites[:maximum]
    journal.info("%d inedites, %d retenues pour le digest", len(inedites), len(retenues))

    statistiques = {
        "total_collecte": total_collecte,
        "sources_actives": sources_actives,
        "deja_vues": len(pertinentes) - len(inedites),
        "pertinentes": len(pertinentes),
        "detail_sources": rapport_sources,
        "genere_le": datetime.now(timezone.utc).isoformat(),
        "demo": bool(arguments.demo),
    }

    # 4. Restitution
    dossier.mkdir(parents=True, exist_ok=True)
    (dossier / "index.html").write_text(rendu.en_html(retenues, statistiques), encoding="utf-8")
    (dossier / "digest.md").write_text(rendu.en_markdown(retenues, statistiques), encoding="utf-8")
    (dossier / "etat").mkdir(exist_ok=True)
    (dossier / "etat" / "dernier-digest.json").write_text(
        json.dumps(
            {"statistiques": statistiques, "offres": [o.en_dict() for o in retenues]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print()
    print(rendu.en_markdown(retenues, statistiques))

    if arguments.email:
        destinataire = arguments.destinataire or profil["identite"]["email"]
        rendu.envoyer_email(retenues, statistiques, destinataire)

    # 5. Memoire : on n'enregistre qu'apres une restitution reussie, et seulement
    # les offres reellement envoyees. Celles coupees par le plafond doivent
    # ressortir dans le digest du lendemain, pas disparaitre.
    if not arguments.ignorer_historique and not arguments.demo:
        historique.enregistrer(retenues, vues)

    return 0


def analyser_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    analyseur = argparse.ArgumentParser(
        description="Veille quotidienne d'offres d'alternance calee sur un CV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    analyseur.add_argument("--jours", type=int, help="fenetre de recence (defaut : profil.json)")
    analyseur.add_argument("--max", type=int, help="nombre d'offres dans le digest")
    analyseur.add_argument(
        "--sources",
        type=lambda v: [s.strip() for s in v.split(",") if s.strip()],
        help=f"sous-ensemble parmi {', '.join(sources.TOUTES_LES_SOURCES)}",
    )
    analyseur.add_argument("--profil", help="chemin d'un profil.json alternatif")
    analyseur.add_argument("--sortie", help="dossier de sortie (defaut : ce dossier)")
    analyseur.add_argument("--destinataire", help="adresse e-mail (defaut : celle du profil)")
    analyseur.add_argument("--demo", action="store_true", help="jeu d'essai hors ligne")
    analyseur.add_argument("--sonde", action="store_true", help="diagnostic cles + reseau")
    analyseur.add_argument("--email", action="store_true", help="envoyer le digest par e-mail")
    analyseur.add_argument(
        "--ignorer-historique",
        action="store_true",
        help="ne pas filtrer les offres deja vues (utile pour tester)",
    )
    analyseur.add_argument("-v", "--verbeux", action="store_true")
    return analyseur.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = analyser_arguments(argv)
    _configurer_journal(arguments.verbeux)
    if arguments.sonde:
        return sonder()
    return executer(arguments)


if __name__ == "__main__":
    raise SystemExit(main())

"""Classement des offres selon le CV et le projet de formation d'Anatoli.

Le score n'a pas de sens dans l'absolu : il sert uniquement a trier, et a
couper la queue des annonces hors sujet via `seuil_score_minimum`.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from modele import Offre, sans_accents

RACINE = Path(__file__).resolve().parent


def charger_profil(chemin: Path | str | None = None) -> dict:
    fichier = Path(chemin) if chemin else RACINE / "profil.json"
    return json.loads(fichier.read_text(encoding="utf-8"))


# Radicaux volontairement compares en sous-chaine : "apprenti" doit attraper
# "apprentissage" comme "apprenti(e)", et "alternan" couvre alternance/alternant.
_MOTS_ALTERNANCE = ("alternan", "apprenti", "professionnalisation")


def est_alternance(offre: Offre) -> bool:
    """Garde-fou : meme quand le filtre de l'API derape, on verifie nous-memes."""
    cible = sans_accents(f"{offre.titre} {offre.contrat} {offre.description[:1500]}")
    return any(mot in cible for mot in _MOTS_ALTERNANCE)


@lru_cache(maxsize=2048)
def _motif(terme: str) -> re.Pattern[str]:
    """Compile un terme en motif borne par des limites de mots.

    Sans ces bornes, "ia" matche "commerc-ia-l" et "git" matche "di-git-al" :
    le score se retrouve pollue par des offres hors sujet. Une etoile libere
    volontairement une borne : "comptab*" attrape comptable et comptabilite,
    "*school" attrape Kaischool et Webschool.
    """
    terme = terme.strip()
    suffixe_libre = terme.startswith("*")
    prefixe_libre = terme.endswith("*")
    noyau = re.escape(terme.strip("*").strip())
    borne_gauche = "" if suffixe_libre else r"(?<![a-z0-9])"
    borne_droite = "" if prefixe_libre else r"(?![a-z0-9])"
    return re.compile(rf"{borne_gauche}{noyau}{borne_droite}")


def _compte_termes(texte: str, termes: list[str]) -> list[str]:
    """Renvoie les termes de la liste reellement presents dans le texte."""
    trouves = []
    for terme in termes:
        propre = terme.strip().strip("*").strip()
        if propre and _motif(terme).search(texte):
            trouves.append(propre)
    return trouves


def scorer(offre: Offre, profil: dict) -> Offre:
    """Attribue un score et les raisons lisibles qui l'expliquent."""
    titre = sans_accents(offre.titre)
    complet = offre.texte_complet
    score = 0
    raisons: list[str] = []

    # --- metier vise : le signal le plus fort
    metiers_trouves = _compte_termes(titre, profil["metiers_vises"])
    if metiers_trouves:
        score += 30
        raisons.append(f"poste vise : {metiers_trouves[0]}")
    else:
        metiers_description = _compte_termes(complet, profil["metiers_vises"])
        if metiers_description:
            score += 10
            raisons.append(f"metier cite : {metiers_description[0]}")

    # --- competences, ponderees par famille
    for famille, bloc in profil["competences"].items():
        trouves = _compte_termes(complet, bloc["termes"])
        if trouves:
            # On plafonne a 3 termes par famille : 10 fois "python" ne vaut pas 10 fois plus.
            score += bloc["poids"] * min(len(trouves), 3)
            raisons.append(f"{famille} : {', '.join(t.strip() for t in trouves[:3])}")

    # --- contrat en alternance affiche clairement
    if any(mot in titre for mot in _MOTS_ALTERNANCE):
        score += profil["bonus"]["alternance_dans_titre"]

    # --- rentree compatible
    if any(terme in complet for terme in profil["bonus"]["termes_rentree"]):
        score += profil["bonus"]["rentree_2026"]
        raisons.append("rentree 2026 mentionnee")

    # --- fraicheur : c'est une veille, l'anciennete compte
    heures = offre.age_heures
    if heures is not None:
        if heures <= 48:
            score += profil["bonus"]["offre_moins_48h"]
        elif heures <= 72:
            score += profil["bonus"]["offre_moins_72h"]
        elif heures <= 24 * 7:
            score += profil["bonus"]["offre_moins_7j"]

    # --- geographie
    geo = profil["geographie"]
    lieu = sans_accents(offre.lieu)
    departement = re.search(r"\((\d{2})\)|^(\d{2})", offre.lieu or "")
    code_departement = (
        (departement.group(1) or departement.group(2)) if departement else offre.departement
    )
    if code_departement in geo["departements_prioritaires"]:
        score += geo["bonus_prioritaire"]
        raisons.append(f"Ile-de-France ({code_departement})")
    elif any(ville in lieu for ville in geo["villes_bonus"]):
        score += 4
    if offre.teletravail or "remote" in complet or "teletravail" in complet:
        score += geo["bonus_teletravail"]
        raisons.append("teletravail possible")

    # --- malus
    exclusions = profil["exclusions"]
    for bloc in ("malus_metier", "malus_organisme_formation", "malus_niveau"):
        regle = exclusions[bloc]
        cible = sans_accents(f"{offre.titre} {offre.entreprise}") if bloc != "malus_niveau" else complet
        trouves = _compte_termes(cible, regle["termes"])
        if trouves:
            score += regle["poids"]
            libelle = {
                "malus_metier": "hors domaine",
                "malus_organisme_formation": "organisme de formation, pas un employeur",
                "malus_niveau": "niveau demande superieur a BAC+3",
            }[bloc]
            raisons.append(f"- {libelle} ({trouves[0].strip()})")

    offre.score = score
    offre.raisons = raisons
    return offre


def classer(offres: list[Offre], profil: dict) -> list[Offre]:
    """Filtre les non-alternances et les hors sujet, puis trie du meilleur au moins bon."""
    retenues = []
    rejet_titre = [sans_accents(t) for t in profil["exclusions"].get("rejet_titre", [])]
    for offre in offres:
        if not est_alternance(offre):
            continue
        if any(motif in sans_accents(offre.titre) for motif in rejet_titre):
            continue
        scorer(offre, profil)
        if offre.score >= profil["seuil_score_minimum"]:
            retenues.append(offre)

    retenues.sort(
        key=lambda o: (o.score, o.date_publication.timestamp() if o.date_publication else 0),
        reverse=True,
    )
    return retenues

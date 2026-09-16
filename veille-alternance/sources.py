"""Connecteurs vers les sources d'annonces d'alternance.

Chaque source est independante : si une cle manque ou si une API tombe, on
journalise et on continue avec les autres. Une veille qui rend 12 offres vaut
mieux qu'une veille qui plante.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

import requests

from modele import Offre, parse_date, pioche, sans_accents

journal = logging.getLogger("veille.sources")

DELAI = 25  # secondes
ENTETE_UA = {"User-Agent": "veille-alternance/1.0 (+github.com/Anatoli-Mikoyan)"}

# Valeurs autorisees par France Travail pour "publieeDepuis".
_RECENCE_FT = (1, 3, 7, 14, 31)


class ResultatSource:
    """Ce qu'une source a produit, plus de quoi diagnostiquer si elle n'a rien produit."""

    def __init__(self, nom: str):
        self.nom = nom
        self.offres: list[Offre] = []
        self.erreurs: list[str] = []
        self.configuree = True

    def __repr__(self) -> str:
        etat = "OK" if not self.erreurs else f"{len(self.erreurs)} erreur(s)"
        return f"<{self.nom}: {len(self.offres)} offres, {etat}>"


def _recence_autorisee(jours: int) -> int:
    for palier in _RECENCE_FT:
        if jours <= palier:
            return palier
    return _RECENCE_FT[-1]


# --------------------------------------------------------------- France Travail

URL_TOKEN_FT = (
    "https://entreprise.francetravail.fr/connexion/oauth2/access_token?realm=%2Fpartenaire"
)
URL_RECHERCHE_FT = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"


def _jeton_france_travail(client_id: str, client_secret: str) -> str | None:
    """Recupere un jeton OAuth2. Le format du scope a change au fil des versions,
    on tente donc les deux formes connues avant d'abandonner."""
    scopes = (
        "api_offresdemploiv2 o2dsoffre",
        f"application_{client_id} api_offresdemploiv2 o2dsoffre",
    )
    for scope in scopes:
        try:
            reponse = requests.post(
                URL_TOKEN_FT,
                data={
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "scope": scope,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded", **ENTETE_UA},
                timeout=DELAI,
            )
        except requests.RequestException as erreur:
            journal.warning("France Travail — jeton injoignable : %s", erreur)
            return None
        if reponse.ok:
            return reponse.json().get("access_token")
        journal.debug("France Travail — scope refuse (%s) : %s", reponse.status_code, scope)
    journal.warning("France Travail — authentification refusee (verifier ID/secret)")
    return None


def _offre_depuis_france_travail(brut: dict) -> Offre:
    lieu = pioche(brut, "lieuTravail.libelle", defaut="")
    description = pioche(brut, "description", defaut="")
    entreprise = pioche(brut, "entreprise.nom", "nomEntreprise", defaut="")
    identifiant = pioche(brut, "id", defaut="")
    return Offre(
        source="France Travail",
        titre=pioche(brut, "intitule", defaut="(sans titre)"),
        entreprise=entreprise,
        lieu=lieu,
        departement=str(lieu)[:2] if str(lieu)[:2].isdigit() else "",
        contrat=pioche(brut, "natureContrat", "typeContratLibelle", defaut=""),
        date_publication=parse_date(pioche(brut, "dateCreation", "dateActualisation")),
        url=pioche(
            brut,
            "origineOffre.urlOrigine",
            defaut=f"https://candidat.francetravail.fr/offres/recherche/detail/{identifiant}",
        ),
        description=description,
        salaire=pioche(brut, "salaire.libelle", defaut=""),
        teletravail="teletravail" in sans_accents(description),
        identifiant_source=identifiant,
    )


def france_travail(profil: dict, jours: int) -> ResultatSource:
    """Source principale : le plus gros volume d'offres d'alternance en France."""
    resultat = ResultatSource("France Travail")
    client_id = os.getenv("FT_CLIENT_ID", "").strip()
    client_secret = os.getenv("FT_CLIENT_SECRET", "").strip()
    if not (client_id and client_secret):
        resultat.configuree = False
        resultat.erreurs.append("FT_CLIENT_ID / FT_CLIENT_SECRET absents")
        return resultat

    jeton = _jeton_france_travail(client_id, client_secret)
    if not jeton:
        resultat.erreurs.append("authentification OAuth2 echouee")
        return resultat

    entetes = {"Authorization": f"Bearer {jeton}", "Accept": "application/json", **ENTETE_UA}
    # E2 = contrat d'apprentissage, FS = contrat de professionnalisation.
    natures = os.getenv("FT_NATURES_CONTRAT", "E2,FS").split(",")

    for mots_cles in profil["mots_cles_recherche"]:
        for nature in natures:
            parametres = {
                "motsCles": mots_cles,
                "natureContrat": nature.strip(),
                "publieeDepuis": _recence_autorisee(jours),
                "range": "0-149",
                "sort": "1",  # tri par date de creation decroissante
            }
            if not profil["geographie"].get("france_entiere", True):
                parametres["commune"] = profil["identite"]["code_insee"]
                parametres["distance"] = profil["geographie"]["rayon_km_autour_domicile"]
            try:
                reponse = requests.get(
                    URL_RECHERCHE_FT, headers=entetes, params=parametres, timeout=DELAI
                )
            except requests.RequestException as erreur:
                resultat.erreurs.append(f"{mots_cles} : {erreur}")
                continue
            # 204 = aucun resultat, 206 = resultats partiels (les deux sont normaux).
            if reponse.status_code == 204:
                continue
            if reponse.status_code not in (200, 206):
                resultat.erreurs.append(f"{mots_cles} : HTTP {reponse.status_code}")
                continue
            try:
                lot = reponse.json().get("resultats", [])
            except ValueError:
                resultat.erreurs.append(f"{mots_cles} : reponse illisible")
                continue
            resultat.offres.extend(_offre_depuis_france_travail(o) for o in lot)

    journal.info("France Travail : %d offres brutes", len(resultat.offres))
    return resultat


# ---------------------------------------------------------- La Bonne Alternance

URL_LBA = "https://api.apprentissage.beta.gouv.fr/api/job/v1/search"


def _offre_depuis_lba(brut: dict) -> Offre:
    ville = pioche(
        brut,
        "workplace.location.address",
        "workplace.location.city",
        "place.city",
        "workplace.name",
        defaut="",
    )
    description = pioche(
        brut, "offer.description", "offer.access_conditions", "job.description", defaut=""
    )
    if isinstance(description, list):
        description = " ".join(str(x) for x in description)
    return Offre(
        source="La Bonne Alternance",
        titre=pioche(brut, "offer.title", "title", "job.title", defaut="(sans titre)"),
        entreprise=pioche(brut, "workplace.name", "company.name", defaut=""),
        lieu=str(ville),
        contrat="alternance",
        date_publication=parse_date(
            pioche(brut, "offer.publication.creation", "offer.publication.created_at", "created_at")
        ),
        url=pioche(brut, "apply.url", "url", "apply.phone", defaut=""),
        description=str(description),
        identifiant_source=str(pioche(brut, "identifier.id", "_id", "id", defaut="")),
    )


def la_bonne_alternance(profil: dict, jours: int) -> ResultatSource:
    """Source specialisee alternance (service public), y compris le marche cache."""
    resultat = ResultatSource("La Bonne Alternance")
    cle = os.getenv("LBA_API_KEY", "").strip()
    if not cle:
        resultat.configuree = False
        resultat.erreurs.append("LBA_API_KEY absente")
        return resultat

    entetes = {"Authorization": f"Bearer {cle}", "Accept": "application/json", **ENTETE_UA}
    parametres = {
        "latitude": profil["identite"]["latitude"],
        "longitude": profil["identite"]["longitude"],
        "radius": max(profil["geographie"]["rayon_km_autour_domicile"], 100),
        "romes": ",".join(profil["codes_rome"]),
        "target_diploma_level": "6",  # BAC+3/4
    }
    try:
        reponse = requests.get(URL_LBA, headers=entetes, params=parametres, timeout=DELAI)
    except requests.RequestException as erreur:
        resultat.erreurs.append(str(erreur))
        return resultat
    if not reponse.ok:
        resultat.erreurs.append(f"HTTP {reponse.status_code}")
        return resultat

    try:
        charge = reponse.json()
    except ValueError:
        resultat.erreurs.append("reponse illisible")
        return resultat

    lot = charge.get("jobs") if isinstance(charge, dict) else charge
    if not isinstance(lot, list):
        lot = pioche(charge, "matchas", "peJobs", "results", defaut=[]) or []
    resultat.offres.extend(_offre_depuis_lba(o) for o in lot if isinstance(o, dict))
    journal.info("La Bonne Alternance : %d offres brutes", len(resultat.offres))
    return resultat


# ----------------------------------------------------------------------- Adzuna

URL_ADZUNA = "https://api.adzuna.com/v1/api/jobs/fr/search/{page}"


def _offre_depuis_adzuna(brut: dict) -> Offre:
    return Offre(
        source="Adzuna",
        titre=pioche(brut, "title", defaut="(sans titre)"),
        entreprise=pioche(brut, "company.display_name", defaut=""),
        lieu=pioche(brut, "location.display_name", defaut=""),
        contrat=pioche(brut, "contract_type", defaut="alternance"),
        date_publication=parse_date(pioche(brut, "created")),
        url=pioche(brut, "redirect_url", defaut=""),
        description=pioche(brut, "description", defaut=""),
        salaire=str(pioche(brut, "salary_min", defaut="") or ""),
        identifiant_source=str(pioche(brut, "id", defaut="")),
    )


def adzuna(profil: dict, jours: int) -> ResultatSource:
    """Agregateur : ratisse les job boards prives que les API publiques ne voient pas."""
    resultat = ResultatSource("Adzuna")
    app_id = os.getenv("ADZUNA_APP_ID", "").strip()
    app_key = os.getenv("ADZUNA_APP_KEY", "").strip()
    if not (app_id and app_key):
        resultat.configuree = False
        resultat.erreurs.append("ADZUNA_APP_ID / ADZUNA_APP_KEY absents")
        return resultat

    for mots_cles in profil["mots_cles_recherche"][:6]:  # quota gratuit : on reste sobre
        parametres = {
            "app_id": app_id,
            "app_key": app_key,
            "what": mots_cles,
            "max_days_old": max(jours, 1),
            "results_per_page": 50,
            "content-type": "application/json",
        }
        if not profil["geographie"].get("france_entiere", True):
            parametres["where"] = profil["identite"]["ville"]
            parametres["distance"] = profil["geographie"]["rayon_km_autour_domicile"]
        try:
            reponse = requests.get(
                URL_ADZUNA.format(page=1), params=parametres, headers=ENTETE_UA, timeout=DELAI
            )
        except requests.RequestException as erreur:
            resultat.erreurs.append(f"{mots_cles} : {erreur}")
            continue
        if not reponse.ok:
            resultat.erreurs.append(f"{mots_cles} : HTTP {reponse.status_code}")
            continue
        try:
            lot = reponse.json().get("results", [])
        except ValueError:
            resultat.erreurs.append(f"{mots_cles} : reponse illisible")
            continue
        resultat.offres.extend(_offre_depuis_adzuna(o) for o in lot)

    journal.info("Adzuna : %d offres brutes", len(resultat.offres))
    return resultat


# -------------------------------------------------------------------- collecte

TOUTES_LES_SOURCES = {
    "france-travail": france_travail,
    "la-bonne-alternance": la_bonne_alternance,
    "adzuna": adzuna,
}


def collecter(profil: dict, jours: int, sources: list[str] | None = None) -> list[ResultatSource]:
    """Interroge toutes les sources demandees et renvoie leurs resultats bruts."""
    choisies = sources or list(TOUTES_LES_SOURCES)
    resultats = []
    for nom in choisies:
        fonction = TOUTES_LES_SOURCES.get(nom)
        if not fonction:
            journal.warning("Source inconnue ignoree : %s", nom)
            continue
        try:
            resultats.append(fonction(profil, jours))
        except Exception as erreur:  # une source cassee ne doit jamais tuer la veille
            journal.exception("Source %s en echec", nom)
            echec = ResultatSource(nom)
            echec.erreurs.append(f"exception : {erreur}")
            resultats.append(echec)
    return resultats


def filtrer_recentes(offres: list[Offre], jours: int) -> list[Offre]:
    """Ne garde que les annonces publiees recemment (le coeur de la demande)."""
    limite = datetime.now(timezone.utc) - timedelta(days=jours)
    gardees = []
    for offre in offres:
        # Sans date, on garde : mieux vaut une offre datee "inconnue" qu'un trou.
        if offre.date_publication is None or offre.date_publication >= limite:
            gardees.append(offre)
    return gardees

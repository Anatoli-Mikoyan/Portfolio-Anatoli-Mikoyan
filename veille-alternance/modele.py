"""Modele commun a toutes les sources d'annonces."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Iterable


def sans_accents(texte: str) -> str:
    """Minuscules sans accents, pour comparer des libelles heterogenes."""
    if not texte:
        return ""
    decompose = unicodedata.normalize("NFD", str(texte))
    plat = "".join(c for c in decompose if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", plat.lower()).strip()


_MENTIONS_GENRE = re.compile(r"\(?\b[hfmx](?:\s*[/-]\s*[hfmx])+\b\)?")


def normaliser_titre(titre: str) -> str:
    """Reduit un intitule a ses mots signifiants.

    La meme annonce republiee ou reprise par un agregateur change de
    ponctuation et de mention de genre : "Alternance - Developpeur IA (H/F)"
    et "Alternance Developpeur IA F/H" doivent donner la meme empreinte.
    """
    plat = _MENTIONS_GENRE.sub(" ", sans_accents(titre))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", plat).split())


def pioche(donnees: Any, *chemins: str, defaut: Any = None) -> Any:
    """Recupere la premiere valeur non vide parmi plusieurs chemins pointes.

    Les API interrogees ne sont pas figees (La Bonne Alternance a change de
    schema plusieurs fois). On essaie donc plusieurs emplacements plutot que
    de planter sur une cle absente.
    """
    for chemin in chemins:
        courant = donnees
        for morceau in chemin.split("."):
            if isinstance(courant, dict):
                courant = courant.get(morceau)
            elif isinstance(courant, list) and morceau.isdigit():
                index = int(morceau)
                courant = courant[index] if index < len(courant) else None
            else:
                courant = None
            if courant is None:
                break
        if courant not in (None, "", [], {}):
            return courant
    return defaut


_FORMATS_DATE = (
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%d/%m/%Y",
)


def parse_date(valeur: Any) -> datetime | None:
    """Convertit une date d'API en datetime UTC, quel que soit son format."""
    if valeur in (None, ""):
        return None
    if isinstance(valeur, datetime):
        return valeur if valeur.tzinfo else valeur.replace(tzinfo=timezone.utc)
    if isinstance(valeur, (int, float)):  # epoch en secondes ou millisecondes
        horodatage = valeur / 1000 if valeur > 1e11 else valeur
        try:
            return datetime.fromtimestamp(horodatage, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None

    texte = str(valeur).strip()
    try:
        date = datetime.fromisoformat(texte.replace("Z", "+00:00"))
        return date if date.tzinfo else date.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    for gabarit in _FORMATS_DATE:
        try:
            date = datetime.strptime(texte, gabarit)
            return date if date.tzinfo else date.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


@dataclass
class Offre:
    """Une annonce d'alternance, normalisee quelle que soit sa source."""

    source: str
    titre: str
    entreprise: str = ""
    lieu: str = ""
    departement: str = ""
    contrat: str = ""
    date_publication: datetime | None = None
    url: str = ""
    description: str = ""
    salaire: str = ""
    teletravail: bool = False
    identifiant_source: str = ""

    score: int = 0
    raisons: list[str] = field(default_factory=list)
    autres_sources: list[str] = field(default_factory=list)

    @property
    def cle(self) -> str:
        """Empreinte stable, utilisee pour ne jamais renvoyer deux fois la meme offre.

        On se base sur l'intitule + l'employeur + la ville plutot que sur l'id
        de la source : la meme annonce remonte souvent via plusieurs agregateurs
        avec des identifiants differents.
        """
        signature = "|".join(
            normaliser_titre(v)
            for v in (self.titre, self.entreprise, self.lieu.split("(")[0])
        )
        return hashlib.sha1(signature.encode("utf-8")).hexdigest()[:16]

    @property
    def age_heures(self) -> float | None:
        if not self.date_publication:
            return None
        delta = datetime.now(timezone.utc) - self.date_publication
        return delta.total_seconds() / 3600

    @property
    def age_texte(self) -> str:
        heures = self.age_heures
        if heures is None:
            return "date inconnue"
        if heures < 1:
            return "a l'instant"
        if heures < 24:
            return f"il y a {int(heures)} h"
        jours = int(heures // 24)
        return "hier" if jours == 1 else f"il y a {jours} jours"

    @property
    def texte_complet(self) -> str:
        return sans_accents(f"{self.titre} {self.entreprise} {self.contrat} {self.description}")

    def en_dict(self) -> dict:
        donnees = asdict(self)
        donnees["date_publication"] = (
            self.date_publication.isoformat() if self.date_publication else None
        )
        donnees["cle"] = self.cle
        donnees["age_texte"] = self.age_texte
        return donnees


def depuis_dict(donnees: dict) -> Offre:
    """Reconstruit une Offre depuis sa forme serialisee (historique JSON)."""
    connus = {champ for champ in Offre.__dataclass_fields__}
    filtres = {k: v for k, v in donnees.items() if k in connus}
    filtres["date_publication"] = parse_date(filtres.get("date_publication"))
    return Offre(**filtres)


def dedoublonner(offres: Iterable[Offre]) -> list[Offre]:
    """Garde une seule occurrence par empreinte, en preferant la mieux remplie."""
    retenues: dict[str, Offre] = {}
    for offre in offres:
        existante = retenues.get(offre.cle)
        if existante is None:
            retenues[offre.cle] = offre
            continue
        # On garde la version la mieux renseignee, sans perdre la trace des
        # autres sources qui publient la meme annonce.
        gardee, ecartee = (
            (offre, existante)
            if len(offre.description) > len(existante.description)
            else (existante, offre)
        )
        for source in [ecartee.source, *ecartee.autres_sources]:
            if source != gardee.source and source not in gardee.autres_sources:
                gardee.autres_sources.append(source)
        retenues[offre.cle] = gardee
    return list(retenues.values())

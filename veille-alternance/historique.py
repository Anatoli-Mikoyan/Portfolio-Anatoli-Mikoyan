"""Memoire de la veille : ne jamais renvoyer deux fois la meme annonce."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from modele import Offre

RACINE = Path(__file__).resolve().parent
FICHIER_HISTORIQUE = RACINE / "etat" / "historique.json"

# Au-dela, on oublie : une annonce reapparue trois mois plus tard merite d'etre revue.
RETENTION_JOURS = 90


def charger(chemin: Path | None = None) -> dict[str, str]:
    """Renvoie {cle_offre: date_iso_premiere_vue}."""
    fichier = chemin or FICHIER_HISTORIQUE
    if not fichier.exists():
        return {}
    try:
        donnees = json.loads(fichier.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    return donnees.get("vues", {}) if isinstance(donnees, dict) else {}


def _purger(vues: dict[str, str]) -> dict[str, str]:
    limite = datetime.now(timezone.utc) - timedelta(days=RETENTION_JOURS)
    gardees = {}
    for cle, vu_le in vues.items():
        try:
            if datetime.fromisoformat(vu_le) >= limite:
                gardees[cle] = vu_le
        except ValueError:
            continue
    return gardees


def nouvelles(offres: list[Offre], vues: dict[str, str]) -> list[Offre]:
    return [offre for offre in offres if offre.cle not in vues]


def enregistrer(offres: list[Offre], vues: dict[str, str], chemin: Path | None = None) -> None:
    fichier = chemin or FICHIER_HISTORIQUE
    fichier.parent.mkdir(parents=True, exist_ok=True)
    maintenant = datetime.now(timezone.utc).isoformat()
    for offre in offres:
        vues.setdefault(offre.cle, maintenant)
    charge = {
        "derniere_execution": maintenant,
        "nombre_offres_connues": len(vues),
        "vues": _purger(vues),
    }
    fichier.write_text(
        json.dumps(charge, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )

"""Tests hors ligne : la logique de tri doit etre verifiable sans appeler une API."""

from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

RACINE = Path(__file__).resolve().parent
sys.path.insert(0, str(RACINE))

import historique  # noqa: E402
from modele import Offre, dedoublonner, depuis_dict, parse_date, sans_accents  # noqa: E402
from scoring import charger_profil, classer, est_alternance, scorer  # noqa: E402
from sources import filtrer_recentes  # noqa: E402


def charger_fixtures() -> list[Offre]:
    """Les dates sont relatives (__H6__ = publiee il y a 6 h) pour que les tests
    restent valides quelle que soit la date d'execution."""
    brut = (RACINE / "fixtures" / "offres-demo.json").read_text(encoding="utf-8")

    def remplacer(correspondance: re.Match) -> str:
        heures = int(correspondance.group(1))
        date = datetime.now(timezone.utc) - timedelta(hours=heures)
        return date.isoformat()

    brut = re.sub(r"__H(\d+)__", remplacer, brut)
    return [depuis_dict(o) for o in json.loads(brut)]


class TestModele(unittest.TestCase):
    def test_normalisation_accents(self):
        self.assertEqual(sans_accents("Développeur  IA"), "developpeur ia")

    def test_dates_multiformats(self):
        for valeur in ("2026-09-15T08:00:00Z", "2026-09-15 08:00:00", "15/09/2026", "2026-09-15"):
            self.assertIsNotNone(parse_date(valeur), f"echec sur {valeur}")
        self.assertIsNone(parse_date(""))
        self.assertIsNone(parse_date("pas une date"))

    def test_cle_stable_entre_sources(self):
        """La meme annonce vue via deux agregateurs doit avoir la meme cle."""
        a = Offre(source="France Travail", titre="Alternance Dev IA", entreprise="Acme", lieu="Paris (75)")
        b = Offre(source="Adzuna", titre="alternance dev ia", entreprise="ACME", lieu="Paris")
        self.assertEqual(a.cle, b.cle)

    def test_source_secondaire_conservee(self):
        """Quand deux agregateurs publient la meme annonce, on garde la version
        la plus complete sans perdre la trace de l'autre source."""
        complete = Offre(
            source="France Travail",
            titre="Alternance Developpeur IA (H/F)",
            entreprise="Acme",
            lieu="Paris (75)",
            description="Une description nettement plus detaillee de l'offre.",
        )
        pauvre = Offre(
            source="Adzuna", titre="Alternance Developpeur IA F/H", entreprise="Acme", lieu="Paris"
        )
        gardees = dedoublonner([complete, pauvre])
        self.assertEqual(len(gardees), 1, "les mentions H/F ne doivent pas creer un doublon")
        self.assertEqual(gardees[0].source, "France Travail")
        self.assertEqual(gardees[0].autres_sources, ["Adzuna"])

    def test_dedoublonnage(self):
        offres = charger_fixtures()
        self.assertEqual(len(offres), 8)
        self.assertEqual(len(dedoublonner(offres)), 7, "le doublon Doctolib doit disparaitre")


class TestScoring(unittest.TestCase):
    def setUp(self):
        self.profil = charger_profil()
        self.offres = dedoublonner(charger_fixtures())
        self.par_entreprise = {o.entreprise: o for o in self.offres}

    def test_detection_alternance(self):
        self.assertTrue(est_alternance(self.par_entreprise["Doctolib"]))
        self.assertFalse(est_alternance(self.par_entreprise["Capgemini"]))

    def test_offre_ideale_en_tete(self):
        classees = classer(self.offres, self.profil)
        self.assertEqual(classees[0].entreprise, "Doctolib")

    def test_cdi_et_commercial_ecartes(self):
        retenues = {o.entreprise for o in classer(self.offres, self.profil)}
        self.assertNotIn("Capgemini", retenues, "un CDI n'est pas une alternance")
        self.assertNotIn("Verisure", retenues, "le commercial est hors domaine")

    def test_organisme_de_formation_degrade(self):
        """ISCOD vend une formation, il ne recrute pas : il doit passer derriere
        les vrais employeurs, voire sortir."""
        iscod = scorer(self.par_entreprise["ISCOD"], self.profil)
        doctolib = scorer(self.par_entreprise["Doctolib"], self.profil)
        self.assertLess(iscod.score, doctolib.score)
        self.assertTrue(any("organisme de formation" in r for r in iscod.raisons))

    def test_etoile_libere_une_borne(self):
        """`comptab*` doit attraper comptabilite, `*school` doit attraper Kaischool,
        sans que `school` seul ne matche a l'interieur d'un autre mot."""
        from scoring import _compte_termes

        self.assertEqual(_compte_termes("poste en comptabilite", ["comptab*"]), ["comptab"])
        self.assertEqual(_compte_termes("recrute chez kaischool", ["*school"]), ["school"])
        self.assertEqual(_compte_termes("recrute chez kaischool", ["school"]), [])
        self.assertEqual(_compte_termes("charge commercial", ["ia"]), [])

    def test_malus_bac5(self):
        thales = scorer(self.par_entreprise["Thales"], self.profil)
        self.assertTrue(any("BAC+3" in r or "superieur" in r for r in thales.raisons))

    def test_province_conservee(self):
        """Lille et Roubaix sont a plus de 200 km : ils doivent rester eligibles."""
        retenues = {o.entreprise for o in classer(self.offres, self.profil)}
        self.assertIn("Decathlon", retenues)
        self.assertIn("OVHcloud", retenues)

    def test_raisons_lisibles(self):
        doctolib = scorer(self.par_entreprise["Doctolib"], self.profil)
        self.assertTrue(doctolib.raisons)
        self.assertTrue(any("poste vise" in r for r in doctolib.raisons))


class TestRecence(unittest.TestCase):
    def test_filtre_recence(self):
        vieille = Offre(
            source="test",
            titre="Alternance dev python",
            date_publication=datetime.now(timezone.utc) - timedelta(days=30),
        )
        fraiche = Offre(
            source="test",
            titre="Alternance dev python",
            date_publication=datetime.now(timezone.utc) - timedelta(hours=3),
        )
        sans_date = Offre(source="test", titre="Alternance dev python")
        gardees = filtrer_recentes([vieille, fraiche, sans_date], jours=7)
        self.assertIn(fraiche, gardees)
        self.assertIn(sans_date, gardees, "une offre sans date ne doit pas etre perdue")
        self.assertNotIn(vieille, gardees)


class TestHistorique(unittest.TestCase):
    def test_pas_deux_fois_la_meme_offre(self):
        with tempfile.TemporaryDirectory() as dossier:
            fichier = Path(dossier) / "historique.json"
            offres = dedoublonner(charger_fixtures())

            vues = historique.charger(fichier)
            self.assertEqual(historique.nouvelles(offres, vues), offres)

            historique.enregistrer(offres, vues, fichier)
            vues = historique.charger(fichier)
            self.assertEqual(historique.nouvelles(offres, vues), [], "tout est deja vu")

            inedite = Offre(source="test", titre="Alternance MLOps", entreprise="Hublo", lieu="Paris")
            self.assertEqual(historique.nouvelles(offres + [inedite], vues), [inedite])

    def test_seules_les_offres_envoyees_sont_memorisees(self):
        """Le digest est plafonne. Les offres pertinentes qui n'ont pas tenu
        dans le digest du jour doivent ressortir le lendemain, pas disparaitre."""
        with tempfile.TemporaryDirectory() as dossier:
            fichier = Path(dossier) / "historique.json"
            pertinentes = dedoublonner(charger_fixtures())
            envoyees = pertinentes[:2]

            vues = historique.charger(fichier)
            historique.enregistrer(envoyees, vues, fichier)

            restantes = historique.nouvelles(pertinentes, historique.charger(fichier))
            self.assertEqual(len(restantes), len(pertinentes) - len(envoyees))
            for offre in envoyees:
                self.assertNotIn(offre, restantes)


if __name__ == "__main__":
    unittest.main(verbosity=2)

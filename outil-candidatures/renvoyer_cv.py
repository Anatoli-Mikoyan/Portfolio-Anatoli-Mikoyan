#!/usr/bin/env python3
"""Renvoie le bon CV aux entreprises qui ont déjà reçu une candidature avec le mauvais.

Les destinataires sont ceux du journal envois.log.csv qui figurent dans les CSV indiqués.
À lancer AVANT de reprendre les envois, sinon les nouveaux destinataires seraient inclus.

Utilisation :
    python renvoyer_cv.py                # aperçu, n'envoie rien
    python renvoyer_cv.py --envoyer      # envoi réel
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import envoyer_candidatures as ec

JOURNAL_CV = ec.DOSSIER / "cv-corriges.log.csv"
OBJET = "Re : Candidature alternance technicien informatique | Anatoli Mikoyan"
CORPS = """Madame, Monsieur,

Je me permets de revenir vers vous au sujet de ma candidature envoyée aujourd'hui. Une erreur s'est glissée dans la pièce jointe : vous trouverez ci-joint mon CV à jour.

Je vous prie de m'excuser pour ce désagrément. Je vous remercie par avance de l'attention que vous porterez à ma candidature.

Je vous prie d'agréer, Madame, Monsieur, l'expression de mes salutations distinguées.

Anatoli Mikoyan"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", nargs="+", type=Path,
                   default=[ec.DOSSIER / "entreprises-technicien-lot2.csv", ec.DOSSIER / "entreprises-technicien-region.csv"])
    pdfs = sorted(ec.DOSSIER.glob("*.pdf"))
    p.add_argument("--cv", default=pdfs[0] if pdfs else None, type=Path)
    p.add_argument("--delai", default=45, type=int)
    p.add_argument("--envoyer", action="store_true")
    args = p.parse_args()

    ec.charger_env(ec.DOSSIER / ".env")
    if len(pdfs) > 1 and args.cv == pdfs[0]:
        sys.exit("Plusieurs PDF dans le dossier : ne garde que le bon CV, ou précise --cv nom.pdf")
    if not (args.cv and args.cv.exists()):
        sys.exit("CV introuvable : place le bon CV en PDF dans ce dossier")

    noms = {}
    for chemin in args.csv:
        if chemin.exists():
            noms.update({l["email"].lower(): l["entreprise"] for l in ec.lire_entreprises(chemin)})
    corriges = set()
    if JOURNAL_CV.exists():
        with open(JOURNAL_CV, newline="", encoding="utf-8") as f:
            corriges = {l["email"].lower() for l in csv.DictReader(f)}
    a_faire = sorted(e for e in ec.deja_envoyes() if e in noms and e not in corriges)

    print(f"{len(a_faire)} entreprise(s) à qui renvoyer le CV ({args.cv.name}).\n")
    if not args.envoyer:
        for e in a_faire:
            print(f"  {noms[e]} → {e}")
        print("\nAperçu seulement. Ajoute --envoyer pour envoyer.")
        return

    expediteur, mot_de_passe = os.environ.get("EXPEDITEUR"), os.environ.get("MOT_DE_PASSE")
    if not (expediteur and mot_de_passe):
        sys.exit("Renseigne EXPEDITEUR et MOT_DE_PASSE dans le fichier .env")
    import smtplib, ssl
    serveur = os.environ.get("SMTP_SERVEUR", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "465"))
    nom = os.environ.get("NOM_EXPEDITEUR", "")

    for i, e in enumerate(a_faire):
        msg = ec.construire_mail(expediteur, nom, e, OBJET, CORPS, args.cv)
        try:
            with smtplib.SMTP_SSL(serveur, port, context=ssl.create_default_context(), timeout=60) as smtp:
                smtp.login(expediteur, mot_de_passe)
                smtp.send_message(msg)
            print(f"✔ CV renvoyé à {noms[e]} → {e}")
            nouveau = not JOURNAL_CV.exists()
            with open(JOURNAL_CV, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if nouveau:
                    w.writerow(["entreprise", "email"])
                w.writerow([noms[e], e])
        except smtplib.SMTPAuthenticationError:
            sys.exit("Gmail refuse la connexion : vérifie EXPEDITEUR et MOT_DE_PASSE dans .env")
        except (smtplib.SMTPException, OSError) as err:
            print(f"✘ {noms[e]} → {e} : {err}")
        if i < len(a_faire) - 1:
            time.sleep(args.delai)


if __name__ == "__main__":
    main()

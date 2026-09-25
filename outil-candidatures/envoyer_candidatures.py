#!/usr/bin/env python3
"""Envoie une candidature personnalisée (lettre + CV en pièce jointe) à chaque entreprise d'un CSV.

Utilisation :
    python envoyer_candidatures.py                 # aperçu, n'envoie rien
    python envoyer_candidatures.py --test moi@x.fr # envoie UN mail de test à ton adresse
    python envoyer_candidatures.py --envoyer       # envoi réel
"""
import argparse
import csv
import mimetypes
import os
import re
import smtplib
import ssl
import sys
import time
import unicodedata
from datetime import datetime
from email.message import EmailMessage
from email.utils import formataddr
from pathlib import Path

DOSSIER = Path(__file__).resolve().parent
JOURNAL = DOSSIER / "envois.log.csv"


def charger_env(chemin):
    if not chemin.exists():
        return
    for ligne in chemin.read_text(encoding="utf-8").splitlines():
        ligne = ligne.strip()
        if ligne and not ligne.startswith("#") and "=" in ligne:
            cle, valeur = ligne.split("=", 1)
            os.environ.setdefault(cle.strip(), valeur.strip())


def lire_entreprises(chemin):
    with open(chemin, newline="", encoding="utf-8-sig") as f:
        echantillon = f.read(2048)
        f.seek(0)
        dialecte = csv.Sniffer().sniff(echantillon, delimiters=",;\t")
        lignes = [
            {k.strip().lower(): (v or "").strip() for k, v in ligne.items() if k}
            for ligne in csv.DictReader(f, dialect=dialecte)
        ]
    return [
        l for l in lignes
        if l.get("entreprise") and l.get("email") and l.get("envoyer", "oui").lower() != "non"
    ]


def deja_envoyes():
    if not JOURNAL.exists():
        return set()
    with open(JOURNAL, newline="", encoding="utf-8") as f:
        return {l["email"].lower() for l in csv.DictReader(f) if l.get("statut") == "envoyé"}


def journaliser(entreprise, email, statut):
    nouveau = not JOURNAL.exists()
    with open(JOURNAL, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nouveau:
            w.writerow(["date", "entreprise", "email", "statut"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), entreprise, email, statut])


def slug(nom):
    nom = unicodedata.normalize("NFKD", nom).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", nom.lower()).strip("-")


def rediger(modele, ligne):
    """Renvoie (objet, corps, source) ; corps vaut None si la lettre n'est pas personnalisée."""
    perso = DOSSIER / "lettres" / f"{slug(ligne['entreprise'])}.txt"
    if perso.exists():
        texte, source = perso.read_text(encoding="utf-8"), perso.name
    elif ligne.get("accroche"):
        texte, source = modele, "lettre.txt + accroche"
    else:
        return None, None, f"aucune lettre : crée lettres/{perso.name} ou remplis la colonne accroche"
    contact = ligne.get("contact", "")
    texte = texte.format(
        entreprise=ligne["entreprise"],
        salutation=f"Bonjour {contact}," if contact else "Bonjour,",
        accroche=ligne.get("accroche", ""),
    )
    premiere, _, corps = texte.partition("\n")
    if premiere.lower().startswith("objet"):
        return premiere.split(":", 1)[1].strip(), corps.strip(), source
    return f"Candidature alternance chez {ligne['entreprise']}", texte.strip(), source


def construire_mail(expediteur, nom, destinataire, objet, corps, cv):
    msg = EmailMessage()
    msg["From"] = formataddr((nom, expediteur))
    msg["To"] = destinataire
    msg["Subject"] = objet
    msg.set_content(corps)
    type_mime, _ = mimetypes.guess_type(cv.name)
    principal, secondaire = (type_mime or "application/octet-stream").split("/")
    msg.add_attachment(cv.read_bytes(), maintype=principal, subtype=secondaire, filename=cv.name)
    return msg


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=DOSSIER / "entreprises.csv", type=Path)
    p.add_argument("--lettre", default=DOSSIER / "lettre.txt", type=Path)
    cv_defaut = DOSSIER / "CV_Anatoli_Mikoyan_alternance.pdf"
    p.add_argument("--cv", default=cv_defaut if cv_defaut.exists() else DOSSIER.parent / "Cv.pdf", type=Path)
    p.add_argument("--delai", default=45, type=int, help="secondes entre deux envois (défaut 45)")
    p.add_argument("--max", default=0, type=int, help="nombre max d'envois cette fois (0 = tous)")
    p.add_argument("--a-partir", metavar="'AAAA-MM-JJ HH:MM'",
                   help="attend cette heure avant d'envoyer (l'ordinateur doit rester allumé)")
    groupe = p.add_mutually_exclusive_group()
    groupe.add_argument("--envoyer", action="store_true", help="envoie réellement les mails")
    groupe.add_argument("--test", metavar="EMAIL", help="envoie le 1er mail à cette adresse, pour vérifier")
    args = p.parse_args()

    charger_env(DOSSIER / ".env")
    for fichier in (args.csv, args.lettre, args.cv):
        if not fichier.exists():
            sys.exit(f"Fichier introuvable : {fichier}")

    modele = args.lettre.read_text(encoding="utf-8")
    entreprises = lire_entreprises(args.csv)
    envoyes = deja_envoyes()
    a_faire = [e for e in entreprises if e["email"].lower() not in envoyes]
    deja = len(entreprises) - len(a_faire)
    if args.max:
        a_faire = a_faire[: args.max]
    print(f"{len(entreprises)} entreprises, {deja} déjà contactées, {len(a_faire)} à traiter.\n")

    if not (args.envoyer or args.test):
        for e in a_faire:
            objet, corps, source = rediger(modele, e)
            if corps is None:
                print(f"=== ⚠ {e['entreprise']} ({e['email']}) : {source}\n")
                continue
            print(f"=== À : {e['email']}   [{source}]\nObjet : {objet}\n\n{corps}\n[PJ : {args.cv.name}]\n")
        print("Aperçu seulement. Ajoute --test ton@mail.fr pour un essai, puis --envoyer.")
        return

    expediteur = os.environ.get("EXPEDITEUR")
    mot_de_passe = os.environ.get("MOT_DE_PASSE")
    if not (expediteur and mot_de_passe):
        sys.exit("Renseigne EXPEDITEUR et MOT_DE_PASSE dans le fichier .env")
    nom = os.environ.get("NOM_EXPEDITEUR", "")
    serveur = os.environ.get("SMTP_SERVEUR", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "465"))

    if args.test:
        a_faire = a_faire[:1]
        if not a_faire:
            sys.exit("Aucune entreprise à utiliser pour le test.")

    if args.a_partir:
        cible = datetime.strptime(args.a_partir, "%Y-%m-%d %H:%M")
        attente = (cible - datetime.now()).total_seconds()
        if attente > 0:
            print(f"Envoi programmé le {cible:%d/%m à %H:%M}. Laisse l'ordinateur allumé et connecté.")
            time.sleep(attente)

    with smtplib.SMTP_SSL(serveur, port, context=ssl.create_default_context()) as smtp:
        smtp.login(expediteur, mot_de_passe)
        for i, e in enumerate(a_faire):
            objet, corps, source = rediger(modele, e)
            if corps is None:
                print(f"⏭ {e['entreprise']} ignorée : {source}")
                continue
            destinataire = args.test or e["email"]
            try:
                smtp.send_message(construire_mail(expediteur, nom, destinataire, objet, corps, args.cv))
                print(f"✔ {e['entreprise']} → {destinataire}")
                if not args.test:
                    journaliser(e["entreprise"], e["email"], "envoyé")
            except smtplib.SMTPException as err:
                print(f"✘ {e['entreprise']} → {destinataire} : {err}")
                if not args.test:
                    journaliser(e["entreprise"], e["email"], f"erreur: {err}")
            if i < len(a_faire) - 1:
                time.sleep(args.delai)


if __name__ == "__main__":
    main()

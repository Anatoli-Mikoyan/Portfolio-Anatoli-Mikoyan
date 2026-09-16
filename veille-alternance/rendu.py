"""Mise en forme du digest : Markdown (console/log), HTML (tableau de bord), e-mail."""

from __future__ import annotations

import html
import logging
import os
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage
from zoneinfo import ZoneInfo

from modele import Offre

journal = logging.getLogger("veille.rendu")
PARIS = ZoneInfo("Europe/Paris")

COULEUR_SOURCE = {
    "France Travail": "#2f6fed",
    "La Bonne Alternance": "#0d7f96",
    "Adzuna": "#8b5cf6",
    "Indeed": "#1d4ed8",
}


def _horodatage() -> str:
    return datetime.now(PARIS).strftime("%d/%m/%Y a %Hh%M")


def _extrait(texte: str, taille: int = 240) -> str:
    propre = " ".join((texte or "").split())
    return propre[:taille].rstrip() + "..." if len(propre) > taille else propre


# ------------------------------------------------------------------- Markdown

def en_markdown(offres: list[Offre], statistiques: dict) -> str:
    lignes = [
        f"# Veille alternance — {_horodatage()}",
        "",
        f"**{len(offres)} nouvelles offres** sur {statistiques.get('total_collecte', 0)} "
        f"annonces collectees ({statistiques.get('sources_actives', 0)} source(s) active(s)).",
        "",
    ]
    if not offres:
        lignes.append("_Aucune nouvelle offre correspondant au profil aujourd'hui._")
        return "\n".join(lignes)

    for rang, offre in enumerate(offres, 1):
        lignes.append(f"## {rang}. [{offre.titre}]({offre.url})")
        entete = " · ".join(
            partie
            for partie in (offre.entreprise, offre.lieu, offre.age_texte, offre.source)
            if partie
        )
        lignes.append(f"*{entete}* — score **{offre.score}**")
        if offre.raisons:
            lignes.append(f"> {' | '.join(offre.raisons[:4])}")
        extrait = _extrait(offre.description)
        if extrait:
            lignes.append("")
            lignes.append(extrait)
        lignes.append("")
    return "\n".join(lignes)


# ------------------------------------------------------- tableau de bord HTML

_CSS = """
:root{--fond:#0e1020;--carte:#171a35;--carte2:#1d2143;--trait:#2c3160;
--texte:#eef0ff;--doux:#a7afd8;--violet:#7c5cff;--cyan:#6fe3f5;--vert:#4ade9b;--orange:#ffb454;}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:Inter,system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--fond);
color:var(--texte);line-height:1.5;padding:0 0 64px}
.bandeau{background:linear-gradient(120deg,#171a35,#241a4d 48%,#10233d);padding:32px 16px 28px;
border-bottom:1px solid var(--trait)}
.dedans{max-width:940px;margin:0 auto}
h1{font-size:1.65rem;letter-spacing:-.02em;margin-bottom:6px}
h1 span{color:var(--cyan)}
.sous{color:var(--doux);font-size:.92rem}
.chiffres{display:flex;flex-wrap:wrap;gap:10px;margin-top:18px}
.chiffre{background:rgba(124,92,255,.14);border:1px solid rgba(124,92,255,.34);
border-radius:9px;padding:7px 13px;font-size:.84rem;color:#dfe3ff}
.chiffre b{color:#fff}
main{max-width:940px;margin:0 auto;padding:26px 16px 0}
.offre{background:var(--carte);border:1px solid var(--trait);border-radius:13px;
padding:17px 18px;margin-bottom:13px;transition:border-color .16s,transform .16s}
.offre:hover{border-color:var(--violet);transform:translateY(-2px)}
.haut{display:flex;justify-content:space-between;gap:14px;align-items:flex-start;flex-wrap:wrap}
.titre{font-family:"Space Grotesk",Inter,sans-serif;font-size:1.06rem;font-weight:700;
color:#fff;text-decoration:none}
.titre:hover{color:var(--cyan)}
.score{font-family:"JetBrains Mono",ui-monospace,monospace;font-size:.76rem;font-weight:600;
padding:3px 9px;border-radius:999px;background:rgba(74,222,155,.15);
border:1px solid rgba(74,222,155,.4);color:var(--vert);white-space:nowrap;flex:none}
.score.moyen{background:rgba(255,180,84,.14);border-color:rgba(255,180,84,.4);color:var(--orange)}
.meta{color:var(--doux);font-size:.85rem;margin:7px 0 9px;display:flex;flex-wrap:wrap;gap:5px 9px}
.meta .frais{color:var(--vert);font-weight:600}
.puce{font-size:.7rem;padding:2px 8px;border-radius:5px;font-weight:600;color:#fff}
.raisons{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:9px}
.raison{font-size:.73rem;padding:2px 8px;border-radius:5px;background:var(--carte2);
border:1px solid var(--trait);color:#c3c9f2}
.raison.negatif{background:rgba(255,99,99,.1);border-color:rgba(255,99,99,.32);color:#ffb0b0}
.extrait{color:var(--doux);font-size:.87rem}
.vide{text-align:center;padding:54px 20px;color:var(--doux)}
.avertissement{background:rgba(255,180,84,.13);border:1px solid rgba(255,180,84,.42);
color:#ffd9a0;border-radius:9px;padding:11px 14px;margin-bottom:16px;font-size:.87rem}
footer{max-width:940px;margin:30px auto 0;padding:0 16px;color:#6d739a;font-size:.78rem}
a.lien{color:var(--cyan)}
@media(max-width:560px){.haut{flex-direction:column}h1{font-size:1.32rem}}
"""


def _puce_source(source: str) -> str:
    couleur = COULEUR_SOURCE.get(source, "#5b6180")
    return f'<span class="puce" style="background:{couleur}">{html.escape(source)}</span>'


def _carte(offre: Offre) -> str:
    classe_score = "score" if offre.score >= 60 else "score moyen"
    frais = (offre.age_heures or 999) <= 48
    raisons = "".join(
        f'<span class="raison{" negatif" if r.startswith("-") else ""}">{html.escape(r.lstrip("- "))}</span>'
        for r in offre.raisons[:5]
    )
    meta = [
        f"<span>{html.escape(offre.entreprise)}</span>" if offre.entreprise else "",
        f"<span>{html.escape(offre.lieu)}</span>" if offre.lieu else "",
        f'<span class="{"frais" if frais else ""}">{html.escape(offre.age_texte)}</span>',
        _puce_source(offre.source),
        *(_puce_source(s) for s in offre.autres_sources),
    ]
    extrait = _extrait(offre.description, 260)
    return f"""<article class="offre">
  <div class="haut">
    <a class="titre" href="{html.escape(offre.url)}" target="_blank" rel="noopener">{html.escape(offre.titre)}</a>
    <span class="{classe_score}">{offre.score} pts</span>
  </div>
  <div class="meta">{''.join(m for m in meta if m)}</div>
  <div class="raisons">{raisons}</div>
  <p class="extrait">{html.escape(extrait)}</p>
</article>"""


def en_html(offres: list[Offre], statistiques: dict) -> str:
    # Un jeu d'essai contient des entreprises fictives : ne jamais le publier
    # sans le dire, la page est servie publiquement par GitHub Pages.
    avertissement = (
        '<div class="avertissement"><b>Jeu de demonstration.</b> Ces annonces sont '
        "fictives et servent uniquement a illustrer le rendu. Les vraies offres "
        "apparaitront apres la configuration des cles d'API (voir le README).</div>"
        if statistiques.get("demo")
        else ""
    )
    corps = (
        "".join(_carte(o) for o in offres)
        if offres
        else '<div class="vide"><p>Aucune nouvelle offre correspondant au profil '
        "aujourd'hui.</p><p>La veille repasse demain matin.</p></div>"
    )
    chiffres = "".join(
        f'<div class="chiffre"><b>{valeur}</b> {html.escape(libelle)}</div>'
        for libelle, valeur in (
            ("nouvelles offres", len(offres)),
            ("annonces analysees", statistiques.get("total_collecte", 0)),
            ("sources actives", statistiques.get("sources_actives", 0)),
            ("deja vues, filtrees", statistiques.get("deja_vues", 0)),
        )
    )
    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex">
<title>Veille alternance — Anatoli Mikoyan</title>
<style>{_CSS}</style>
</head>
<body>
<header class="bandeau"><div class="dedans">
  <h1>Veille <span>alternance</span></h1>
  <p class="sous">Developpeur IA / Python / Data — rentree octobre 2026 · mise a jour du {_horodatage()}</p>
  <div class="chiffres">{chiffres}</div>
</div></header>
<main>{avertissement}{corps}</main>
<footer>
  <p>Genere automatiquement chaque matin par
  <a class="lien" href="https://github.com/Anatoli-Mikoyan/Portfolio-Anatoli-Mikoyan/tree/main/veille-alternance">veille-alternance</a>
  — classement etabli a partir du CV et du projet de formation.</p>
</footer>
</body>
</html>"""


# ----------------------------------------------------------------------- email

def en_email_html(offres: list[Offre], statistiques: dict) -> str:
    """HTML volontairement simple : tableaux et styles en ligne, seul format
    que Gmail et Outlook rendent de maniere fiable."""
    if offres:
        blocs = []
        for rang, offre in enumerate(offres, 1):
            meta = " · ".join(
                p for p in (offre.entreprise, offre.lieu, offre.age_texte, offre.source) if p
            )
            raisons = " | ".join(r for r in offre.raisons[:3] if not r.startswith("-"))
            blocs.append(f"""
<tr><td style="padding:12px 0;border-bottom:1px solid #e3e6f2;">
  <div style="font-size:11px;color:#8b90ad;font-family:monospace;">#{rang} — {offre.score} pts</div>
  <a href="{html.escape(offre.url)}" style="font-size:16px;font-weight:700;color:#3b2bb5;text-decoration:none;">
    {html.escape(offre.titre)}</a>
  <div style="font-size:13px;color:#5a6080;margin:4px 0;">{html.escape(meta)}</div>
  <div style="font-size:12px;color:#7a80a0;">{html.escape(raisons)}</div>
  <div style="font-size:13px;color:#454a66;margin-top:6px;">{html.escape(_extrait(offre.description, 200))}</div>
</td></tr>""")
        corps = "".join(blocs)
    else:
        corps = (
            '<tr><td style="padding:22px 0;color:#5a6080;">Aucune nouvelle offre '
            "correspondant au profil aujourd'hui.</td></tr>"
        )

    return f"""<html><body style="margin:0;background:#f4f5fb;font-family:Arial,Helvetica,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f4f5fb;padding:20px 12px;">
<tr><td align="center">
<table width="620" cellpadding="0" cellspacing="0" style="max-width:620px;background:#ffffff;border-radius:11px;overflow:hidden;">
  <tr><td style="background:#1d1a44;padding:22px 24px;">
    <div style="color:#ffffff;font-size:20px;font-weight:700;">Veille alternance</div>
    <div style="color:#a7afd8;font-size:13px;margin-top:4px;">
      {len(offres)} nouvelle(s) offre(s) — {_horodatage()}</div>
  </td></tr>
  <tr><td style="padding:6px 24px 20px;"><table width="100%" cellpadding="0" cellspacing="0">{corps}</table></td></tr>
  <tr><td style="background:#f4f5fb;padding:14px 24px;color:#8b90ad;font-size:11px;">
    {statistiques.get('total_collecte', 0)} annonces analysees ·
    {statistiques.get('sources_actives', 0)} source(s) · offres deja vues filtrees automatiquement.
  </td></tr>
</table>
</td></tr></table></body></html>"""


def envoyer_email(offres: list[Offre], statistiques: dict, destinataire: str) -> bool:
    """Envoie le digest par SMTP. Renvoie False (sans lever) si non configure."""
    hote = os.getenv("SMTP_HOST", "smtp.gmail.com").strip()
    port = int(os.getenv("SMTP_PORT", "587"))
    utilisateur = os.getenv("SMTP_USER", "").strip()
    secret = os.getenv("SMTP_PASS", "").strip()

    if not (utilisateur and secret):
        journal.warning("E-mail non envoye : SMTP_USER / SMTP_PASS absents")
        return False

    message = EmailMessage()
    sujet = (
        f"{len(offres)} offres d'alternance — {datetime.now(PARIS):%d/%m}"
        if offres
        else f"Veille alternance — rien de neuf le {datetime.now(PARIS):%d/%m}"
    )
    message["Subject"] = sujet
    message["From"] = utilisateur
    message["To"] = destinataire
    message.set_content(en_markdown(offres, statistiques))
    message.add_alternative(en_email_html(offres, statistiques), subtype="html")

    try:
        with smtplib.SMTP(hote, port, timeout=30) as serveur:
            serveur.starttls()
            serveur.login(utilisateur, secret)
            serveur.send_message(message)
    except (smtplib.SMTPException, OSError) as erreur:
        journal.error("Envoi e-mail echoue : %s", erreur)
        return False

    journal.info("E-mail envoye a %s", destinataire)
    return True

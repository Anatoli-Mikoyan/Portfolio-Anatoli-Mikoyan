#!/usr/bin/env python3
"""Crée une page de recherche LinkedIn : pour chaque entreprise de tes listes, des liens qui
ouvrent directement la recherche des recruteurs et des responsables techniques de cette entreprise.

Utilisation :
    python chercher_prospects.py ..\\outil-candidatures\\entreprises.csv ..\\outil-candidatures\\entreprises-lot2.csv
    python chercher_prospects.py                 # prend tous les entreprises*.csv du dossier outil-candidatures
"""
import csv
import html
import sys
import webbrowser
from pathlib import Path
from urllib.parse import quote

DOSSIER = Path(__file__).resolve().parent
CANDIDATURES = DOSSIER.parent / "outil-candidatures"

RECHERCHES = [
    ("Recruteurs / RH", 'recrutement OR "talent acquisition" OR RH OR recruteur'),
    ("Tech (CTO, lead dev)", 'CTO OR "lead developer" OR "responsable technique" OR "head of data"'),
]
# Recherches générales, sans entreprise
GENERALES = [
    ("Posts récents : entreprises qui cherchent un alternant", "content", 'alternance développeur recrute OR "on recrute" OR "nous recrutons" IA OR python OR data'),
    ("Offres d'emploi LinkedIn : alternance IA / Python", "jobs", "alternance développeur IA python"),
    ("Recruteurs tech en Île-de-France", "people", '"talent acquisition" tech OR IT OR informatique'),
]


def url_personnes(mots):
    return "https://www.linkedin.com/search/results/people/?keywords=" + quote(mots)


def url_generale(kind, mots):
    if kind == "jobs":
        return "https://www.linkedin.com/jobs/search/?keywords=" + quote(mots) + "&location=France&sortBy=DD"
    return f"https://www.linkedin.com/search/results/{kind}/?keywords=" + quote(mots) + ("&sortBy=%22date_posted%22" if kind == "content" else "")


def entreprises(fichiers):
    vus = {}
    for f in fichiers:
        with open(f, newline="", encoding="utf-8-sig") as fh:
            ech = fh.read(2048); fh.seek(0)
            for l in csv.DictReader(fh, dialect=csv.Sniffer().sniff(ech, delimiters=",;\t")):
                nom = (l.get("entreprise") or "").strip()
                if nom and nom.lower() not in vus:
                    vus[nom.lower()] = (nom, (l.get("ville") or "").strip())
    return list(vus.values())


def main():
    fichiers = [Path(a) for a in sys.argv[1:]] or sorted(p for p in CANDIDATURES.glob("entreprises*.csv") if "exemple" not in p.name)
    if not fichiers:
        raise SystemExit("Aucune liste trouvée : indique les fichiers CSV en argument.")
    e = html.escape
    lignes = []
    for nom, ville in entreprises(fichiers):
        liens = " ".join(f'<a href="{e(url_personnes(f"{nom} {mots}"))}" target="_blank" rel="noopener">{e(lbl)}</a>' for lbl, mots in RECHERCHES)
        lignes.append(f'<tr><td><b>{e(nom)}</b><br><span>{e(ville)}</span></td><td>{liens}</td></tr>')
    gen = "".join(f'<li><a href="{e(url_generale(k, m))}" target="_blank" rel="noopener">{e(lbl)}</a></li>' for lbl, k, m in GENERALES)
    page = f"""<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trouver des prospects</title><style>
:root{{--bg:#f5f6fa;--card:#fff;--ink:#1c1f26;--mut:#5a6275;--acc:#0a66c2;--line:#dfe3ec}}
@media (prefers-color-scheme:dark){{:root{{--bg:#14161c;--card:#1d2029;--ink:#e8eaf0;--mut:#9aa3b5;--line:#2c3140}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,Segoe UI,Arial,sans-serif}}main{{max-width:820px;margin:0 auto;padding:16px}}
table{{width:100%;border-collapse:collapse;background:var(--card)}}td{{border-bottom:1px solid var(--line);padding:8px;vertical-align:top}}
span,.mut{{color:var(--mut);font-size:13px}}a{{display:inline-block;margin:2px 6px 2px 0;background:var(--acc);color:#fff;padding:5px 10px;border-radius:14px;text-decoration:none;font-size:13px}}
a:visited{{background:#6b7a90}}li a{{margin:4px 0}}ul{{padding-left:18px}}</style></head><body><main>
<h1>Trouver des prospects LinkedIn</h1>
<p class="mut">Connecte-toi à LinkedIn dans ce navigateur, puis clique. Pour chaque bonne personne trouvée, copie l'adresse de son profil dans <b>prospects.csv</b> (prénom, nom, entreprise, poste, type rh ou tech, profil). Les liens déjà ouverts deviennent gris.</p>
<h2>Recherches générales</h2><ul>{gen}</ul>
<h2>Dans les entreprises de tes listes ({len(lignes)})</h2>
<p class="mut">Ce sont les entreprises auxquelles tu as déjà écrit : un message LinkedIn à leur recruteur double tes chances.</p>
<table>{''.join(lignes)}</table></main></body></html>"""
    sortie = DOSSIER / "recherche_prospects.html"
    sortie.write_text(page, encoding="utf-8")
    print(f"{len(lignes)} entreprises, page créée : {sortie.name}")
    try:
        webbrowser.open(sortie.as_uri())
    except Exception:
        pass


if __name__ == "__main__":
    main()

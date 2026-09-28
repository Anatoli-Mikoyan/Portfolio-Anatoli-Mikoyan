#!/usr/bin/env python3
"""Prépare une page de prospection LinkedIn : une fiche par personne avec la note d'invitation
déjà rédigée, un bouton pour ouvrir le profil et un bouton pour copier la note.

Rien n'est envoyé automatiquement : c'est toi qui cliques sur « Se connecter » dans LinkedIn,
ce qui évite toute restriction de compte.

Utilisation :
    python preparer_linkedin.py                      # lit prospects.csv, crée linkedin.html
    python preparer_linkedin.py --csv autre.csv
"""
import argparse
import csv
import html
import json
import webbrowser
from pathlib import Path

DOSSIER = Path(__file__).resolve().parent
LIMITE_NOTE = 200  # longueur maximale d'une note avec un compte LinkedIn gratuit

SIGNATURE = "Anatoli"
PORTFOLIO = "https://anatoli-mikoyan.github.io/Portfolio-Anatoli-Mikoyan/"

# Notes d'invitation (courtes). {accroche} est remplacée par la colonne accroche si elle existe.
NOTES = {
    "rh": ("Bonjour {prenom}, je cherche une alternance de développeur IA / Python (12 mois, dès octobre). "
           "{entreprise} m'intéresse beaucoup : puis-je vous envoyer mon CV ? Merci, " + SIGNATURE),
    "tech": ("Bonjour {prenom}, développeur Python et IA, je cherche une alternance de 12 mois dès octobre. "
             "Votre travail chez {entreprise} m'intéresse : ravi d'échanger avec vous. " + SIGNATURE),
    "accroche": ("Bonjour {prenom}, {accroche}. Je cherche une alternance de développeur IA / Python dès octobre : "
                 "ravi d'échanger avec vous. " + SIGNATURE),
}

# Message à envoyer une fois l'invitation acceptée (pas de limite stricte).
MESSAGE_APRES = (
    "Bonjour {prenom},\n\n"
    "Merci d'avoir accepté mon invitation.\n\n"
    "Titulaire d'un BTS SIO, j'entre en octobre en 3e année de Bachelor en intelligence artificielle. "
    "Je recherche une alternance de 12 mois en développement IA / Python.\n\n"
    "Depuis un an, je développe seul TradeVIQ, une application SaaS en production dans laquelle j'ai intégré "
    "un modèle de langage, avec un service Python de collecte de données. Mes projets sont visibles ici : "
    + PORTFOLIO + "\n\n"
    "{demande}\n\n"
    "Je vous remercie par avance, bonne journée.\n\n"
    "Anatoli Mikoyan"
)
DEMANDES = {
    "rh": "Auriez-vous une opportunité d'alternance chez {entreprise}, ou pourriez-vous m'indiquer la bonne personne à contacter ? Je peux vous envoyer mon CV.",
    "tech": "Accepteriez-vous d'échanger quelques minutes sur votre équipe chez {entreprise} ? Si une alternance est possible, je serais ravi de vous envoyer mon CV.",
}


def lire_prospects(chemin):
    with open(chemin, newline="", encoding="utf-8-sig") as f:
        echantillon = f.read(2048)
        f.seek(0)
        dialecte = csv.Sniffer().sniff(echantillon, delimiters=",;\t")
        lignes = [
            {k.strip().lower(): (v or "").strip() for k, v in ligne.items() if k}
            for ligne in csv.DictReader(f, dialect=dialecte)
        ]
    return [l for l in lignes if l.get("prenom") and l.get("profil")]


def rediger(p):
    type_ = "tech" if p.get("type", "").lower().startswith("tech") else "rh"
    valeurs = {
        "prenom": p["prenom"],
        "entreprise": p.get("entreprise") or "votre entreprise",
        "accroche": p.get("accroche", "").rstrip(". "),
    }
    if valeurs["accroche"][:2] in ("J'", "Je") or valeurs["accroche"][:1] in "JV":
        valeurs["accroche"] = valeurs["accroche"][:1].lower() + valeurs["accroche"][1:]
    modele = NOTES["accroche"] if valeurs["accroche"] else NOTES[type_]
    note = modele.format(**valeurs)
    if len(note) > LIMITE_NOTE and valeurs["accroche"]:
        note = NOTES[type_].format(**valeurs)  # accroche trop longue : on revient à la note standard
    message = MESSAGE_APRES.format(demande=DEMANDES[type_].format(**valeurs), **valeurs)
    return note, message


def fiche(i, p, note, message):
    trop_long = len(note) > LIMITE_NOTE
    e = html.escape
    return f"""
<article class="fiche" id="f{i}">
  <header>
    <label class="done"><input type="checkbox" data-id="{e(p['profil'])}"> Envoyé</label>
    <div><b>{e(p['prenom'])} {e(p.get('nom', ''))}</b><br><span>{e(p.get('poste', ''))} · {e(p.get('entreprise', ''))}</span></div>
    <a class="btn" href="{e(p['profil'])}" target="_blank" rel="noopener">Ouvrir le profil</a>
  </header>
  <p class="lbl">1. Note d'invitation <span class="cnt{' ko' if trop_long else ''}">{len(note)} / {LIMITE_NOTE}</span></p>
  <textarea rows="3">{e(note)}</textarea>
  <button class="copy">Copier la note</button>
  <details><summary>2. Message après acceptation</summary>
    <textarea rows="9">{e(message)}</textarea>
    <button class="copy">Copier le message</button>
  </details>
</article>"""


PAGE = """<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Prospection LinkedIn</title>
<style>
:root{--bg:#f5f6fa;--card:#fff;--ink:#1c1f26;--mut:#5a6275;--acc:#0a66c2;--ok:#2f9a5a;--ko:#c0392b;--line:#dfe3ec}
@media (prefers-color-scheme:dark){:root{--bg:#14161c;--card:#1d2029;--ink:#e8eaf0;--mut:#9aa3b5;--line:#2c3140}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,Segoe UI,Arial,sans-serif}
main{max-width:760px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:6px 0}.intro{color:var(--mut);margin:0 0 14px}
.bar{position:sticky;top:0;background:var(--bg);padding:8px 0;z-index:2;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.bar input{flex:1;min-width:180px;padding:8px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink)}
.fiche{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:10px 0}
.fiche.fait{opacity:.5}
header{display:flex;gap:12px;align-items:center}header div{flex:1}header span{color:var(--mut);font-size:13px}
.btn,button{background:var(--acc);color:#fff;border:0;border-radius:16px;padding:6px 12px;font-size:13px;text-decoration:none;cursor:pointer}
button.ok{background:var(--ok)}
textarea{width:100%;box-sizing:border-box;margin:4px 0;padding:8px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--ink);font:inherit}
.lbl{margin:10px 0 0;font-size:13px;color:var(--mut)}.cnt{float:right}.cnt.ko{color:var(--ko);font-weight:bold}
details{margin-top:8px}summary{cursor:pointer;color:var(--mut);font-size:13px}
.done{font-size:13px;white-space:nowrap}
</style></head><body><main>
<h1>Prospection LinkedIn</h1>
<p class="intro">Pour chaque personne : <b>Ouvrir le profil</b>, cliquer sur <b>Se connecter</b> puis <b>Ajouter une note</b>, coller la note, envoyer, puis cocher <b>Envoyé</b>. Quand la personne accepte, envoie-lui le message 2. Reste sous 15 à 20 invitations par jour.</p>
<div class="bar"><input id="q" placeholder="Filtrer par nom ou entreprise…"><span id="stat"></span></div>
__FICHES__
</main>
<script>
const KEY="prospection-linkedin";
let etat={};try{etat=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
function save(){try{localStorage.setItem(KEY,JSON.stringify(etat))}catch(e){}}
function maj(){const t=document.querySelectorAll(".fiche").length,f=document.querySelectorAll(".fiche.fait").length;document.getElementById("stat").textContent=f+" / "+t+" envoyées"}
document.querySelectorAll(".done input").forEach(c=>{const art=c.closest(".fiche");c.checked=!!etat[c.dataset.id];art.classList.toggle("fait",c.checked);
 c.addEventListener("change",()=>{etat[c.dataset.id]=c.checked;save();art.classList.toggle("fait",c.checked);maj()})});
document.querySelectorAll("button.copy").forEach(b=>b.addEventListener("click",async()=>{const t=b.previousElementSibling;
 try{await navigator.clipboard.writeText(t.value)}catch(e){t.select();document.execCommand("copy")}
 const old=b.textContent;b.textContent="Copié !";b.classList.add("ok");setTimeout(()=>{b.textContent=old;b.classList.remove("ok")},1500)}));
document.querySelectorAll("textarea").forEach(t=>t.addEventListener("input",()=>{const c=t.parentElement.querySelector(".cnt");if(c&&t.previousElementSibling.classList.contains("lbl")){c.textContent=t.value.length+" / __LIM__";c.classList.toggle("ko",t.value.length>__LIM__)}}));
document.getElementById("q").addEventListener("input",e=>{const q=e.target.value.toLowerCase();document.querySelectorAll(".fiche").forEach(f=>{f.style.display=f.querySelector("header div").textContent.toLowerCase().includes(q)?"":"none"})});
maj();
</script></body></html>"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=DOSSIER / "prospects.csv", type=Path)
    p.add_argument("--sortie", default=DOSSIER / "linkedin.html", type=Path)
    args = p.parse_args()
    if not args.csv.exists():
        raise SystemExit(f"Fichier introuvable : {args.csv} (copie prospects.exemple.csv en prospects.csv)")

    prospects = lire_prospects(args.csv)
    fiches, trop_longues = [], 0
    for i, pr in enumerate(prospects):
        note, message = rediger(pr)
        trop_longues += len(note) > LIMITE_NOTE
        fiches.append(fiche(i, pr, note, message))
    page = PAGE.replace("__FICHES__", "\n".join(fiches)).replace("__LIM__", str(LIMITE_NOTE))
    args.sortie.write_text(page, encoding="utf-8")
    print(f"{len(prospects)} fiches créées dans {args.sortie.name}.")
    if trop_longues:
        print(f"⚠ {trop_longues} note(s) dépassent {LIMITE_NOTE} caractères : raccourcis-les dans la page avant de coller.")
    try:
        webbrowser.open(args.sortie.as_uri())
    except Exception:
        pass


if __name__ == "__main__":
    main()

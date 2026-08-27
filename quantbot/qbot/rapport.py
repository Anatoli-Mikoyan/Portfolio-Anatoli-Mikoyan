"""Rapport HTML de fin de parcours, écrit par `scripts/start.py`.

Un seul fichier, aucune dépendance réseau : il s'ouvre d'un double-clic, s'archive et se
relit dans deux ans. Les courbes sont du SVG produit à la main.

Le parti pris de lecture : le chiffre que l'on retient en premier doit être celui qui
engage de l'argent, et l'intervalle de confiance doit être aussi visible que lui. Un
rapport qui affiche « +12 % » en gros et l'incertitude en note de bas de page fait
prendre des décisions sur du bruit.
"""
from __future__ import annotations

import html
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

__all__ = ["ecrire_rapport", "rapport_html"]

_CSS = """
:root{--bg:#eceef0;--carte:#f8f9fa;--bord:#ccd2d8;--txt:#15191d;--doux:#5a646e;
--ok:#2c6449;--ko:#96271f;--att:#7e5a12;--acc:#0e5257}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--bg:#0e1114;--carte:#171b21;--bord:#2a3138;--txt:#e3e7ea;--doux:#8b96a2;
--ok:#6bbe92;--ko:#e28078;--att:#d6ac55;--acc:#54bec4}}
:root[data-theme="dark"]{--bg:#0e1114;--carte:#171b21;--bord:#2a3138;--txt:#e3e7ea;
--doux:#8b96a2;--ok:#6bbe92;--ko:#e28078;--att:#d6ac55;--acc:#54bec4}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);font:16px/1.6 system-ui,-apple-system,
"Segoe UI",Roboto,sans-serif}
.p{max-width:900px;margin:0 auto;padding:40px 22px 80px}
h1{font-size:29px;margin:0 0 6px;letter-spacing:-.02em}
.sous{color:var(--doux);font-size:14px;margin-bottom:32px}
h2{font-size:17px;margin:40px 0 14px;letter-spacing:-.01em;
border-bottom:2px solid var(--txt);padding-bottom:8px}
.verdict{background:var(--carte);border:1px solid var(--bord);border-radius:10px;
padding:26px;margin-bottom:26px}
.verdict .lbl{color:var(--doux);font-size:12px;letter-spacing:.09em;text-transform:uppercase}
.gros{font-size:clamp(38px,9vw,62px);font-weight:700;letter-spacing:-.03em;line-height:1.05;
margin:8px 0 2px;font-variant-numeric:tabular-nums}
.gros.ok{color:var(--ok)}.gros.ko{color:var(--ko)}
.pct{font-size:19px;color:var(--doux);font-variant-numeric:tabular-nums}
.cmp{display:flex;flex-wrap:wrap;gap:22px;margin-top:20px;padding-top:18px;
border-top:1px solid var(--bord)}
.cmp div{flex:1 1 150px}
.cmp .v{font-size:20px;font-weight:600;font-variant-numeric:tabular-nums}
.cmp .k{color:var(--doux);font-size:12px;margin-top:2px}
.grille{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));
margin:18px 0}
.kpi{background:var(--carte);border:1px solid var(--bord);border-radius:8px;padding:13px 15px}
.kpi .k{color:var(--doux);font-size:11px;letter-spacing:.07em;text-transform:uppercase}
.kpi .v{font-size:20px;font-weight:600;margin-top:4px;font-variant-numeric:tabular-nums}
.kpi .v.ok{color:var(--ok)}.kpi .v.ko{color:var(--ko)}.kpi .v.att{color:var(--att)}
.note{background:var(--carte);border-left:3px solid var(--acc);border-radius:0 8px 8px 0;
padding:16px 20px;margin:20px 0}
.note.alerte{border-left-color:var(--ko)}
.note .t{display:block;font-weight:700;margin-bottom:6px}
table{width:100%;border-collapse:collapse;font-size:14px;background:var(--carte);
border:1px solid var(--bord);border-radius:8px;overflow:hidden}
th{text-align:left;color:var(--doux);font-size:11px;letter-spacing:.06em;
text-transform:uppercase;padding:11px 15px;border-bottom:1px solid var(--bord)}
td{padding:11px 15px;border-bottom:1px solid var(--bord)}
tr:last-child td{border-bottom:none}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.enveloppe{background:var(--carte);border:1px solid var(--bord);border-radius:8px;
padding:20px;margin:18px 0}
.barre{position:relative;height:38px;margin:16px 0 8px}
.barre .piste{position:absolute;top:15px;left:0;right:0;height:8px;border-radius:4px;
background:linear-gradient(90deg,var(--ko),var(--att),var(--ok))}
.barre .marq{position:absolute;top:6px;width:3px;height:26px;background:var(--txt);
border-radius:2px}
.barre .zero{position:absolute;top:2px;width:1px;height:34px;background:var(--doux);
opacity:.6}
.bornes{display:flex;justify-content:space-between;color:var(--doux);font-size:12.5px;
font-variant-numeric:tabular-nums}
.pied{color:var(--doux);font-size:13px;margin-top:44px;padding-top:18px;
border-top:1px solid var(--bord)}
pre{background:var(--carte);border:1px solid var(--bord);border-radius:8px;padding:14px 16px;
overflow-x:auto;font-size:12.5px;line-height:1.55;font-family:ui-monospace,Menlo,Consolas,
monospace;color:var(--doux)}
"""


def _euro(v: Any) -> str:
    try:
        return f"{float(v):,.2f} €".replace(",", " ")
    except (TypeError, ValueError):
        return "n/a"


def _pct(v: Any, nd: int = 2) -> str:
    try:
        f = float(v)
        return "n/a" if not math.isfinite(f) else f"{f:+.{nd}%}"
    except (TypeError, ValueError):
        return "n/a"


def _num(v: Any, nd: int = 2) -> str:
    try:
        f = float(v)
        return "n/a" if not math.isfinite(f) else f"{f:,.{nd}f}"
    except (TypeError, ValueError):
        return "n/a"


def _courbe(valeurs: Sequence[float], capital: float, largeur: int = 860,
            hauteur: int = 190) -> str:
    v = [float(x) for x in valeurs if isinstance(x, (int, float)) and math.isfinite(x)]
    if len(v) < 2:
        return '<p style="color:var(--doux)">Pas assez de points pour tracer.</p>'
    # Sous-échantillonnage : au-delà de ~900 points, le SVG grossit sans rien montrer de plus.
    if len(v) > 900:
        pas = len(v) // 900 + 1
        v = v[::pas]
    lo, hi = min(min(v), capital), max(max(v), capital)
    if hi - lo < 1e-9:
        hi, lo = hi + 1, lo - 1
    marge = 22
    ih = hauteur - 2 * marge

    def y(val: float) -> float:
        return marge + ih * (1 - (val - lo) / (hi - lo))

    pas_x = (largeur - 2 * marge) / (len(v) - 1)
    pts = " ".join(f"{marge + i * pas_x:.1f},{y(x):.1f}" for i, x in enumerate(v))
    couleur = "var(--ok)" if v[-1] >= capital else "var(--ko)"
    y0 = y(capital)
    return (
        f'<svg viewBox="0 0 {largeur} {hauteur}" width="100%" height="{hauteur}" '
        f'preserveAspectRatio="none" role="img" aria-label="Courbe du capital">'
        f'<line x1="{marge}" y1="{y0:.1f}" x2="{largeur - marge}" y2="{y0:.1f}" '
        f'stroke="var(--doux)" stroke-width="1" stroke-dasharray="4,4"/>'
        f'<polygon points="{marge},{y0:.1f} {pts} {largeur - marge},{y0:.1f}" '
        f'fill="{couleur}" opacity="0.14"/>'
        f'<polyline points="{pts}" fill="none" stroke="{couleur}" stroke-width="2"/>'
        f'<text x="{marge}" y="14" fill="var(--doux)" font-size="11">'
        f'{_euro(hi)}</text>'
        f'<text x="{marge}" y="{hauteur - 4}" fill="var(--doux)" font-size="11">'
        f'{_euro(lo)}</text>'
        f'<text x="{largeur - marge}" y="{y0 - 6:.1f}" fill="var(--doux)" font-size="11" '
        f'text-anchor="end">capital de départ</text></svg>')


def _enveloppe(bas: float, med: float, haut: float, capital: float) -> str:
    """Barre montrant où tombe le résultat et où se situe zéro."""
    lo, hi = min(bas, 0.0), max(haut, 0.0)
    if hi - lo < 1e-9:
        hi, lo = hi + 0.01, lo - 0.01
    def pos(x: float) -> float:
        return 100.0 * (x - lo) / (hi - lo)
    entier_neg = haut < 0
    entier_pos = bas > 0
    if entier_neg:
        msg = ("<b class=\"t\">Ce n'est pas de la malchance.</b>"
               "L'intervalle <b>entier</b> est négatif : le résultat est systématique.")
        cls = " alerte"
    elif entier_pos:
        msg = ("<b class=\"t\">Résultat inhabituellement solide.</b>"
               "L'intervalle entier est positif — vérifier qu'aucune donnée future n'a fuité.")
        cls = ""
    else:
        msg = ("<b class=\"t\">Ce résultat ne prouve rien.</b>"
               "L'intervalle contient zéro : ni dans un sens, ni dans l'autre.")
        cls = ""
    return (
        '<div class="enveloppe">'
        '<div style="color:var(--doux);font-size:12px;letter-spacing:.08em;'
        'text-transform:uppercase">Et si cette période s\'était déroulée un peu autrement ?</div>'
        '<div class="barre"><div class="piste"></div>'
        f'<div class="zero" style="left:{pos(0.0):.1f}%"></div>'
        f'<div class="marq" style="left:{pos(med):.1f}%"></div></div>'
        f'<div class="bornes"><span>{_euro(capital * (1 + bas))} ({_pct(bas)})</span>'
        f'<span>médiane {_euro(capital * (1 + med))}</span>'
        f'<span>{_euro(capital * (1 + haut))} ({_pct(haut)})</span></div>'
        f'<div class="note{cls}" style="margin:16px 0 0">{msg}</div></div>')


# =======================================================================================
def rapport_html(r: Dict[str, Any]) -> str:
    capital = float(r.get("capital", 1000.0))
    final = float(r.get("final", capital))
    gain = final - capital
    bh = float(r.get("buy_hold", capital))
    brut = float(r.get("brut_sans_frais", final))
    bas, med, haut = (float(r.get("ci_bas", 0.0)), float(r.get("ci_median", 0.0)),
                      float(r.get("ci_haut", 0.0)))
    cls = "ok" if gain >= 0 else "ko"

    frais = float(r.get("frais_annuels", 0.0))
    dsr = float(r.get("deflated_sharpe", 0.0))

    lecture = []
    if brut > capital and final < capital:
        lecture.append(
            "<b class=\"t\">Le modèle a un peu de flair, les frais le mangent entièrement.</b>"
            f" Sans aucun frais il aurait fait {_euro(brut)} ; avec les frais réels, "
            f"{_euro(final)}. Le signal existe, il est trop petit pour payer le trading "
            "qu'il déclenche. Le levier le plus efficace n'est pas un meilleur modèle : "
            "c'est <b>trader beaucoup moins souvent</b>.")
    elif final >= capital and dsr < 0.95:
        lecture.append(
            "<b class=\"t\">Résultat positif, mais pas établi.</b> Le Deflated Sharpe vaut "
            f"{_num(dsr, 3)} — en dessous de 0.95, on ne distingue pas ce résultat du "
            "hasard une fois corrigé du nombre d'essais. Ne pas déployer sur cette base.")
    elif final < capital:
        lecture.append(
            "<b class=\"t\">Le modèle perd de l'argent sur une période qu'il n'avait jamais vue.</b>"
            "C'est le résultat le plus fréquent, et il est probablement juste : le marché "
            "ne doit rien à personne.")
    if frais > 0.05:
        lecture.append(
            f"<b class=\"t\">Les frais ont coûté {_pct(frais).lstrip(chr(43))} du capital sur l'année.</b>"
            "Au-delà de 5 %, c'est presque toujours la fréquence de trading qui est en "
            "cause, pas la qualité des prédictions.")

    kpis = [
        ("Sharpe", _num(r.get("sharpe"), 2),
         "ok" if float(r.get("sharpe", 0)) > 0 else "ko"),
        ("Drawdown max", _pct(r.get("drawdown")), "ko"),
        ("Transactions", f"{int(r.get('n_trades', 0)):,}".replace(",", " "), ""),
        ("Frais / an", _pct(frais), "att" if frais > 0.05 else ""),
        ("Taux de réussite", _pct(r.get("hit_rate"), 1), ""),
        ("Deflated Sharpe", _num(dsr, 3), "ok" if dsr >= 0.95 else "ko"),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><div class="k">{html.escape(k)}</div>'
        f'<div class="v {c}">{v}</div></div>' for k, v, c in kpis)

    genere = datetime.now(timezone.utc).strftime("%d/%m/%Y à %H:%M UTC")
    sonde = html.escape(str(r.get("sonde", "")))

    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rapport QBot</title><style>{_CSS}</style></head><body><div class="p">

<h1>Rapport QBot</h1>
<div class="sous">Données du {html.escape(str(r.get('periode', '')))} ·
{int(r.get('n_barres', 0)):,} barres ·
période de test du {html.escape(str(r.get('test_debut', '')))} au
{html.escape(str(r.get('test_fin', '')))}, jamais vue par le modèle ·
généré le {genere}</div>

<div class="verdict">
  <div class="lbl">{_euro(capital)} placés sur la période de test</div>
  <div class="gros {cls}">{_euro(final)}</div>
  <div class="pct">{'+' if gain >= 0 else ''}{_euro(gain).replace('+', '')} —
     {_pct(final / capital - 1)}</div>
  <div class="cmp">
    <div><div class="v">{_euro(bh)}</div><div class="k">acheter et conserver</div></div>
    <div><div class="v">{_euro(capital)}</div><div class="k">ne rien faire</div></div>
    <div><div class="v">{_euro(brut)}</div><div class="k">le bot, frais mis à zéro</div></div>
  </div>
</div>

{_enveloppe(bas, med, haut, capital)}

<h2>Comment lire ce résultat</h2>
{''.join(f'<div class="note">{t}</div>' for t in lecture) if lecture else
 '<div class="note">Résultat sans signal particulier à commenter.</div>'}

<h2>Le détail</h2>
<div class="grille">{kpi_html}</div>

<h2>Évolution du capital</h2>
{_courbe(r.get('equity', []), capital)}

<h2>Le mur des coûts</h2>
<p>Spread 1 pip + commission ECN = <b>{_num(r.get('cout_bps'), 2)} bps</b> par unité de
turnover. Volatilité annualisée du marché : <b>{_pct(r.get('vol_marche'), 2).lstrip('+')}</b>.</p>
<table><tr><th>Si le bot trade</th><th class="n">Frais par an</th>
<th class="n">Sharpe à produire pour les couvrir</th></tr>
{''.join(f'<tr><td>{lab}</td><td class="n">{_pct(t * float(r.get("cout_bps", 1.11)) / 1e4 * 6240).lstrip("+")}</td>'
         f'<td class="n">{_num(t * float(r.get("cout_bps", 1.11)) / 1e4 * 6240 / max(float(r.get("vol_marche", 0.06)), 1e-9), 2)}</td></tr>'
         for lab, t in [("à chaque barre", 1.0), ("une fois par jour", 1 / 24),
                        ("une fois par semaine", 1 / 120)])}
</table>
<p style="color:var(--doux);font-size:14px">Les meilleurs fonds du monde tournent à un
Sharpe de 2 à 3. Un bot qui trade à chaque heure devrait en produire bien davantage
<b>avant</b> de gagner le premier euro. C'est la raison la plus fréquente pour laquelle
un bot retail perd : pas de mauvaises prédictions, trop de transactions.</p>

<h2>Les stratégies classiques</h2>
<p><b>{html.escape(str(r.get('survivantes', 'n/a')))}</b> hypothèses classiques survivent
au criblage, frais compris.</p>

<h2>Y avait-il un signal ?</h2>
<pre>{sonde}</pre>

<div class="pied">
Ce rapport est un outil de recherche. Un résultat positif sur une période ne prouve pas
qu'il se reproduira : c'est une trajectoire parmi celles qui étaient possibles, et
l'intervalle affiché plus haut dit à quel point elle aurait pu être différente.
Rien ici ne constitue un conseil en investissement. Le trading à effet de levier fait
perdre de l'argent à la majorité des comptes particuliers.
</div>
</div></body></html>"""


def ecrire_rapport(resultats: Dict[str, Any], chemin: str | Path) -> Path:
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(rapport_html(resultats), encoding="utf-8")
    return chemin

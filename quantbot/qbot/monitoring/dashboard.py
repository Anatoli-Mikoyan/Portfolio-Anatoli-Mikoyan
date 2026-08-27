"""Tableau de bord HTML autonome (cahier des charges §17).

Un seul fichier, aucune dépendance : ni CDN, ni serveur, ni bibliothèque de graphiques.
Les courbes sont du SVG produit à la main. Trois raisons, toutes opérationnelles :

  * un tableau de bord qui dépend du réseau est indisponible exactement le jour où
    quelque chose ne va pas ;
  * un fichier unique s'archive, s'envoie par courriel et se relit dans deux ans ;
  * aucune donnée de compte ne sort de la machine.

Le parti pris de lecture : ce qui va mal remonte en haut. Les alertes sont au-dessus des
courbes, et le bandeau supérieur donne un état global lisible en une seconde. Un tableau
de bord où il faut chercher l'anomalie n'est pas un tableau de bord, c'est une archive.
"""
from __future__ import annotations

import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = ["render_dashboard", "dashboard_html"]

_CSS = """
:root{--bg:#0e1116;--panel:#161b22;--line:#262d38;--txt:#e6edf3;--dim:#8b949e;
--ok:#3fb950;--warn:#d29922;--crit:#f85149;--accent:#58a6ff;--grid:#21262d}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);
font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.wrap{max-width:1180px;margin:0 auto;padding:20px}
h1{font-size:19px;margin:0 0 2px;letter-spacing:.02em}
.sub{color:var(--dim);font-size:12px;margin-bottom:18px}
.banner{padding:12px 16px;border-radius:8px;margin-bottom:18px;font-weight:600;
border:1px solid}
.banner.ok{background:rgba(63,185,80,.10);border-color:var(--ok);color:var(--ok)}
.banner.warn{background:rgba(210,153,34,.10);border-color:var(--warn);color:var(--warn)}
.banner.crit{background:rgba(248,81,73,.12);border-color:var(--crit);color:var(--crit)}
.grid{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(178px,1fr));
margin-bottom:18px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.kpi .k{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.06em}
.kpi .v{font-size:21px;font-weight:600;margin-top:4px}
.kpi .n{color:var(--dim);font-size:11px;margin-top:2px}
.v.ok{color:var(--ok)}.v.warn{color:var(--warn)}.v.crit{color:var(--crit)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;
padding:14px 16px;margin-bottom:18px}
.panel h2{font-size:13px;margin:0 0 12px;color:var(--dim);text-transform:uppercase;
letter-spacing:.08em;font-weight:600}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th{text-align:left;color:var(--dim);font-weight:600;padding:5px 8px;
border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:5px 8px;border-bottom:1px solid var(--grid)}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
.tag{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;font-weight:600}
.tag.ok{background:rgba(63,185,80,.14);color:var(--ok)}
.tag.warn{background:rgba(210,153,34,.14);color:var(--warn)}
.tag.crit{background:rgba(248,81,73,.16);color:var(--crit)}
.tag.info{background:rgba(88,166,255,.14);color:var(--accent)}
.scroll{overflow-x:auto}
.empty{color:var(--dim);font-style:italic;padding:6px 2px}
.foot{color:var(--dim);font-size:11px;margin-top:22px;border-top:1px solid var(--line);
padding-top:12px}
.bar{height:6px;background:var(--grid);border-radius:3px;overflow:hidden;margin-top:6px}
.bar>i{display:block;height:100%}
"""


# =======================================================================================
# Graphiques SVG
# =======================================================================================
def _sparkline(values: Sequence[float], width: int = 1100, height: int = 130,
               color: str = "#58a6ff", fill: bool = True, zero: bool = False,
               label_fmt: str = "{:.4g}") -> str:
    """Courbe SVG minimale, échelle automatique, sans dépendance."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size < 2:
        return '<div class="empty">Pas assez de points pour tracer.</div>'

    lo, hi = float(v.min()), float(v.max())
    if zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    if hi - lo < 1e-15:
        hi, lo = hi + 1e-9, lo - 1e-9
    pad = 16
    inner_h = height - 2 * pad

    def y(val: float) -> float:
        return pad + inner_h * (1.0 - (val - lo) / (hi - lo))

    xs = np.linspace(pad, width - pad, v.size)
    pts = " ".join(f"{x:.1f},{y(val):.1f}" for x, val in zip(xs, v))

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
             f'preserveAspectRatio="none" role="img">']
    for frac in (0.0, 0.5, 1.0):
        gy = pad + inner_h * frac
        parts.append(f'<line x1="{pad}" y1="{gy:.1f}" x2="{width - pad}" y2="{gy:.1f}" '
                     f'stroke="#21262d" stroke-width="1"/>')
    if zero and lo < 0.0 < hi:
        parts.append(f'<line x1="{pad}" y1="{y(0.0):.1f}" x2="{width - pad}" '
                     f'y2="{y(0.0):.1f}" stroke="#8b949e" stroke-width="1" '
                     f'stroke-dasharray="3,3"/>')
    if fill:
        base = y(max(lo, 0.0)) if zero else height - pad
        parts.append(f'<polygon points="{pad},{base:.1f} {pts} {width - pad},{base:.1f}" '
                     f'fill="{color}" opacity="0.13"/>')
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                 f'stroke-width="1.8" stroke-linejoin="round"/>')
    parts.append(f'<text x="{pad}" y="11" fill="#8b949e" font-size="10">'
                 f'{html.escape(label_fmt.format(hi))}</text>')
    parts.append(f'<text x="{pad}" y="{height - 3}" fill="#8b949e" font-size="10">'
                 f'{html.escape(label_fmt.format(lo))}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _histogram(values: Sequence[float], bins: int = 40, width: int = 1100,
               height: int = 120, color: str = "#58a6ff") -> str:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size < 10:
        return '<div class="empty">Pas assez de points pour un histogramme.</div>'
    counts, edges = np.histogram(v, bins=bins)
    pad, top = 16, float(counts.max()) or 1.0
    bw = (width - 2 * pad) / len(counts)
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
             f'preserveAspectRatio="none">']
    for i, c in enumerate(counts):
        h = (height - 2 * pad) * (c / top)
        x = pad + i * bw
        neg = edges[i] < 0
        parts.append(f'<rect x="{x:.1f}" y="{height - pad - h:.1f}" width="{max(bw - 1, 1):.1f}" '
                     f'height="{h:.1f}" fill="{"#f85149" if neg else color}" opacity="0.75"/>')
    parts.append(f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" '
                 f'y2="{height - pad}" stroke="#262d38"/>')
    parts.append("</svg>")
    return "".join(parts)


# =======================================================================================
# Rendu
# =======================================================================================
def _fmt(v: Any, kind: str = "num", nd: int = 3) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return html.escape(str(v)) if v not in (None, "") else "n/a"
    if not math.isfinite(f):
        return "n/a"
    if kind == "pct":
        return f"{f:.2%}"
    if kind == "int":
        return f"{int(f):,}"
    if kind == "money":
        return f"{f:,.2f}"
    return f"{f:,.{nd}f}"


def _kpi(key: str, value: str, note: str = "", tone: str = "") -> str:
    cls = f" {tone}" if tone else ""
    note_html = f'<div class="n">{html.escape(note)}</div>' if note else ""
    return (f'<div class="kpi"><div class="k">{html.escape(key)}</div>'
            f'<div class="v{cls}">{value}</div>{note_html}</div>')


def _tone_from(value: float, warn: float, crit: float, higher_is_worse: bool = True) -> str:
    if not math.isfinite(value):
        return ""
    if higher_is_worse:
        return "crit" if value >= crit else ("warn" if value >= warn else "ok")
    return "crit" if value <= crit else ("warn" if value <= warn else "ok")


def dashboard_html(snapshot: Dict[str, Any], curves: Dict[str, List[float]],
                   returns: Optional[Sequence[float]] = None,
                   title: str = "QBot — supervision",
                   text_report: str = "") -> str:
    """Construit la page complète à partir d'un instantané et des séries."""
    alerts = snapshot.get("alerts") or {}
    worst = alerts.get("worst_level")
    recent = alerts.get("recent") or []
    n_recent_crit = sum(1 for a in recent if a.get("level") == "critical")

    if n_recent_crit:
        banner_cls, banner = "crit", f"‼ {n_recent_crit} alerte(s) critique(s) récente(s)"
    elif any(a.get("level") == "warn" for a in recent):
        banner_cls, banner = "warn", "! Points de vigilance signalés"
    elif worst == "critical":
        banner_cls, banner = "warn", "Alertes critiques dans l'historique, rien de récent"
    else:
        banner_cls, banner = "ok", "✓ Aucune anomalie signalée"

    dd = abs(float(snapshot.get("drawdown") or 0.0))
    sharpe = snapshot.get("sharpe_rolling", float("nan"))
    lat = snapshot.get("p99_latency_ms", float("nan"))

    kpis = [
        _kpi("Équité", _fmt(snapshot.get("equity"), "money")),
        _kpi("Rendement cumulé", _fmt(snapshot.get("total_return"), "pct"),
             tone="ok" if float(snapshot.get("total_return") or 0) >= 0 else "crit"),
        _kpi("Drawdown courant", _fmt(snapshot.get("drawdown"), "pct"),
             f"max {_fmt(snapshot.get('max_drawdown'), 'pct')}",
             _tone_from(dd, 0.08, 0.15)),
        _kpi("Sharpe glissant", _fmt(sharpe, nd=2),
             f"global {_fmt(snapshot.get('sharpe'), nd=2)}",
             _tone_from(float(sharpe) if sharpe is not None else float("nan"), 0.0, -1.0, False)),
        _kpi("Sortino", _fmt(snapshot.get("sortino"), nd=2)),
        _kpi("Volatilité ann.", _fmt(snapshot.get("ann_volatility"), "pct")),
        _kpi("Taux de réussite", _fmt(snapshot.get("hit_rate"), "pct"),
             f"profit factor {_fmt(snapshot.get('profit_factor'), nd=2)}"),
        _kpi("Exposition", _fmt(snapshot.get("exposure"), "pct"),
             f"à plat {_fmt(snapshot.get('flat_rate'), 'pct')}"),
        _kpi("Turnover / barre", _fmt(snapshot.get("turnover_per_bar"), nd=4),
             f"{_fmt(snapshot.get('n_trades'), 'int')} transactions"),
        _kpi("Décisions contraintes", _fmt(snapshot.get("constraint_rate"), "pct"),
             "par les garde-fous",
             _tone_from(float(snapshot.get("constraint_rate") or 0), 0.25, 0.40)),
        _kpi("Latence p99", _fmt(lat, nd=0) + " ms",
             f"max {_fmt(snapshot.get('max_latency_ms'), nd=0)} ms · "
             f"hors délai {_fmt(snapshot.get('latency_breach_rate'), 'pct')}",
             _tone_from(float(lat) if lat is not None else float("nan"), 250.0, 1000.0)),
        _kpi("Barres observées", _fmt(snapshot.get("n_bars"), "int"),
             f"régime : {html.escape(str((snapshot.get('regime') or {}).get('label') or '—'))}"),
    ]

    # ---- alertes -----------------------------------------------------------------------
    if recent:
        rows = "".join(
            f'<tr><td><span class="tag {html.escape(str(a.get("level", "info")))}">'
            f'{html.escape(str(a.get("level", "")))}</span></td>'
            f'<td>{html.escape(str(a.get("code", "")))}</td>'
            f'<td>{html.escape(str(a.get("message", "")))}</td>'
            f'<td class="num">{_fmt(a.get("value"), nd=3)}</td>'
            f'<td class="num">{_fmt(a.get("threshold"), nd=3)}</td>'
            f'<td class="num">{_fmt(a.get("bar"), "int")}</td></tr>'
            for a in reversed(recent))
        alerts_html = ('<div class="scroll"><table><tr><th>Niveau</th><th>Code</th>'
                       '<th>Message</th><th class="num">Valeur</th><th class="num">Seuil</th>'
                       f'<th class="num">Barre</th></tr>{rows}</table></div>'
                       f'<div class="n" style="color:var(--dim);margin-top:8px">'
                       f'{_fmt(alerts.get("n_alerts"), "int")} alertes émises, '
                       f'{_fmt(alerts.get("n_suppressed"), "int")} répétitions supprimées '
                       f'par temporisation.</div>')
    else:
        alerts_html = '<div class="empty">Aucune alerte émise.</div>'

    # ---- dérive ------------------------------------------------------------------------
    drift = snapshot.get("drift")
    if drift:
        feats = sorted(drift.get("features", []),
                       key=lambda f: (-(f.get("psi") if math.isfinite(f.get("psi", float("nan")))
                                        else -1)))[:15]
        rows = "".join(
            f'<tr><td>{html.escape(str(f.get("name", "")))}</td>'
            f'<td class="num">{_fmt(f.get("psi"), nd=4)}</td>'
            f'<td class="num">{_fmt(f.get("js"), nd=4)}</td>'
            f'<td class="num">{_fmt(f.get("ks_pvalue"), nd=4)}</td>'
            f'<td class="num">{_fmt(f.get("z_shift"), nd=2)}</td>'
            f'<td class="num">{_fmt(f.get("ref_mean"), nd=3)}</td>'
            f'<td class="num">{_fmt(f.get("live_mean"), nd=3)}</td>'
            f'<td><span class="tag {"crit" if f.get("verdict") == "critique" else ("warn" if f.get("verdict") == "modéré" else "ok")}">'
            f'{html.escape(str(f.get("verdict", "")))}</span></td></tr>'
            for f in feats)
        status = str(drift.get("status", ""))
        tone = "crit" if status == "critique" else ("warn" if status == "modéré" else "ok")
        drift_html = (
            f'<div style="margin-bottom:10px">État : <span class="tag {tone}">{html.escape(status)}</span> '
            f'&nbsp;·&nbsp; score global {_fmt(drift.get("global_score"), nd=4)} '
            f'&nbsp;·&nbsp; {_fmt(drift.get("n_critical"), "int")} critiques, '
            f'{_fmt(drift.get("n_moderate"), "int")} modérées '
            f'&nbsp;·&nbsp; fenêtre de {_fmt(drift.get("n_live"), "int")} barres</div>'
            '<div class="scroll"><table><tr><th>Feature</th><th class="num">PSI</th>'
            '<th class="num">JS</th><th class="num">p (KS)</th><th class="num">Δ (σ)</th>'
            '<th class="num">µ réf.</th><th class="num">µ live</th><th>Verdict</th></tr>'
            f'{rows}</table></div>')
    else:
        drift_html = ('<div class="empty">Aucun verdict de dérive : distribution de '
                      'référence absente ou fenêtre insuffisamment remplie.</div>')

    # ---- attendu vs réalisé --------------------------------------------------------------
    rec = snapshot.get("reconciliation")
    if rec:
        rec_html = "".join([
            f'<div style="margin-bottom:8px"><span class="tag '
            f'{"crit" if rec.get("degraded") and rec.get("sequential_alarm") else ("warn" if rec.get("degraded") or rec.get("sequential_alarm") else "ok")}">'
            f'{html.escape(str(rec.get("verdict", "")))}</span></div>',
            '<div class="scroll"><table>',
            f'<tr><th>Sharpe réalisé</th><td class="num">{_fmt(rec.get("live_sharpe"), nd=3)}</td>'
            f'<th>Sharpe attendu</th><td class="num">{_fmt(rec.get("expected_sharpe"), nd=3)}</td></tr>',
            f'<tr><th>Centile du Sharpe</th><td class="num">{_fmt(rec.get("sharpe_percentile"), "pct")}</td>'
            f'<th>Centile du rendement</th><td class="num">{_fmt(rec.get("return_percentile"), "pct")}</td></tr>',
            f'<tr><th>Drawdown réalisé</th><td class="num">{_fmt(rec.get("live_drawdown"), "pct")}</td>'
            f'<th>Statistique séquentielle</th><td class="num">{_fmt(rec.get("sequential_stat"), nd=2)}</td></tr>',
            "</table></div>",
        ])
    else:
        rec_html = ('<div class="empty">Enveloppe de performance non fournie : '
                    'la confrontation attendu/réalisé est inactive.</div>')

    # ---- coûts ---------------------------------------------------------------------------
    tca = snapshot.get("tca")
    if tca and int(tca.get("n_fills") or 0) > 0:
        ratio = float(tca.get("cost_ratio") or float("nan"))
        tone = _tone_from(ratio, 1.5, 2.5)
        tca_html = "".join([
            f'<div style="margin-bottom:8px"><span class="tag {tone or "ok"}">'
            f'{html.escape(str(tca.get("verdict", "")))}</span></div>',
            '<div class="scroll"><table>',
            f'<tr><th>Exécutions</th><td class="num">{_fmt(tca.get("n_fills"), "int")}</td>'
            f'<th>Shortfall moyen</th><td class="num">{_fmt(tca.get("mean_is_bps"), nd=2)} bps</td></tr>',
            f'<tr><th>Délai</th><td class="num">{_fmt(tca.get("mean_delay_bps"), nd=2)} bps</td>'
            f'<th>Spread</th><td class="num">{_fmt(tca.get("mean_spread_bps"), nd=2)} bps</td></tr>',
            f'<tr><th>Commission</th><td class="num">{_fmt(tca.get("mean_commission_bps"), nd=2)} bps</td>'
            f'<th>Impact / résidu</th><td class="num">{_fmt(tca.get("mean_impact_bps"), nd=2)} bps</td></tr>',
            f'<tr><th>Coût modélisé</th><td class="num">{_fmt(tca.get("mean_expected_bps"), nd=2)} bps</td>'
            f'<th>Ratio réalisé / prévu</th><td class="num">{_fmt(ratio, nd=2)}×</td></tr>',
            f'<tr><th>Excès (p-value)</th><td class="num">{_fmt(tca.get("excess_bps"), nd=2)} bps '
            f'({_fmt(tca.get("excess_pvalue"), nd=4)})</td>'
            f'<th>Rabais de Sharpe</th><td class="num">{_fmt(tca.get("sharpe_haircut"), nd=3)}</td></tr>',
            f'<tr><th>Turnover par barre</th><td class="num">{_fmt(tca.get("turnover_per_bar"), nd=4)}</td>'
            f'<th>Coût excédentaire annuel</th><td class="num">{_fmt(tca.get("excess_cost_annual"), "pct")}</td></tr>',
            "</table></div>",
        ])
    else:
        tca_html = '<div class="empty">Aucune exécution enregistrée.</div>'

    # ---- journal ---------------------------------------------------------------------------
    jr = snapshot.get("journal")
    if jr:
        ok = jr.get("verified")
        tag = "ok" if ok else ("crit" if ok is False else "info")
        state = "intègre" if ok else ("COMPROMIS" if ok is False else "non vérifié")
        journal_html = (
            f'<table><tr><th>Fichier</th><td>{html.escape(str(jr.get("path", "")))}</td></tr>'
            f'<tr><th>Entrées</th><td class="num">{_fmt(jr.get("n_entries"), "int")}</td></tr>'
            f'<tr><th>Empreinte de tête</th><td>{html.escape(str(jr.get("head", ""))[:32])}…</td></tr>'
            f'<tr><th>Chaîne</th><td><span class="tag {tag}">{state}</span></td></tr></table>')
    else:
        journal_html = '<div class="empty">Journal d\'audit désactivé.</div>'

    # ---- statuts ---------------------------------------------------------------------------
    sc = snapshot.get("status_counts") or {}
    total = sum(sc.values()) or 1
    status_html = "".join(
        f'<tr><td>{html.escape(str(k))}</td><td class="num">{v:,}</td>'
        f'<td class="num">{v / total:.1%}</td></tr>' for k, v in sc.items()) or \
        '<tr><td colspan="3" class="empty">aucun</td></tr>'

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    eq = curves.get("equity") or []
    ddc = curves.get("drawdown") or []
    ex = curves.get("exposure") or []

    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{_CSS}</style></head>
<body><div class="wrap">
<h1>{html.escape(title)}</h1>
<div class="sub">Modèle {html.escape(str(snapshot.get('model_id') or 'n/a'))} ·
barre {_fmt(snapshot.get('bar'), 'int')} ·
{html.escape(str(snapshot.get('first_ts') or ''))} → {html.escape(str(snapshot.get('last_ts') or ''))} ·
généré le {generated}</div>

<div class="banner {banner_cls}">{html.escape(banner)}</div>

<div class="grid">{''.join(kpis)}</div>

<div class="panel"><h2>Alertes</h2>{alerts_html}</div>

<div class="panel"><h2>Équité</h2>{_sparkline(eq, color="#58a6ff")}</div>
<div class="panel"><h2>Drawdown</h2>{_sparkline(ddc, color="#f85149", zero=True)}</div>
<div class="panel"><h2>Exposition nette</h2>{_sparkline(ex, color="#3fb950", zero=True, fill=False)}</div>
<div class="panel"><h2>Distribution des rendements par barre</h2>
{_histogram(returns if returns is not None else [])}</div>

<div class="panel"><h2>Dérive des features (entraînement → production)</h2>{drift_html}</div>
<div class="panel"><h2>Performance attendue vs réalisée</h2>{rec_html}</div>
<div class="panel"><h2>Coûts d'exécution</h2>{tca_html}</div>
<div class="panel"><h2>Journal d'audit</h2>{journal_html}</div>
<div class="panel"><h2>Statuts de décision</h2>
<table><tr><th>Statut</th><th class="num">Barres</th><th class="num">Part</th></tr>
{status_html}</table></div>

<div class="foot">
Tableau de bord autonome — aucune ressource externe, aucune donnée transmise.
Les indicateurs statistiques (PSI, KS, Page-Hinkley, enveloppe bootstrap) sont décrits
dans <code>docs/METHODOLOGIE.md</code>. Un Sharpe glissant calculé sur peu de barres est
dominé par le bruit&nbsp;: se référer au nombre de barres avant d'en tirer une conclusion.
</div>
</div></body></html>"""


def render_dashboard(monitor, path: str | Path, title: str = "QBot — supervision") -> Path:
    """Écrit le tableau de bord d'un `LiveMonitor` dans un fichier HTML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = monitor.refresh()
    curves = monitor.store.curves()
    returns = monitor.store.returns().tolist()
    path.write_text(dashboard_html(snapshot, curves, returns, title=title),
                    encoding="utf-8")
    return path

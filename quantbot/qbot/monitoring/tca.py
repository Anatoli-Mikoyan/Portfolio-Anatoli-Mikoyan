"""Analyse des coûts de transaction — TCA (cahier des charges §17).

Le backtest suppose un modèle de coûts. La production révèle les vrais. L'écart entre
les deux est la première cause de mort d'une stratégie qui « marchait sur historique » :
le Sharpe ne s'effondre pas d'un coup, il fond de quelques dixièmes, silencieusement,
jusqu'à passer sous zéro.

La mesure de référence est l'**implementation shortfall** de Perold (1988) : l'écart
entre la performance du portefeuille papier — celui qui se serait exécuté
instantanément au prix observé au moment de la décision — et celle du portefeuille réel.
Elle a un avantage décisif sur la comparaison au VWAP : elle n'est pas manipulable. Un
exécutant peut battre le VWAP en retardant ses ordres ; il ne peut pas battre le prix
qui existait au moment où la décision a été prise.

Décomposition retenue, du plus contrôlable au moins contrôlable :

    IS = coût de délai + coût de spread + impact/résidu

  * **délai** — le prix a bougé entre la décision et l'envoi de l'ordre. Symptôme d'une
    infrastructure lente, d'un modèle trop long à répondre, ou d'un EA qui attend la
    clôture de barre.
  * **spread** — la moitié de la fourchette payée pour franchir le carnet. Compressible
    seulement en changeant de courtier ou de type d'ordre.
  * **impact/résidu** — ce qui reste : l'effet de l'ordre sur le prix, plus le bruit.
    Sur les tailles d'un compte de détail, ce terme est dominé par le bruit ; il ne
    devient une vraie composante d'impact qu'à partir d'une fraction notable du volume.

Le seul chiffre qui compte au bout : de combien faut-il rabattre le Sharpe du backtest
pour que sa politique de coûts corresponde à la réalité (`sharpe_haircut`). Ce chiffre
répond à la seule question qui vaille — « la stratégie survit-elle à ses propres frais
d'exécution ? » — et il est calculable dès la première centaine d'exécutions.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy import stats

__all__ = ["Fill", "TCAReport", "analyse_fills", "slippage_test"]

BPS = 1e4


@dataclass
class Fill:
    """Une exécution, telle qu'on peut la reconstituer depuis le journal + le rapport EA.

    `decision_price` est le prix de référence au moment où le modèle a décidé — c'est
    le point d'ancrage de l'implementation shortfall. `arrival_price` est le prix au
    moment où l'ordre atteint le marché : la différence entre les deux isole le coût de
    délai, imputable au système et non au marché.
    """
    ts: str = ""
    side: int = 0                       # +1 achat, -1 vente
    qty: float = 0.0                    # taille en unités d'exposition (valeur absolue)
    decision_price: float = 0.0
    arrival_price: float = 0.0
    fill_price: float = 0.0
    half_spread: float = 0.0            # demi-fourchette au moment de l'exécution, en prix
    commission: float = 0.0             # en monnaie, pour la taille exécutée
    expected_cost_bps: float = 0.0      # ce que le modèle de coûts du backtest prévoyait
    latency_ms: float = 0.0

    # -- décomposition, en points de base du prix de décision ---------------------------
    @property
    def implementation_shortfall_bps(self) -> float:
        if self.decision_price <= 0 or self.side == 0:
            return float("nan")
        px_cost = (self.fill_price - self.decision_price) * self.side
        commission_px = (self.commission / self.qty) if self.qty > 1e-12 else 0.0
        return float((px_cost + commission_px) / self.decision_price * BPS)

    @property
    def delay_bps(self) -> float:
        if self.decision_price <= 0 or self.arrival_price <= 0 or self.side == 0:
            return float("nan")
        return float((self.arrival_price - self.decision_price) * self.side
                     / self.decision_price * BPS)

    @property
    def spread_bps(self) -> float:
        if self.decision_price <= 0:
            return float("nan")
        return float(self.half_spread / self.decision_price * BPS)

    @property
    def commission_bps(self) -> float:
        if self.decision_price <= 0 or self.qty <= 1e-12:
            return 0.0
        return float(self.commission / self.qty / self.decision_price * BPS)

    @property
    def impact_bps(self) -> float:
        """Résidu : IS − délai − spread − commission. Peut être négatif (exécution favorable)."""
        is_ = self.implementation_shortfall_bps
        if not np.isfinite(is_):
            return float("nan")
        delay = self.delay_bps if np.isfinite(self.delay_bps) else 0.0
        spread = self.spread_bps if np.isfinite(self.spread_bps) else 0.0
        return float(is_ - delay - spread - self.commission_bps)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.update(
            is_bps=self.implementation_shortfall_bps,
            delay_bps=self.delay_bps,
            spread_bps=self.spread_bps,
            commission_bps=self.commission_bps,
            impact_bps=self.impact_bps,
        )
        return d


# =======================================================================================
def slippage_test(realized_bps: np.ndarray, expected_bps: np.ndarray) -> tuple[float, float]:
    """Le coût réalisé dépasse-t-il systématiquement le coût modélisé ?

    Test de Student unilatéral sur la série des écarts (réalisé − modélisé). On teste
    bien la MOYENNE des écarts et non la différence de deux moyennes : chaque exécution
    fournit une paire appariée, ce qui élimine la variance due à la taille et au moment
    des ordres, et rend le test bien plus puissant à nombre d'exécutions égal.

    Retourne (écart moyen en bps, p-value de H₀ : « le modèle n'est pas optimiste »).
    """
    a = np.asarray(realized_bps, dtype=float)
    b = np.asarray(expected_bps, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    if d.size < 3:
        return float(np.mean(d)) if d.size else float("nan"), float("nan")
    sd = d.std(ddof=1)
    if sd < 1e-14:
        return float(d.mean()), 0.0 if d.mean() > 0 else 1.0
    t = d.mean() / (sd / np.sqrt(d.size))
    return float(d.mean()), float(stats.t.sf(t, df=d.size - 1))


@dataclass
class TCAReport:
    n_fills: int
    total_qty: float
    mean_is_bps: float
    median_is_bps: float
    p95_is_bps: float
    mean_delay_bps: float
    mean_spread_bps: float
    mean_commission_bps: float
    mean_impact_bps: float
    mean_expected_bps: float
    cost_ratio: float               # réalisé / modélisé
    excess_bps: float               # réalisé − modélisé
    excess_pvalue: float
    mean_latency_ms: float
    turnover_per_bar: float
    excess_cost_annual: float
    sharpe_haircut: float
    verdict: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:  # pragma: no cover - affichage
        from ..utils.text import render_box

        def f(v: float, unit: str = "", nd: int = 2) -> str:
            return "n/a" if not np.isfinite(v) else f"{v:,.{nd}f}{unit}"

        return render_box("ANALYSE DES COÛTS DE TRANSACTION", [
            (None, [("Exécutions", f"{self.n_fills:,}"),
                    ("Volume total", f(self.total_qty))]),
            ("IMPLEMENTATION SHORTFALL", [("Moyenne", f(self.mean_is_bps, " bps")),
                                          ("Médiane", f(self.median_is_bps, " bps")),
                                          ("95e centile", f(self.p95_is_bps, " bps"))]),
            ("DÉCOMPOSITION", [("Délai (infrastructure)", f(self.mean_delay_bps, " bps")),
                               ("Spread (franchissement)", f(self.mean_spread_bps, " bps")),
                               ("Commission", f(self.mean_commission_bps, " bps")),
                               ("Impact / résidu", f(self.mean_impact_bps, " bps"))]),
            ("CONFRONTATION AU MODÈLE DE BACKTEST", [
                ("Coût modélisé", f(self.mean_expected_bps, " bps")),
                ("Ratio réalisé / prévu", f(self.cost_ratio, "×")),
                ("Excès de coût", f"{f(self.excess_bps, ' bps')}  (p={f(self.excess_pvalue, '', 4)})"),
                ("Turnover par barre", f(self.turnover_per_bar, "", 4)),
                ("Coût excédentaire annuel", "n/a" if not np.isfinite(self.excess_cost_annual)
                 else f"{self.excess_cost_annual:.2%} du capital"),
                ("Rabais de Sharpe induit", f(self.sharpe_haircut, "", 3)),
                ("Latence moyenne", f(self.mean_latency_ms, " ms", 1)),
                ("VERDICT", self.verdict)]),
        ], width=78)

def analyse_fills(fills: Sequence[Fill], ann_volatility: Optional[float] = None,
                  bars_per_year: float = 252.0, n_bars: Optional[int] = None,
                  ratio_warn: float = 1.5, ratio_critical: float = 2.5) -> TCAReport:
    """Agrège une série d'exécutions en un rapport de coûts exploitable.

    `ann_volatility` et `n_bars` servent au rabais de Sharpe : l'excès de coût par barre,
    annualisé et rapporté à la volatilité annuelle de la stratégie, donne directement le
    nombre de points de Sharpe que le backtest surestime. Sans eux, le rabais n'est pas
    calculé — on préfère un NaN explicite à un chiffre inventé.
    """
    fl = [f for f in fills if np.isfinite(f.implementation_shortfall_bps)]
    n = len(fl)
    if n == 0:
        nan = float("nan")
        return TCAReport(
            n_fills=0, total_qty=0.0, mean_is_bps=nan, median_is_bps=nan, p95_is_bps=nan,
            mean_delay_bps=nan, mean_spread_bps=nan, mean_commission_bps=nan,
            mean_impact_bps=nan, mean_expected_bps=nan, cost_ratio=nan, excess_bps=nan,
            excess_pvalue=nan, mean_latency_ms=nan, turnover_per_bar=nan,
            excess_cost_annual=nan, sharpe_haircut=nan, verdict="aucune exécution",
        )

    is_bps = np.array([f.implementation_shortfall_bps for f in fl], dtype=float)
    exp_bps = np.array([f.expected_cost_bps for f in fl], dtype=float)
    qty = np.array([f.qty for f in fl], dtype=float)

    def _m(vals: List[float]) -> float:
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if arr.size else float("nan")

    excess, pval = slippage_test(is_bps, exp_bps)
    mean_exp = _m([f.expected_cost_bps for f in fl])
    ratio = float(is_bps.mean() / mean_exp) if abs(mean_exp) > 1e-9 else float("nan")

    # Rabais de Sharpe : coût excédentaire par barre × barres/an ÷ volatilité annuelle.
    haircut = float("nan")
    turnover = float("nan")
    excess_annual = float("nan")
    if n_bars and n_bars > 0:
        turnover = float(qty.sum() / n_bars)
        # Coût excédentaire annualisé, en fraction du capital : turnover annuel × excès.
        # C'est le chiffre à lire en premier — il se compare directement au rendement
        # espéré, là où le rabais de Sharpe demande de connaître la volatilité.
        excess_annual = float((excess / BPS) * turnover * bars_per_year)
        if ann_volatility and ann_volatility > 1e-9:
            haircut = float(excess_annual / ann_volatility)

    if not np.isfinite(ratio):
        verdict = "modèle de coûts non renseigné"
    elif ratio >= ratio_critical and np.isfinite(pval) and pval < 0.05:
        verdict = "COÛTS RÉELS ≫ MODÈLE (significatif)"
    elif ratio >= ratio_warn and np.isfinite(pval) and pval < 0.10:
        verdict = "coûts réels > modèle"
    elif ratio <= 0.8:
        verdict = "modèle de coûts conservateur"
    else:
        verdict = "coûts conformes au modèle"

    return TCAReport(
        n_fills=n,
        total_qty=float(qty.sum()),
        mean_is_bps=float(is_bps.mean()),
        median_is_bps=float(np.median(is_bps)),
        p95_is_bps=float(np.percentile(is_bps, 95)),
        mean_delay_bps=_m([f.delay_bps for f in fl]),
        mean_spread_bps=_m([f.spread_bps for f in fl]),
        mean_commission_bps=_m([f.commission_bps for f in fl]),
        mean_impact_bps=_m([f.impact_bps for f in fl]),
        mean_expected_bps=mean_exp,
        cost_ratio=ratio,
        excess_bps=excess,
        excess_pvalue=pval,
        mean_latency_ms=_m([f.latency_ms for f in fl]),
        turnover_per_bar=turnover,
        excess_cost_annual=excess_annual,
        sharpe_haircut=haircut,
        verdict=verdict,
    )

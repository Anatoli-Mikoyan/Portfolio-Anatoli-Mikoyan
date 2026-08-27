"""Confrontation entre performance attendue et performance réalisée (§17).

La dérive de covariables (`drift.py`) se voit sur les entrées du modèle. La **dérive de
concept** — les mêmes features ne produisent plus le même résultat — ne se voit nulle
part ailleurs que dans la performance. C'est la dérive qui compte, et c'est la plus
difficile à détecter, pour une raison mathématique nette : le rapport signal/bruit d'une
série de rendements est catastrophique. Une stratégie de Sharpe 1.0 a besoin d'environ
quatre ans de données pour que son Sharpe soit distinguable de zéro à 95 %. Détecter
qu'elle est *passée* de 1.0 à 0.0 demande, naïvement, du même ordre.

Trois outils, du plus lent au plus rapide :

  1. **Enveloppe bootstrap.** On ne compare pas le Sharpe live à celui du backtest — ce
     serait comparer une réalisation à une moyenne. On simule par bootstrap par blocs
     des milliers de trajectoires de la MÊME longueur que la période live, et on regarde
     dans quel centile de cette distribution tombe le résultat réel. Un Sharpe live de
     0.2 face à un backtest à 1.5 n'est pas une anomalie si un tiers des trajectoires
     simulées de 200 barres font pire.
  2. **Test séquentiel de Page-Hinkley** sur les rendements standardisés. Il ne demande
     pas d'attendre la fin d'une fenêtre : il accumule et déclenche dès que le cumul est
     inexplicable. C'est le détecteur le plus rapide à taux de fausse alarme donné.
  3. **Rejeu des décisions.** On réexécute les entrées journalisées à travers le modèle
     courant et on compte les désaccords. Zéro désaccord attendu ; toute valeur non
     nulle signale un décalage de version, de configuration ou de features — une panne
     qui ne se manifeste par aucune erreur et que rien d'autre ne détecte.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ..backtest.metrics import max_drawdown, sharpe_ratio
from ..validation.monte_carlo import stationary_bootstrap_indices
from .drift import PageHinkley

__all__ = ["PerformanceEnvelope", "ReconciliationReport", "reconcile",
           "DegradationDetector", "replay_mismatch", "sharpe_drop_to_sigma"]


# =======================================================================================
@dataclass
class PerformanceEnvelope:
    """Distribution de référence des résultats attendus, à horizon donné.

    Construite UNE FOIS à partir des rendements hors échantillon du backtest, puis
    figée avec le modèle. La reconstruire à partir des données récentes reviendrait à
    déplacer la cible : une stratégie qui se dégrade verrait son enveloppe se dégrader
    avec elle et resterait éternellement « conforme ».
    """
    horizon: int
    sharpe_quantiles: Dict[str, float]
    return_quantiles: Dict[str, float]
    drawdown_quantiles: Dict[str, float]
    mean_return: float
    std_return: float
    n_reference: int
    mean_block: int
    n_paths: int

    QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)

    @classmethod
    def build(cls, backtest_returns: np.ndarray, horizon: int, bars_per_year: float = 252.0,
              n_paths: int = 2000, mean_block: Optional[int] = None,
              seed: int = 0) -> "PerformanceEnvelope":
        r = np.asarray(backtest_returns, dtype=float)
        r = r[np.isfinite(r)]
        if r.size < 50:
            raise ValueError("Au moins 50 rendements de référence sont nécessaires.")
        horizon = int(max(horizon, 10))
        # Longueur de bloc ~ n^(1/3) : compromis usuel entre conservation de la structure
        # de dépendance (blocs longs) et variabilité du rééchantillonnage (blocs courts).
        block = int(mean_block or max(5, round(r.size ** (1.0 / 3.0))))
        rng = np.random.default_rng(seed)

        sharpes = np.empty(n_paths)
        totals = np.empty(n_paths)
        dds = np.empty(n_paths)
        for k in range(n_paths):
            idx = stationary_bootstrap_indices(r.size, block, rng, size=horizon)
            path = r[idx]
            sharpes[k] = sharpe_ratio(path, bars_per_year)
            totals[k] = float(np.prod(1.0 + path) - 1.0)
            dds[k] = max_drawdown(path)

        def q(a: np.ndarray) -> Dict[str, float]:
            return {f"q{int(p * 100):02d}": float(np.quantile(a, p)) for p in cls.QUANTILES}

        return cls(horizon=horizon, sharpe_quantiles=q(sharpes), return_quantiles=q(totals),
                   drawdown_quantiles=q(dds), mean_return=float(r.mean()),
                   std_return=float(r.std(ddof=1)), n_reference=int(r.size),
                   mean_block=block, n_paths=int(n_paths))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "PerformanceEnvelope":
        raw = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        return cls(**raw)


def _percentile_of(value: float, quantiles: Dict[str, float]) -> float:
    """Centile approché d'une valeur dans une distribution résumée par ses quantiles."""
    levels = sorted((int(k[1:]) / 100.0, v) for k, v in quantiles.items())
    ps = np.array([p for p, _ in levels])
    vs = np.array([v for _, v in levels])
    if value <= vs[0]:
        return float(ps[0])
    if value >= vs[-1]:
        return float(ps[-1])
    return float(np.interp(value, vs, ps))


# =======================================================================================
@dataclass
class ReconciliationReport:
    n_live: int
    live_sharpe: float
    live_return: float
    live_drawdown: float
    expected_sharpe: float          # médiane de l'enveloppe
    sharpe_percentile: float
    return_percentile: float
    drawdown_percentile: float
    degraded: bool
    sequential_alarm: bool
    sequential_stat: float
    verdict: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:  # pragma: no cover - affichage
        from ..utils.text import render_box

        def f(v: float, nd: int = 3) -> str:
            return "n/a" if not np.isfinite(v) else f"{v:,.{nd}f}"

        return render_box("ATTENDU vs RÉALISÉ", [
            (None, [("Barres de production", f"{self.n_live:,}")]),
            ("RÉALISÉ", [("Sharpe", f(self.live_sharpe)),
                         ("Rendement cumulé", f"{self.live_return:.2%}"),
                         ("Drawdown maximal", f"{self.live_drawdown:.2%}")]),
            ("POSITION DANS L'ENVELOPPE DE RÉFÉRENCE", [
                ("Sharpe attendu (médiane)", f(self.expected_sharpe)),
                ("Centile du Sharpe réalisé", f"{self.sharpe_percentile:.1%}"),
                ("Centile du rendement", f"{self.return_percentile:.1%}"),
                ("Centile du drawdown", f"{self.drawdown_percentile:.1%}")]),
            ("DÉTECTION SÉQUENTIELLE", [
                ("Statistique de Page-Hinkley", f(self.sequential_stat, 2)),
                ("Alarme", "OUI" if self.sequential_alarm else "non")]),
            ("CONCLUSION", [("VERDICT", self.verdict)]),
        ], width=78)


def sharpe_drop_to_sigma(delta_sharpe: float, bars_per_year: float) -> float:
    """Traduit une chute de Sharpe annualisé en dérive par barre, en écarts-types.

    C'est la conversion qui rend le problème lisible : une chute de 2 points de Sharpe
    en barres horaires (6240 par an) ne représente que 0.025 σ de dérive par barre. Tout
    ce qui suit — délai de détection, puissance des tests — découle de ce chiffre, et
    aucune astuce statistique ne le rendra plus grand.
    """
    return float(abs(delta_sharpe) / np.sqrt(max(bars_per_year, 1.0)))


def reconcile(live_returns: np.ndarray, envelope: PerformanceEnvelope,
              bars_per_year: float = 252.0, alarm_percentile: float = 0.05,
              delta_sharpe: float = 2.0, arl0: float = 31_200.0,
              detector: Optional[PageHinkley] = None) -> ReconciliationReport:
    """Situe la performance réalisée dans l'enveloppe attendue et rend un verdict.

    `alarm_percentile` est le seuil de gravité : un Sharpe réalisé sous le 5e centile de
    ce que le backtest laissait attendre sur cette durée est un événement à 1 chance sur
    20 — pas une preuve d'échec, mais un signal qui mérite d'être regardé. Le verdict
    combine ce test de fenêtre avec le détecteur séquentiel, précisément parce qu'ils
    échouent différemment : le premier est aveugle en début de période, le second est
    aveugle à une sous-performance stable mais faible.

    `delta_sharpe` est la chute de Sharpe qu'on veut être capable de détecter, et `arl0`
    le budget de fausses alarmes du détecteur séquentiel (une tous les N pas sous H₀).
    Les défauts — chute de 2 points, une fausse alarme tous les 31 200 pas, soit cinq ans
    en barres horaires — ne sont pas hérités d'ailleurs : ils sortent d'une mesure. Taux
    de fausses alarmes et puissance mesurés sur 150 tirages, backtest à Sharpe 1.2,
    production à Sharpe −1.5 (chute de 2.7 points), barres horaires :

        ARL₀       horizon    fausses alarmes    détection
        12 480     6 240 (1 an)      18 %          93 %
        31 200     6 240 (1 an)       8 %          86 %          ← retenu
        62 400     6 240 (1 an)       3 %          60 %
        31 200    12 480 (2 ans)     17 %         100 %

    Ce tableau dit aussi la vérité qu'aucun réglage ne contournera : **il faut environ un
    an de production pour établir statistiquement qu'une stratégie horaire s'est
    effondrée de deux points de Sharpe.** C'est la raison d'être des couches amont —
    dérive des features et coûts de transaction détectent la CAUSE en quelques centaines
    de barres, bien avant que l'EFFET ne devienne significatif.
    """
    r = np.asarray(live_returns, dtype=float)
    r = r[np.isfinite(r)]
    n = int(r.size)

    if n < 10:
        return ReconciliationReport(
            n_live=n, live_sharpe=float("nan"), live_return=float("nan"),
            live_drawdown=float("nan"),
            expected_sharpe=envelope.sharpe_quantiles.get("q50", float("nan")),
            sharpe_percentile=float("nan"), return_percentile=float("nan"),
            drawdown_percentile=float("nan"), degraded=False, sequential_alarm=False,
            sequential_stat=0.0, verdict="historique insuffisant",
        )

    live_sharpe = sharpe_ratio(r, bars_per_year)
    live_return = float(np.prod(1.0 + r) - 1.0)
    live_dd = max_drawdown(r)

    p_sharpe = _percentile_of(live_sharpe, envelope.sharpe_quantiles)
    p_ret = _percentile_of(live_return, envelope.return_quantiles)
    p_dd = _percentile_of(live_dd, envelope.drawdown_quantiles)

    # Détection séquentielle. Deux précautions, faute de quoi le détecteur ne détecte
    # rien : (1) la référence est FIXE — celle du backtest — car un détecteur qui
    # réapprend sa moyenne suit la dégradation au lieu de la signaler ; (2) l'amplitude
    # visée est convertie de « points de Sharpe » en « σ par barre », seule échelle où
    # δ et λ ont un sens.
    sd = envelope.std_return if envelope.std_return > 1e-14 else 1.0
    ph = detector or PageHinkley.calibrate(
        sharpe_drop_to_sigma(delta_sharpe, bars_per_year), arl0=arl0,
        ref_mean=0.0, ref_std=1.0,
    )
    for x in r:
        ph.update((x - envelope.mean_return) / sd)

    degraded = bool(p_sharpe <= alarm_percentile)
    alarm = bool(ph.triggered and ph.direction == "baisse")

    if degraded and alarm:
        verdict = "DÉGRADATION CONFIRMÉE (fenêtre + séquentiel)"
    elif degraded:
        verdict = "sous-performance au-delà du seuil de l'enveloppe"
    elif alarm:
        verdict = "décrochage séquentiel détecté"
    elif p_sharpe >= 0.95:
        verdict = "surperformance atypique — vérifier les données"
    else:
        verdict = "conforme à l'attendu"

    return ReconciliationReport(
        n_live=n, live_sharpe=live_sharpe, live_return=live_return, live_drawdown=live_dd,
        expected_sharpe=envelope.sharpe_quantiles.get("q50", float("nan")),
        sharpe_percentile=p_sharpe, return_percentile=p_ret, drawdown_percentile=p_dd,
        degraded=degraded, sequential_alarm=alarm, sequential_stat=ph.statistic,
        verdict=verdict,
    )


# =======================================================================================
class DegradationDetector:
    """Détecteur séquentiel persistant, alimenté barre par barre en production.

    Différence avec `reconcile`, qui rejoue tout l'historique à chaque appel : celui-ci
    conserve son état. Il coûte O(1) par barre et n'oublie rien — deux propriétés
    nécessaires pour tourner en continu à l'intérieur du serveur d'inférence.
    """

    def __init__(self, envelope: PerformanceEnvelope, bars_per_year: float = 252.0,
                 delta_sharpe: float = 2.0, arl0: float = 31_200.0):
        self.envelope = envelope
        self.bars_per_year = float(bars_per_year)
        self.delta_sharpe = float(delta_sharpe)
        self.ph = PageHinkley.calibrate(
            sharpe_drop_to_sigma(delta_sharpe, bars_per_year), arl0=arl0,
            ref_mean=0.0, ref_std=1.0,
        )
        self._sd = envelope.std_return if envelope.std_return > 1e-14 else 1.0
        self.n = 0

    @property
    def expected_delay(self) -> float:
        """Barres nécessaires en moyenne pour détecter la chute visée. À lire avant de
        se fier au détecteur : si le délai dépasse l'horizon de décision, ce n'est pas
        lui qui protégera le compte, ce sont les couches en amont (dérive, coûts)."""
        return self.ph.expected_delay(sharpe_drop_to_sigma(self.delta_sharpe, self.bars_per_year))

    def update(self, ret: float) -> bool:
        self.n += 1
        return self.ph.update((float(ret) - self.envelope.mean_return) / self._sd)

    @property
    def triggered(self) -> bool:
        return self.ph.triggered and self.ph.direction == "baisse"

    @property
    def statistic(self) -> float:
        return self.ph.statistic

    def reset(self) -> None:
        self.ph.reset()
        self.n = 0


# =======================================================================================
def replay_mismatch(entries: Sequence[Dict[str, Any]],
                    predict: Callable[[Dict[str, Any]], Dict[str, Any]],
                    key: str = "action", limit: Optional[int] = None) -> Dict[str, Any]:
    """Rejoue des décisions journalisées et compte les désaccords avec le modèle courant.

    Sert à répondre à une question à laquelle rien d'autre ne répond : « le serveur qui
    tourne est-il bien le modèle qu'on a validé ? ». Un déploiement partiel, un fichier
    de features d'une version antérieure, une bibliothèque mise à jour sous le pied du
    modèle — aucun de ces incidents ne produit d'erreur. Ils produisent des décisions
    différentes, silencieusement.

    Attente stricte : **zéro désaccord**. Le seuil n'est pas « faible », il est nul ; un
    modèle déterministe rejoué sur ses propres entrées doit reproduire ses sorties à
    l'identique. Toute valeur non nulle est un incident, pas une statistique.
    """
    rows = list(entries)[-limit:] if limit else list(entries)
    checked = 0
    mismatches: List[Dict[str, Any]] = []

    for row in rows:
        request = row.get("request")
        if not isinstance(request, dict):
            continue
        expected = row.get("response", {}).get(key) if isinstance(row.get("response"), dict) \
            else row.get(key)
        if expected is None:
            continue
        try:
            got = predict(request)
        except Exception as exc:                       # une exception EST un désaccord
            mismatches.append({"ts": row.get("ts", ""), "expected": expected,
                               "got": f"{type(exc).__name__}: {exc}"})
            checked += 1
            continue
        actual = got.get(key) if isinstance(got, dict) else got
        checked += 1
        if actual != expected:
            mismatches.append({"ts": row.get("ts", ""), "expected": expected, "got": actual})

    rate = float(len(mismatches) / checked) if checked else float("nan")
    return {
        "n_checked": checked,
        "n_mismatch": len(mismatches),
        "mismatch_rate": rate,
        "reproducible": checked > 0 and not mismatches,
        "examples": mismatches[:10],
    }

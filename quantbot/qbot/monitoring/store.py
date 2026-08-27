"""Mémoire de production : ce que le bot a réellement fait, barre après barre.

Ce module ne juge rien — il enregistre. La séparation est volontaire : la couche qui
mesure ne doit pas être la couche qui décide si c'est grave, sinon on finit par ajuster
la mesure pour éteindre l'alerte.

Ce qui est conservé pour chaque barre est exactement ce qu'il faut pour reconstituer une
décision *et* pour la juger : l'horodatage, l'équité, l'exposition demandée et
l'exposition finalement autorisée par les garde-fous, l'action discrète, la confiance du
modèle, le statut, la latence, le spread. L'écart entre `target` et `applied` mérite
d'être suivi en soi : c'est la mesure de combien le risk management contraint le modèle,
et un modèle contraint 40 % du temps ne se comporte plus du tout comme à l'entraînement.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional

import numpy as np

from ..backtest.metrics import (
    PerformanceReport, compute_report, drawdown_series, equity_curve, sharpe_ratio,
)

__all__ = ["DecisionRecord", "LiveMetricsStore"]


@dataclass
class DecisionRecord:
    """Une barre de production."""
    ts: str = ""
    equity: float = 0.0
    balance: float = 0.0
    price: float = 0.0
    target: float = 0.0          # exposition voulue par le modèle
    applied: float = 0.0         # exposition après garde-fous
    action: int = 0
    confidence: float = 0.0
    status: str = "ok"
    latency_ms: float = 0.0
    spread_bps: float = 0.0
    data_age_s: float = 0.0
    reasons: List[str] = field(default_factory=list)
    regime: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LiveMetricsStore:
    """Tampon circulaire des décisions + calcul des indicateurs de tableau de bord.

    La persistance est en JSON Lines et volontairement *séparée* du journal d'audit :
    ce fichier-ci est un cache de travail, réécrit et purgé sans conséquence ; le
    journal, lui, ne se réécrit jamais. Confondre les deux revient à rendre la trace
    d'audit modifiable par le code de tableau de bord.
    """

    def __init__(self, maxlen: int = 5000, path: Optional[str | Path] = None,
                 bars_per_year: float = 252.0):
        self.records: Deque[DecisionRecord] = deque(maxlen=int(maxlen))
        self.bars_per_year = float(bars_per_year)
        self.path = Path(path) if path else None
        self.n_total = 0
        self._peak_equity = 0.0
        if self.path and self.path.exists():
            self.load(self.path)

    # -- alimentation -------------------------------------------------------------------
    def append(self, record: DecisionRecord) -> None:
        self.records.append(record)
        self.n_total += 1
        self._peak_equity = max(self._peak_equity, float(record.equity))
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record.to_dict(), separators=(",", ":"),
                                    default=str) + "\n")

    def extend(self, records: Iterable[DecisionRecord]) -> None:
        for r in records:
            self.append(r)

    def load(self, path: str | Path) -> int:
        n = 0
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.records.append(DecisionRecord(**json.loads(line)))
                    n += 1
                except (json.JSONDecodeError, TypeError):
                    continue
        self.n_total = max(self.n_total, n)
        if self.records:
            self._peak_equity = max(float(r.equity) for r in self.records)
        return n

    # -- séries -------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    def column(self, name: str) -> np.ndarray:
        return np.array([getattr(r, name) for r in self.records], dtype=float)

    def returns(self) -> np.ndarray:
        """Rendements de l'équité, barre à barre.

        On dérive les rendements de l'équité rapportée par le courtier plutôt que de les
        recalculer depuis les positions et les prix. C'est la seule série qui intègre
        tout : slippage réel, swap, commissions, exécutions partielles. Une série
        reconstruite « proprement » afficherait la performance du modèle, pas celle du
        compte — et c'est le compte qui paie.
        """
        eq = self.column("equity")
        eq = eq[np.isfinite(eq) & (eq > 0)]
        return np.diff(eq) / eq[:-1] if eq.size >= 2 else np.zeros(0)

    def turnover(self) -> np.ndarray:
        applied = self.column("applied")
        return np.abs(np.diff(applied)) if applied.size >= 2 else np.zeros(0)

    # -- indicateurs --------------------------------------------------------------------
    @property
    def equity(self) -> float:
        return float(self.records[-1].equity) if self.records else 0.0

    @property
    def drawdown(self) -> float:
        if not self.records:
            return 0.0
        return float(self.equity / max(self._peak_equity, 1e-12) - 1.0)

    def rolling_sharpe(self, window: int = 250) -> float:
        r = self.returns()
        return sharpe_ratio(r[-window:], self.bars_per_year) if r.size >= 20 else float("nan")

    def constraint_rate(self) -> float:
        """Fraction des barres où le garde-fou a modifié la décision du modèle."""
        if not self.records:
            return float("nan")
        diffs = [abs(r.target - r.applied) > 1e-9 for r in self.records]
        return float(np.mean(diffs))

    def status_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self.records:
            counts[r.status] = counts.get(r.status, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def report(self, n_trials: int = 1) -> Optional[PerformanceReport]:
        """Rapport de performance complet, ou None si l'historique est trop court.

        Le seuil de 30 barres n'est pas cosmétique : un Sharpe calculé sur 10 points a un
        écart-type d'environ 1/√10 ≈ 0.32 par unité de Sharpe annualisé — il n'informe
        sur rien. Publier un tel chiffre sur un tableau de bord, c'est garantir qu'on
        prendra une décision sur du bruit.
        """
        r = self.returns()
        if r.size < 30:
            return None
        return compute_report(
            r, bars_per_year=self.bars_per_year, n_trials=n_trials,
            turnover=self.turnover(), positions=self.column("applied"),
        )

    def latency_breach_rate(self, deadline_ms: float) -> float:
        """Fraction des réponses arrivées après l'échéance. Complète le p99, qui rate par
        construction tout incident touchant moins de 1 % des barres."""
        lat = self.column("latency_ms")
        lat = lat[np.isfinite(lat)]
        return float(np.mean(lat >= deadline_ms)) if lat.size else float("nan")

    def snapshot(self, window: int = 250, latency_deadline_ms: float = 1000.0) -> Dict[str, Any]:
        """Instantané dense, sérialisable, destiné au tableau de bord et au protocole."""
        r = self.returns()
        eq = self.column("equity")
        lat = self.column("latency_ms")
        conf = self.column("confidence")
        applied = self.column("applied")
        to = self.turnover()

        snap: Dict[str, Any] = {
            "n_bars": len(self.records),
            "n_total": self.n_total,
            "first_ts": self.records[0].ts if self.records else "",
            "last_ts": self.records[-1].ts if self.records else "",
            "equity": self.equity,
            "peak_equity": float(self._peak_equity),
            "drawdown": self.drawdown,
            "total_return": float(eq[-1] / eq[0] - 1.0) if eq.size >= 2 and eq[0] > 0 else 0.0,
            "exposure": float(np.mean(np.abs(applied))) if applied.size else 0.0,
            "net_exposure": float(applied[-1]) if applied.size else 0.0,
            "flat_rate": float(np.mean(np.abs(applied) < 1e-9)) if applied.size else float("nan"),
            "turnover_per_bar": float(np.mean(to)) if to.size else 0.0,
            "n_trades": int(np.sum(to > 1e-9)),
            "constraint_rate": self.constraint_rate(),
            "mean_confidence": float(np.mean(conf)) if conf.size else float("nan"),
            "mean_latency_ms": float(np.mean(lat)) if lat.size else float("nan"),
            "p99_latency_ms": float(np.percentile(lat, 99)) if lat.size else float("nan"),
            "max_latency_ms": float(np.max(lat)) if lat.size else float("nan"),
            "latency_breach_rate": self.latency_breach_rate(latency_deadline_ms),
            "max_data_age_s": float(np.max(self.column("data_age_s"))) if self.records else 0.0,
            "status_counts": self.status_counts(),
            "sharpe_rolling": self.rolling_sharpe(window),
        }

        if r.size >= 30:
            rep = self.report()
            assert rep is not None
            snap.update({
                "sharpe": rep.sharpe, "sortino": rep.sortino, "calmar": rep.calmar,
                "ann_volatility": rep.ann_volatility, "hit_rate": rep.hit_rate,
                "profit_factor": rep.profit_factor, "max_drawdown": rep.max_drawdown,
                "max_dd_duration": rep.max_dd_duration, "psr": rep.psr,
                "var_95": rep.var_95, "cvar_95": rep.cvar_95,
            })
        else:
            for k in ("sharpe", "sortino", "calmar", "ann_volatility", "hit_rate",
                      "profit_factor", "max_drawdown", "psr", "var_95", "cvar_95"):
                snap[k] = float("nan")
            snap["max_dd_duration"] = 0
        return snap

    # -- séries pour les graphiques -----------------------------------------------------
    def curves(self, max_points: int = 400) -> Dict[str, List[float]]:
        """Séries sous-échantillonnées pour le tableau de bord (équité, DD, exposition)."""
        eq = self.column("equity")
        if eq.size == 0:
            return {"equity": [], "drawdown": [], "exposure": [], "index": []}
        dd = drawdown_series(eq / eq[0]) if eq[0] > 0 else np.zeros_like(eq)
        expo = self.column("applied")
        step = max(1, int(np.ceil(eq.size / max_points)))
        idx = np.arange(0, eq.size, step)
        return {
            "equity": [float(v) for v in eq[idx]],
            "drawdown": [float(v) for v in dd[idx]],
            "exposure": [float(v) for v in expo[idx]],
            "index": [int(i) for i in idx],
        }

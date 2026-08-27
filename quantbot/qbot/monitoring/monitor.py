"""Orchestrateur de surveillance (cahier des charges §17).

Assemble les couches en un seul objet appelé une fois par barre depuis le serveur
d'inférence. L'ordre des couches n'est pas arbitraire : il va de la cause à l'effet,
c'est-à-dire du signal le plus rapide au plus lent.

    entrées du modèle  →  dérive des features        (quelques centaines de barres)
    exécution          →  coûts réels vs modélisés   (quelques dizaines d'exécutions)
    comportement       →  contrainte, inactivité      (immédiat)
    sorties du modèle  →  attendu vs réalisé          (des milliers de barres)

Cette hiérarchie est le cœur du dispositif. La performance est le seul juge qui compte,
et c'est le plus lent : il faut environ un an de barres horaires pour établir qu'une
stratégie a perdu deux points de Sharpe. Les autres couches ne remplacent pas ce
verdict — elles préviennent avant qu'il ne tombe, en surveillant ce qui *cause* la perte
plutôt que la perte elle-même.

Contrainte d'exécution : `observe()` tourne dans la boucle de trading. Elle doit être en
O(1) amorti et ne jamais lever d'exception vers l'appelant. Une couche de surveillance
qui fait tomber le serveur qu'elle surveille est un défaut, pas une protection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..config import MonitorConfig
from ..utils.logging import get_logger
from .alerts import Alert, AlertLevel, AlertManager, evaluate_rules, log_sink
from .drift import DriftMonitor, DriftReport, ReferenceDistribution
from .journal import DecisionJournal
from .reconciliation import (
    DegradationDetector, PerformanceEnvelope, ReconciliationReport, reconcile,
)
from .store import DecisionRecord, LiveMetricsStore
from .tca import Fill, TCAReport, analyse_fills

log = get_logger("monitoring")

__all__ = ["RegimeTracker", "LiveMonitor"]


# =======================================================================================
@dataclass
class RegimeTracker:
    """Suit l'état de régime et n'acte un changement qu'une fois confirmé.

    La confirmation sur plusieurs barres n'est pas de la prudence décorative : un
    détecteur causal qui hésite entre deux états produit des allers-retours à chaque
    barre, et chaque aller-retour émettrait une alerte. On exige donc `confirm_bars`
    observations concordantes — au prix d'un retard de détection assumé et connu.
    """
    threshold: float = 0.7
    confirm_bars: int = 3

    state: int = field(default=-1, init=False)
    previous: int = field(default=-1, init=False)
    _candidate: int = field(default=-1, init=False)
    _streak: int = field(default=0, init=False)
    n_changes: int = field(default=0, init=False)
    bars_in_state: int = field(default=0, init=False)

    def update(self, state: int, proba: float = 1.0,
               label: str = "") -> Optional[Dict[str, Any]]:
        """Retourne un descripteur de changement, ou None si rien n'est confirmé."""
        state = int(state)
        self.bars_in_state += 1
        if state == self.state:
            self._candidate, self._streak = -1, 0
            return None

        if state == self._candidate:
            self._streak += 1
        else:
            self._candidate, self._streak = state, 1

        if self._streak < self.confirm_bars or float(proba) < self.threshold:
            return None

        first = self.state < 0            # première identification, pas un changement
        self.previous, self.state = self.state, state
        self.bars_in_state = 0
        self._candidate, self._streak = -1, 0
        if first:
            # Passer de « aucun régime connu » au premier régime identifié n'est pas un
            # changement de régime : c'est l'initialisation. En faire une alerte
            # garantirait une notification inutile au démarrage de chaque session.
            return None
        self.n_changes += 1
        return {"state": state, "previous": self.previous, "proba": float(proba),
                "label": label or f"état {state}", "bars_in_previous": self.bars_in_state}


# =======================================================================================
class LiveMonitor:
    """Surveillance de production : mesure, confronte, alerte — et n'agit pas.

    Le moniteur ne ferme aucune position. Il expose `should_halt`, que le serveur est
    libre de consulter et que la configuration relie ou non au coupe-circuit
    (`halt_on_critical`, faux par défaut). Cette séparation est délibérée : une couche
    d'observation qui peut liquider un portefeuille devient elle-même un risque
    opérationnel, et le premier bug de seuil coûterait un compte.
    """

    def __init__(
        self,
        cfg: Optional[MonitorConfig] = None,
        reference: Optional[ReferenceDistribution] = None,
        envelope: Optional[PerformanceEnvelope] = None,
        bars_per_year: float = 252.0,
        model_id: str = "",
        journal: Optional[DecisionJournal] = None,
        store_path: Optional[str | Path] = None,
        regime_labels: Optional[Dict[int, str]] = None,
    ):
        self.cfg = cfg or MonitorConfig()
        self.bars_per_year = float(bars_per_year)
        self.model_id = model_id
        self.regime_labels = dict(regime_labels or {})

        self.store = LiveMetricsStore(maxlen=max(self.cfg.window, 1000), path=store_path,
                                      bars_per_year=self.bars_per_year)
        self.drift = (DriftMonitor(reference, window=self.cfg.drift_window,
                                   min_samples=self.cfg.drift_min_samples,
                                   psi_warn=self.cfg.psi_warn,
                                   psi_critical=self.cfg.psi_critical)
                      if reference is not None else None)
        self.envelope = envelope
        self.degradation = (
            DegradationDetector(envelope, bars_per_year=self.bars_per_year,
                                delta_sharpe=self.cfg.delta_sharpe, arl0=self.cfg.arl0)
            if envelope is not None else None)
        self.alerts = AlertManager(cooldown_bars=self.cfg.alert_cooldown_bars,
                                   sinks=[log_sink(log)])
        self.regime = RegimeTracker(threshold=self.cfg.regime_change_threshold,
                                    confirm_bars=self.cfg.regime_confirm_bars)

        if journal is not None:
            self.journal: Optional[DecisionJournal] = journal
        elif self.cfg.journal_path:
            self.journal = DecisionJournal(self.cfg.journal_path)
        else:
            self.journal = None

        self.fills: List[Fill] = []
        self.bar = 0
        self._last_drift: Optional[DriftReport] = None
        self._last_reconciliation: Optional[ReconciliationReport] = None
        self._last_tca: Optional[TCAReport] = None
        self._journal_ok: Optional[bool] = None
        self.n_errors = 0

    # ---------------------------------------------------------------------------------
    def observe(
        self,
        record: DecisionRecord,
        features: Optional[Sequence[float] | Dict[str, float]] = None,
        fill: Optional[Fill] = None,
        regime_state: Optional[int] = None,
        regime_proba: float = 1.0,
        request: Optional[Dict[str, Any]] = None,
    ) -> List[Alert]:
        """Enregistre une barre de production et retourne les alertes nouvellement émises.

        Ne lève jamais : toute exception est capturée, comptée et journalisée. Le pire
        scénario acceptable est un tableau de bord dégradé, jamais une session de trading
        interrompue par son propre observateur.
        """
        try:
            return self._observe(record, features, fill, regime_state, regime_proba, request)
        except Exception:  # pragma: no cover - filet de sécurité
            self.n_errors += 1
            log.exception("Erreur de surveillance à la barre %d (ignorée)", self.bar)
            return []

    def _observe(self, record, features, fill, regime_state, regime_proba, request) -> List[Alert]:
        self.bar += 1
        prev_equity = self.store.records[-1].equity if self.store.records else None
        self.store.append(record)

        if features is not None and self.drift is not None:
            self.drift.push(features)

        if fill is not None:
            self.fills.append(fill)
            if len(self.fills) > 2000:
                del self.fills[:-2000]

        # -- détecteur séquentiel : alimenté au rendement réalisé, barre par barre -------
        if self.degradation is not None and prev_equity and prev_equity > 0:
            self.degradation.update(record.equity / prev_equity - 1.0)

        # -- régime ---------------------------------------------------------------------
        regime_change = None
        if regime_state is not None:
            regime_change = self.regime.update(
                regime_state, regime_proba, self.regime_labels.get(int(regime_state), ""))

        # -- journal d'audit ------------------------------------------------------------
        if self.journal is not None:
            payload: Dict[str, Any] = {"bar": self.bar, "decision": record.to_dict(),
                                       "model_id": self.model_id}
            if request is not None:
                payload["request"] = request
            self.journal.append("decision", payload)

        # -- couches coûteuses : cadencées ----------------------------------------------
        if self.drift is not None and self.bar % max(self.cfg.drift_every, 1) == 0:
            self._last_drift = self.drift.report()
        if self.fills and self.bar % max(self.cfg.drift_every, 1) == 0:
            self._last_tca = self._compute_tca()
        if self.envelope is not None and self.bar % max(self.cfg.drift_every, 1) == 0:
            self._last_reconciliation = self._compute_reconciliation()
        if self.journal is not None and self.bar % 500 == 0:
            self._journal_ok = self.journal.verify().valid

        # -- règles ---------------------------------------------------------------------
        snapshot = self.store.snapshot(window=self.cfg.window,
                                       latency_deadline_ms=self.cfg.latency_critical_ms)
        raised = evaluate_rules(
            snapshot, self.cfg, bar=self.bar, drift=self._last_drift,
            reconciliation=self._last_reconciliation, tca=self._last_tca,
            regime_change=regime_change, journal_ok=self._journal_ok,
        )
        emitted = self.alerts.submit(raised, bar=self.bar)

        if emitted and self.journal is not None:
            for a in emitted:
                self.journal.append("alert", a.to_dict())
        return emitted

    # ---------------------------------------------------------------------------------
    def _compute_tca(self) -> Optional[TCAReport]:
        rep = self.store.report()
        return analyse_fills(
            self.fills,
            ann_volatility=rep.ann_volatility if rep else None,
            bars_per_year=self.bars_per_year,
            n_bars=len(self.store) or None,
            ratio_warn=self.cfg.cost_ratio_warn,
            ratio_critical=self.cfg.cost_ratio_critical,
        )

    def _compute_reconciliation(self) -> Optional[ReconciliationReport]:
        if self.envelope is None:
            return None
        r = self.store.returns()
        if r.size < 10:
            return None
        return reconcile(r, self.envelope, bars_per_year=self.bars_per_year,
                         delta_sharpe=self.cfg.delta_sharpe, arl0=self.cfg.arl0,
                         detector=self.degradation.ph if self.degradation else None)

    # ---------------------------------------------------------------------------------
    @property
    def should_halt(self) -> bool:
        """Vrai si une alerte critique récente justifie l'arrêt — sous réserve de config.

        Consulté par le serveur, jamais appliqué ici. `halt_on_critical` est faux par
        défaut : on veut d'abord observer le comportement du dispositif d'alerte sur un
        compte réel avant de lui confier le droit de couper.
        """
        return bool(self.cfg.halt_on_critical
                    and self.alerts.has_critical(self.bar, window=self.cfg.alert_cooldown_bars))

    def snapshot(self) -> Dict[str, Any]:
        """État complet, sérialisable — c'est ce que renvoie le message `status`."""
        snap = self.store.snapshot(window=self.cfg.window,
                                   latency_deadline_ms=self.cfg.latency_critical_ms)
        snap.update({
            "bar": self.bar,
            "model_id": self.model_id,
            "bars_per_year": self.bars_per_year,
            "monitor_errors": self.n_errors,
            "should_halt": self.should_halt,
            "regime": {"state": self.regime.state, "previous": self.regime.previous,
                       "label": self.regime_labels.get(self.regime.state, ""),
                       "n_changes": self.regime.n_changes,
                       "bars_in_state": self.regime.bars_in_state},
            "alerts": self.alerts.summary(),
            "drift": self._last_drift.to_dict() if self._last_drift else None,
            "reconciliation": (self._last_reconciliation.to_dict()
                               if self._last_reconciliation else None),
            "tca": self._last_tca.to_dict() if self._last_tca else None,
            "journal": ({"path": str(self.journal.path), "n_entries": len(self.journal),
                         "head": self.journal.head, "verified": self._journal_ok}
                        if self.journal else None),
        })
        snap.update(self.flat_summary())
        return snap

    def flat_summary(self) -> Dict[str, Any]:
        """Résumé à plat, à clés uniques — destiné à l'EA MetaTrader.

        L'analyseur JSON de l'EA est volontairement minimal (pas de bibliothèque, pas de
        DLL) : il cherche une clé n'importe où dans la chaîne. Une clé comme `status`,
        présente à la fois dans le bloc dérive et dans le bloc réconciliation, lui
        renverrait la première trouvée — c'est-à-dire au hasard. On expose donc des noms
        uniques, préfixés, plutôt que de complexifier l'EA.
        """
        drift = self._last_drift
        rec = self._last_reconciliation
        tca = self._last_tca
        alerts = self.alerts
        return {
            "drift_status": drift.status if drift else "n/a",
            "drift_critical": int(drift.n_critical) if drift else 0,
            "drift_worst": (drift.worst.name if drift and drift.worst else ""),
            "alert_count": len(alerts.history),
            "alert_worst": alerts.worst_level.value if alerts.worst_level else "aucune",
            "recon_verdict": rec.verdict if rec else "n/a",
            "tca_verdict": tca.verdict if tca else "n/a",
            "tca_ratio": float(tca.cost_ratio) if tca else float("nan"),
            "journal_ok": bool(self._journal_ok) if self._journal_ok is not None else True,
        }

    def refresh(self) -> Dict[str, Any]:
        """Force le recalcul des couches cadencées (utile pour un rapport à la demande)."""
        if self.drift is not None:
            self._last_drift = self.drift.report()
        if self.fills:
            self._last_tca = self._compute_tca()
        if self.envelope is not None:
            self._last_reconciliation = self._compute_reconciliation()
        if self.journal is not None:
            self._journal_ok = self.journal.verify().valid
        return self.snapshot()

    # ---------------------------------------------------------------------------------
    def to_html(self, path: str | Path, title: str = "QBot — supervision") -> Path:
        from .dashboard import render_dashboard

        return render_dashboard(self, path, title=title)

    def text_report(self) -> str:
        """Rapport texte pour la console et les journaux."""
        from ..utils.text import render_box

        snap = self.snapshot()

        def f(key: str, fmt: str = ".3f", pct: bool = False) -> str:
            v = snap.get(key, float("nan"))
            try:
                v = float(v)
            except (TypeError, ValueError):
                return str(v)
            if not np.isfinite(v):
                return "n/a"
            return f"{v:.2%}" if pct else f"{v:{fmt}}"

        drift_line = (f"{self._last_drift.status} "
                      f"({self._last_drift.n_critical} critiques)" if self._last_drift
                      else "pas encore de verdict")
        sections = [
            (None, [("Barres observées", f"{snap['n_bars']:,}"),
                    ("Modèle", self.model_id or "n/a"),
                    ("Régime courant", snap["regime"]["label"] or str(snap["regime"]["state"]))]),
            ("COMPTE", [("Équité", f("equity", ",.2f")),
                        ("Rendement cumulé", f("total_return", pct=True)),
                        ("Drawdown courant", f("drawdown", pct=True)),
                        ("Drawdown maximal", f("max_drawdown", pct=True))]),
            ("PERFORMANCE", [("Sharpe", f("sharpe")), ("Sortino", f("sortino")),
                             ("Sharpe glissant", f("sharpe_rolling")),
                             ("Volatilité annualisée", f("ann_volatility", pct=True)),
                             ("Taux de réussite", f("hit_rate", pct=True)),
                             ("Profit factor", f("profit_factor"))]),
            ("ACTIVITÉ", [("Exposition moyenne", f("exposure", pct=True)),
                          ("Turnover par barre", f("turnover_per_bar", ".4f")),
                          ("Transactions", f"{snap['n_trades']:,}"),
                          ("Barres à plat", f("flat_rate", pct=True)),
                          ("Décisions contraintes", f("constraint_rate", pct=True))]),
            ("INFRASTRUCTURE", [("Latence moyenne", f("mean_latency_ms", ".1f") + " ms"),
                                ("Latence p99", f("p99_latency_ms", ".1f") + " ms"),
                                ("Latence maximale", f("max_latency_ms", ".1f") + " ms"),
                                ("Réponses hors délai", f("latency_breach_rate", pct=True)),
                                ("Confiance moyenne", f("mean_confidence")),
                                ("Erreurs de surveillance", str(self.n_errors))]),
            ("SURVEILLANCE", [("Dérive des features", drift_line),
                              ("Attendu vs réalisé",
                               self._last_reconciliation.verdict if self._last_reconciliation
                               else "enveloppe non fournie"),
                              ("Coûts d'exécution",
                               self._last_tca.verdict if self._last_tca else "aucune exécution"),
                              ("Journal d'audit",
                               "intègre" if self._journal_ok else
                               ("COMPROMIS" if self._journal_ok is False else "non vérifié")),
                              ("Alertes émises", f"{snap['alerts']['n_alerts']:,}"),
                              ("Niveau le plus grave",
                               snap["alerts"]["worst_level"] or "aucune")]),
        ]
        out = render_box("SUPERVISION DE PRODUCTION", sections, width=78)
        recent = self.alerts.history[-8:]
        if recent:
            out += "\n\nDernières alertes :\n" + "\n".join(f"  {a}" for a in recent)
        return out

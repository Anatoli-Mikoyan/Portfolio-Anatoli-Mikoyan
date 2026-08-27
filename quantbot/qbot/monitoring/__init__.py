"""Surveillance de production (cahier des charges §17).

Le principe qui structure tout le paquet : **surveiller la cause avant l'effet**. La
perte d'argent est le seul juge qui compte, et c'est le plus lent — il faut de l'ordre
d'un an de barres horaires pour établir statistiquement qu'une stratégie a perdu deux
points de Sharpe. Les couches amont existent pour prévenir avant que ce verdict ne
tombe :

    drift.py           dérive des distributions d'entrée (PSI, KL, JS, KS, Page-Hinkley)
    tca.py             coûts d'exécution réels vs modélisés (implementation shortfall)
    store.py           mémoire de production et indicateurs de tableau de bord
    reconciliation.py  attendu vs réalisé (enveloppe bootstrap, test séquentiel, rejeu)
    journal.py         trace d'audit chaînée par empreinte, inaltérable de fait
    alerts.py          règles déterministes, niveaux, temporisation
    monitor.py         orchestrateur appelé une fois par barre
    dashboard.py       tableau de bord HTML autonome
"""
from .drift import (
    DriftMonitor, DriftReport, FeatureDrift, PageHinkley, ReferenceDistribution,
    effective_sample_size, jensen_shannon_distance, kl_divergence, ks_one_sample,
    ks_two_sample, population_stability_index,
)
from .journal import ChainVerification, DecisionJournal, JournalEntry
from .store import DecisionRecord, LiveMetricsStore
from .tca import Fill, TCAReport, analyse_fills, slippage_test
from .reconciliation import (
    DegradationDetector, PerformanceEnvelope, ReconciliationReport, reconcile,
    replay_mismatch, sharpe_drop_to_sigma,
)
from .alerts import Alert, AlertLevel, AlertManager, evaluate_rules
from .monitor import LiveMonitor, RegimeTracker
from .dashboard import dashboard_html, render_dashboard

__all__ = [
    "DriftMonitor", "DriftReport", "FeatureDrift", "PageHinkley", "ReferenceDistribution",
    "effective_sample_size", "jensen_shannon_distance", "kl_divergence", "ks_one_sample",
    "ks_two_sample", "population_stability_index",
    "ChainVerification", "DecisionJournal", "JournalEntry",
    "DecisionRecord", "LiveMetricsStore",
    "Fill", "TCAReport", "analyse_fills", "slippage_test",
    "DegradationDetector", "PerformanceEnvelope", "ReconciliationReport", "reconcile",
    "replay_mismatch", "sharpe_drop_to_sigma",
    "Alert", "AlertLevel", "AlertManager", "evaluate_rules",
    "LiveMonitor", "RegimeTracker",
    "dashboard_html", "render_dashboard",
]

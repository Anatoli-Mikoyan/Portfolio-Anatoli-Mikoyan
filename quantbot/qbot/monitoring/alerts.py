"""Règles d'alerte et gestion des alarmes (cahier des charges §17).

Un système d'alerte se juge à une seule chose : est-il encore lu au bout de six mois ?
Trois principes le déterminent, et ils sont tous des contraintes de conception, pas des
options.

  1. **Trois niveaux, trois conduites à tenir.** INFO se consulte, WARN se regarde dans
     la journée, CRITIQUE interrompt ce qu'on est en train de faire. Une échelle plus
     fine ne fait qu'ajouter des niveaux qu'on n'utilise pas ; une échelle plus grossière
     force à mettre au même rang une latence en hausse et un drawdown hors limite.
  2. **Anti-répétition obligatoire, avec temporisation croissante.** Une condition vraie
     pendant 2 000 barres doit produire quelques alertes, pas quarante. Un cooldown fixe
     ne suffit pas : à 50 barres d'intervalle, une dérive installée en produit encore
     quarante. Chaque répétition double donc la temporisation (50, 100, 200, 400…), si
     bien qu'une condition persistante coûte un nombre d'alertes logarithmique en sa
     durée. Sans cela, la seule réaction rationnelle de l'opérateur est de couper les
     notifications — et il coupera aussi celles qui comptaient.
  3. **Règles déterministes.** Aucun seuil appris, aucun modèle dans la couche de
     surveillance. Le jour où le modèle surveillé se trompe, on a besoin d'un juge dont
     le comportement est prévisible et lisible dans le code.

Séparation nette avec `RiskGuard` : le garde-fou agit dans la boucle de décision, barre
par barre, et coupe. Le moniteur observe et *signale*. Le seul pont entre les deux est
explicite et désactivé par défaut (`halt_on_critical`), parce qu'une couche
d'observation qui peut fermer les positions devient un risque à elle seule.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

__all__ = ["AlertLevel", "Alert", "AlertManager", "evaluate_rules"]


class AlertLevel(str, Enum):
    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {"info": 0, "warn": 1, "critical": 2}[self.value]


@dataclass
class Alert:
    code: str                       # identifiant stable : c'est lui qui porte le cooldown
    level: AlertLevel
    message: str
    value: float = float("nan")
    threshold: float = float("nan")
    context: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    bar: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["level"] = self.level.value
        return d

    def __str__(self) -> str:  # pragma: no cover - affichage
        mark = {"info": "·", "warn": "!", "critical": "‼"}[self.level.value]
        return f"[{mark}] {self.code}: {self.message}"


# =======================================================================================
# Règles
# =======================================================================================
def _n(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f


def evaluate_rules(snapshot: Dict[str, Any], cfg, bar: int = 0,
                   drift: Optional[Any] = None, reconciliation: Optional[Any] = None,
                   tca: Optional[Any] = None, regime_change: Optional[Dict[str, Any]] = None,
                   journal_ok: Optional[bool] = None) -> List[Alert]:
    """Évalue toutes les règles et retourne les alertes déclenchées, non dédupliquées.

    Chaque règle est écrite pour être lisible seule : condition, valeur, seuil, message.
    La déduplication et le cooldown sont l'affaire de `AlertManager` — les mélanger ici
    rendrait impossible de tester les règles indépendamment de l'état du gestionnaire.
    """
    out: List[Alert] = []

    def add(code: str, level: AlertLevel, message: str, value: float = float("nan"),
            threshold: float = float("nan"), **ctx: Any) -> None:
        out.append(Alert(code=code, level=level, message=message, value=value,
                         threshold=threshold, context=ctx, bar=bar))

    # ---- infrastructure : ce qui casse en premier et se répare le plus vite ------------
    # La latence se juge sur un TAUX de dépassement, pas seulement sur un centile fixe :
    # avec 0.8 % de réponses à 900 ms, le p99 reste bas et ne dit rien, alors qu'une
    # réponse sur cent arrivée après la barre est un incident d'exécution bien réel.
    lat = _n(snapshot.get("p99_latency_ms"))
    breach = _n(snapshot.get("latency_breach_rate"))
    if np.isfinite(breach) and breach >= 0.01:
        add("latence_critique", AlertLevel.CRITICAL,
            f"{breach:.1%} des réponses au-delà de {cfg.latency_critical_ms:.0f} ms : "
            "le modèle répond après la barre.", breach, 0.01,
            p99_ms=lat, max_ms=_n(snapshot.get("max_latency_ms")))
    elif np.isfinite(lat):
        if cfg.latency_critical_ms and lat >= cfg.latency_critical_ms:
            add("latence_critique", AlertLevel.CRITICAL,
                f"Latence p99 de {lat:.0f} ms : le modèle répond après la barre.",
                lat, cfg.latency_critical_ms)
        elif cfg.latency_warn_ms and lat >= cfg.latency_warn_ms:
            add("latence_elevee", AlertLevel.WARN,
                f"Latence p99 de {lat:.0f} ms, en hausse.", lat, cfg.latency_warn_ms)

    age = _n(snapshot.get("max_data_age_s"))
    if cfg.max_data_age_s and np.isfinite(age) and age >= cfg.max_data_age_s:
        add("flux_prix_perime", AlertLevel.CRITICAL,
            f"Dernière barre reçue il y a {age:.0f} s : flux de prix probablement mort.",
            age, cfg.max_data_age_s)

    if journal_ok is False:
        add("journal_compromis", AlertLevel.CRITICAL,
            "Le journal d'audit ne vérifie plus : entrée modifiée ou supprimée.", 0.0, 0.0)

    # ---- risque : le compte ------------------------------------------------------------
    dd = abs(_n(snapshot.get("drawdown")))
    if np.isfinite(dd):
        if cfg.live_dd_critical and dd >= cfg.live_dd_critical:
            add("drawdown_critique", AlertLevel.CRITICAL,
                f"Drawdown de {dd:.1%} au-delà de la limite.", dd, cfg.live_dd_critical)
        elif cfg.live_dd_warn and dd >= cfg.live_dd_warn:
            add("drawdown_eleve", AlertLevel.WARN,
                f"Drawdown de {dd:.1%}.", dd, cfg.live_dd_warn)

    raw_bars = _n(snapshot.get("n_bars"))
    n_bars = int(raw_bars) if np.isfinite(raw_bars) else 0
    sharpe = _n(snapshot.get("sharpe_rolling"))
    if (np.isfinite(sharpe) and n_bars >= cfg.min_bars_for_performance
            and cfg.sharpe_floor is not None and sharpe <= cfg.sharpe_floor):
        add("sharpe_glissant_bas", AlertLevel.WARN,
            f"Sharpe glissant de {sharpe:.2f} sur {n_bars} barres.",
            sharpe, cfg.sharpe_floor)

    # ---- entrées du modèle : la dérive -------------------------------------------------
    if drift is not None:
        n_crit = int(getattr(drift, "n_critical", 0))
        worst = getattr(drift, "worst", None)
        if n_crit > cfg.max_drifted_features:
            add("derive_generalisee", AlertLevel.CRITICAL,
                f"{n_crit} features au-delà de PSI {cfg.psi_critical:.2f} : "
                "le modèle est interrogé hors de sa distribution d'entraînement.",
                float(n_crit), float(cfg.max_drifted_features),
                worst=getattr(worst, "name", ""))
        elif n_crit > 0:
            add("derive_feature", AlertLevel.WARN,
                f"{n_crit} feature(s) en dérive critique"
                + (f", la pire étant {worst.name} (PSI {worst.psi:.2f})." if worst else "."),
                float(n_crit), 0.0, worst=getattr(worst, "name", ""))
        elif int(getattr(drift, "n_moderate", 0)) > 0:
            add("derive_moderee", AlertLevel.INFO,
                f"{drift.n_moderate} feature(s) en dérive modérée.",
                float(drift.n_moderate), 0.0)

    # ---- sorties du modèle : la performance --------------------------------------------
    if reconciliation is not None:
        if getattr(reconciliation, "degraded", False) and getattr(reconciliation, "sequential_alarm", False):
            add("degradation_confirmee", AlertLevel.CRITICAL,
                f"Performance sous le seuil de l'enveloppe ET décrochage séquentiel : "
                f"Sharpe réalisé {reconciliation.live_sharpe:.2f} au "
                f"{reconciliation.sharpe_percentile:.0%} centile.",
                _n(reconciliation.live_sharpe), _n(reconciliation.expected_sharpe))
        elif getattr(reconciliation, "degraded", False):
            add("sous_performance", AlertLevel.WARN,
                f"Sharpe réalisé {reconciliation.live_sharpe:.2f}, au "
                f"{reconciliation.sharpe_percentile:.0%} centile de l'attendu.",
                _n(reconciliation.live_sharpe), _n(reconciliation.expected_sharpe))
        elif getattr(reconciliation, "sequential_alarm", False):
            add("decrochage_sequentiel", AlertLevel.WARN,
                "Le détecteur séquentiel signale un décrochage des rendements.",
                _n(getattr(reconciliation, "sequential_stat", np.nan)), 0.0)

    # ---- exécution : les coûts ---------------------------------------------------------
    if tca is not None:
        ratio = _n(getattr(tca, "cost_ratio", np.nan))
        pval = _n(getattr(tca, "excess_pvalue", np.nan))
        if np.isfinite(ratio) and np.isfinite(pval) and pval < 0.05:
            if ratio >= cfg.cost_ratio_critical:
                add("couts_execution_critiques", AlertLevel.CRITICAL,
                    f"Coûts réalisés {ratio:.1f}× le modèle "
                    f"(rabais de Sharpe {_n(getattr(tca, 'sharpe_haircut', np.nan)):.2f}).",
                    ratio, cfg.cost_ratio_critical)
            elif ratio >= cfg.cost_ratio_warn:
                add("couts_execution_eleves", AlertLevel.WARN,
                    f"Coûts réalisés {ratio:.1f}× le modèle.", ratio, cfg.cost_ratio_warn)

    # ---- régime ------------------------------------------------------------------------
    if regime_change:
        proba = _n(regime_change.get("proba"))
        if np.isfinite(proba) and proba >= cfg.regime_change_threshold:
            add("changement_regime", AlertLevel.INFO,
                f"Passage vers le régime « {regime_change.get('label', regime_change.get('state'))} » "
                f"(probabilité {proba:.0%}).", proba, cfg.regime_change_threshold,
                **{k: v for k, v in regime_change.items() if k in ("state", "label", "previous")})

    # ---- comportement : le modèle est-il encore lui-même ? -----------------------------
    constraint = _n(snapshot.get("constraint_rate"))
    if np.isfinite(constraint) and constraint >= 0.40 and n_bars >= 50:
        add("modele_sur_contraint", AlertLevel.WARN,
            f"Les garde-fous modifient {constraint:.0%} des décisions : le comportement "
            "réel s'écarte de celui qui a été validé.", constraint, 0.40)

    flat = _n(snapshot.get("flat_rate"))
    if np.isfinite(flat) and flat >= 0.98 and n_bars >= 100:
        add("modele_inactif", AlertLevel.WARN,
            f"Aucune exposition sur {flat:.0%} des barres : modèle bloqué, "
            "garde-fou permanent ou signal éteint.", flat, 0.98)

    return out


# =======================================================================================
# Gestionnaire
# =======================================================================================
class AlertManager:
    """Déduplique, temporise et distribue les alertes.

    Le cooldown porte sur le CODE, pas sur le message : deux formulations de la même
    condition ne doivent pas contourner la temporisation. C'est aussi pourquoi les codes
    sont des identifiants stables et non des phrases.
    """

    def __init__(self, cooldown_bars: int = 30,
                 sinks: Optional[Sequence[Callable[[Alert], None]]] = None,
                 history: int = 500, backoff: float = 2.0,
                 max_cooldown_bars: int = 2000, reset_factor: float = 3.0):
        self.cooldown_bars = int(cooldown_bars)
        self.backoff = float(backoff)
        self.max_cooldown_bars = int(max_cooldown_bars)
        self.reset_factor = float(reset_factor)
        self.sinks: List[Callable[[Alert], None]] = list(sinks or [])
        self.history: List[Alert] = []
        self._max_history = int(history)
        self._last_bar: Dict[str, int] = {}
        self._streak: Dict[str, int] = {}
        self.n_suppressed = 0
        self.counts: Dict[str, int] = {}

    def cooldown_for(self, code: str) -> int:
        """Temporisation courante d'un code : cooldown × backoff^(répétitions−1), plafonnée.

        L'exposant est `répétitions − 1` pour que la PREMIÈRE répétition attende
        exactement `cooldown_bars` : le paramètre affiché doit être celui qu'on observe,
        sinon le réglage ne veut plus rien dire.
        """
        streak = max(self._streak.get(code, 0) - 1, 0)
        return int(min(self.cooldown_bars * (self.backoff ** streak), self.max_cooldown_bars))

    def add_sink(self, sink: Callable[[Alert], None]) -> None:
        self.sinks.append(sink)

    def submit(self, alerts: Sequence[Alert], bar: int = 0) -> List[Alert]:
        """Filtre par cooldown, distribue les survivantes, retourne celles qui sont sorties."""
        emitted: List[Alert] = []
        for alert in alerts:
            last = self._last_bar.get(alert.code)
            if last is not None:
                gap = bar - last
                current = self.cooldown_for(alert.code)
                if gap < current:
                    self.n_suppressed += 1
                    continue
                # Condition revenue après un long silence : on repart de la temporisation
                # de base. Une alerte espacée de plusieurs mois est une NOUVELLE alerte, et
                # l'escalade accumulée ne doit pas la retarder. Le seuil est un MULTIPLE de
                # la temporisation courante, jamais le plafond lui-même : au plafond, les
                # deux se confondraient et l'escalade serait remise à zéro à chaque fois.
                if gap >= self.reset_factor * current:
                    self._streak[alert.code] = 0
            self._streak[alert.code] = self._streak.get(alert.code, 0) + 1
            alert.context.setdefault("repetition", self._streak[alert.code])
            self._last_bar[alert.code] = bar
            self.counts[alert.code] = self.counts.get(alert.code, 0) + 1
            self.history.append(alert)
            emitted.append(alert)
            for sink in self.sinks:
                try:
                    sink(alert)
                except Exception:      # pragma: no cover - un puits cassé ne doit rien casser
                    # Un canal de notification indisponible (réseau, fichier verrouillé) ne
                    # doit jamais interrompre la boucle de trading. On perd la notification,
                    # pas la session — et l'alerte reste dans l'historique.
                    pass
        if len(self.history) > self._max_history:
            del self.history[: -self._max_history]
        return emitted

    # -- lecture ------------------------------------------------------------------------
    @property
    def worst_level(self) -> Optional[AlertLevel]:
        if not self.history:
            return None
        return max((a.level for a in self.history), key=lambda l: l.rank)

    def active(self, bar: int, window: int = 100) -> List[Alert]:
        """Alertes émises dans les `window` dernières barres."""
        return [a for a in self.history if bar - a.bar <= window]

    def has_critical(self, bar: int, window: int = 100) -> bool:
        return any(a.level is AlertLevel.CRITICAL for a in self.active(bar, window))

    def summary(self) -> Dict[str, Any]:
        by_level: Dict[str, int] = {}
        for a in self.history:
            by_level[a.level.value] = by_level.get(a.level.value, 0) + 1
        return {
            "n_alerts": len(self.history),
            "n_suppressed": self.n_suppressed,
            "by_level": by_level,
            "by_code": dict(sorted(self.counts.items(), key=lambda kv: -kv[1])),
            "worst_level": self.worst_level.value if self.worst_level else None,
            "recent": [a.to_dict() for a in self.history[-20:]],
        }

    def clear(self) -> None:
        self.history.clear()
        self._last_bar.clear()
        self._streak.clear()
        self.counts.clear()
        self.n_suppressed = 0


def log_sink(logger) -> Callable[[Alert], None]:
    """Puits qui écrit dans un logger standard, au niveau correspondant."""
    def _sink(alert: Alert) -> None:
        if alert.level is AlertLevel.CRITICAL:
            logger.error("ALERTE %s", alert)
        elif alert.level is AlertLevel.WARN:
            logger.warning("Alerte %s", alert)
        else:
            logger.info("%s", alert)
    return _sink

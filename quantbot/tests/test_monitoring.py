"""Tests de la couche de surveillance (cahier des charges §17).

Les tests portent sur ce qui peut réellement rendre un dispositif de surveillance
inutile : un détecteur muet face à ce qu'il est censé voir, un seuil qui crie sans
raison, une trace d'audit modifiable, un observateur qui fait tomber ce qu'il observe.
Plusieurs tests sont des mesures de PUISSANCE et de TAUX DE FAUSSES ALARMES, pas de
simples vérifications de type : un détecteur qu'on n'a pas calibré n'est pas un
détecteur.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest

from qbot.config import MonitorConfig
from qbot.monitoring import (
    Alert, AlertLevel, AlertManager, DecisionJournal, DecisionRecord, DriftMonitor,
    Fill, LiveMetricsStore, LiveMonitor, PageHinkley, PerformanceEnvelope,
    ReferenceDistribution, analyse_fills, dashboard_html, effective_sample_size,
    evaluate_rules, jensen_shannon_distance, kl_divergence, ks_one_sample,
    population_stability_index, reconcile, replay_mismatch, sharpe_drop_to_sigma,
)
from qbot.monitoring.monitor import RegimeTracker
from qbot.utils.text import render_box

BPY = 6240.0
SD = 1.5e-3


def _returns(sharpe: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Rendements de Sharpe annualisé donné, en barres horaires."""
    return rng.normal(sharpe * SD / np.sqrt(BPY), SD, n)


@pytest.fixture(scope="module")
def reference() -> ReferenceDistribution:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(0, 1, (5000, 6)), columns=[f"f{i}" for i in range(6)])
    X["const"] = 0.0
    return ReferenceDistribution.fit(X, n_bins=10, model_id="test")


# =======================================================================================
# Distances et tests de distribution
# =======================================================================================
def test_psi_is_zero_for_identical_distributions():
    p = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    assert population_stability_index(p, p) == pytest.approx(0.0, abs=1e-12)
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-12)
    assert jensen_shannon_distance(p, p) == pytest.approx(0.0, abs=1e-9)


def test_psi_is_symmetric_and_positive():
    a = np.array([0.4, 0.3, 0.2, 0.1])
    b = np.array([0.1, 0.2, 0.3, 0.4])
    assert population_stability_index(a, b) == pytest.approx(population_stability_index(b, a))
    assert population_stability_index(a, b) > 0.0


def test_jensen_shannon_is_bounded_by_one():
    """La distance JS est bornée : deux lois à supports disjoints valent 1, jamais plus.
    C'est ce qui permet de l'agréger en un score global, contrairement au PSI."""
    a = np.array([1.0 - 3e-9, 1e-9, 1e-9, 1e-9])
    b = np.array([1e-9, 1e-9, 1e-9, 1.0 - 3e-9])
    d = jensen_shannon_distance(a, b)
    assert 0.99 <= d <= 1.0


def test_drift_detects_mean_shift_and_ignores_stable_features(reference):
    rng = np.random.default_rng(1)
    live = pd.DataFrame(rng.normal(0, 1, (400, 6)), columns=[f"f{i}" for i in range(6)])
    live["const"] = 0.0
    live["f0"] = rng.normal(1.5, 1.0, 400)

    rep = reference.compare(live)
    by_name = {f.name: f for f in rep.features}
    assert by_name["f0"].verdict == "critique"
    assert by_name["f0"].ks_pvalue < 1e-6
    assert all(by_name[f"f{i}"].verdict == "stable" for i in range(1, 6))
    assert rep.status == "critique" and rep.worst.name == "f0"


def test_drift_detects_variance_change_without_mean_shift(reference):
    """Une explosion de volatilité laisse la moyenne intacte : un simple test de moyenne
    la manquerait. Le PSI, qui compare les distributions entières, la voit."""
    rng = np.random.default_rng(2)
    live = pd.DataFrame(rng.normal(0, 1, (400, 6)), columns=[f"f{i}" for i in range(6)])
    live["const"] = 0.0
    live["f2"] = rng.normal(0.0, 3.0, 400)

    f2 = {f.name: f for f in reference.compare(live).features}["f2"]
    assert abs(f2.z_shift) < 0.5           # aucune dérive de moyenne
    assert f2.verdict == "critique"        # dérive de distribution détectée malgré tout


def test_drift_detects_a_frozen_feature_coming_back_to_life(reference):
    """Une feature constante à l'entraînement qui se met à bouger est le cas que les
    découpages naïfs manquent : toute la masse tombe dans la même case."""
    rng = np.random.default_rng(3)
    live = pd.DataFrame(rng.normal(0, 1, (300, 6)), columns=[f"f{i}" for i in range(6)])
    live["const"] = rng.normal(5.0, 1.0, 300)

    const = {f.name: f for f in reference.compare(live).features}["const"]
    assert const.psi > 1.0 and const.verdict == "critique"


def test_drift_is_quiet_when_nothing_changes(reference):
    rng = np.random.default_rng(4)
    live = pd.DataFrame(rng.normal(0, 1, (400, 6)), columns=[f"f{i}" for i in range(6)])
    live["const"] = 0.0
    rep = reference.compare(live)
    assert rep.status == "stable" and rep.n_critical == 0
    assert rep.global_score < 0.15


def test_reference_distribution_round_trips(tmp_path, reference):
    path = reference.save(tmp_path / "ref.json")
    loaded = ReferenceDistribution.load(path)
    assert loaded.feature_names == reference.feature_names
    rng = np.random.default_rng(5)
    live = pd.DataFrame(rng.normal(0, 1, (200, 6)), columns=[f"f{i}" for i in range(6)])
    live["const"] = 0.0
    a = {f.name: f.psi for f in reference.compare(live).features}
    b = {f.name: f.psi for f in loaded.compare(live).features}
    assert a == pytest.approx(b)


def test_effective_sample_size_shrinks_with_autocorrelation():
    rng = np.random.default_rng(6)
    iid = rng.normal(size=1000)
    ar = np.zeros(1000)
    for i in range(1, 1000):
        ar[i] = 0.95 * ar[i - 1] + rng.normal()
    assert effective_sample_size(iid) > 800
    assert effective_sample_size(ar) < 60      # 1000 observations ≈ quelques dizaines


def test_ks_pvalue_is_not_anti_conservative_on_autocorrelated_data(reference):
    """Sans correction d'autocorrélation, le KS déclarerait une dérive en permanence sur
    des features financières. Le test le vérifie sur des données SANS dérive."""
    rng = np.random.default_rng(7)
    n_false = 0
    for _ in range(40):
        x = np.zeros(300)
        for i in range(1, 300):
            x[i] = 0.9 * x[i - 1] + rng.normal(0, np.sqrt(1 - 0.81))
        live = pd.DataFrame({f"f{i}": rng.normal(0, 1, 300) for i in range(6)})
        live["f0"] = x                        # même loi marginale, mais autocorrélée
        live["const"] = 0.0
        f0 = {f.name: f for f in reference.compare(live).features}["f0"]
        n_false += f0.ks_pvalue < 0.05
    assert n_false <= 8                       # ≤ 20 % de fausses détections


# =======================================================================================
# Page-Hinkley
# =======================================================================================
def test_page_hinkley_calibration_matches_its_own_arl_formula():
    ph = PageHinkley.calibrate(0.5, arl0=10_000.0)
    assert ph.arl0() == pytest.approx(10_000.0, rel=1e-3)
    assert ph.delta == pytest.approx(0.25)


def test_page_hinkley_false_alarm_rate_matches_the_advertised_budget():
    """Le taux de fausses alarmes mesuré doit correspondre à l'ARL₀ annoncé, sans quoi
    le budget affiché sur le tableau de bord est un mensonge."""
    rng = np.random.default_rng(8)
    n_series, length = 120, 2000
    fired = 0
    for _ in range(n_series):
        ph = PageHinkley.calibrate(0.5, arl0=10_000.0, ref_mean=0.0, ref_std=1.0)
        for _ in range(length):
            if ph.update(rng.normal()):
                fired += 1
                break
    # Attendu ≈ 1 − exp(−2000/10000) ≈ 18 %.
    assert 0.05 <= fired / n_series <= 0.40


def test_page_hinkley_detects_a_real_shift_within_the_predicted_delay():
    rng = np.random.default_rng(9)
    delays = []
    for _ in range(60):
        ph = PageHinkley.calibrate(0.5, arl0=10_000.0, ref_mean=0.0, ref_std=1.0)
        detected = None
        for i in range(1200):
            x = rng.normal(0, 1) if i < 200 else rng.normal(-0.5, 1)
            if ph.update(x) and detected is None:
                detected = i - 200
                break
        delays.append(detected if detected is not None else np.nan)
    delays = np.asarray(delays, dtype=float)
    assert np.isfinite(delays).mean() >= 0.90
    assert np.nanmedian(delays) <= 2.5 * PageHinkley.calibrate(0.5, 10_000.0).expected_delay(0.5)


def test_adaptive_page_hinkley_is_blind_to_the_drift_it_learns():
    """Justifie l'existence du mode à référence fixe : en mode adaptatif, la moyenne
    courante absorbe la dégradation, et le détecteur reste muet pendant l'effondrement.
    C'est exactement le piège que la surveillance de performance doit éviter."""
    rng = np.random.default_rng(10)
    x = np.concatenate([rng.normal(0.0, 1.0, 400), rng.normal(-0.6, 1.0, 3000)])

    adaptive = PageHinkley.calibrate(0.3, arl0=20_000.0)
    fixed = PageHinkley.calibrate(0.3, arl0=20_000.0, ref_mean=0.0, ref_std=1.0)
    for v in x:
        adaptive.update(v)
        fixed.update(v)

    assert fixed.triggered and fixed.direction == "baisse"
    assert fixed.statistic > adaptive.statistic


# =======================================================================================
# Journal d'audit
# =======================================================================================
def test_journal_chain_is_valid_and_survives_restart(tmp_path):
    path = tmp_path / "audit.jsonl"
    j = DecisionJournal(path)
    for i in range(20):
        j.append("decision", {"bar": i, "action": i % 3})
    assert j.verify().valid and len(j) == 20

    # Un redémarrage doit reprendre la chaîne, pas la recommencer.
    j2 = DecisionJournal(path)
    j2.append("decision", {"bar": 20})
    result = j2.verify()
    assert result.valid and result.n_entries == 21


def test_journal_detects_a_modified_entry(tmp_path):
    path = tmp_path / "audit.jsonl"
    j = DecisionJournal(path)
    for i in range(10):
        j.append("decision", {"bar": i, "exposure": 0.1 * i})

    lines = path.read_text(encoding="utf-8").splitlines()
    raw = json.loads(lines[4])
    raw["payload"]["exposure"] = 99.0
    lines[4] = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = DecisionJournal(path).verify()
    assert not result.valid and result.first_broken_seq == 4


def test_journal_detects_a_deleted_entry(tmp_path):
    path = tmp_path / "audit.jsonl"
    j = DecisionJournal(path)
    for i in range(10):
        j.append("decision", {"bar": i})

    lines = path.read_text(encoding="utf-8").splitlines()
    del lines[5]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = DecisionJournal(path).verify()
    assert not result.valid and result.first_broken_seq == 6


def test_journal_head_changes_with_every_entry(tmp_path):
    j = DecisionJournal(tmp_path / "audit.jsonl")
    heads = set()
    for i in range(15):
        j.append("decision", {"bar": i})
        heads.add(j.head)
    assert len(heads) == 15


def test_journal_replay_recovers_payloads(tmp_path):
    j = DecisionJournal(tmp_path / "audit.jsonl")
    for i in range(12):
        j.append("decision", {"bar": i})
        j.append("alert", {"code": "x"})
    assert [p["bar"] for p in j.read(kind="decision")] == list(range(12))
    assert len(j.read(kind="alert")) == 12


# =======================================================================================
# Coûts d'exécution
# =======================================================================================
def _fills(n: int, extra_bps: float, rng: np.random.Generator, delay_bps: float = 0.0,
           half_spread_bps: float = 1.0) -> list:
    out = []
    for _ in range(n):
        px = 1.10 + rng.normal(0, 5e-4)
        side = 1 if rng.random() < 0.5 else -1
        arrival = px * (1 + side * delay_bps / 1e4)
        fill = arrival * (1 + side * (half_spread_bps + extra_bps) / 1e4)
        out.append(Fill(side=side, qty=1.0, decision_price=px, arrival_price=arrival,
                        fill_price=fill, half_spread=px * half_spread_bps / 1e4,
                        expected_cost_bps=half_spread_bps, latency_ms=40.0))
    return out


def test_tca_decomposition_sums_to_implementation_shortfall():
    rng = np.random.default_rng(11)
    f = _fills(1, extra_bps=2.0, rng=rng, delay_bps=0.8)[0]
    total = f.delay_bps + f.spread_bps + f.commission_bps + f.impact_bps
    # La décomposition est EXACTE par construction : le résidu absorbe tout le reste.
    assert total == pytest.approx(f.implementation_shortfall_bps, abs=1e-9)
    # Les composantes retrouvent les coûts injectés. La tolérance de 1e-3 bps couvre le
    # terme croisé du composé (1+8e-5)(1+3e-4) − 1, soit 2.4e-4 bps : c'est un effet réel
    # de la composition des rendements, pas une approximation du calcul.
    assert f.delay_bps == pytest.approx(0.8, abs=1e-3)
    assert f.spread_bps == pytest.approx(1.0, abs=1e-3)
    assert f.impact_bps == pytest.approx(2.0, abs=1e-3)


def test_tca_is_quiet_when_costs_match_the_model():
    rng = np.random.default_rng(12)
    rep = analyse_fills(_fills(200, extra_bps=0.0, rng=rng))
    assert rep.cost_ratio == pytest.approx(1.0, abs=0.05)
    assert rep.excess_pvalue > 0.05
    assert "conformes" in rep.verdict


def test_tca_flags_costs_that_exceed_the_model():
    rng = np.random.default_rng(13)
    rep = analyse_fills(_fills(200, extra_bps=2.0, rng=rng),
                        ann_volatility=0.10, bars_per_year=BPY, n_bars=5000)
    assert rep.cost_ratio > 2.5 and rep.excess_pvalue < 1e-6
    assert "≫" in rep.verdict
    # Le rabais de Sharpe doit être arithmétiquement cohérent avec l'excès mesuré.
    expected = (rep.excess_bps / 1e4) * rep.turnover_per_bar * BPY / 0.10
    assert rep.sharpe_haircut == pytest.approx(expected, rel=1e-6)


def test_tca_handles_no_fills():
    rep = analyse_fills([])
    assert rep.n_fills == 0 and rep.verdict == "aucune exécution"


# =======================================================================================
# Attendu vs réalisé
# =======================================================================================
@pytest.fixture(scope="module")
def envelope() -> PerformanceEnvelope:
    rng = np.random.default_rng(14)
    return PerformanceEnvelope.build(_returns(1.2, 30_000, rng), horizon=6240,
                                     bars_per_year=BPY, n_paths=800, seed=1)


def test_envelope_is_wide_at_short_horizons():
    """Résultat central et contre-intuitif : à horizon court, l'enveloppe est si large
    qu'aucune sous-performance n'est détectable. Le test fige ce fait pour qu'aucune
    évolution ne laisse croire à une détection rapide."""
    rng = np.random.default_rng(15)
    short = PerformanceEnvelope.build(_returns(1.2, 20_000, rng), horizon=300,
                                      bars_per_year=BPY, n_paths=600, seed=2)
    width = short.sharpe_quantiles["q95"] - short.sharpe_quantiles["q05"]
    assert width > 8.0        # plus de 8 points de Sharpe entre le 5e et le 95e centile


def test_reconcile_accepts_performance_consistent_with_the_backtest(envelope):
    rng = np.random.default_rng(16)
    n_flagged = sum(reconcile(_returns(1.2, 6240, rng), envelope,
                              bars_per_year=BPY).degraded for _ in range(40))
    assert n_flagged <= 8            # taux de fausses alarmes de fenêtre sous contrôle


def test_reconcile_detects_a_collapse_over_a_year(envelope):
    rng = np.random.default_rng(17)
    hits = sum(bool(reconcile(_returns(-1.5, 6240, rng), envelope, bars_per_year=BPY).degraded)
               for _ in range(30))
    assert hits >= 24                # ≥ 80 % de détection à un an


def test_sharpe_drop_conversion_is_exact():
    assert sharpe_drop_to_sigma(2.0, 6240.0) == pytest.approx(2.0 / np.sqrt(6240.0))


def test_reconcile_reports_insufficient_history(envelope):
    rep = reconcile(np.zeros(3), envelope, bars_per_year=BPY)
    assert rep.verdict == "historique insuffisant" and not rep.degraded


def test_replay_detects_a_model_that_no_longer_reproduces_its_decisions():
    entries = [{"ts": str(i), "request": {"x": i}, "response": {"action": i % 3}}
               for i in range(50)]
    faithful = replay_mismatch(entries, lambda req: {"action": req["x"] % 3})
    assert faithful["reproducible"] and faithful["n_mismatch"] == 0

    drifted = replay_mismatch(entries, lambda req: {"action": (req["x"] + 1) % 3})
    assert not drifted["reproducible"] and drifted["mismatch_rate"] == 1.0

    crashing = replay_mismatch(entries, lambda req: (_ for _ in ()).throw(RuntimeError("boom")))
    assert crashing["n_mismatch"] == 50      # une exception EST un désaccord


# =======================================================================================
# Alertes
# =======================================================================================
def test_alert_backoff_makes_a_persistent_condition_logarithmic():
    """Une condition vraie 4 000 barres ne doit pas produire 80 alertes. C'est la
    différence entre un système lu et un système coupé."""
    mgr = AlertManager(cooldown_bars=50, backoff=2.0, sinks=[])
    emitted = 0
    for bar in range(4000):
        alerts = [Alert(code="derive", level=AlertLevel.WARN, message="x", bar=bar)]
        emitted += len(mgr.submit(alerts, bar=bar))
    assert emitted <= 10
    assert mgr.cooldown_for("derive") > 50


def test_alert_cooldown_resets_after_a_long_silence():
    mgr = AlertManager(cooldown_bars=10, backoff=2.0, max_cooldown_bars=100, sinks=[])
    mgr.submit([Alert(code="c", level=AlertLevel.WARN, message="x", bar=0)], bar=0)
    assert mgr.cooldown_for("c") == 10       # la première répétition attend la base

    for bar in range(0, 500, 10):
        mgr.submit([Alert(code="c", level=AlertLevel.WARN, message="x", bar=bar)], bar=bar)
    assert mgr.cooldown_for("c") == 100      # escalade jusqu'au plafond

    mgr.submit([Alert(code="c", level=AlertLevel.WARN, message="x", bar=5000)], bar=5000)
    assert mgr.cooldown_for("c") == 10       # revenue après un long silence => remise à zéro


def test_rules_fire_on_each_family_of_problem():
    cfg = MonitorConfig()
    snap = {"p99_latency_ms": 1500.0, "latency_breach_rate": 0.05, "max_data_age_s": 900.0,
            "drawdown": -0.20, "sharpe_rolling": -1.8, "n_bars": 500,
            "constraint_rate": 0.60, "flat_rate": 0.10}
    codes = {a.code for a in evaluate_rules(snap, cfg, bar=1, journal_ok=False)}
    assert {"latence_critique", "flux_prix_perime", "drawdown_critique",
            "sharpe_glissant_bas", "modele_sur_contraint", "journal_compromis"} <= codes


def test_rules_stay_silent_on_a_healthy_system():
    cfg = MonitorConfig()
    snap = {"p99_latency_ms": 40.0, "latency_breach_rate": 0.0, "max_data_age_s": 5.0,
            "drawdown": -0.01, "sharpe_rolling": 1.4, "n_bars": 500,
            "constraint_rate": 0.02, "flat_rate": 0.30}
    assert evaluate_rules(snap, cfg, bar=1, journal_ok=True) == []


def test_latency_rule_catches_a_rare_but_real_breach():
    """0.8 % de réponses hors délai laissent le p99 muet. C'est pourtant un incident."""
    cfg = MonitorConfig()
    quiet_p99 = {"p99_latency_ms": 50.0, "latency_breach_rate": 0.008}
    assert {a.code for a in evaluate_rules(quiet_p99, cfg)} == set()
    breaching = {"p99_latency_ms": 50.0, "latency_breach_rate": 0.02}
    assert "latence_critique" in {a.code for a in evaluate_rules(breaching, cfg)}


def test_a_broken_sink_never_breaks_the_manager():
    def bad(_: Alert) -> None:
        raise RuntimeError("canal indisponible")

    mgr = AlertManager(cooldown_bars=1, sinks=[bad])
    emitted = mgr.submit([Alert(code="c", level=AlertLevel.CRITICAL, message="x")], bar=1)
    assert len(emitted) == 1 and len(mgr.history) == 1


# =======================================================================================
# Régime
# =======================================================================================
def test_regime_tracker_requires_confirmation_and_ignores_initialisation():
    tr = RegimeTracker(threshold=0.6, confirm_bars=3)
    assert tr.update(0, 0.9) is None          # première identification : pas un changement
    assert tr.update(0, 0.9) is None
    assert tr.update(0, 0.9) is None
    assert tr.state == 0 and tr.n_changes == 0

    assert tr.update(1, 0.9) is None          # une seule barre : non confirmé
    assert tr.update(1, 0.9) is None
    change = tr.update(1, 0.9)
    assert change is not None and change["state"] == 1 and change["previous"] == 0
    assert tr.n_changes == 1


def test_regime_tracker_ignores_low_confidence_transitions():
    tr = RegimeTracker(threshold=0.8, confirm_bars=2)
    for _ in range(4):
        tr.update(0, 0.95)
    for _ in range(10):
        assert tr.update(1, 0.5) is None       # probabilité insuffisante
    assert tr.state == 0


# =======================================================================================
# Mémoire de production
# =======================================================================================
def test_store_metrics_match_a_hand_computed_case():
    store = LiveMetricsStore(maxlen=100, bars_per_year=BPY)
    equities = [100.0, 110.0, 99.0, 99.0]
    exposures = [0.0, 1.0, 1.0, -1.0]
    for eq, ex in zip(equities, exposures):
        store.append(DecisionRecord(equity=eq, target=ex, applied=ex))

    assert store.returns() == pytest.approx([0.10, -0.10, 0.0])
    assert store.turnover() == pytest.approx([1.0, 0.0, 2.0])
    assert store.equity == 99.0
    assert store.drawdown == pytest.approx(99.0 / 110.0 - 1.0)


def test_store_constraint_rate_counts_guard_interventions():
    store = LiveMetricsStore(maxlen=100, bars_per_year=BPY)
    for i in range(10):
        store.append(DecisionRecord(equity=100.0, target=1.0,
                                    applied=1.0 if i < 7 else 0.0))
    assert store.constraint_rate() == pytest.approx(0.3)


def test_store_refuses_to_report_on_too_few_bars():
    store = LiveMetricsStore(maxlen=100, bars_per_year=BPY)
    for i in range(10):
        store.append(DecisionRecord(equity=100.0 + i))
    assert store.report() is None
    assert not np.isfinite(store.snapshot()["sharpe"])


def test_store_persists_and_reloads(tmp_path):
    path = tmp_path / "decisions.jsonl"
    store = LiveMetricsStore(maxlen=100, path=path, bars_per_year=BPY)
    for i in range(30):
        store.append(DecisionRecord(ts=str(i), equity=100.0 + i, applied=0.5))
    reloaded = LiveMetricsStore(maxlen=100, bars_per_year=BPY)
    assert reloaded.load(path) == 30
    assert reloaded.equity == store.equity


def test_latency_breach_rate():
    store = LiveMetricsStore(maxlen=1000, bars_per_year=BPY)
    for i in range(1000):
        store.append(DecisionRecord(equity=100.0, latency_ms=1500.0 if i % 200 == 0 else 40.0))
    assert store.latency_breach_rate(1000.0) == pytest.approx(0.005)
    assert np.percentile(store.column("latency_ms"), 99) == pytest.approx(40.0)


# =======================================================================================
# Orchestrateur de bout en bout
# =======================================================================================
def _run_monitor(n_healthy: int, n_broken: int, reference, envelope, tmp_path,
                 seed: int = 21) -> LiveMonitor:
    rng = np.random.default_rng(seed)
    cfg = MonitorConfig(window=4000, drift_window=250, drift_every=25,
                        alert_cooldown_bars=50)
    mon = LiveMonitor(cfg, reference=reference, envelope=envelope, bars_per_year=BPY,
                      model_id="test-v1", journal=DecisionJournal(tmp_path / "audit.jsonl"),
                      regime_labels={0: "calme", 1: "agité"})
    mon.alerts.sinks = []
    equity, exposure = 10_000.0, 0.0

    for t in range(n_healthy + n_broken):
        broken = t >= n_healthy
        x = rng.normal(0, 1, 6)
        if broken:
            # Quatre features décalées : au-delà de `max_drifted_features` (3), ce qui
            # doit faire passer l'alerte de « dérive isolée » à « dérive généralisée ».
            x[:4] += 1.5
        equity *= 1.0 + float(_returns(-2.0 if broken else 1.2, 1, rng)[0])

        target = float(rng.choice([-1.0, 0.0, 1.0])) if rng.random() < 0.15 else exposure
        fill = None
        if abs(target - exposure) > 1e-9:
            px, slip = 1.10, (3.4 if broken else 1.0)
            side = int(np.sign(target - exposure))
            fill = Fill(side=side, qty=abs(target - exposure), decision_price=px,
                        arrival_price=px, fill_price=px * (1 + side * slip / 1e4),
                        half_spread=px * 1.0 / 1e4, expected_cost_bps=1.0, latency_ms=45.0)
        exposure = target

        mon.observe(
            DecisionRecord(ts=f"2026-01-01T00:00:{t % 60:02d}", equity=equity, price=1.10,
                           target=target, applied=target, confidence=0.6, latency_ms=45.0,
                           status="ok", data_age_s=3.0),
            features=np.concatenate([x, [0.0]]), fill=fill,
            regime_state=1 if broken else 0, regime_proba=0.9)
    mon.refresh()
    return mon


def test_monitor_stays_silent_on_a_healthy_run(reference, envelope, tmp_path):
    mon = _run_monitor(1200, 0, reference, envelope, tmp_path, seed=31)
    codes = {a.code for a in mon.alerts.history}
    assert "derive_generalisee" not in codes
    assert "couts_execution_critiques" not in codes
    assert mon.n_errors == 0
    assert mon._last_drift is not None and mon._last_drift.n_critical == 0


def test_monitor_raises_the_full_chain_when_everything_degrades(reference, envelope, tmp_path):
    mon = _run_monitor(600, 1800, reference, envelope, tmp_path, seed=32)
    codes = {a.code for a in mon.alerts.history}
    assert "derive_generalisee" in codes            # entrées hors distribution
    assert {"couts_execution_eleves", "couts_execution_critiques"} & codes
    assert "changement_regime" in codes
    assert mon.alerts.worst_level is AlertLevel.CRITICAL
    assert mon.n_errors == 0
    # Le backoff doit contenir le volume malgré 1 800 barres de conditions vraies.
    assert len(mon.alerts.history) < 40 and mon.alerts.n_suppressed > 100


def test_monitor_journal_stays_verifiable_after_a_full_run(reference, envelope, tmp_path):
    mon = _run_monitor(300, 300, reference, envelope, tmp_path, seed=33)
    result = mon.journal.verify()
    assert result.valid and result.n_entries == len(mon.journal)
    assert len(mon.journal.read(kind="decision")) == 600


def test_monitor_never_raises_on_malformed_input(reference, envelope, tmp_path):
    mon = LiveMonitor(MonitorConfig(), reference=reference, envelope=envelope,
                      bars_per_year=BPY)
    mon.alerts.sinks = []
    # Nombre de features incohérent : le moniteur doit encaisser, compter et continuer.
    assert mon.observe(DecisionRecord(equity=1.0), features=np.zeros(3)) == []
    assert mon.n_errors == 1
    assert mon.observe(DecisionRecord(equity=1.0), features=np.zeros(7)) == []
    assert mon.n_errors == 1                      # la barre valide passe sans erreur


def test_monitor_does_not_halt_by_default(reference, envelope, tmp_path):
    mon = _run_monitor(200, 1000, reference, envelope, tmp_path, seed=34)
    assert mon.alerts.has_critical(mon.bar, window=10_000)
    assert mon.should_halt is False                # halt_on_critical est faux par défaut

    mon.cfg.halt_on_critical = True
    mon.cfg.alert_cooldown_bars = 10_000
    assert mon.should_halt is True


def test_snapshot_is_json_serialisable(reference, envelope, tmp_path):
    mon = _run_monitor(300, 300, reference, envelope, tmp_path, seed=35)
    payload = json.dumps(mon.snapshot(), default=str)
    assert len(payload) > 500 and "drift" in payload


def test_text_report_lines_are_all_the_same_width(reference, envelope, tmp_path):
    mon = _run_monitor(200, 200, reference, envelope, tmp_path, seed=36)
    box = [l for l in mon.text_report().splitlines() if l.startswith(("┌", "│", "├", "└"))]
    assert len({len(l) for l in box}) == 1


# =======================================================================================
# Tableau de bord
# =======================================================================================
def test_dashboard_is_self_contained_and_well_formed(reference, envelope, tmp_path):
    mon = _run_monitor(300, 400, reference, envelope, tmp_path, seed=37)
    path = mon.to_html(tmp_path / "dash.html")
    page = path.read_text(encoding="utf-8")

    assert page.startswith("<!doctype html>") and page.rstrip().endswith("</html>")
    # Aucune ressource externe : le tableau de bord doit fonctionner hors ligne.
    for forbidden in ("http://", "https://", "<script", "src="):
        assert forbidden not in page
    assert "<svg" in page and "supervision" in page.lower()


def test_dashboard_handles_an_empty_history():
    page = dashboard_html({"n_bars": 0, "alerts": {}}, {"equity": [], "drawdown": [],
                                                        "exposure": []}, [])
    assert "<!doctype html>" in page and "Pas assez de points" in page


def test_dashboard_escapes_hostile_text():
    """Les messages d'alerte peuvent contenir des noms de features venus des données.
    Ils sont injectés dans du HTML : ils doivent être échappés."""
    snap = {"n_bars": 1, "alerts": {"recent": [
        {"level": "warn", "code": "<img onerror=x>", "message": "a & b <script>",
         "value": 1.0, "threshold": 0.0, "bar": 1}]}}
    page = dashboard_html(snap, {"equity": [], "drawdown": [], "exposure": []}, [])
    assert "<img onerror" not in page and "&lt;img onerror" in page
    assert "<script>" not in page


# =======================================================================================
# Rendu texte
# =======================================================================================
def test_render_box_is_width_exact_even_with_overlong_content():
    out = render_box("T", [(None, [("x" * 200, "y" * 200)]), ("S", [("a", "b")])], width=60)
    assert {len(l) for l in out.splitlines()} == {60}

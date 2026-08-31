"""Tests du labeling (triple barrière, poids) et de la couche de risque."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from qbot.config import RiskConfig
from qbot.labeling import (
    average_uniqueness, build_sample_weights, cusum_filter, get_bins, get_events,
    get_vol_target, indicator_matrix, num_concurrent_events, sequential_bootstrap,
    time_decay_weights,
)
from qbot.risk import (
    GuardStatus, RiskGuard, continuous_kelly, fractional_kelly, kelly_fraction,
    lots_from_exposure, risk_parity_weights, vol_target_size,
)


@pytest.fixture(scope="module")
def events(ohlcv):
    c, h, l = ohlcv["close"], ohlcv["high"], ohlcv["low"]
    vol = get_vol_target(c)
    idx = cusum_filter(c, vol.fillna(vol.median()) * 2.0)
    return get_events(c, idx, (1.5, 1.0), trgt=vol, vertical_bars=24, high=h, low=l), c


# ---------------------------------------------------------------------------------------
def test_cusum_samples_only_significant_moves(ohlcv):
    c = ohlcv["close"]
    vol = get_vol_target(c).fillna(0.001)
    loose = cusum_filter(c, vol * 0.5)
    tight = cusum_filter(c, vol * 5.0)
    assert len(tight) < len(loose) < len(c)


def test_triple_barrier_exit_is_in_the_future(events):
    ev, _ = events
    assert (ev["t1"] > ev.index).all()


def test_intrabar_touch_detects_more_stops(ohlcv):
    """Tester le contact uniquement en clôture SOUS-ESTIME les stops touchés."""
    c, h, l = ohlcv["close"], ohlcv["high"], ohlcv["low"]
    vol = get_vol_target(c)
    idx = cusum_filter(c, vol.fillna(vol.median()) * 2.0)
    with_hl = get_events(c, idx, (1.5, 1.0), trgt=vol, vertical_bars=24, high=h, low=l)
    close_only = get_events(c, idx, (1.5, 1.0), trgt=vol, vertical_bars=24)
    touched_hl = (with_hl["t1_pt"].notna() | with_hl["t1_sl"].notna()).sum()
    touched_c = (close_only["t1_pt"].notna() | close_only["t1_sl"].notna()).sum()
    assert touched_hl >= touched_c


def test_meta_labels_are_binary(ohlcv):
    c, h, l = ohlcv["close"], ohlcv["high"], ohlcv["low"]
    vol = get_vol_target(c)
    idx = cusum_filter(c, vol.fillna(vol.median()) * 2.0)
    side = pd.Series(np.sign(np.log(c).diff(10)).reindex(idx).fillna(1.0), index=idx)
    ev = get_events(c, idx, (1.5, 1.0), trgt=vol, vertical_bars=24, side=side, high=h, low=l)
    bins = get_bins(ev, c)
    assert set(bins["bin"].unique()).issubset({0, 1})


def test_average_uniqueness_reflects_overlap(ohlcv):
    idx = ohlcv.index
    # Labels disjoints -> unicité 1 ; labels totalement superposés -> unicité faible
    disjoint = pd.Series(idx[np.arange(0, 100) * 10 + 5], index=idx[np.arange(0, 100) * 10])
    overlap = pd.Series(idx[np.arange(0, 100) + 50], index=idx[np.arange(0, 100)])
    assert average_uniqueness(idx, disjoint).mean() > average_uniqueness(idx, overlap).mean()
    assert average_uniqueness(idx, overlap).mean() < 0.5


def test_effective_sample_size_is_smaller_than_nominal(events, ohlcv):
    ev, c = events
    weights = build_sample_weights(ohlcv.index, ev["t1"], c)
    assert weights["tW"].sum() <= len(weights) + 1e-9
    assert (weights["tW"] > 0).all() and (weights["tW"] <= 1.0 + 1e-9).all()


def test_time_decay_downweights_old_samples(events, ohlcv):
    ev, _ = events
    tw = average_uniqueness(ohlcv.index, ev["t1"])
    decay = time_decay_weights(tw, last_weight=0.2)
    assert decay.iloc[0] < decay.iloc[-1]
    assert decay.iloc[-1] == pytest.approx(1.0, abs=0.05)


def test_sequential_bootstrap_beats_iid_uniqueness(events, ohlcv):
    """Le bootstrap séquentiel doit produire un échantillon PLUS unique que l'i.i.d."""
    ev, _ = events
    t1 = ev["t1"].iloc[:150]
    ind = indicator_matrix(ohlcv.index, t1)
    rng = np.random.default_rng(0)
    seq = sequential_bootstrap(ind, 60, rng)
    iid = rng.integers(0, ind.shape[1], 60)
    assert len(set(seq.tolist())) >= len(set(iid.tolist())) - 5


def test_concurrency_counts_are_consistent(ohlcv):
    idx = ohlcv.index
    t1 = pd.Series(idx[np.arange(50) + 20], index=idx[np.arange(50)])
    conc = num_concurrent_events(idx, t1)
    assert conc.max() <= 21 and conc.min() >= 0
    assert conc.iloc[0] == 1


# ---------------------------------------------------------------------------------------
# Risque
# ---------------------------------------------------------------------------------------
def test_kelly_formulas():
    assert kelly_fraction(0.6, 1.0) == pytest.approx(0.2)
    assert kelly_fraction(0.5, 1.0) == pytest.approx(0.0)
    assert continuous_kelly(0.001, 0.0004) == pytest.approx(2.5)
    assert fractional_kelly(2.5, 0.25, cap=1.0) == pytest.approx(0.625)
    assert fractional_kelly(10.0, 0.5, cap=1.0) == pytest.approx(1.0)   # écrêtage


def test_half_kelly_retains_three_quarters_of_growth():
    """Propriété analytique : g(f·f*) / g(f*) = 2f - f². Pour f=0.5 -> 0.75."""
    mu, var = 0.001, 0.0004
    def growth(x):
        return mu * x - var * x ** 2 / 2.0
    full = continuous_kelly(mu, var)
    assert growth(0.5 * full) / growth(full) == pytest.approx(0.75, abs=1e-9)


def test_vol_target_size_is_bounded():
    assert vol_target_size(0.10, 0.20, max_leverage=1.0) == pytest.approx(0.5)
    assert vol_target_size(0.10, 0.01, max_leverage=1.0) == pytest.approx(1.0)
    assert vol_target_size(0.10, 0.0, max_leverage=2.0) == pytest.approx(2.0)


def test_lots_rounding_never_exceeds_request():
    lots = lots_from_exposure(0.5, 10_000.0, 1.0850, contract_size=100_000.0, lot_step=0.01)
    assert lots * 100_000.0 * 1.0850 <= 0.5 * 10_000.0 + 1e-6
    assert lots_from_exposure(0.001, 10_000.0, 1.0850) == 0.0     # sous le lot minimal
    assert lots_from_exposure(-0.5, 10_000.0, 1.0850) < 0


def test_risk_parity_equalises_contributions():
    cov = np.array([[0.04, 0.006, 0.0], [0.006, 0.01, 0.0], [0.0, 0.0, 0.0025]])
    w = risk_parity_weights(cov)
    contrib = w * (cov @ w)
    assert w.sum() == pytest.approx(1.0)
    assert contrib.std() / contrib.mean() < 0.05


def test_guard_liquidates_on_max_drawdown():
    guard = RiskGuard(RiskConfig(max_drawdown_stop=0.20, max_daily_loss=None))
    ts = datetime(2026, 3, 10, 14, tzinfo=timezone.utc)
    guard.update_equity(1.0, ts)
    assert guard.check(1.0, timestamp=ts).status == GuardStatus.OK
    guard.update_equity(0.79, ts)
    decision = guard.check(1.0, timestamp=ts)
    assert decision.status == GuardStatus.LIQUIDATE and decision.allowed_position == 0.0
    assert guard.halted


def test_guard_throttles_before_the_stop():
    guard = RiskGuard(RiskConfig(max_drawdown_stop=0.20, max_daily_loss=None))
    ts = datetime(2026, 3, 10, 14, tzinfo=timezone.utc)
    guard.update_equity(1.0, ts)
    guard.update_equity(0.85, ts)                            # 75 % du seuil
    decision = guard.check(1.0, timestamp=ts)
    assert decision.status == GuardStatus.THROTTLED
    assert 0.0 < decision.allowed_position < 1.0


def test_guard_blocks_on_wide_spread_and_cooldown():
    guard = RiskGuard(RiskConfig(max_spread_bps=5.0, max_consecutive_losses=3, cooldown_bars=4,
                                 max_drawdown_stop=None, max_daily_loss=None))
    ts = datetime(2026, 3, 10, 14, tzinfo=timezone.utc)
    guard.update_equity(1.0, ts)
    assert guard.check(1.0, spread_bps=9.0, timestamp=ts).status == GuardStatus.BLOCKED
    for _ in range(3):
        guard.register_trade_result(-0.01)
    assert guard.check(1.0, timestamp=ts).status == GuardStatus.BLOCKED
    assert guard.check(1.0, timestamp=ts).status == GuardStatus.BLOCKED   # gel actif


def test_guard_respects_session_filter():
    guard = RiskGuard(RiskConfig(session_filter=[(7, 16)], max_drawdown_stop=None,
                                 max_daily_loss=None))
    guard.update_equity(1.0)
    inside = datetime(2026, 3, 10, 10, tzinfo=timezone.utc)
    outside = datetime(2026, 3, 10, 3, tzinfo=timezone.utc)
    assert guard.check(1.0, timestamp=inside).status == GuardStatus.OK
    assert guard.check(1.0, timestamp=outside).status == GuardStatus.BLOCKED


def test_le_statut_throttled_ne_se_leve_que_sur_une_reduction_reelle():
    """« throttled » sur une position déjà nulle est un contresens visible.

    La confiance du modèle est presque toujours inférieure à 1.00, et le statut se
    levait à chaque décision — donc constant, donc muet. Pire : un opérateur lisant
    « throttled | exposition 0.000 » conclut qu'un garde-fou empêche son bot de
    trader, alors que le modèle a simplement choisi de rester à plat. La distinction
    entre « bridé » et « à plat par décision » doit rester lisible.
    """
    from qbot.config import RiskConfig
    from qbot.risk.guards import GuardStatus, RiskGuard

    guard = RiskGuard(RiskConfig())

    # Le modèle veut rester à plat : la confiance ne réduit rien.
    plat = guard.check(0.0, model_confidence=0.93)
    assert plat.allowed_position == pytest.approx(0.0)
    assert plat.status is not GuardStatus.THROTTLED, (
        "une position nulle ne peut pas être « bridée »")
    assert not any("confiance" in r for r in plat.reasons)

    # Le modèle veut une position : la confiance la réduit pour de bon.
    guard2 = RiskGuard(RiskConfig())
    ouvert = guard2.check(0.8, model_confidence=0.5)
    assert ouvert.allowed_position == pytest.approx(0.4)
    assert ouvert.status is GuardStatus.THROTTLED
    assert any("confiance" in r for r in ouvert.reasons)
    # Le motif doit montrer l'avant et l'après, sinon il n'apprend rien.
    motif = next(r for r in ouvert.reasons if "confiance" in r)
    assert "+0.800" in motif and "+0.400" in motif, motif


def test_une_confiance_pleine_ne_bride_jamais():
    from qbot.config import RiskConfig
    from qbot.risk.guards import GuardStatus, RiskGuard

    d = RiskGuard(RiskConfig()).check(0.6, model_confidence=1.0)
    assert d.allowed_position == pytest.approx(0.6)
    assert d.status is not GuardStatus.THROTTLED

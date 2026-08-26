"""Tests de la machinerie anti-overfitting.

Chaque test vérifie une propriété STATISTIQUE connue, pas seulement l'absence d'exception :
un estimateur qui tourne sans erreur mais renvoie une valeur fausse est plus dangereux
qu'un estimateur qui plante.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qbot.backtest import (
    deflated_sharpe_ratio, expected_max_sharpe, max_drawdown,
    min_track_record_length, probabilistic_sharpe_ratio, sharpe_ratio,
)
from qbot.validation import (
    CombinatorialPurgedCV, PurgedKFold, compute_pbo, monte_carlo_drawdown,
    stationary_bootstrap_indices, train_valid_test_split, whites_reality_check,
)


@pytest.fixture(scope="module")
def index() -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=2000, freq="h", tz="UTC")


@pytest.fixture(scope="module")
def t1(index) -> pd.Series:
    return pd.Series(index[np.minimum(np.arange(len(index)) + 24, len(index) - 1)], index=index)


# ---------------------------------------------------------------------------------------
# Purge / embargo
# ---------------------------------------------------------------------------------------
def test_purged_kfold_has_no_overlap(index, t1):
    for train, test in PurgedKFold(5, 0.02, t1).split(index):
        assert len(np.intersect1d(train, test)) == 0


def test_purge_removes_overlapping_labels(index, t1):
    """L'écart entre la fin du train et le début du test doit couvrir l'horizon du label."""
    for train, test in PurgedKFold(5, 0.0, t1).split(index):
        before = train[train < test.min()]
        if before.size:
            assert test.min() - before.max() > 24, "labels chevauchants non purgés"


def test_embargo_creates_forward_gap(index, t1):
    embargo_bars = int(len(index) * 0.03)
    for train, test in PurgedKFold(5, 0.03, t1).split(index):
        after = train[train > test.max()]
        if after.size:
            assert after.min() - test.max() > embargo_bars


def test_cpcv_path_count(index, t1):
    cv = CombinatorialPurgedCV(6, 2, 0.01, t1)
    assert cv.n_splits == 15
    assert cv.n_paths == 5
    folds = list(cv.split(index))
    assert len(folds) == 15
    for train, test, _ in folds:
        assert len(np.intersect1d(train, test)) == 0


def test_train_valid_test_split_is_chronological(index):
    tr, va, te = train_valid_test_split(index, 0.6, 0.2, 0.01)
    assert tr.max() < va.min() < va.max() < te.min()


# ---------------------------------------------------------------------------------------
# Métriques inférentielles
# ---------------------------------------------------------------------------------------
def test_expected_max_sharpe_matches_simulation():
    """Le Sharpe maximal attendu sous H0 doit correspondre à la simulation."""
    rng = np.random.default_rng(0)
    sharpes = np.array([sharpe_ratio(rng.normal(0, 0.01, 1000), 252) for _ in range(3000)])
    sd = float(sharpes.std())
    for n in (10, 100, 1000):
        empirical = np.mean([sharpes[rng.integers(0, sharpes.size, n)].max() for _ in range(400)])
        theory = expected_max_sharpe(n, sd)
        assert abs(theory - empirical) < 0.12, f"n={n}: théorie {theory:.3f} vs empirique {empirical:.3f}"


def test_deflated_sharpe_penalises_multiple_trials():
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0008, 0.008, 2000)
    dsr = [deflated_sharpe_ratio(returns, n, 1.0, 252) for n in (1, 10, 100, 1000)]
    assert dsr == sorted(dsr, reverse=True), "le DSR doit décroître avec le nombre d'essais"
    assert dsr[0] > 0.9 and dsr[-1] < 0.1


def test_psr_penalises_negative_skew():
    """Deux séries de même Sharpe, l'une à queue gauche épaisse : le PSR doit les séparer."""
    rng = np.random.default_rng(2)
    symmetric = rng.normal(0.0005, 0.01, 3000)
    skewed = -rng.gamma(2.0, 0.005, 3000) + 0.0105
    skewed = (skewed - skewed.mean()) / skewed.std() * symmetric.std() + symmetric.mean()
    assert sharpe_ratio(skewed, 252) == pytest.approx(sharpe_ratio(symmetric, 252), abs=0.05)
    assert probabilistic_sharpe_ratio(skewed, 0.0, 252) < probabilistic_sharpe_ratio(symmetric, 0.0, 252)


def test_min_track_record_length_infinite_without_edge():
    rng = np.random.default_rng(3)
    assert min_track_record_length(rng.normal(-0.001, 0.01, 500), 0.0, 0.95, 252) == float("inf")
    assert np.isfinite(min_track_record_length(rng.normal(0.002, 0.005, 2000), 0.0, 0.95, 252))


# ---------------------------------------------------------------------------------------
# PBO
# ---------------------------------------------------------------------------------------
def test_pbo_near_half_under_null():
    """Sans edge, la sélection in-sample vaut un tirage au sort : PBO ≈ 0.5."""
    values = []
    for rep in range(12):
        rng = np.random.default_rng(500 + rep)
        values.append(compute_pbo(rng.normal(0, 0.01, (1500, 40)), n_partitions=10,
                                  max_combinations=252).pbo)
    assert 0.35 < float(np.mean(values)) < 0.65


def test_pbo_low_with_real_edge():
    rng = np.random.default_rng(4)
    matrix = rng.normal(0, 0.01, (2000, 40))
    matrix[:, 3] += 0.003                                   # une config a un vrai edge
    assert compute_pbo(matrix, n_partitions=10, max_combinations=252).pbo < 0.15


def test_pbo_warns_on_small_n():
    rng = np.random.default_rng(5)
    with pytest.warns(RuntimeWarning):
        compute_pbo(rng.normal(0, 0.01, (800, 5)), n_partitions=8, max_combinations=70)


# ---------------------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------------------
def test_block_bootstrap_preserves_autocorrelation():
    rng = np.random.default_rng(6)
    series = np.zeros(4000)
    noise = rng.normal(0, 0.01, 4000)
    for i in range(1, 4000):
        series[i] = 0.6 * series[i - 1] + noise[i]

    def ac(x):
        return float(np.corrcoef(x[:-1], x[1:])[0, 1])

    block = ac(series[stationary_bootstrap_indices(4000, 30, rng)])
    iid = ac(series[rng.integers(0, 4000, 4000)])
    assert block > 0.4, f"le bootstrap par blocs a détruit l'autocorrélation ({block:.3f})"
    assert abs(iid) < 0.1


def test_reality_check_controls_false_positives():
    rng = np.random.default_rng(7)
    p_null = whites_reality_check(rng.normal(0, 0.01, (1000, 40)), n_samples=500)["p_value"]
    matrix = rng.normal(0, 0.01, (1000, 40))
    matrix[:, 11] += 0.0025
    res = whites_reality_check(matrix, n_samples=500)
    assert p_null > 0.05
    assert res["p_value"] < 0.05 and res["best_strategy"] == 11


def test_monte_carlo_drawdown_is_conservative():
    rng = np.random.default_rng(8)
    mc = monte_carlo_drawdown(rng.normal(0.0004, 0.008, 2500), 600, 20, 0)
    assert mc["p95_worst"] <= mc["median"] <= 0.0
    assert mc["p99_worst"] <= mc["p95_worst"]

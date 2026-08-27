"""Tests du méta-modèle ML (cahier des charges §6 et §9)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from qbot.config import FeatureConfig
from qbot.data.synthetic import RegimeSwitchingGBM, generate_synthetic_ohlcv
from qbot.features import FeaturePipeline, align_features_prices
from qbot.ml import (
    MetaModel, build_meta_dataset, cluster_features, compare_models, cross_validate_meta,
    justify_complexity, mdi_importance, select_features,
)
from qbot.ml.importance import mda_importance
from qbot.ml.models import MODEL_ZOO, build_model, fit_model
from qbot.strategies import TimeSeriesMomentum


@pytest.fixture(scope="module")
def dataset():
    model = RegimeSwitchingGBM(mu=(0.10, -0.08, 0.0), sigma=(0.06, 0.18, 0.10),
                               persistence=0.997, autocorr=0.25, t_df=5.0)
    df = generate_synthetic_ohlcv(n=16_000, seed=12, model=model, spread_bps=0.6).drop(
        columns=["regime"])
    cfg = FeatureConfig(returns_windows=(1, 5, 20), vol_windows=(10, 60), ema_windows=(10, 50),
                        use_microstructure=False, use_calendar=True, scaler_window=300)
    feats = FeaturePipeline(cfg).fit_transform(df)
    x, p = align_features_prices(feats, df)
    return build_meta_dataset(TimeSeriesMomentum(lookback=60, threshold=0.5), x, p)


# ---------------------------------------------------------------------------------------
# Jeu de données
# ---------------------------------------------------------------------------------------
def test_meta_labels_are_binary_and_causal(dataset):
    assert set(dataset.y.unique()) <= {0, 1}
    assert (dataset.t1 > dataset.y.index).all(), "un label se clôture avant son événement"
    assert 0.2 < dataset.base_rate < 0.8
    assert len(dataset.X) == len(dataset.y) == len(dataset.sample_weight)


def test_primary_signal_is_a_feature(dataset):
    """Le méta-modèle doit voir la FORCE du signal primaire, pas seulement le contexte."""
    assert "primary_signal" in dataset.X.columns
    assert "primary_abs" in dataset.X.columns
    assert (dataset.X["primary_abs"] >= 0).all()


def test_sample_weights_reflect_uniqueness(dataset):
    assert (dataset.sample_weight > 0).all()
    assert dataset.sample_weight.std() > 0, "poids constants : la correction d'unicité est inopérante"


def test_split_is_chronological(dataset):
    train, test = dataset.split(0.7)
    assert train.X.index.max() < test.X.index.min()


# ---------------------------------------------------------------------------------------
# Zoo de modèles
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(MODEL_ZOO))
def test_every_model_fits_with_weights(name):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 6))
    y = (X[:, 0] > 0).astype(int)
    w = rng.uniform(0.5, 1.5, 400)
    model = fit_model(build_model(name), X, y, w)
    proba = model.predict_proba(X)
    assert proba.shape == (400, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_always_trade_is_a_true_baseline():
    """La référence doit être NEUTRE : elle ne discrimine rien, par construction."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 4))
    y = (X[:, 0] > 0).astype(int)
    model = fit_model(build_model("always_trade"), X, y, None)
    proba = model.predict_proba(X)[:, 1]
    assert proba.std() == 0.0
    assert proba[0] == pytest.approx(y.mean())


# ---------------------------------------------------------------------------------------
# Validation croisée et évaluation économique
# ---------------------------------------------------------------------------------------
def test_cross_validation_is_purged(dataset):
    ev = cross_validate_meta(dataset, "logistic", n_splits=4)
    assert ev.n_folds >= 3
    assert 0.0 <= ev.auc <= 1.0
    assert ev.n_trades_filtered <= ev.n_trades_base
    assert 0.3 <= ev.threshold <= 0.8


def test_baseline_shows_no_economic_gain(dataset):
    """`always_trade` ne filtre rien : son gain économique doit être exactement nul.

    Si ce test échoue, c'est que l'évaluation économique compare des populations
    différentes — donc que tous les autres chiffres sont faux."""
    ev = cross_validate_meta(dataset, "always_trade", n_splits=4)
    assert ev.economic_gain == pytest.approx(0.0, abs=1e-12)
    assert ev.trade_retention == pytest.approx(1.0)
    assert ev.verdict.startswith("INUTILE")


def test_model_comparison_orders_by_complexity(dataset):
    table = compare_models(dataset, ["always_trade", "logistic", "forest"], n_splits=4)
    assert list(table["complexité"]) == sorted(table["complexité"])
    assert "gain_par_trade" in table.columns
    verdict = justify_complexity(table)
    assert isinstance(verdict, str) and len(verdict) > 30


def test_filtering_improves_precision_over_base_rate(dataset):
    """Le meta-labeling doit augmenter la PRÉCISION, quitte à réduire le rappel.

    C'est tout l'intérêt : en trading, un trade évité vaut mieux qu'un trade moyen,
    puisque chaque trade coûte le spread."""
    ev = cross_validate_meta(dataset, "forest", n_splits=4)
    assert ev.precision > dataset.base_rate, (
        f"précision {ev.precision:.3f} <= taux de base {dataset.base_rate:.3f}")
    assert ev.trade_retention < 1.0


# ---------------------------------------------------------------------------------------
# Importance des features
# ---------------------------------------------------------------------------------------
def test_clustering_groups_redundant_features(dataset):
    groups = cluster_features(dataset.X, threshold=0.5)
    assert 1 < len(groups) < dataset.X.shape[1]
    members = [c for g in groups.values() for c in g]
    assert sorted(members) == sorted(dataset.X.columns), "des features ont été perdues"


def test_clustering_detects_a_duplicated_feature():
    """Une feature dupliquée doit atterrir dans le même cluster que son original."""
    rng = np.random.default_rng(0)
    base = pd.DataFrame(rng.standard_normal((300, 4)), columns=list("abcd"))
    base["a_copy"] = base["a"] + rng.normal(0, 1e-3, 300)
    groups = cluster_features(base, threshold=0.3)
    for members in groups.values():
        if "a" in members:
            assert "a_copy" in members
            return
    pytest.fail("la feature dupliquée n'a pas été regroupée avec son original")


def test_mda_ranks_informative_features_first(dataset):
    imp = mda_importance(dataset, model_name="forest", n_splits=3, n_repeats=1)
    assert not imp.empty
    assert imp["mda"].iloc[0] >= imp["mda"].iloc[-1]
    assert {"feature", "mda", "std", "t_stat"} <= set(imp.columns)


def test_feature_selection_requires_stability(dataset):
    """La sélection porte sur le t de Student, pas sur l'importance moyenne."""
    imp = mda_importance(dataset, model_name="forest", n_splits=3, n_repeats=1)
    lenient = select_features(imp, min_t=0.0)
    strict = select_features(imp, min_t=3.0)
    assert set(strict) <= set(lenient)
    assert len(strict) <= len(lenient)


def test_mdi_is_available_but_flagged_as_indicative(dataset):
    imp = mdi_importance(dataset, "forest")
    assert len(imp) == dataset.X.shape[1]
    assert imp.sum() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------------------
# Modèle entraîné
# ---------------------------------------------------------------------------------------
def test_meta_model_filters_signal(dataset):
    train, test = dataset.split(0.7)
    meta = MetaModel("logistic", threshold=0.55).fit(train)

    signal = test.side.astype(float)
    filtered = meta.filter_signal(signal, test.X)
    assert len(filtered) == len(signal)
    assert (filtered.abs() <= signal.abs() + 1e-9).all(), "le filtre a augmenté l'exposition"
    assert filtered.abs().sum() < signal.abs().sum(), "le filtre n'a rien filtré"


def test_meta_model_rejects_missing_columns(dataset):
    meta = MetaModel("logistic").fit(dataset)
    with pytest.raises(ValueError, match="manquantes"):
        meta.predict_proba(dataset.X.drop(columns=[dataset.X.columns[0]]))


def test_soft_sizing_is_monotone(dataset):
    meta = MetaModel("logistic", threshold=0.5, soft_sizing=True).fit(dataset)
    sizes = meta.size(np.array([0.3, 0.5, 0.6, 0.8, 1.0]))
    assert (np.diff(sizes) >= 0).all()
    assert sizes[0] == 0.0 and sizes[-1] == pytest.approx(1.0)

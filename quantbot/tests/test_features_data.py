"""Tests des données et du pipeline de features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qbot.config import FeatureConfig
from qbot.data import generate_synthetic_ohlcv, load_ohlcv, validate_ohlcv
from qbot.data.bars import build_bars, dollar_bars, imbalance_bars, tick_bars, volume_bars
from qbot.features import FeaturePipeline, adf_stat, find_min_ffd, frac_diff_ffd, frac_diff_weights
from qbot.features.regime import plugin_entropy, trend_strength, variance_ratio


# ---------------------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------------------
def test_validate_rejects_broken_bars(ohlcv):
    bad = ohlcv.copy()
    bad.iloc[10, bad.columns.get_loc("high")] = bad["low"].iloc[10] - 1.0
    with pytest.raises(ValueError, match="incohérentes"):
        validate_ohlcv(bad)


def test_validate_rejects_unsorted_index(ohlcv):
    shuffled = ohlcv.iloc[np.random.default_rng(0).permutation(len(ohlcv))]
    with pytest.raises(ValueError, match="croissant"):
        validate_ohlcv(shuffled)


def test_loader_roundtrip(tmp_path, ohlcv):
    path = tmp_path / "data.csv"
    ohlcv.reset_index().rename(columns={"time": "Date"}).to_csv(path, index=False)
    loaded = load_ohlcv(path)
    assert len(loaded) == len(ohlcv)
    assert np.allclose(loaded["close"].to_numpy(), ohlcv["close"].to_numpy())


def test_dollar_bars_normalise_returns(ohlcv):
    """Les barres à information constante réduisent fortement l'aplatissement —
    c'est leur raison d'être (López de Prado, ch. 2)."""
    from scipy import stats

    db = dollar_bars(ohlcv, float((ohlcv["close"] * ohlcv["volume"]).sum() / 200))
    k_time = stats.kurtosis(np.diff(np.log(ohlcv["close"].to_numpy())))
    k_dollar = stats.kurtosis(np.diff(np.log(db["close"].to_numpy())))
    assert k_dollar < k_time


@pytest.mark.parametrize("kind", ["tick", "volume", "dollar", "imbalance"])
def test_all_bar_types_produce_valid_ohlcv(ohlcv, kind):
    bars = build_bars(ohlcv, kind)
    validate_ohlcv(bars)
    assert 5 < len(bars) < len(ohlcv)


def test_bar_timestamp_is_the_close(ohlcv):
    """L'horodatage d'une barre agrégée doit être celui de sa DERNIÈRE observation :
    la barre n'est exploitable qu'une fois close."""
    tb = tick_bars(ohlcv, 20)
    assert tb.index[0] == ohlcv.index[19]


# ---------------------------------------------------------------------------------------
# Différenciation fractionnaire
# ---------------------------------------------------------------------------------------
def test_fracdiff_weights_shrink_with_d():
    assert len(frac_diff_weights(1.0)) == 2          # d=1 -> simple différence
    assert len(frac_diff_weights(0.4)) > 50          # mémoire longue conservée


def test_adf_distinguishes_stationarity(ohlcv):
    logp = np.log(ohlcv["close"].to_numpy())
    assert adf_stat(logp) > -2.86                    # prix : non stationnaire
    assert adf_stat(np.diff(logp)) < -3.43           # rendements : stationnaires


def test_fracdiff_finds_stationarity_while_keeping_memory(ohlcv):
    d, stat, corr = find_min_ffd(ohlcv["close"])
    assert 0.0 < d < 1.0
    assert stat < -2.5
    assert corr > 0.85, "trop de mémoire perdue : l'intérêt du fracdiff disparaît"


def test_fracdiff_is_causal(ohlcv):
    series = np.log(ohlcv["close"])
    full = frac_diff_ffd(series, 0.4)
    truncated = frac_diff_ffd(series.iloc[:2000], 0.4)
    common = truncated.dropna().index
    assert np.allclose(full.loc[common].to_numpy(), truncated.loc[common].to_numpy(), atol=1e-10)


# ---------------------------------------------------------------------------------
# Features de régime
# ---------------------------------------------------------------------------------
def test_entropy_separates_noise_from_structure():
    rng = np.random.default_rng(0)
    random_walk = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, 4000))))
    alternating = pd.Series(np.exp(np.cumsum(np.where(np.arange(4000) % 2 == 0, 0.01, -0.009))))
    assert plugin_entropy(random_walk).dropna().mean() > 0.9
    assert plugin_entropy(alternating).dropna().mean() < 0.5


def test_trend_strength_signs_correctly():
    up = pd.Series(np.exp(np.linspace(0, 0.5, 500)))
    down = pd.Series(np.exp(np.linspace(0, -0.5, 500)))
    assert trend_strength(up, 60).dropna().mean() > 0.9
    assert trend_strength(down, 60).dropna().mean() < -0.9


def test_variance_ratio_detects_trend_and_reversion():
    rng = np.random.default_rng(1)
    trending = pd.Series(np.exp(np.cumsum(rng.normal(0.002, 0.005, 3000))))
    noise = rng.normal(0, 0.01, 3000)
    reverting = pd.Series(np.exp(np.cumsum(noise - 0.7 * np.roll(noise, 1))))
    assert variance_ratio(trending, 5, 252).dropna().mean() > 0.9
    assert variance_ratio(reverting, 5, 252).dropna().mean() < 0.9


# ---------------------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------------------
def test_pipeline_output_is_finite_and_bounded(ohlcv, feature_cfg):
    x = FeaturePipeline(feature_cfg).fit_transform(ohlcv)
    assert x.shape[0] > 1000 and x.shape[1] > 20
    assert np.isfinite(x.to_numpy()).all()
    assert np.abs(x.to_numpy()).max() <= feature_cfg.winsorize_sigma + 1e-6


def test_transform_enforces_same_schema(ohlcv, feature_cfg):
    pipe = FeaturePipeline(feature_cfg)
    train = pipe.fit_transform(ohlcv.iloc[:4000])
    test = pipe.transform(ohlcv.iloc[3000:])
    assert list(test.columns) == list(train.columns)


def test_zero_variance_feature_becomes_zero_not_nan(ohlcv, feature_cfg):
    """Une feature constante doit valoir 0 après normalisation. Si elle devenait NaN,
    un `dropna()` pourrait vider silencieusement toute la matrice."""
    flat = ohlcv.copy()
    flat["open"] = flat["close"].shift(1).bfill()     # rend `gap` identiquement nul
    pipe = FeaturePipeline(feature_cfg)
    x = pipe.fit_transform(flat)
    assert len(x) > 500


def test_min_history_guarantees_serving_parity(ohlcv, feature_cfg):
    """Contrat central du live : `min_history` barres suffisent à reproduire à
    l'identique les features calculées sur l'historique complet."""
    pipe = FeaturePipeline(feature_cfg)
    pipe.fit_transform(ohlcv)
    full = pipe.transform(ohlcv).tail(10).to_numpy(dtype=np.float32)
    truncated = pipe.transform_latest(ohlcv.iloc[-pipe.min_history:], n_rows=10)
    assert np.abs(full - truncated).max() < 1e-4


def test_pipeline_save_load(tmp_path, ohlcv, feature_cfg):
    pipe = FeaturePipeline(feature_cfg)
    x = pipe.fit_transform(ohlcv)
    restored = FeaturePipeline.load(pipe.save(tmp_path / "pipeline.json"))
    assert restored.feature_names == pipe.feature_names
    assert restored.fracdiff_d == pipe.fracdiff_d
    assert np.allclose(restored.transform(ohlcv).to_numpy(), x.to_numpy(), atol=1e-8)


def test_transform_latest_refuses_short_history(ohlcv, feature_cfg):
    pipe = FeaturePipeline(feature_cfg)
    pipe.fit_transform(ohlcv)
    with pytest.raises(ValueError, match="Historique insuffisant"):
        pipe.transform_latest(ohlcv.iloc[-50:], n_rows=1)

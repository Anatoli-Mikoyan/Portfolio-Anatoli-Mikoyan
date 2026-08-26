"""Tests anti-look-ahead — les plus importants du dépôt.

Principe : on PERTURBE le futur et on vérifie que rien dans le passé ne bouge. Toute
dépendance au futur, même indirecte (une moyenne centrée, un `fit` global, un extrême
glissant non décalé), se traduit immédiatement par un échec.

Ces tests attrapent la classe de bugs qui produit des backtests spectaculaires et des
comptes vides — la seule qui compte vraiment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qbot.backtest import run_backtest
from qbot.config import CostConfig, EnvConfig
from qbot.features import FeaturePipeline, align_features_prices
from qbot.features.technical import build_technical_features, donchian
from qbot.labeling import cusum_filter, get_events, get_vol_target


def _corrupt_future(df: pd.DataFrame, from_idx: int, factor: float = 1.5) -> pd.DataFrame:
    """Multiplie violemment tous les prix à partir de `from_idx`."""
    out = df.copy()
    cols = ["open", "high", "low", "close"]
    out.iloc[from_idx:, [out.columns.get_loc(c) for c in cols]] *= factor
    return out


def test_features_do_not_depend_on_future(ohlcv, feature_cfg):
    cut = 3000
    pipe = FeaturePipeline(feature_cfg)
    base = pipe.fit_transform(ohlcv)

    pipe2 = FeaturePipeline(feature_cfg)
    pipe2.fracdiff_d = pipe.fracdiff_d          # même d, sinon la comparaison n'a pas de sens
    corrupted = pipe2.fit_transform(_corrupt_future(ohlcv, cut))

    common = base.index.intersection(corrupted.index)
    common = common[common < ohlcv.index[cut]]
    assert len(common) > 500, "trop peu de lignes communes pour conclure"

    a = base.loc[common, base.columns]
    b = corrupted.loc[common, base.columns]
    diff = (a - b).abs().max()
    offenders = diff[diff > 1e-8]
    assert offenders.empty, f"Features dépendant du futur : {offenders.to_dict()}"


def test_technical_indicators_are_causal(ohlcv, feature_cfg):
    cut = 2000
    a = build_technical_features(ohlcv, feature_cfg).iloc[:cut]
    b = build_technical_features(_corrupt_future(ohlcv, cut), feature_cfg).iloc[:cut]
    diff = (a - b).abs().max()
    offenders = diff[diff > 1e-10]
    assert offenders.empty, f"Indicateurs non causaux : {offenders.to_dict()}"


def test_donchian_excludes_current_bar(ohlcv):
    """Sans `shift(1)`, la barre courante entre dans son propre extrême : le breakout
    devient trivialement détectable et le backtest explose artificiellement."""
    h, l, c = ohlcv["high"], ohlcv["low"], ohlcv["close"]
    dc = donchian(h, l, c, 20)
    # Un nouveau plus-haut absolu DOIT être signalé comme cassure haussière.
    idx = int(np.argmax(c.to_numpy()[100:2000])) + 100
    window_max = h.iloc[idx - 20: idx].max()
    if c.iloc[idx] > window_max:
        assert dc["dc_break_up"].iloc[idx] == 1.0


def test_labels_never_use_past(ohlcv):
    c, h, l = ohlcv["close"], ohlcv["high"], ohlcv["low"]
    vol = get_vol_target(c)
    events_idx = cusum_filter(c, vol.fillna(vol.median()) * 2.0)
    ev = get_events(c, events_idx, (1.5, 1.0), trgt=vol, vertical_bars=24, high=h, low=l)
    assert (ev["t1"] > ev.index).all(), "une barrière de sortie précède son événement"


def test_backtest_positions_are_shifted(ohlcv, zero_cost):
    """Une position parfaitement corrélée au rendement de la MÊME barre ne doit
    RIEN rapporter : le moteur doit décaler d'une barre."""
    cheat = np.sign(ohlcv["close"].pct_change().fillna(0.0)).to_numpy()
    res = run_backtest(cheat, ohlcv, zero_cost, EnvConfig(vol_target=None), 6240.0)
    # Si le décalage manquait, on obtiendrait un Sharpe astronomique (> 50).
    assert res.report.sharpe < 5.0, (
        f"Sharpe={res.report.sharpe:.1f} : le moteur exploite le rendement de la barre courante."
    )


def test_perfect_foresight_is_detectable(ohlcv, zero_cost):
    """Contrôle inverse : avec une VRAIE vision du futur, le Sharpe doit exploser.
    Ce test valide que le test précédent a bien du pouvoir de détection."""
    future = np.sign(ohlcv["close"].shift(-1) / ohlcv["close"] - 1.0).fillna(0.0).to_numpy()
    res = run_backtest(future, ohlcv, zero_cost, EnvConfig(vol_target=None), 6240.0)
    assert res.report.sharpe > 20.0, "le détecteur de look-ahead est aveugle"


def test_env_observation_uses_only_past(ohlcv, feature_cfg, env_cfg, zero_cost):
    from qbot.env import make_env_from_frames

    cut = 3000
    pipe = FeaturePipeline(feature_cfg)
    x = pipe.fit_transform(ohlcv)
    xa, pa = align_features_prices(x, ohlcv)

    # `transform` (et non `fit_transform`) impose le MÊME schéma de colonnes : sans cela
    # une colonne dégénérée sur un jeu et pas sur l'autre changerait la dimension d'observation
    # et masquerait la comparaison.
    corrupted_df = _corrupt_future(ohlcv, cut)
    x2 = pipe.transform(corrupted_df)
    xa2, pa2 = align_features_prices(x2, corrupted_df)

    common = xa.index.intersection(xa2.index)
    common = common[common < ohlcv.index[cut]]
    n = len(common)
    assert n > 500

    e1 = make_env_from_frames(xa.loc[common], pa.loc[common], env_cfg, zero_cost)
    e2 = make_env_from_frames(xa2.loc[common], pa2.loc[common], env_cfg, zero_cost)
    o1, o2 = e1.reset(), e2.reset()
    assert np.allclose(o1, o2, atol=1e-6), "l'observation initiale dépend de données futures"

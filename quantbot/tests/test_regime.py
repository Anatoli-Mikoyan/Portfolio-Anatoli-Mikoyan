"""Tests de la couche de détection de régime (cahier des charges §7).

Deux exigences y sont vérifiées, et la seconde est la plus importante :

1. les détecteurs retrouvent les régimes quand ceux-ci sont réellement observables ;
2. **aucune inférence causale n'utilise le futur** — le piège numéro un de cette couche,
   parce qu'un HMM lissé produit des régimes superbes et des backtests faux.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("hmmlearn")

from sklearn.metrics import adjusted_rand_score

from qbot.backtest import run_backtest
from qbot.config import CostConfig, EnvConfig
from qbot.data.synthetic import RegimeSwitchingGBM, generate_synthetic_ohlcv
from qbot.regime import (
    ClusteringDetector, HMMDetector, LookaheadError, RuleBasedDetector, build_detector,
    build_regime_matrix, compare_detectors, conditional_performance, lookahead_gain,
    looks_rolling_normalized, regime_usefulness, strategy_regime_map,
)
from qbot.strategies import default_strategies

BPY = 6240.0


def _market(sigmas, mus=None, n=20_000, seed=21, persistence=0.999, autocorr=0.0):
    model = RegimeSwitchingGBM(mu=mus or tuple(0.0 for _ in sigmas), sigma=sigmas,
                               persistence=persistence, autocorr=autocorr, t_df=None)
    return generate_synthetic_ohlcv(n=n, seed=seed, model=model, spread_bps=0.6)


@pytest.fixture(scope="module")
def ambiguous():
    """Régimes de volatilité proches (10 % / 13 %) : le cas réaliste, et le seul où le
    lissage change quelque chose."""
    full = _market((0.10, 0.13), n=20_000, persistence=0.997, autocorr=0.15)
    return full["regime"], build_regime_matrix(full.drop(columns=["regime"]))


@pytest.fixture(scope="module")
def separable():
    """Deux régimes de volatilité 2 % et 40 % : trivialement séparables."""
    full = _market((0.02, 0.40))
    return full["regime"], build_regime_matrix(full.drop(columns=["regime"]))


# ---------------------------------------------------------------------------------------
# Le piège des features normalisées
# ---------------------------------------------------------------------------------------
def test_rolling_zscore_destroys_regime_information():
    """LE test qui documente le défaut de conception corrigé dans ce module.

    Un z-score glissant plus court que la durée d'un régime efface le NIVEAU, qui est
    précisément l'information de régime. Sur un marché 2 % / 40 % — impossible à rater —
    la détection tombe au niveau du hasard.
    """
    from qbot.config import FeatureConfig
    from qbot.features import FeaturePipeline, align_features_prices

    full = _market((0.02, 0.40))
    truth, prices = full["regime"], full.drop(columns=["regime"])

    cfg = FeatureConfig(returns_windows=(1, 5, 20), vol_windows=(10, 60), ema_windows=(10, 50),
                        use_microstructure=False, use_calendar=False, scaler_window=300)
    scaled, _ = align_features_prices(FeaturePipeline(cfg).fit_transform(prices), prices)
    cols = [c for c in ("vol_10", "vol_60", "vol_pctile", "trend_strength", "hurst")
            if c in scaled.columns]

    level = build_regime_matrix(prices)
    results = {}
    for name, matrix in (("z-scorées", scaled[cols]), ("niveaux", level)):
        cut = int(len(matrix) * 0.6)
        det = ClusteringDetector(n_states=2).fit(matrix.iloc[:cut])
        states = det.filter(matrix.iloc[cut:]).states
        results[name] = adjusted_rand_score(truth.reindex(matrix.index[cut:]), states)

    assert results["niveaux"] > 0.7, f"features de niveau inopérantes (ARI={results['niveaux']:.3f})"
    assert results["z-scorées"] < 0.2, "le piège du z-score glissant a disparu : test obsolète"
    assert results["niveaux"] > results["z-scorées"] * 4


def test_normalization_guard_detects_zscored_input():
    rng = np.random.default_rng(0)
    zscored = pd.DataFrame(rng.standard_normal((3_000, 5)),
                           index=pd.date_range("2020", periods=3_000, freq="h", tz="UTC"))
    assert looks_rolling_normalized(zscored)
    levels = zscored + np.linspace(0, 50, 3_000)[:, None]
    assert not looks_rolling_normalized(levels)


# ---------------------------------------------------------------------------------------
# Causalité — le point critique
# ---------------------------------------------------------------------------------------
def test_hmm_filter_is_causal(separable):
    """Perturber le futur ne doit pas modifier les états filtrés du passé."""
    _, X = separable
    cut = len(X) // 2
    det = HMMDetector(n_states=2).fit(X.iloc[: cut // 2])

    corrupted = X.copy()
    corrupted.iloc[cut:] = corrupted.iloc[cut:] * 3.0 + 5.0

    a = det.filter(X).states.iloc[:cut]
    b = det.filter(corrupted).states.iloc[:cut]
    assert (a.to_numpy() == b.to_numpy()).all(), "le filtrage HMM regarde le futur"


def test_hmm_smoothing_is_not_causal(ambiguous):
    """Contrôle inverse : le lissage DOIT être affecté par le futur.

    Le test procède par TRONCATURE plutôt que par corruption : rendre le futur aberrant
    le rend également improbable sous tous les états, si bien que le message arrière
    devient uniforme et que le lissage redevient accidentellement causal. Comparer
    « lisser en voyant la suite » à « lisser sans la voir » isole exactement l'effet.

    Le marché choisi est AMBIGU à dessein : quand les régimes sont tranchés, le présent
    détermine déjà l'état et le futur n'apporte rien (mesuré : 0 % de désaccord sur un
    marché 2 %/40 %, contre 2.7 % sur un marché 10 %/13 %).
    """
    _, X = ambiguous
    cut = len(X) // 2
    det = HMMDetector(n_states=2).fit(X.iloc[:cut])

    # Le test porte sur les PROBABILITÉS a posteriori, pas sur l'argmax. La persistance
    # élevée du HMM fait que le message arrière ne remonte que quelques centaines de
    # barres et déplace les probabilités sans toujours faire basculer l'état retenu :
    # un test sur les états seuls conclurait à tort que le lissage est causal.
    with_future = det.smooth(X).proba.iloc[:cut].to_numpy()
    without_future = det.smooth(X.iloc[:cut]).proba.to_numpy()
    delta = np.abs(with_future - without_future).max()
    assert delta > 1e-6, f"le lissage devrait dépendre du futur (écart max {delta:.2e})"

    # Le filtrage causal, lui, doit être insensible à la troncature — au bit près.
    f_with = det.filter(X).proba.iloc[:cut].to_numpy()
    f_without = det.filter(X.iloc[:cut]).proba.to_numpy()
    assert np.abs(f_with - f_without).max() < 1e-12, "le filtrage causal dépend du futur"


def test_smoothed_regimes_are_refused_downstream(separable):
    _, X = separable
    det = HMMDetector(n_states=2).fit(X.iloc[: len(X) // 2])
    smoothed = det.smooth(X)
    assert smoothed.causal is False
    with pytest.raises(LookaheadError, match="filter"):
        smoothed.require_causal()


def test_lookahead_illusion_grows_when_detection_is_hard():
    """Le biais de lissage est NÉGLIGEABLE quand c'est facile, MAXIMAL quand c'est dur.

    C'est l'enseignement contre-intuitif de cette couche : le lissage trompe le plus
    précisément là où l'on aurait le plus besoin d'y voir clair."""
    disagreements = {}
    for label, sigmas in (("tranché", (0.02, 0.40)), ("ambigu", (0.10, 0.13))):
        full = _market(sigmas, n=20_000, persistence=0.997, autocorr=0.15)
        prices = full.drop(columns=["regime"])
        X = build_regime_matrix(prices)
        signal = default_strategies()[1].signal(prices)
        returns = run_backtest(signal, prices, CostConfig(spread_bps=0.6),
                               EnvConfig(vol_target=None), BPY).frame["net_return"]
        det = HMMDetector(n_states=2).fit(X.iloc[: len(X) // 2])
        gain = lookahead_gain(det, X, returns, BPY)
        disagreements[label] = gain["taux_desaccord"]
        assert 0.0 <= gain["agreement"] <= 1.0
        assert set(gain) >= {"dispersion_causale", "dispersion_lissee", "separation_illusoire"}

    assert disagreements["ambigu"] > disagreements["tranché"], (
        "le désaccord filtrage/lissage devrait croître quand la détection devient difficile")


# ---------------------------------------------------------------------------------------
# Capacité de détection
# ---------------------------------------------------------------------------------------
def test_detectors_recover_clearly_separable_regimes(separable):
    truth, X = separable
    cut = int(len(X) * 0.6)
    for kind in ("kmeans", "gmm", "hmm"):
        det = build_detector(kind, n_states=2).fit(X.iloc[:cut])
        ari = adjusted_rand_score(truth.reindex(X.index[cut:]), det.filter(X.iloc[cut:]).states)
        assert ari > 0.7, f"{kind} ne retrouve pas des régimes 2%/40% (ARI={ari:.3f})"


def test_drift_only_regimes_are_undetectable():
    """Résultat honnête à conserver : on détecte la VOLATILITÉ, pas la dérive.

    Deux régimes de dérive opposée (+30 %/-30 % annualisés) à volatilité identique sont
    indétectables à l'échelle de la barre : la dérive par barre y est deux ordres de
    grandeur sous le bruit. Toute couche de régime qui prétendrait le contraire ment.
    """
    full = _market((0.12, 0.12), mus=(0.30, -0.30))
    truth, X = full["regime"], build_regime_matrix(full.drop(columns=["regime"]))
    cut = int(len(X) * 0.6)
    det = HMMDetector(n_states=2).fit(X.iloc[:cut])
    ari = adjusted_rand_score(truth.reindex(X.index[cut:]), det.filter(X.iloc[cut:]).states)
    assert abs(ari) < 0.15, f"la dérive seule serait détectable (ARI={ari:.3f}) : vérifier le test"


def test_hmm_learns_persistence(separable):
    """Le HMM doit apprendre que les régimes DURENT — c'est sa valeur ajoutée."""
    _, X = separable
    det = HMMDetector(n_states=2).fit(X.iloc[: int(len(X) * 0.6)])
    assert (det.persistence > 0.9).all(), f"persistance apprise trop faible : {det.persistence}"


def test_hmm_switches_less_than_clustering(separable):
    """Un détecteur trop nerveux est inexploitable : chaque bascule coûte de la rotation."""
    _, X = separable
    cut = int(len(X) * 0.6)
    hmm = HMMDetector(n_states=2).fit(X.iloc[:cut]).filter(X.iloc[cut:])
    km = ClusteringDetector(n_states=2).fit(X.iloc[:cut]).filter(X.iloc[cut:])
    assert hmm.transition_rate() <= km.transition_rate() * 1.5


def test_rule_detector_needs_no_fitting():
    full = _market((0.05, 0.25))
    X = build_regime_matrix(full.drop(columns=["regime"]))
    det = RuleBasedDetector().fit(X)
    regimes = det.filter(X)
    assert regimes.causal
    assert 1 <= regimes.states.nunique() <= 4


def test_rule_detector_reports_missing_columns():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((200, 3)), columns=list("abc"))
    with pytest.raises(ValueError, match="absentes"):
        RuleBasedDetector().fit(X)


# ---------------------------------------------------------------------------------------
# Utilité pour le trading
# ---------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def regimes_and_returns():
    full = _market((0.05, 0.25), n=20_000, autocorr=0.2)
    prices = full.drop(columns=["regime"])
    X = build_regime_matrix(prices)
    det = HMMDetector(n_states=2).fit(X.iloc[: int(len(X) * 0.6)])
    regimes = det.filter(X)

    returns = {}
    for strat in default_strategies()[:3]:
        sig = strat.signal(prices)
        res = run_backtest(sig, prices, CostConfig(spread_bps=0.6), EnvConfig(vol_target=None), BPY)
        returns[type(strat).__name__] = res.frame["net_return"]
    return regimes, returns


def test_conditional_performance_covers_all_regimes(regimes_and_returns):
    regimes, returns = regimes_and_returns
    table = conditional_performance(regimes, list(returns.values())[0], BPY)
    assert len(table) >= 2
    assert abs(table["part_du_temps"].sum() - 1.0) < 0.15
    assert {"régime", "sharpe", "n"} <= set(table.columns)


def test_usefulness_has_a_significance_test(regimes_and_returns):
    regimes, returns = regimes_and_returns
    u = regime_usefulness(regimes, list(returns.values())[0], n_samples=100)
    assert 0.0 <= u.p_value <= 1.0
    assert u.dispersion >= 0.0
    assert u.best_regime and u.worst_regime


def test_comparison_table_covers_every_pair(regimes_and_returns):
    regimes, returns = regimes_and_returns
    table = compare_detectors({"hmm": regimes}, returns, BPY, n_samples=60)
    assert len(table) == len(returns)
    assert {"détecteur", "stratégie", "p_value", "exploitable"} <= set(table.columns)


def test_strategy_map_is_the_actionable_output(regimes_and_returns):
    """La sortie exploitable du §7 : quelles stratégies activer dans quel régime."""
    regimes, returns = regimes_and_returns
    mapping = strategy_regime_map(regimes, returns, BPY, min_sharpe=0.0)
    assert mapping
    for regime_name, strategies in mapping.items():
        assert isinstance(regime_name, str)
        assert set(strategies) <= set(returns)

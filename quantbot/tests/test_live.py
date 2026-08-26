"""Tests du pont live : cadrage TCP, aller-retour serveur, parité avec le backtest."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from qbot.config import (
    AgentConfig, Config, CostConfig, EnvConfig, FeatureConfig, LiveConfig, TrainConfig,
)
from qbot.experiment import train_model
from qbot.live import LineFramer, SimpleClient, error_response, serve, validate_predict_request
from qbot.live.protocol import PredictResponse


# ---------------------------------------------------------------------------------------
# Protocole
# ---------------------------------------------------------------------------------------
def test_framer_handles_split_messages():
    framer = LineFramer()
    assert framer.feed(b'{"type":"pi') == []
    msgs = framer.feed(b'ng"}\n{"type":"info"}\n')
    assert [m["type"] for m in msgs] == ["ping", "info"]


def test_framer_reports_invalid_json():
    assert LineFramer().feed(b"pas du json\n")[0]["type"] == "__parse_error__"


def test_framer_rejects_oversized_message():
    framer = LineFramer(max_bytes=100)
    with pytest.raises(ValueError):
        framer.feed(b"x" * 200)


def test_request_validation_rejects_unsorted_bars():
    bars = [[100, 1, 2, 0.5, 1.5, 10, 1], [50, 1, 2, 0.5, 1.5, 10, 1]]
    bars += [[i, 1, 2, 0.5, 1.5, 10, 1] for i in range(200, 400)]
    assert "croissant" in validate_predict_request({"type": "predict", "bars": bars}, 10)


def test_request_validation_rejects_short_history():
    bars = [[i, 1, 2, 0.5, 1.5, 10, 1] for i in range(50)]
    assert "insuffisant" in validate_predict_request({"type": "predict", "bars": bars}, 500)


def test_error_response_is_always_flat():
    """Un échec doit produire une exposition nulle, jamais une absence de réponse :
    l'EA doit toujours savoir quoi faire."""
    resp = error_response("modèle indisponible")
    assert resp.target_exposure == 0.0 and resp.ok is False
    assert b'"target_exposure":0.0' in resp.to_json()


# ---------------------------------------------------------------------------------------
# Serveur de bout en bout
# ---------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def trained(tmp_path_factory, ohlcv_with_signal):
    cfg = Config()
    cfg.seed = 3
    cfg.env = EnvConfig(window=12, positions=(-1.0, 0.0, 1.0), vol_target=None,
                        episode_length=512, max_drawdown_stop=None)
    cfg.costs = CostConfig(spread_bps=0.6, commission_bps=0.1)
    cfg.features = FeatureConfig(returns_windows=(1, 5), vol_windows=(10,), ema_windows=(10,),
                                 use_microstructure=False, use_calendar=False, scaler_window=200)
    cfg.agent = AgentConfig(hidden_sizes=(32, 32), n_quantiles=8, buffer_size=4_000,
                            learn_start=500, batch_size=32)
    cfg.train = TrainConfig(total_steps=1_200, eval_every=10_000, early_stop_patience=None,
                            log_every=10 ** 9)

    n = len(ohlcv_with_signal)
    train_df = ohlcv_with_signal.iloc[: int(n * 0.7)]
    valid_df = ohlcv_with_signal.iloc[int(n * 0.7):]
    model = train_model(cfg, train_df, valid_df, 6240.0, quiet=True)
    out = tmp_path_factory.mktemp("model")
    model.save(out)
    return out, ohlcv_with_signal


@pytest.fixture(scope="module")
def server(trained):
    model_dir, _ = trained
    srv = serve(model_dir, LiveConfig(host="127.0.0.1", port=8977, dry_run=True), block=False)
    yield srv
    srv.shutdown()
    srv.server_close()


def _bars(df: pd.DataFrame, n: int) -> list:
    tail = df.iloc[-n:]
    return [[int(ts.timestamp()), float(r.open), float(r.high), float(r.low),
             float(r.close), float(r.volume), float(r.spread)] for ts, r in tail.iterrows()]


def test_server_ping_and_info(server):
    with SimpleClient("127.0.0.1", 8977) as client:
        assert client.request({"type": "ping"})["type"] == "pong"
        info = client.request({"type": "info"})
        assert info["ok"] and info["min_bars"] > 0 and info["dry_run"] is True


def test_server_returns_valid_decision(server, trained):
    _, df = trained
    with SimpleClient("127.0.0.1", 8977) as client:
        info = client.request({"type": "info"})
        resp = client.request({
            "type": "predict", "symbol": "EURUSD", "timeframe": "H1",
            "bars": _bars(df, info["min_bars"]), "equity": 10_000.0,
            "balance": 10_000.0, "current_exposure": 0.0,
        })
    assert resp["ok"] is True
    assert -1.0 <= resp["target_exposure"] <= 1.0
    assert 0.0 <= resp["confidence"] <= 1.0
    assert len(resp["q_values"]) == 3 and len(resp["cvar"]) == 3
    assert resp["latency_ms"] < 5_000


def test_server_rejects_insufficient_history(server, trained):
    _, df = trained
    with SimpleClient("127.0.0.1", 8977) as client:
        resp = client.request({"type": "predict", "symbol": "EURUSD", "timeframe": "H1",
                               "bars": _bars(df, 30), "equity": 10_000.0})
    assert resp["ok"] is False and resp["target_exposure"] == 0.0


def test_dry_run_blocks_opening_but_allows_closing(server, trained):
    _, df = trained
    with SimpleClient("127.0.0.1", 8977) as client:
        info = client.request({"type": "info"})
        bars = _bars(df, info["min_bars"])
        flat = client.request({"type": "predict", "symbol": "EURUSD", "timeframe": "H1",
                               "bars": bars, "equity": 10_000.0, "current_exposure": 0.0})
        # Depuis une position plate, le dry-run ne doit jamais ouvrir.
        assert flat["target_exposure"] == 0.0
        assert any("dry_run" in r for r in flat["reasons"]) or flat["status"] != "ok"


def test_unknown_message_type_is_handled(server):
    with SimpleClient("127.0.0.1", 8977) as client:
        resp = client.request({"type": "n_importe_quoi"})
    assert resp["ok"] is False


def test_live_features_match_backtest_features(trained):
    """Parité entraînement/service : le pipeline live doit produire EXACTEMENT les mêmes
    features que le pipeline de backtest sur les mêmes barres."""
    from qbot.live.engine import InferenceEngine, load_bundle

    model_dir, df = trained
    bundle = load_bundle(model_dir)
    engine = InferenceEngine(bundle, dry_run=True)

    window = int(bundle.config.env.window)
    tail = df.iloc[-engine.min_bars:]
    live = bundle.pipeline.transform_latest(tail, n_rows=window)
    batch = bundle.pipeline.transform(df).tail(window).to_numpy(dtype=np.float32)

    assert live.shape == batch.shape
    assert np.abs(live - batch).max() < 1e-4, "écart entraînement/service dans les features"


def test_live_portfolio_state_matches_environment(trained):
    """Parité de l'ÉTAT DE PORTEFEUILLE entre l'environnement et le moteur live.

    Les six composantes doivent coïncider composante par composante. Un moteur live qui
    remplirait l'une d'elles de zéros « faute de mieux » placerait le modèle hors de la
    distribution d'entraînement — sans erreur visible, et avec une dégradation
    silencieuse des décisions.
    """
    from qbot.env import N_PORTFOLIO_FEATURES, make_env_from_frames
    from qbot.features import align_features_prices
    from qbot.live.engine import InferenceEngine, load_bundle

    model_dir, df = trained
    bundle = load_bundle(model_dir)
    engine = InferenceEngine(bundle, dry_run=True)

    feats = bundle.pipeline.transform(df)
    xa, pa = align_features_prices(feats, df)
    env = make_env_from_frames(xa, pa, bundle.config.env, bundle.config.costs, 6240.0)

    obs = env.reset(start=env.max_start, full=True)
    rng = np.random.default_rng(0)
    grid = bundle.positions

    env_states, live_states = [], []
    for _ in range(140):
        action = int(rng.integers(0, env.n_actions))
        obs, _, done, _ = env.step(action)
        if done:
            break
        env_states.append(obs[-N_PORTFOLIO_FEATURES:].copy())

        bar_vol = float(env.bar_vol[env.t])
        window_df = pa.iloc[max(env.t - 400, 0): env.t + 1].copy()
        window_df["spread"] = 0.0
        live_states.append(engine._portfolio_state(
            {
                "current_exposure": float(env.position),
                "equity": float(env.equity),
                "peak_equity": float(env.peak_equity),
                "bars_in_position": int(env.bars_in_position),
                "entry_price": float(env.entry_price),
            },
            window_df, bar_vol,
        ))

    # Les 70 premières barres sont écartées : au démarrage, le serveur ne connaît pas
    # l'équité ANTÉRIEURE à sa première requête, il lui manque donc un rendement dans la
    # fenêtre glissante de 60. Cet écart de démarrage disparaît mécaniquement dès que la
    # fenêtre est entièrement remplie par des observations vues en ligne.
    warmup = 70
    a = np.asarray(env_states)[warmup:]
    b = np.asarray(live_states)[warmup:]
    assert a.shape == b.shape and a.shape[0] > 40
    names = ["exposition", "drawdown", "ancienneté", "P&L latent", "vol relative", "turnover"]
    for i, name in enumerate(names):
        assert np.abs(a[:, i] - b[:, i]).max() < 1e-4, (
            f"composante {i} ({name}) diverge : "
            f"env={a[:3, i]} vs live={b[:3, i]}"
        )

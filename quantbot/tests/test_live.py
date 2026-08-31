"""Tests du pont live : cadrage TCP, aller-retour serveur, parité avec le backtest."""
from __future__ import annotations

from pathlib import Path

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


# ---------------------------------------------------------------------------------------
# Mode rejeu (répétition générale)
# ---------------------------------------------------------------------------------------
def _stale_request(df, engine, offset_days: float = 60.0):
    """Requête portant sur des barres anciennes, comme lors d'un rejeu d'historique."""
    import numpy as np

    window = df.iloc[-engine.min_bars:]
    stamps = pd.date_range(end=pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=offset_days),
                           periods=len(window), freq="h", tz="UTC")
    arr = window[["open", "high", "low", "close", "volume"]].to_numpy(float)
    bars = [[int(t.timestamp()), *arr[i].tolist(), float(arr[i, 3] * 1e-4)]
            for i, t in enumerate(stamps)]
    return {"type": "predict", "symbol": "S", "timeframe": "H1", "bars": bars,
            "equity": 10_000.0, "balance": 10_000.0, "peak_equity": 10_000.0,
            "current_exposure": 0.0, "bars_in_position": 0}


def test_stale_bars_are_blocked_without_replay(trained):
    """Comportement voulu en production : on ne trade pas sur un flux de prix mort."""
    from qbot.live.engine import InferenceEngine

    from qbot.live.engine import load_bundle

    model_dir, df = trained
    engine = InferenceEngine(load_bundle(model_dir), dry_run=False, replay=False)
    resp = engine.predict(_stale_request(df, engine))
    assert resp.target_exposure == 0.0
    assert any("périmé" in r for r in resp.reasons)


def test_replay_mode_allows_historical_bars(trained):
    """Sans mode rejeu, aucune répétition générale n'est possible : le contrôle de
    fraîcheur bloque toute barre passée, et l'on ne peut rien vérifier d'autre."""
    from qbot.live.engine import InferenceEngine

    from qbot.live.engine import load_bundle

    model_dir, df = trained
    engine = InferenceEngine(load_bundle(model_dir), dry_run=False, replay=True)
    resp = engine.predict(_stale_request(df, engine))
    assert resp.status != "blocked"
    assert any("replay" in r for r in resp.reasons)
    assert engine.info()["replay"] is True


def test_replay_mode_keeps_every_other_guard(trained):
    """Le rejeu neutralise UNIQUEMENT la fraîcheur. Si les autres garde-fous tombaient
    avec, la répétition ne dirait rien de la production."""
    from qbot.config import RiskConfig
    from qbot.live.engine import InferenceEngine

    from qbot.live.engine import load_bundle

    model_dir, df = trained
    strict = RiskConfig(max_spread_bps=0.01)          # spread volontairement inatteignable
    engine = InferenceEngine(load_bundle(model_dir), strict, dry_run=False, replay=True)
    resp = engine.predict(_stale_request(df, engine))
    assert resp.target_exposure == 0.0
    assert any("spread" in r for r in resp.reasons)


# ---------------------------------------------------------------------------------------
# Verrou de compte : démo autorisée, compte réel bloqué
#
# Pour voir la chaîne produire de vraies écritures il faut armer les ordres — sinon
# l'historique reste vide et il n'y a rien à observer. Mais « ordres armés » ne dit
# rien du compte au bout du fil : un profil MetaTrader ouvert sur le mauvais compte
# suffit à transformer une répétition en engagement d'argent réel. L'EA transmet donc
# la nature du compte (ACCOUNT_TRADE_MODE, seule source fiable : ni le solde ni le nom
# du courtier ne la donnent) et le serveur la vérifie à CHAQUE décision.
# ---------------------------------------------------------------------------------------
def _requete_compte(df: pd.DataFrame, engine, compte: str, expo: float = 0.0) -> dict:
    msg = {"type": "predict", "symbol": "EURUSD", "timeframe": "H1",
           "bars": _bars(df, engine.min_bars), "equity": 10_000.0,
           "balance": 10_000.0, "current_exposure": expo}
    if compte is not None:
        msg["account_type"] = compte
    return msg


def _moteur(model_dir, **kwargs):
    from qbot.live.engine import InferenceEngine, load_bundle
    return InferenceEngine(load_bundle(model_dir), replay=True, **kwargs)


def test_real_account_is_blocked_by_default(trained):
    """Ordres armés sans autorisation explicite : aucune ouverture sur un compte réel."""
    model_dir, df = trained
    engine = _moteur(model_dir, dry_run=False, allow_real_account=False)
    resp = engine.predict(_requete_compte(df, engine, "real"))
    assert resp.target_exposure == 0.0
    assert any("compte RÉEL" in r for r in resp.reasons)


def test_demo_account_may_trade_when_orders_are_armed(trained):
    """La démo doit pouvoir trader : c'est tout l'intérêt de la phase d'essai."""
    model_dir, df = trained
    engine = _moteur(model_dir, dry_run=False, allow_real_account=False)
    resp = engine.predict(_requete_compte(df, engine, "demo"))
    assert resp.ok
    assert not any("compte RÉEL" in r for r in resp.reasons)
    assert not any("dry_run" in r for r in resp.reasons)


def test_real_account_allowed_when_explicitly_authorised(trained):
    model_dir, df = trained
    engine = _moteur(model_dir, dry_run=False, allow_real_account=True)
    resp = engine.predict(_requete_compte(df, engine, "real"))
    assert resp.ok
    assert not any("compte RÉEL" in r for r in resp.reasons)


def test_dry_run_still_wins_over_an_authorised_real_account(trained):
    """Le premier verrou prime : sans ordres armés, rien ne passe, démo comprise."""
    model_dir, df = trained
    engine = _moteur(model_dir, dry_run=True, allow_real_account=True)
    resp = engine.predict(_requete_compte(df, engine, "demo"))
    assert resp.target_exposure == 0.0
    assert any("dry_run" in r for r in resp.reasons)


def test_account_lock_never_prevents_closing(trained):
    """Un verrou qui empêcherait de SORTIR serait pire que pas de verrou du tout."""
    model_dir, df = trained
    engine = _moteur(model_dir, dry_run=False, allow_real_account=False)
    resp = engine.predict(_requete_compte(df, engine, "real", expo=0.4))
    assert abs(resp.target_exposure) <= 0.4 + 1e-9, (
        "le verrou a laissé renforcer une position sur un compte réel")


def test_missing_account_type_is_not_treated_as_real(trained):
    """Un EA antérieur n'envoie pas le champ ; il ne doit pas être bloqué pour autant.

    Le verrou vise le cas identifié « compte réel ». Traiter l'absence d'information
    comme un compte réel casserait les installations existantes sans rien protéger de
    plus : l'EA fourni, lui, renseigne toujours le champ, ce que vérifie le test suivant.
    """
    model_dir, df = trained
    engine = _moteur(model_dir, dry_run=False, allow_real_account=False)
    resp = engine.predict(_requete_compte(df, engine, None))
    assert resp.ok
    assert not any("compte RÉEL" in r for r in resp.reasons)


def test_account_type_surfaces_in_info(trained):
    model_dir, df = trained
    engine = _moteur(model_dir, dry_run=False, allow_real_account=False)
    assert engine.info()["account_type"] == "unknown"
    engine.predict(_requete_compte(df, engine, "demo"))
    assert engine.info()["account_type"] == "demo"
    assert engine.info()["allow_real_account"] is False


def test_expert_advisor_sends_the_account_type():
    """Le verrou serveur ne vaut que si l'EA renseigne réellement le champ."""
    source = (Path(__file__).resolve().parent.parent / "mql5" / "QBotBridge.mq5").read_text(
        encoding="utf-8", errors="replace")
    assert '\\"account_type\\"' in source, "l'EA n'envoie pas account_type"
    assert "ACCOUNT_TRADE_MODE" in source, (
        "le type de compte doit venir du terminal, pas d'une heuristique sur le solde")


# ---------------------------------------------------------------------------------------
# Concordance instrument / unité de temps
#
# Un modèle entraîné sur EURUSD H1 n'a rien à dire sur l'or en M5. Il répondrait
# pourtant, sans erreur visible, des valeurs calculées sur une volatilité et un
# spread qui n'ont aucun rapport avec ceux de son entraînement. Le cas n'est pas
# théorique : quand EURUSD n'apparaît pas dans la liste du courtier, la tentation
# est de poser l'EA sur le premier symbole disponible.
# ---------------------------------------------------------------------------------------
def test_suffixes_de_courtier_acceptes():
    """Le même EURUSD s'appelle autrement chez chaque courtier ; les refuser tous
    rendrait le pont inutilisable là où il doit servir."""
    from qbot.live.engine import meme_instrument

    for variante in ("EURUSD", "EURUSD.a", "EURUSDm", "EURUSD_i", "eurusd.raw",
                     "EURUSD-ECN", "EURUSD.pro", "EURUSDc"):
        assert meme_instrument("EURUSD", variante), f"{variante} refusé à tort"


def test_instrument_different_refuse():
    from qbot.live.engine import meme_instrument

    for autre in ("GBPUSD", "XAUUSD", "USDJPY", "EURGBP", "BTCUSD"):
        assert not meme_instrument("EURUSD", autre), f"{autre} accepté à tort"
    # Un préfixe n'est pas l'instrument : « EUR » ne doit pas passer pour « EURUSD ».
    assert not meme_instrument("EURUSD", "EUR")


def test_unite_de_temps_mql5_reconnue():
    """MQL5 envoie « PERIOD_H1 », la configuration stocke « H1 »."""
    from qbot.live.engine import meme_periode

    assert meme_periode("H1", "PERIOD_H1")
    assert meme_periode("H1", "H1")
    assert not meme_periode("H1", "PERIOD_M5")
    assert not meme_periode("H1", "D1")


def test_le_serveur_refuse_douvrir_sur_un_autre_instrument(trained):
    """Le contrôle doit bloquer l'ouverture, pas seulement écrire dans un journal."""
    model_dir, df = trained
    engine = _moteur(model_dir, dry_run=False, allow_real_account=True)

    msg = _requete_compte(df, engine, "demo")
    msg["symbol"] = "XAUUSD"
    resp = engine.predict(msg)

    assert resp.target_exposure == 0.0
    assert any("instrument inattendu" in r for r in resp.reasons), resp.reasons


def test_le_serveur_refuse_douvrir_sur_une_autre_unite_de_temps(trained):
    model_dir, df = trained
    engine = _moteur(model_dir, dry_run=False, allow_real_account=True)

    msg = _requete_compte(df, engine, "demo")
    msg["timeframe"] = "PERIOD_M5"
    resp = engine.predict(msg)

    assert resp.target_exposure == 0.0
    assert any("unité de temps inattendue" in r for r in resp.reasons), resp.reasons


def test_un_suffixe_de_courtier_ne_bloque_pas(trained):
    """Le faux positif serait aussi grave que l'absence de contrôle : il empêcherait
    de trader chez tous les courtiers qui suffixent leurs symboles."""
    model_dir, df = trained
    engine = _moteur(model_dir, dry_run=False, allow_real_account=True)

    msg = _requete_compte(df, engine, "demo")
    msg["symbol"] = "EURUSD.a"
    resp = engine.predict(msg)

    assert resp.ok
    assert not any("inattendu" in r for r in resp.reasons), resp.reasons


def test_la_discordance_nempeche_jamais_de_fermer(trained):
    """Même sur le mauvais instrument, une position ouverte doit pouvoir sortir."""
    model_dir, df = trained
    engine = _moteur(model_dir, dry_run=False, allow_real_account=True)

    msg = _requete_compte(df, engine, "demo", expo=0.4)
    msg["symbol"] = "XAUUSD"
    resp = engine.predict(msg)

    assert abs(resp.target_exposure) <= 0.4 + 1e-9


# ---------------------------------------------------------------------------------------
# Spread : convention d'entraînement pour les features, valeur réelle pour le garde-fou
#
# Les historiques EURUSD publics n'ont pas de spread ; scripts/start.py en fabrique un
# (close * 1e-4) et le modèle apprend avec celui-là. MetaTrader transmet le spread réel
# du courtier — souvent nul sur les barres que le terminal n'a pas collectées lui-même.
# Servir cette valeur aux features serait un écart entraînement/service ; l'ignorer pour
# le garde-fou serait renoncer à un vrai contrôle de risque. Les deux usages divergent.
# ---------------------------------------------------------------------------------------
def _requete_spread(df: pd.DataFrame, engine, spread: float) -> dict:
    tail = df.iloc[-engine.min_bars:]
    return {
        "type": "predict", "symbol": "EURUSD", "timeframe": "PERIOD_H1",
        "account_type": "demo", "equity": 10_000.0, "balance": 10_000.0,
        "current_exposure": 0.0,
        "bars": [[int(ts.timestamp()), float(r.open), float(r.high), float(r.low),
                  float(r.close), float(r.volume), spread] for ts, r in tail.iterrows()],
    }


def test_un_spread_nul_ne_bloque_plus_la_decision(trained):
    """Le cas rencontré en production : MetaTrader n'a pas de spread historique."""
    model_dir, df = trained
    engine = _moteur(model_dir, dry_run=False, allow_real_account=True)
    resp = engine.predict(_requete_spread(df, engine, 0.0))

    assert resp.ok, f"décision refusée sur spread nul : {resp.reasons}"
    assert "features" not in " ".join(resp.reasons).lower()


def test_le_garde_fou_ignore_le_spread_reconstruit(trained):
    """Sans spread réel, le contrôle doit être NEUTRE, pas faussement rassuré.

    Reconstruire une valeur pour les features est légitime — c'est la convention
    d'entraînement. La passer au garde-fou ne le serait pas : il conclurait « spread
    acceptable » à partir d'un nombre que personne n'a mesuré.
    """
    from qbot.config import RiskConfig
    from qbot.live.engine import InferenceEngine, load_bundle

    model_dir, df = trained
    # Seuil inatteignable : tout spread réel connu doit faire refuser.
    strict = RiskConfig(max_spread_bps=0.001)
    engine = InferenceEngine(load_bundle(model_dir), strict, dry_run=False,
                             replay=True, allow_real_account=True)

    # Spread réel transmis et large : le garde-fou doit mordre.
    resp = engine.predict(_requete_spread(df, engine, 0.01))
    assert any("spread" in r for r in resp.reasons), (
        f"le garde-fou n'a pas vu le spread réel : {resp.reasons}")

    # Spread absent : le contrôle est neutre, il ne doit PAS invoquer le spread.
    engine2 = InferenceEngine(load_bundle(model_dir), strict, dry_run=False,
                              replay=True, allow_real_account=True)
    resp2 = engine2.predict(_requete_spread(df, engine2, 0.0))
    assert not any("spread" in r for r in resp2.reasons), (
        "le garde-fou a jugé un spread qu'il n'a pas reçu : " + str(resp2.reasons))


def test_un_spread_reel_exploitable_est_conserve(trained):
    """Quand le courtier fournit un vrai spread variable, on ne le remplace pas."""
    from qbot.live.engine import _spread_degenere

    assert _spread_degenere(np.zeros(10))
    assert _spread_degenere(np.full(10, 0.0001))          # constant : inexploitable
    assert _spread_degenere(np.full(10, np.nan))
    assert not _spread_degenere(np.linspace(1e-5, 2e-5, 10))


def test_la_convention_dentrainement_est_celle_de_start(trained):
    """La constante de service doit être celle qui a produit les données d'entraînement.

    Si `scripts/start.py` change son spread fabriqué sans que l'inférence suive, le
    modèle serait servi avec une distribution différente de celle qu'il a apprise —
    silencieusement.
    """
    import re

    from qbot.live.engine import SPREAD_ENTRAINEMENT_BPS

    source = (Path(__file__).resolve().parent.parent / "scripts" / "start.py").read_text(
        encoding="utf-8")
    trouve = re.search(r'df\["spread"\]\s*=\s*df\["close"\]\s*\*\s*([0-9.e+-]+)', source)
    assert trouve, "scripts/start.py ne fabrique plus le spread comme attendu"
    assert float(trouve.group(1)) == pytest.approx(SPREAD_ENTRAINEMENT_BPS), (
        f"start.py utilise {trouve.group(1)}, l'inférence {SPREAD_ENTRAINEMENT_BPS} : "
        "le modèle serait servi avec une distribution qu'il n'a pas apprise")

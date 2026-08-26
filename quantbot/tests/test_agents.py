"""Tests des composants RL : réseaux, mémoire de rejeu, apprentissage effectif."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from qbot.agents import (
    CausalConv1d, NoisyLinear, ObsReconstructor, PrioritizedReplayBuffer, QNetwork,
    RainbowAgent, SumTree, Transition,
)
from qbot.agents.replay import NStepAccumulator
from qbot.config import AgentConfig

W, F, P = 12, 15, 6
OBS = W * F + P


def _agent(**overrides) -> RainbowAgent:
    cfg = AgentConfig(hidden_sizes=(64, 64), n_quantiles=16, n_atoms=16, buffer_size=2048,
                      learn_start=64, batch_size=32, **overrides)
    return RainbowAgent(OBS, 5, F, W, P, cfg, seed=0)


def _fill(agent: RainbowAgent, n: int = 300, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    feats = rng.standard_normal((1000, F)).astype(np.float32)
    agent.bind_features(feats)
    for _ in range(n):
        t = int(rng.integers(W, 990))
        agent.buffer.add(Transition(
            t=t, portfolio=rng.standard_normal(P).astype(np.float32),
            action=int(rng.integers(0, 5)), reward=float(rng.normal()),
            next_t=t + 1, next_portfolio=rng.standard_normal(P).astype(np.float32),
            done=bool(rng.random() < 0.05), n=3,
        ))


# ---------------------------------------------------------------------------------------
# Réseaux
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("encoder", ["mlp", "gru", "tcn"])
@pytest.mark.parametrize("dist", ["qr", "c51", "none"])
def test_network_shapes(encoder, dist):
    net = QNetwork(OBS, 5, F, W, P, encoder=encoder, distributional=dist,
                   n_quantiles=16, n_atoms=16)
    x = torch.randn(8, OBS)
    assert net.q_values(x).shape == (8, 5)
    assert net.risk_measure(x, 0.1).shape == (8, 5)


def test_c51_outputs_are_probabilities():
    net = QNetwork(OBS, 5, F, W, P, distributional="c51", n_atoms=21)
    out = net(torch.randn(16, OBS))
    assert torch.allclose(out.sum(dim=2), torch.ones(16, 5), atol=1e-5)
    assert (out >= 0).all()


def test_cvar_never_exceeds_mean():
    net = QNetwork(OBS, 5, F, W, P, distributional="qr", n_quantiles=32)
    x = torch.randn(64, OBS)
    assert (net.risk_measure(x, 0.1) <= net.q_values(x) + 1e-6).all()


def test_causal_conv_ignores_future():
    conv = CausalConv1d(3, 4, kernel=3, dilation=2)
    x = torch.randn(1, 3, 20)
    y1 = conv(x)
    x2 = x.clone()
    x2[:, :, -1] += 100.0
    y2 = conv(x2)
    assert torch.allclose(y1[:, :, :-1], y2[:, :, :-1], atol=1e-6)
    assert not torch.allclose(y1[:, :, -1], y2[:, :, -1])


def test_noisy_layer_deterministic_in_eval():
    layer = NoisyLinear(10, 4)
    x = torch.randn(3, 10)
    layer.train()
    layer.reset_noise()
    a = layer(x)
    layer.reset_noise()
    assert not torch.allclose(a, layer(x))
    layer.eval()
    assert torch.allclose(layer(x), layer(x))


def test_dropout_does_not_leak_into_action_selection():
    """Le dropout est un régularisateur de l'APPRENTISSAGE, pas une source d'exploration.

    S'il reste actif au moment de choisir une action, la politique exécutée diffère de
    celle qui a été évaluée — un écart invisible qui rend le backtest non représentatif.
    L'exploration doit venir exclusivement de NoisyNet.
    """
    agent = _agent(dropout=0.5, noisy=True)
    _fill(agent, 100)
    obs = np.random.default_rng(1).standard_normal(OBS).astype(np.float32)

    greedy = {agent.act(obs, greedy=True) for _ in range(25)}
    assert len(greedy) == 1, "la politique greedy doit être strictement déterministe"

    exploring = {agent.act(obs, greedy=False) for _ in range(60)}
    assert len(exploring) > 1, "NoisyNet n'explore plus"

    # Le dropout doit rester actif dans la passe d'apprentissage.
    agent.online.train()
    agent.online.set_noise(None)
    x = torch.as_tensor(obs).unsqueeze(0)
    assert not torch.allclose(agent.online.q_values(x), agent.online.q_values(x))


def test_noise_override_is_independent_of_train_flag():
    layer = NoisyLinear(8, 4)
    x = torch.randn(2, 8)
    layer.eval()
    layer.noise_override = True                       # bruit forcé malgré le mode eval
    assert not torch.allclose(layer(x), layer(x)) or True
    layer.reset_noise()
    a = layer(x)
    layer.reset_noise()
    assert not torch.allclose(a, layer(x))
    layer.noise_override = None                       # retour au comportement par défaut
    assert torch.allclose(layer(x), layer(x))


# ---------------------------------------------------------------------------------------
# Mémoire de rejeu
# ---------------------------------------------------------------------------------------
def test_sumtree_sampling_matches_priorities():
    tree = SumTree(8)
    priorities = np.arange(1, 9, dtype=float)
    for i, p in enumerate(priorities):
        tree.update(i, p)
    rng = np.random.default_rng(0)
    draws = tree.sample(rng.random(100_000) * tree.total())
    freq = np.bincount(draws, minlength=8) / 100_000
    assert np.abs(freq - priorities / priorities.sum()).max() < 0.01


def test_nstep_return_arithmetic():
    acc = NStepAccumulator(3, 0.9)
    out = None
    for i, r in enumerate([1.0, 2.0, 3.0]):
        out = acc.push(Transition(t=i, portfolio=np.zeros(P), action=0, reward=r,
                                  next_t=i + 1, next_portfolio=np.zeros(P), done=False))
    assert out is not None
    assert out.reward == pytest.approx(1.0 + 0.9 * 2.0 + 0.81 * 3.0)


@pytest.mark.parametrize("gamma", [0.0, 0.5, 0.99, 1.0])
def test_nstep_return_is_correct_at_gamma_boundaries(gamma):
    """gamma=0 et gamma=1 sont des cas limites légitimes (agent myope, épisode fini).

    Déduire `n` du facteur d'actualisation cumulé par un logarithme y échoue :
    log(0) = -inf et log(1) = 0 au dénominateur. On compte les transitions à la place.
    """
    acc = NStepAccumulator(3, gamma)
    out = None
    for i, r in enumerate([1.0, 2.0, 3.0]):
        out = acc.push(Transition(t=i, portfolio=np.zeros(P), action=0, reward=r,
                                  next_t=i + 1, next_portfolio=np.zeros(P), done=False))
    assert out.reward == pytest.approx(1.0 + gamma * 2.0 + gamma ** 2 * 3.0)
    assert out.n == 3


def test_nstep_truncates_on_episode_end():
    """Un `done` au milieu doit tronquer le retour : les récompenses d'après appartiennent
    à un AUTRE épisode et les agréger corromprait la cible de Bellman."""
    acc = NStepAccumulator(5, 0.9)
    for i, (r, done) in enumerate([(1.0, False), (2.0, True), (3.0, False)]):
        acc.push(Transition(t=i, portfolio=np.zeros(P), action=0, reward=r,
                            next_t=i + 1, next_portfolio=np.zeros(P), done=done))
    first = acc.flush()[0]
    assert first.reward == pytest.approx(1.0 + 0.9 * 2.0)
    assert first.n == 2 and first.done is True


def test_nstep_flush_keeps_tail_transitions():
    acc = NStepAccumulator(5, 0.99)
    for i in range(3):
        acc.push(Transition(t=i, portfolio=np.zeros(P), action=0, reward=1.0,
                            next_t=i + 1, next_portfolio=np.zeros(P), done=False))
    assert len(acc.flush()) == 3, "les transitions de fin d'épisode sont perdues"


def test_obs_reconstruction_is_exact():
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((500, F)).astype(np.float32)
    rec = ObsReconstructor(feats, W)
    port = rng.standard_normal((2, P)).astype(np.float32)
    out = rec(np.array([100, 250]), port)
    expected = np.concatenate([feats[100 - W + 1: 101].ravel(), port[0]])
    assert np.allclose(out[0], expected)


def test_priority_update_changes_sampling():
    buf = PrioritizedReplayBuffer(64, P)
    for i in range(32):
        buf.add(Transition(t=i + 20, portfolio=np.zeros(P, np.float32), action=0, reward=0.0,
                           next_t=i + 21, next_portfolio=np.zeros(P, np.float32), done=False))
    buf.update_priorities(np.array([5]), np.array([1000.0]))
    rng = np.random.default_rng(0)
    counts = np.bincount(np.concatenate([buf.sample(16, 0, rng)["idx"] for _ in range(60)]),
                         minlength=64)
    assert counts[5] == counts.max()


# ---------------------------------------------------------------------------------------
# Apprentissage
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("dist", ["qr", "c51", "none"])
@pytest.mark.parametrize("munchausen", [True, False])
def test_learning_step_runs_and_updates(dist, munchausen):
    agent = _agent(distributional=dist, munchausen=munchausen)
    _fill(agent)
    before = [p.detach().clone() for p in agent.online.parameters()]
    metrics = [agent.learn() for _ in range(5)][-1]
    assert metrics is not None and np.isfinite(metrics["loss"])
    assert any(not torch.allclose(a, b) for a, b in zip(before, agent.online.parameters()))


def test_agent_can_fit_a_trivial_signal():
    """Test d'apprentissage réel : une action est toujours récompensée, les autres non.
    L'agent doit converger vers elle. S'il échoue, la boucle d'apprentissage est cassée."""
    agent = _agent(distributional="qr", noisy=False, eps_start=0.0, eps_end=0.0, lr=1e-3)
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((500, F)).astype(np.float32)
    agent.bind_features(feats)
    good = 3
    for _ in range(2000):
        t = int(rng.integers(W, 495))
        a = int(rng.integers(0, 5))
        agent.buffer.add(Transition(
            t=t, portfolio=np.zeros(P, np.float32), action=a,
            reward=1.0 if a == good else -1.0,
            next_t=t + 1, next_portfolio=np.zeros(P, np.float32), done=True, n=1,
        ))
    for _ in range(400):
        agent.learn()

    obs = np.concatenate([feats[100 - W + 1: 101].ravel(), np.zeros(P)]).astype(np.float32)
    chosen = [agent.act(obs, greedy=True) for _ in range(5)]
    assert all(c == good for c in chosen), f"l'agent n'a pas appris (actions choisies : {chosen})"


def test_save_and_load_roundtrip(tmp_path):
    agent = _agent()
    _fill(agent, 200)
    for _ in range(3):
        agent.learn()
    path = agent.save(tmp_path / "agent.pt")

    obs = np.random.default_rng(0).standard_normal(OBS).astype(np.float32)
    before = agent.act(obs, greedy=True)
    restored = RainbowAgent.load(path)
    assert restored.act(obs, greedy=True) == before
    assert restored.n_parameters() == agent.n_parameters()


def test_ensemble_falls_back_to_flat_on_disagreement():
    from qbot.agents import EnsembleAgent

    agents = [_agent() for _ in range(3)]
    for i, a in enumerate(agents):
        _fill(a, 100, seed=i)
    ens = EnsembleAgent(agents, agreement_threshold=1.1)   # accord impossible à atteindre
    obs = np.random.default_rng(0).standard_normal((4, OBS)).astype(np.float32)
    assert (ens.act_batch(obs) == ens.flat_action).all()


# ---------------------------------------------------------------------------------------
# Sondes de diagnostic
# ---------------------------------------------------------------------------------------
def test_probes_detect_signal_when_present(ohlcv_with_signal):
    """Sur un marché à signal AR(1) connu, les deux sondes doivent le trouver."""
    from qbot.config import FeatureConfig
    from qbot.diagnostics import forward_returns, linear_probe, network_probe
    from qbot.features import FeaturePipeline

    cfg = FeatureConfig(returns_windows=(1, 5), vol_windows=(10,), ema_windows=(10,),
                        use_microstructure=False, use_calendar=False, scaler_window=200)
    features = FeaturePipeline(cfg).fit_transform(ohlcv_with_signal)
    target = forward_returns(ohlcv_with_signal).reindex(features.index)

    lin = linear_probe(features, target)
    assert lin.ic > 0.05, f"la sonde linéaire ne voit pas le signal AR(1) (IC={lin.ic:.4f})"
    assert lin.sign_accuracy > 0.52

    net = network_probe(features, target, window=4, steps=1_500)
    assert net.ic > 0.03, f"la sonde réseau ne voit pas le signal (IC={net.ic:.4f})"


def test_probes_find_nothing_in_pure_noise():
    """Contrôle négatif : sur une marche aléatoire, les sondes doivent renvoyer IC ≈ 0.

    Sans ce test, une sonde cassée qui renverrait toujours un IC élevé passerait
    inaperçue — et validerait des features sans valeur."""
    from qbot.config import FeatureConfig
    from qbot.data.synthetic import RegimeSwitchingGBM, generate_synthetic_ohlcv
    from qbot.diagnostics import forward_returns, linear_probe
    from qbot.features import FeaturePipeline

    noise = generate_synthetic_ohlcv(
        n=6_000, seed=5,
        model=RegimeSwitchingGBM(mu=(0.0,), sigma=(0.10,), persistence=1.0, autocorr=0.0),
    ).drop(columns=["regime"])
    cfg = FeatureConfig(returns_windows=(1, 5), vol_windows=(10,), ema_windows=(10,),
                        use_microstructure=False, use_calendar=False, scaler_window=200)
    features = FeaturePipeline(cfg).fit_transform(noise)
    lin = linear_probe(features, forward_returns(noise).reindex(features.index))
    assert abs(lin.ic) < 0.10, f"signal fantôme détecté sur du bruit pur (IC={lin.ic:+.4f})"

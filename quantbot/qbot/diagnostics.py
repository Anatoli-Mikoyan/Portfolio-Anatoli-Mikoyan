"""Sondes de diagnostic : y a-t-il du signal, et l'architecture peut-elle l'extraire ?

Motivation. Un entraînement RL coûte des heures et échoue silencieusement : l'agent
converge vers « ne rien faire », ou vers du bruit, sans jamais lever d'erreur. Quand cela
arrive, trois causes sont possibles et il est impossible de les distinguer en regardant
la courbe de Sharpe :

  1. les features ne contiennent aucun signal exploitable ;
  2. elles en contiennent, mais l'architecture ne peut pas l'extraire ;
  3. les deux vont bien, mais la boucle RL (exploration, coûts, crédit temporel) échoue.

Ces sondes séparent les trois en quelques secondes, en posant la question sous forme
SUPERVISÉE — la version la plus facile du problème. Le raisonnement est un encadrement :

  * Une régression linéaire sur les features donne un **plancher** de ce qui est extractible.
  * Le réseau réellement utilisé, entraîné en supervisé, donne le **plafond** de ce que le
    RL peut espérer atteindre : le RL résout un problème strictement plus dur (récompense
    différée, exploration, données non i.i.d.).

Si le réseau fait moins bien que la régression linéaire, inutile de lancer le RL : c'est
la représentation qu'il faut corriger, pas l'algorithme.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .utils.logging import get_logger

log = get_logger("diagnostics")


@dataclass
class ProbeResult:
    name: str
    ic: float                    # corrélation prédiction / rendement futur, HORS échantillon
    ic_train: float              # la même, DANS l'échantillon
    sign_accuracy: float
    r2_oos: float
    n_train: int
    n_test: int
    seconds: float = 0.0

    @property
    def overfit_gap(self) -> float:
        """IC in-sample moins IC out-of-sample.

        C'est LA quantité à surveiller en finance : un écart élevé signifie que la capacité
        du modèle dépasse ce que le rapport signal/bruit des données peut supporter.
        Le remède est de réduire le modèle ou de le régulariser — jamais de l'entraîner plus.
        """
        return self.ic_train - self.ic

    def __str__(self) -> str:  # pragma: no cover - affichage
        return (f"{self.name:<26} IC={self.ic:+.4f} (train {self.ic_train:+.4f}, "
                f"écart {self.overfit_gap:+.4f})  signe={self.sign_accuracy:.3f}")


def _ic(pred: np.ndarray, truth: np.ndarray) -> float:
    pred = np.asarray(pred, float).ravel()
    truth = np.asarray(truth, float).ravel()
    ok = np.isfinite(pred) & np.isfinite(truth)
    pred, truth = pred[ok], truth[ok]
    if pred.size < 3 or pred.std() < 1e-14 or truth.std() < 1e-14:
        return 0.0
    return float(np.corrcoef(pred, truth)[0, 1])


def _metrics(name: str, pred: np.ndarray, truth: np.ndarray, n_train: int,
             pred_train: Optional[np.ndarray] = None,
             truth_train: Optional[np.ndarray] = None,
             seconds: float = 0.0) -> ProbeResult:
    pred = np.asarray(pred, float).ravel()
    truth = np.asarray(truth, float).ravel()
    ok = np.isfinite(pred) & np.isfinite(truth)
    pred, truth = pred[ok], truth[ok]
    return ProbeResult(
        name=name,
        ic=_ic(pred, truth),
        ic_train=_ic(pred_train, truth_train) if pred_train is not None else float("nan"),
        sign_accuracy=float(np.mean(np.sign(pred) == np.sign(truth))),
        r2_oos=float(1.0 - np.var(truth - pred) / max(np.var(truth), 1e-30)),
        n_train=n_train, n_test=int(pred.size), seconds=seconds,
    )


def forward_returns(prices: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Rendement de la barre SUIVANTE — la cible que la stratégie cherche à prédire."""
    return (prices["close"].shift(-horizon) / prices["close"] - 1.0).fillna(0.0)


def linear_probe(
    features: pd.DataFrame, target: pd.Series, train_frac: float = 0.7, ridge: float = 1e-6
) -> ProbeResult:
    """Régression linéaire régularisée : le PLANCHER de ce qui est extractible.

    Si cette sonde ne trouve rien, aucun modèle plus complexe n'a de raison d'y arriver —
    sauf non-linéarité forte, ce que la sonde réseau ci-dessous permet de vérifier.
    """
    import time

    t0 = time.perf_counter()
    x = features.to_numpy(dtype=float)
    y = target.reindex(features.index).to_numpy(dtype=float)
    cut = int(len(x) * train_frac)

    a = np.column_stack([x[:cut], np.ones(cut)])
    gram = a.T @ a + ridge * np.eye(a.shape[1])
    beta = np.linalg.solve(gram, a.T @ y[:cut])
    pred = np.column_stack([x[cut:], np.ones(len(x) - cut)]) @ beta
    return _metrics("régression linéaire", pred, y[cut:], cut,
                    pred_train=a @ beta, truth_train=y[:cut],
                    seconds=time.perf_counter() - t0)


def network_probe(
    features: pd.DataFrame,
    target: pd.Series,
    window: int = 16,
    encoder: str = "mlp",
    hidden_sizes: Sequence[int] = (128, 128),
    steps: int = 8_000,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    layer_norm: bool = True,
    n_portfolio: int = 6,
    train_frac: float = 0.7,
    seed: int = 0,
    scale: float = 100.0,
) -> ProbeResult:
    """Entraîne EXACTEMENT le réseau du Rainbow, mais en supervisé : le PLAFOND du RL.

    On lui demande de prédire la récompense immédiate de chaque action (position × rendement
    futur). C'est le sous-problème le plus simple contenu dans le RL : pas d'exploration,
    pas de crédit différé, données i.i.d. Si le réseau échoue ici, il échouera à coup sûr
    en RL — et le correctif est du côté de la représentation, pas de l'algorithme.
    """
    import time

    import torch

    from .agents.networks import QNetwork
    from .utils.seeding import seed_everything

    seed_everything(seed)
    matrix = features.to_numpy(dtype=np.float32)
    y_all = target.reindex(features.index).to_numpy(dtype=np.float32) * scale
    n_feat = matrix.shape[1]

    offsets = np.arange(-window + 1, 1)
    idx = np.arange(window - 1, len(matrix))
    rows = np.clip(idx[:, None] + offsets[None, :], 0, len(matrix) - 1)
    obs = np.concatenate(
        [matrix[rows].reshape(len(idx), -1), np.zeros((len(idx), n_portfolio), np.float32)],
        axis=1,
    )
    y = y_all[idx]

    cut = int(len(obs) * train_frac)
    x_tr = torch.as_tensor(obs[:cut])
    y_tr = torch.as_tensor(y[:cut])
    x_te = torch.as_tensor(obs[cut:])

    net = QNetwork(obs.shape[1], 3, n_feat, window, n_portfolio, encoder=encoder,
                   hidden_sizes=tuple(hidden_sizes), noisy=False, dueling=True,
                   distributional="none", dropout=0.0, layer_norm=layer_norm)
    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1.5e-4, weight_decay=weight_decay)

    t0 = time.perf_counter()
    net.train()
    for _ in range(steps):
        i = torch.randint(0, len(x_tr), (batch_size,))
        q = net(x_tr[i])
        yb = y_tr[i]
        # Actions [-1, 0, +1] : la récompense immédiate est directement -y, 0, +y.
        tgt = torch.stack([-yb, torch.zeros_like(yb), yb], dim=1)
        loss = torch.nn.functional.smooth_l1_loss(q, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    net.eval()
    net.set_noise(False)

    def _edge(x: "torch.Tensor") -> np.ndarray:
        with torch.no_grad():
            q = torch.cat([net(x[i: i + 4096]) for i in range(0, len(x), 4096)]).numpy()
        # L'« edge » prédit = Q(long) - Q(short), homogène au rendement futur.
        return (q[:, 2] - q[:, 0]) / (2.0 * scale)

    return _metrics(f"réseau {encoder} (W={window})", _edge(x_te), y[cut:] / scale, cut,
                    pred_train=_edge(x_tr), truth_train=y[:cut] / scale,
                    seconds=time.perf_counter() - t0)


def signal_report(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    windows: Sequence[int] = (1, 4, 16),
    encoder: str = "mlp",
    steps: int = 8_000,
    horizon: int = 1,
) -> Tuple[ProbeResult, list[ProbeResult]]:
    """Compare le plancher linéaire au plafond réseau et rend un verdict actionnable."""
    target = forward_returns(prices, horizon)
    idx = features.index.intersection(prices.index)
    features, target = features.loc[idx], target.loc[idx]

    lin = linear_probe(features, target)
    nets = [network_probe(features, target, window=w, encoder=encoder, steps=steps)
            for w in windows]

    print()
    print("┌─ SONDES DE SIGNAL ───────────────────────────────────────────┐")
    print(f"│ {str(lin):<60} │")
    for r in nets:
        print(f"│ {str(r):<60} │")
    print("├──────────────────────────────────────────────────────────────┤")

    best = max(nets, key=lambda r: r.ic)
    if lin.ic < 0.01 and best.ic < 0.01:
        verdict = ("Aucun signal détectable, même en linéaire : le problème est dans les "
                   "FEATURES. Entraîner un modèle plus gros ne changera rien.")
    elif best.ic < lin.ic * 0.7:
        gap = best.overfit_gap
        if np.isfinite(gap) and gap > 0.05:
            verdict = (f"Le réseau (IC {best.ic:+.3f}) fait moins bien que le linéaire "
                       f"(IC {lin.ic:+.3f}) avec un écart train/test de {gap:+.3f} : il "
                       f"SUR-APPREND. Réduire hidden_sizes, augmenter weight_decay, "
                       f"entraîner MOINS longtemps.")
        else:
            verdict = (f"Le réseau (IC {best.ic:+.3f}) fait moins bien que le linéaire "
                       f"(IC {lin.ic:+.3f}) sans sur-apprendre : problème de conditionnement "
                       f"(échelle de la cible, taux d'apprentissage).")
    else:
        verdict = (f"Signal extractible par le réseau (IC {best.ic:+.3f}, plancher linéaire "
                   f"{lin.ic:+.3f}) : l'entraînement RL a une chance d'aboutir.")
    for line in _wrap(verdict, 60):
        print(f"│ {line:<60} │")
    print("└──────────────────────────────────────────────────────────────┘")
    return lin, nets


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines

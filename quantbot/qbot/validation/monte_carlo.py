"""Tests de robustesse par rééchantillonnage.

Un backtest produit UNE trajectoire. Or l'histoire n'est qu'une réalisation parmi
d'innombrables possibles : le vrai ordre des barres aurait pu être légèrement différent,
un trade décisif aurait pu ne pas se présenter. Ces méthodes construisent la distribution
des résultats plausibles et répondent à « quel est le pire scénario raisonnable ? ».

Le bootstrap PAR BLOCS est obligatoire ici : rééchantillonner les rendements un par un
détruirait le clustering de volatilité et l'autocorrélation, produisant des intervalles
de confiance beaucoup trop étroits — donc faussement rassurants.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from ..backtest.metrics import max_drawdown, sharpe_ratio


def stationary_bootstrap_indices(
    n: int, mean_block: int, rng: np.random.Generator, size: Optional[int] = None
) -> np.ndarray:
    """Bootstrap stationnaire de Politis & Romano (1994).

    Les blocs ont une longueur GÉOMÉTRIQUE aléatoire (moyenne `mean_block`) plutôt que
    fixe : la série rééchantillonnée reste alors strictement stationnaire, ce que le
    bootstrap par blocs de longueur fixe ne garantit pas.
    """
    size = size or n
    p = 1.0 / max(mean_block, 1)
    idx = np.empty(size, dtype=np.int64)
    i = int(rng.integers(0, n))
    for k in range(size):
        idx[k] = i
        if rng.random() < p:
            i = int(rng.integers(0, n))       # nouveau bloc
        else:
            i = (i + 1) % n                    # continuité du bloc (avec bouclage)
    return idx


@dataclass
class BootstrapResult:
    metric_name: str
    observed: float
    samples: np.ndarray
    ci_low: float
    ci_high: float
    p_value_positive: float

    def __str__(self) -> str:  # pragma: no cover - affichage
        return (
            f"{self.metric_name}: observé={self.observed:.4f} | "
            f"IC95%=[{self.ci_low:.4f}, {self.ci_high:.4f}] | "
            f"P({self.metric_name}>0)={self.p_value_positive:.3f}"
        )


def bootstrap_metric(
    returns: np.ndarray,
    metric_fn: Callable[[np.ndarray], float],
    n_samples: int = 2000,
    block_size: int = 20,
    seed: int = 0,
    metric_name: str = "metric",
) -> BootstrapResult:
    """Distribution bootstrap d'une métrique quelconque."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    rng = np.random.default_rng(seed)
    samples = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        idx = stationary_bootstrap_indices(r.size, block_size, rng)
        samples[i] = metric_fn(r[idx])
    return BootstrapResult(
        metric_name=metric_name,
        observed=float(metric_fn(r)),
        samples=samples,
        ci_low=float(np.quantile(samples, 0.025)),
        ci_high=float(np.quantile(samples, 0.975)),
        p_value_positive=float((samples > 0).mean()),
    )


def monte_carlo_drawdown(
    returns: np.ndarray, n_samples: int = 2000, block_size: int = 20, seed: int = 0
) -> Dict[str, float]:
    """Distribution du drawdown maximal sur trajectoires rééchantillonnées.

    Le drawdown observé dans un backtest est presque toujours OPTIMISTE : c'est le
    drawdown d'un seul chemin. Le quantile 95 % de cette distribution est une estimation
    bien plus honnête de ce qu'il faudra encaisser en production.
    """
    res = bootstrap_metric(returns, max_drawdown, n_samples, block_size, seed, "max_drawdown")
    return {
        "observed": res.observed,
        "median": float(np.median(res.samples)),
        "p95_worst": float(np.quantile(res.samples, 0.05)),   # DD le plus profond (valeur négative)
        "p99_worst": float(np.quantile(res.samples, 0.01)),
        "prob_worse_than_observed": float((res.samples < res.observed).mean()),
    }


def shuffle_trades_test(
    trade_returns: np.ndarray, n_samples: int = 5000, seed: int = 0
) -> Dict[str, float]:
    """Permute l'ORDRE des trades en conservant leur distribution.

    Sépare deux sources de performance : la qualité moyenne des trades (invariante par
    permutation) et l'enchaînement temporel (détruit par la permutation). Si le drawdown
    observé est bien meilleur que la médiane des permutations, c'est que le backtest a
    bénéficié d'un ordonnancement chanceux — et non d'un timing reproductible.
    """
    r = np.asarray(trade_returns, dtype=float)
    rng = np.random.default_rng(seed)
    dds = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        dds[i] = max_drawdown(rng.permutation(r))
    observed = max_drawdown(r)
    return {
        "observed_dd": float(observed),
        "median_shuffled_dd": float(np.median(dds)),
        "p05_shuffled_dd": float(np.quantile(dds, 0.05)),
        "luck_percentile": float((dds < observed).mean()),
    }


def whites_reality_check(
    strategy_returns: np.ndarray,
    benchmark_returns: Optional[np.ndarray] = None,
    n_samples: int = 2000,
    block_size: int = 20,
    seed: int = 0,
) -> Dict[str, float]:
    """Reality Check de White (2000) — version bootstrap stationnaire.

    Teste H0 : « la MEILLEURE des N stratégies n'a pas de performance supérieure au
    benchmark », en tenant compte du fait qu'on a regardé N stratégies. Sans cette
    correction, tester N stratégies au seuil de 5 % produit en moyenne 0.05·N faux
    positifs — avec N=100, cinq stratégies « significatives » purement par hasard.

    `strategy_returns` : (T, N).
    """
    m = np.atleast_2d(np.asarray(strategy_returns, dtype=float))
    if m.shape[0] < m.shape[1]:
        m = m.T
    t, n = m.shape
    bench = np.zeros(t) if benchmark_returns is None else np.asarray(benchmark_returns, dtype=float)

    excess = m - bench[:, None]
    f_bar = excess.mean(axis=0)
    v_observed = float(np.sqrt(t) * f_bar.max())

    rng = np.random.default_rng(seed)
    v_star = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        idx = stationary_bootstrap_indices(t, block_size, rng)
        # Re-centrage sur f_bar : impose H0 dans la distribution bootstrap.
        boot = excess[idx].mean(axis=0) - f_bar
        v_star[i] = float(np.sqrt(t) * boot.max())

    return {
        "v_observed": v_observed,
        "p_value": float((v_star >= v_observed).mean()),
        "critical_95": float(np.quantile(v_star, 0.95)),
        "n_strategies": int(n),
        "best_strategy": int(np.argmax(f_bar)),
    }


def confidence_band(
    returns: np.ndarray, n_samples: int = 500, block_size: int = 20, seed: int = 0
) -> Dict[str, np.ndarray]:
    """Enveloppe de confiance de la courbe d'équité (utile pour le reporting visuel)."""
    r = np.asarray(returns, dtype=float)
    rng = np.random.default_rng(seed)
    curves = np.empty((n_samples, r.size), dtype=float)
    for i in range(n_samples):
        idx = stationary_bootstrap_indices(r.size, block_size, rng)
        curves[i] = np.cumprod(1.0 + r[idx])
    return {
        "median": np.median(curves, axis=0),
        "p05": np.quantile(curves, 0.05, axis=0),
        "p95": np.quantile(curves, 0.95, axis=0),
        "observed": np.cumprod(1.0 + r),
    }

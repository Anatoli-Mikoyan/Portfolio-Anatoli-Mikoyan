"""Générateurs de marchés synthétiques.

Deux usages, tous deux indispensables :

1. **Tests** : valider le pipeline de bout en bout sans dépendre d'un fichier de données.
2. **Diagnostic d'overfitting** : entraîner l'agent sur un marché dont on CONNAÎT la
   structure (par ex. un processus purement aléatoire sans signal). Si l'agent trouve
   un « edge » sur une marche aléatoire, le pipeline fuit — c'est un test négatif
   qu'aucune stratégie ne devrait passer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class RegimeSwitchingGBM:
    """Mouvement brownien géométrique à régimes markoviens (drift/vol par régime).

    Reproduit trois faits stylisés majeurs : clustering de volatilité, régimes persistants
    (tendance/range) et queues épaisses via un mélange de gaussiennes.
    """
    mu: Sequence[float] = (0.08, -0.05, 0.0)         # drift annualisé par régime
    sigma: Sequence[float] = (0.08, 0.22, 0.13)      # vol annualisée par régime
    persistence: float = 0.995                        # probabilité de rester dans le régime
    autocorr: float = 0.0                             # AR(1) sur les rendements (signal exploitable)
    t_df: Optional[float] = 4.0                       # ddl de la Student (None => gaussien)

    def simulate(self, n: int, bars_per_year: float, s0: float = 1.1000,
                 rng: Optional[np.random.Generator] = None) -> tuple[np.ndarray, np.ndarray]:
        rng = rng or np.random.default_rng(0)
        k = len(self.mu)
        dt = 1.0 / bars_per_year

        # Chaîne de Markov des régimes
        states = np.empty(n, dtype=np.int64)
        s = 0
        off = (1.0 - self.persistence) / max(k - 1, 1)
        for i in range(n):
            states[i] = s
            if rng.random() > self.persistence:
                probs = np.full(k, off / max(1e-12, off * (k - 1)) if k > 1 else 1.0)
                probs = np.ones(k) / (k - 1) if k > 1 else np.ones(1)
                probs[s] = 0.0
                probs = probs / probs.sum()
                s = int(rng.choice(k, p=probs))

        mu = np.asarray(self.mu, float)[states]
        sig = np.asarray(self.sigma, float)[states]

        if self.t_df is not None:
            shocks = rng.standard_t(self.t_df, size=n) / np.sqrt(self.t_df / (self.t_df - 2.0))
        else:
            shocks = rng.standard_normal(n)

        eps = sig * np.sqrt(dt) * shocks
        if abs(self.autocorr) > 1e-12:  # AR(1) : introduit un signal momentum/mean-reversion réel
            filt = np.empty(n)
            prev = 0.0
            for i in range(n):
                prev = self.autocorr * prev + eps[i]
                filt[i] = prev
            eps = filt

        log_ret = (mu - 0.5 * sig**2) * dt + eps
        prices = s0 * np.exp(np.cumsum(log_ret))
        return prices, states


def generate_synthetic_ohlcv(
    n: int = 20_000,
    freq: str = "1h",
    start: str = "2019-01-01",
    seed: int = 0,
    model: Optional[RegimeSwitchingGBM] = None,
    bars_per_year: float = 6240.0,
    s0: float = 1.1000,
    spread_bps: float = 1.2,
    intrabar_noise: float = 0.35,
) -> pd.DataFrame:
    """Construit un DataFrame OHLCV complet (avec volume et spread) à partir du modèle."""
    rng = np.random.default_rng(seed)
    model = model or RegimeSwitchingGBM()
    close, states = model.simulate(n, bars_per_year, s0=s0, rng=rng)

    open_ = np.empty(n)
    open_[0] = s0
    open_[1:] = close[:-1]

    # Amplitude intra-barre proportionnelle au mouvement de la barre + bruit
    body = np.abs(close - open_)
    scale = intrabar_noise * (body + np.abs(close) * 1e-4)
    hi_ext = np.abs(rng.normal(0.0, 1.0, n)) * scale
    lo_ext = np.abs(rng.normal(0.0, 1.0, n)) * scale
    high = np.maximum(open_, close) + hi_ext
    low = np.minimum(open_, close) - lo_ext

    # Volume corrélé à la volatilité réalisée (relation empirique robuste)
    rel_range = (high - low) / close
    volume = np.maximum(1.0, rng.gamma(2.0, 1.0, n) * 500.0 * (1.0 + 40.0 * rel_range))

    index = pd.date_range(start=start, periods=n, freq=freq, tz="UTC")
    df = pd.DataFrame(
        {
            "open": open_, "high": high, "low": low, "close": close,
            "volume": volume,
            "spread": close * (spread_bps / 1e4) * (1.0 + 0.5 * np.abs(rng.normal(0, 1, n))),
            "regime": states,
        },
        index=index,
    )
    df.index.name = "time"
    return df

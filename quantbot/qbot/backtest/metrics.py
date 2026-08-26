"""Métriques de performance, y compris les tests de significativité statistique.

La distinction essentielle :

  * Les métriques DESCRIPTIVES (Sharpe, Calmar, profit factor…) racontent ce qui s'est
    passé dans le backtest. Elles ne disent rien sur la reproductibilité.
  * Les métriques INFÉRENTIELLES (PSR, Deflated Sharpe, MinTRL) répondent à la seule
    question qui compte : « ce résultat est-il distinguable de la chance, compte tenu du
    nombre d'essais effectués et de la forme de la distribution des rendements ? »

Un Sharpe de 2.5 obtenu après 500 configurations testées est statistiquement moins
crédible qu'un Sharpe de 1.1 obtenu du premier coup. Le Deflated Sharpe Ratio quantifie
exactement cette intuition.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329


# =======================================================================================
# Bloc descriptif
# =======================================================================================
def sharpe_ratio(returns: np.ndarray, bars_per_year: float = 252.0, rf: float = 0.0) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return 0.0
    excess = r - rf / bars_per_year
    sd = excess.std(ddof=1)
    return float(excess.mean() / sd * np.sqrt(bars_per_year)) if sd > 1e-14 else 0.0


def sortino_ratio(returns: np.ndarray, bars_per_year: float = 252.0, target: float = 0.0) -> float:
    """Ne pénalise que la volatilité BAISSIÈRE — la volatilité haussière n'est pas un risque."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return 0.0
    downside = np.minimum(r - target, 0.0)
    dd = np.sqrt(np.mean(downside ** 2))
    return float((r.mean() - target) / dd * np.sqrt(bars_per_year)) if dd > 1e-14 else 0.0


def equity_curve(returns: np.ndarray, initial: float = 1.0) -> np.ndarray:
    return initial * np.cumprod(1.0 + np.asarray(returns, dtype=float))


def drawdown_series(equity: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(equity)
    return equity / peak - 1.0


def max_drawdown(returns: np.ndarray) -> float:
    return float(drawdown_series(equity_curve(returns)).min())


def drawdown_duration(returns: np.ndarray) -> int:
    """Plus longue série de barres sans nouveau plus-haut.

    Métrique très sous-estimée : c'est la durée du drawdown, pas sa profondeur, qui fait
    abandonner une stratégie (et qui déclenche les rachats d'investisseurs)."""
    eq = equity_curve(returns)
    peak = np.maximum.accumulate(eq)
    under = eq < peak - 1e-15
    longest = current = 0
    for u in under:
        current = current + 1 if u else 0
        longest = max(longest, current)
    return int(longest)


def ulcer_index(returns: np.ndarray) -> float:
    """Racine de la moyenne des drawdowns au carré : pénalise profondeur ET durée."""
    dd = drawdown_series(equity_curve(returns))
    return float(np.sqrt(np.mean(dd ** 2)))


def calmar_ratio(returns: np.ndarray, bars_per_year: float = 252.0) -> float:
    mdd = abs(max_drawdown(returns))
    return float(cagr(returns, bars_per_year) / mdd) if mdd > 1e-12 else 0.0


def cagr(returns: np.ndarray, bars_per_year: float = 252.0) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    total = float(np.prod(1.0 + r))
    years = r.size / bars_per_year
    if years <= 0 or total <= 0:
        return -1.0
    return float(total ** (1.0 / years) - 1.0)


def omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    r = np.asarray(returns, dtype=float) - threshold
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    return float(gains / losses) if losses > 1e-14 else float("inf")


def tail_ratio(returns: np.ndarray, q: float = 0.05) -> float:
    """|P95| / |P5| : > 1 signifie que les meilleurs jours dépassent les pires."""
    r = np.asarray(returns, dtype=float)
    lo = abs(np.quantile(r, q))
    return float(abs(np.quantile(r, 1 - q)) / lo) if lo > 1e-14 else 0.0


def value_at_risk(returns: np.ndarray, alpha: float = 0.05) -> float:
    return float(np.quantile(np.asarray(returns, dtype=float), alpha))


def conditional_var(returns: np.ndarray, alpha: float = 0.05) -> float:
    """CVaR / Expected Shortfall : perte moyenne dans le pire α des cas."""
    r = np.asarray(returns, dtype=float)
    var = np.quantile(r, alpha)
    tail = r[r <= var]
    return float(tail.mean()) if tail.size else float(var)


# =======================================================================================
# Bloc inférentiel — López de Prado / Bailey
# =======================================================================================
def probabilistic_sharpe_ratio(
    returns: np.ndarray, sr_benchmark: float = 0.0, bars_per_year: float = 252.0
) -> float:
    """PSR : probabilité que le vrai Sharpe dépasse `sr_benchmark` (annualisé).

        PSR = Φ[ (SR̂ - SR*)·√(n-1) / √(1 - γ₃·SR̂ + (γ₄-1)/4·SR̂²) ]

    Corrige explicitement l'asymétrie et l'aplatissement. C'est important : une stratégie
    de vente d'options affiche un Sharpe élevé avec une skewness très négative — le Sharpe
    brut la surévalue massivement, le PSR remet les choses en place.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 3:
        return 0.0
    sd = r.std(ddof=1)
    if sd <= 1e-14:
        return 0.0

    sr = r.mean() / sd                                   # Sharpe par barre
    sr_star = sr_benchmark / np.sqrt(bars_per_year)      # benchmark ramené par barre
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))        # kurtosis non centrée (normale = 3)

    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if denom <= 1e-14:
        return 0.0
    z = (sr - sr_star) * math.sqrt(n - 1) / math.sqrt(denom)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, sharpe_std: float) -> float:
    """Sharpe maximal ATTENDU sous l'hypothèse nulle « aucune stratégie n'a d'edge ».

        E[max SR_N] ≈ σ_SR · [ (1-γ)·Z⁻¹(1 - 1/N) + γ·Z⁻¹(1 - 1/(N·e)) ]

    Résultat contre-intuitif mais fondamental : en testant 1 000 stratégies SANS AUCUN
    edge, dont les Sharpes ont un écart-type de 1, le meilleur affichera ~3.2 de Sharpe.
    Tout backtest issu d'une recherche doit être comparé à CE seuil, pas à zéro.
    """
    n = max(int(n_trials), 1)
    if n == 1 or sharpe_std <= 0:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n * math.e))
    return float(sharpe_std * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2))


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    sharpe_std: Optional[float] = None,
    bars_per_year: float = 252.0,
) -> float:
    """DSR = PSR évalué contre le Sharpe maximal attendu sous l'hypothèse nulle.

    Interprétation : probabilité que la stratégie ait un vrai Sharpe positif APRÈS
    correction du biais de sélection. DSR < 0.95 ⇒ le résultat n'est pas distinguable
    du meilleur tirage d'une famille de stratégies sans edge.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 3:
        return 0.0
    if sharpe_std is None:
        # Faute d'échantillon de Sharpes, on utilise l'erreur-type asymptotique du Sharpe
        # sous l'hypothèse nulle : σ ≈ √(1/(n-1)) par barre, annualisée.
        sharpe_std = float(np.sqrt(1.0 / (r.size - 1)) * np.sqrt(bars_per_year))
    sr_star = expected_max_sharpe(n_trials, sharpe_std)
    return probabilistic_sharpe_ratio(r, sr_benchmark=sr_star, bars_per_year=bars_per_year)


def min_track_record_length(
    returns: np.ndarray, sr_benchmark: float = 0.0, confidence: float = 0.95,
    bars_per_year: float = 252.0
) -> float:
    """Nombre MINIMAL d'observations pour que le Sharpe soit significatif au seuil donné.

    Si votre backtest est plus court que cette valeur, le résultat n'est pas exploitable —
    quelle que soit sa beauté sur le graphique.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 3:
        return float("inf")
    sd = r.std(ddof=1)
    if sd <= 1e-14:
        return float("inf")
    sr = r.mean() / sd
    sr_star = sr_benchmark / np.sqrt(bars_per_year)
    if sr <= sr_star:
        return float("inf")
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    z = stats.norm.ppf(confidence)
    return float(1.0 + (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2) * (z / (sr - sr_star)) ** 2)


# =======================================================================================
# Agrégation
# =======================================================================================
@dataclass
class PerformanceReport:
    n_obs: int
    total_return: float
    cagr: float
    ann_volatility: float
    sharpe: float
    sortino: float
    calmar: float
    omega: float
    max_drawdown: float
    max_dd_duration: int
    ulcer_index: float
    skew: float
    kurtosis: float
    var_95: float
    cvar_95: float
    tail_ratio: float
    hit_rate: float
    profit_factor: float
    psr: float
    deflated_sharpe: float
    min_track_record: float
    turnover_per_bar: float = 0.0
    cost_drag_annual: float = 0.0
    exposure: float = 0.0
    n_trades: int = 0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def __str__(self) -> str:  # pragma: no cover - affichage
        lines = [
            "┌─ PERFORMANCE ────────────────────────────────────────────────┐",
            f"│ Observations        {self.n_obs:>12,}                            │",
            f"│ Rendement total     {self.total_return:>11.2%}                            │",
            f"│ CAGR                {self.cagr:>11.2%}                            │",
            f"│ Volatilité ann.     {self.ann_volatility:>11.2%}                            │",
            "├─ RATIOS ─────────────────────────────────────────────────────┤",
            f"│ Sharpe              {self.sharpe:>11.3f}                            │",
            f"│ Sortino             {self.sortino:>11.3f}                            │",
            f"│ Calmar              {self.calmar:>11.3f}                            │",
            f"│ Omega               {self.omega:>11.3f}                            │",
            "├─ RISQUE ─────────────────────────────────────────────────────┤",
            f"│ Max drawdown        {self.max_drawdown:>11.2%}                            │",
            f"│ Durée max DD        {self.max_dd_duration:>11,} barres                     │",
            f"│ Ulcer index         {self.ulcer_index:>11.4f}                            │",
            f"│ VaR 95%             {self.var_95:>11.4%}                            │",
            f"│ CVaR 95%            {self.cvar_95:>11.4%}                            │",
            f"│ Skewness            {self.skew:>11.3f}                            │",
            f"│ Kurtosis            {self.kurtosis:>11.3f}                            │",
            "├─ TRADING ────────────────────────────────────────────────────┤",
            f"│ Taux de réussite    {self.hit_rate:>11.2%}                            │",
            f"│ Profit factor       {self.profit_factor:>11.3f}                            │",
            f"│ Turnover / barre    {self.turnover_per_bar:>11.4f}                            │",
            f"│ Coût annuel         {self.cost_drag_annual:>11.2%}                            │",
            f"│ Exposition moyenne  {self.exposure:>11.2%}                            │",
            "├─ SIGNIFICATIVITÉ STATISTIQUE ────────────────────────────────┤",
            f"│ PSR (vs 0)          {self.psr:>11.3f}   (> 0.95 souhaitable)     │",
            f"│ Deflated Sharpe     {self.deflated_sharpe:>11.3f}   (> 0.95 requis)          │",
            f"│ Track record min.   {self.min_track_record:>11,.0f} barres nécessaires        │",
            "└──────────────────────────────────────────────────────────────┘",
        ]
        return "\n".join(lines)


def compute_report(
    returns: np.ndarray,
    bars_per_year: float = 252.0,
    n_trials: int = 1,
    turnover: Optional[np.ndarray] = None,
    costs: Optional[np.ndarray] = None,
    positions: Optional[np.ndarray] = None,
    n_trades: int = 0,
    sharpe_std: Optional[float] = None,
) -> PerformanceReport:
    """Calcule le rapport complet à partir d'une série de rendements nets."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        raise ValueError("Série de rendements vide.")

    wins, losses = r[r > 0], r[r < 0]
    gross_win, gross_loss = float(wins.sum()), float(-losses.sum())

    return PerformanceReport(
        n_obs=int(r.size),
        total_return=float(np.prod(1.0 + r) - 1.0),
        cagr=cagr(r, bars_per_year),
        ann_volatility=float(r.std(ddof=1) * np.sqrt(bars_per_year)),
        sharpe=sharpe_ratio(r, bars_per_year),
        sortino=sortino_ratio(r, bars_per_year),
        calmar=calmar_ratio(r, bars_per_year),
        omega=omega_ratio(r),
        max_drawdown=max_drawdown(r),
        max_dd_duration=drawdown_duration(r),
        ulcer_index=ulcer_index(r),
        skew=float(stats.skew(r)),
        kurtosis=float(stats.kurtosis(r, fisher=False)),
        var_95=value_at_risk(r, 0.05),
        cvar_95=conditional_var(r, 0.05),
        tail_ratio=tail_ratio(r),
        hit_rate=float((r > 0).mean()),
        profit_factor=float(gross_win / gross_loss) if gross_loss > 1e-14 else float("inf"),
        psr=probabilistic_sharpe_ratio(r, 0.0, bars_per_year),
        deflated_sharpe=deflated_sharpe_ratio(r, n_trials, sharpe_std, bars_per_year),
        min_track_record=min_track_record_length(r, 0.0, 0.95, bars_per_year),
        turnover_per_bar=float(np.mean(turnover)) if turnover is not None else 0.0,
        cost_drag_annual=float(np.mean(costs) * bars_per_year) if costs is not None else 0.0,
        exposure=float(np.mean(np.abs(positions))) if positions is not None else 0.0,
        n_trades=int(n_trades),
    )

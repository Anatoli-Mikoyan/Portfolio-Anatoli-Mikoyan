"""Banc de criblage : traiter chaque stratégie comme une hypothèse à réfuter (§8, §13).

Deux protocoles, délibérément distincts, parce qu'ils ne mesurent pas la même chose :

**`screen_fixed`** — paramètres figés d'avance, évaluation walk-forward. Mesure l'edge
de l'hypothèse elle-même. Un seul essai : le Deflated Sharpe est indulgent.

**`screen_family`** — à chaque fold, on choisit les meilleurs paramètres sur la fenêtre
d'entraînement puis on trade la fenêtre suivante. C'est ce qu'un praticien fait
réellement, et cela intègre le coût de la sélection. Le Deflated Sharpe est ici calculé
avec le nombre RÉEL de combinaisons balayées.

L'écart entre les deux chiffres EST la mesure du data-snooping. Quand `screen_family`
s'effondre alors que `screen_fixed` tient, cela signifie que l'optimisation de
paramètres détruit l'edge au lieu de l'améliorer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Type

import numpy as np
import pandas as pd

from ..backtest import PerformanceReport, compute_report, run_backtest, sharpe_ratio
from ..config import CostConfig, EnvConfig, ValidationConfig
from ..utils.logging import get_logger
from ..validation import make_folds
from .base import Strategy

log = get_logger("strategies.screening")


@dataclass
class ScreeningResult:
    name: str
    hypothesis: str
    fails_when: str
    n_trials: int
    report: PerformanceReport
    oos_returns: pd.Series
    oos_positions: pd.Series
    fold_sharpes: np.ndarray = field(default_factory=lambda: np.array([]))
    selected_params: List[Dict] = field(default_factory=list)

    @property
    def pct_folds_positive(self) -> float:
        return float((self.fold_sharpes > 0).mean()) if self.fold_sharpes.size else 0.0

    @property
    def verdict(self) -> str:
        """Verdict volontairement sévère : le coût d'un faux positif dépasse celui d'un faux négatif."""
        r = self.report
        if r.sharpe <= 0:
            return "REJETÉE (Sharpe OOS négatif)"
        if r.deflated_sharpe < 0.95:
            return f"REJETÉE (Deflated Sharpe {r.deflated_sharpe:.2f} < 0.95 sur {self.n_trials} essais)"
        if self.fold_sharpes.size and self.pct_folds_positive < 0.6:
            return f"REJETÉE (positive sur seulement {self.pct_folds_positive:.0%} des folds)"
        if r.n_obs < r.min_track_record:
            return f"INDÉCIDABLE (track record trop court : {r.n_obs:.0f} < {r.min_track_record:.0f})"
        return "RETENUE"


# =======================================================================================
def _oos_returns_fixed(
    strategy: Strategy, df: pd.DataFrame, folds, cost_cfg: CostConfig,
    env_cfg: EnvConfig, bpy: float,
) -> tuple[pd.Series, pd.Series, np.ndarray]:
    """Évalue une stratégie à paramètres figés sur les seules fenêtres de test."""
    rets, poss, sharpes = [], [], []
    for fold in folds:
        # Le contexte inclut la fenêtre d'entraînement : le signal a besoin de son
        # warm-up, sinon les premières barres du test seraient neutralisées à tort.
        ctx_start = max(fold.train_slice.start, 0)
        segment = df.iloc[ctx_start: fold.test_slice.stop]
        signal = strategy.signal(segment)
        res = run_backtest(signal, segment, cost_cfg, env_cfg, bpy)

        test_index = df.index[fold.test_slice]
        sub = res.frame.reindex(res.frame.index.intersection(test_index))
        if sub.empty:
            continue
        rets.append(sub["net_return"])
        poss.append(sub["position"])
        sharpes.append(sharpe_ratio(sub["net_return"].to_numpy(), bpy))

    if not rets:
        raise ValueError("Aucun fold exploitable : série trop courte pour ce découpage.")
    return (pd.concat(rets).sort_index(), pd.concat(poss).sort_index(),
            np.asarray(sharpes, dtype=float))


def screen_fixed(
    strategy: Strategy,
    df: pd.DataFrame,
    cost_cfg: Optional[CostConfig] = None,
    env_cfg: Optional[EnvConfig] = None,
    val_cfg: Optional[ValidationConfig] = None,
    bpy: float = 6240.0,
) -> ScreeningResult:
    """Une hypothèse, un seul jeu de paramètres, évaluation hors échantillon."""
    cost_cfg = cost_cfg or CostConfig()
    env_cfg = env_cfg or EnvConfig()
    val_cfg = val_cfg or ValidationConfig()

    folds = make_folds(len(df), df.index, val_cfg.train_bars, val_cfg.test_bars,
                       valid_pct=0.0, anchored=False,
                       embargo_bars=int(len(df) * val_cfg.embargo_pct))
    if not folds:
        raise ValueError(
            f"Aucun fold constructible : {len(df)} barres pour train={val_cfg.train_bars} "
            f"+ test={val_cfg.test_bars}."
        )

    rets, poss, sharpes = _oos_returns_fixed(strategy, df, folds, cost_cfg, env_cfg, bpy)
    report = compute_report(rets.to_numpy(), bpy, n_trials=1, positions=poss.to_numpy())
    return ScreeningResult(
        name=strategy.name, hypothesis=strategy.hypothesis, fails_when=strategy.fails_when,
        n_trials=1, report=report, oos_returns=rets, oos_positions=poss, fold_sharpes=sharpes,
    )


# =======================================================================================
def screen_family(
    cls: Type[Strategy],
    df: pd.DataFrame,
    cost_cfg: Optional[CostConfig] = None,
    env_cfg: Optional[EnvConfig] = None,
    val_cfg: Optional[ValidationConfig] = None,
    bpy: float = 6240.0,
    selection_metric: str = "sharpe",
) -> ScreeningResult:
    """Sélection des paramètres in-sample à chaque fold, exécution out-of-sample.

    C'est le protocole honnête : il facture le coût de l'optimisation. Un résultat obtenu
    ainsi est directement comparable à ce que produirait la stratégie en production.
    """
    cost_cfg = cost_cfg or CostConfig()
    env_cfg = env_cfg or EnvConfig()
    val_cfg = val_cfg or ValidationConfig()

    candidates = cls.enumerate()
    folds = make_folds(len(df), df.index, val_cfg.train_bars, val_cfg.test_bars,
                       valid_pct=0.0, anchored=False,
                       embargo_bars=int(len(df) * val_cfg.embargo_pct))
    if not folds:
        raise ValueError(f"Aucun fold constructible pour {cls.__name__}.")

    rets, poss, sharpes, chosen = [], [], [], []
    for fold in folds:
        train_df = df.iloc[fold.train_slice]
        best, best_score = None, -np.inf
        for cand in candidates:
            sig = cand.signal(train_df)
            if sig.abs().sum() == 0:
                continue
            r = run_backtest(sig, train_df, cost_cfg, env_cfg, bpy)
            score = {"sharpe": r.report.sharpe,
                     "calmar": r.report.calmar}.get(selection_metric, r.report.sharpe)
            if np.isfinite(score) and score > best_score:
                best, best_score = cand, score
        if best is None:
            continue

        chosen.append({"fold": fold.idx, "is_sharpe": float(best_score), **best.params})
        ctx_start = max(fold.train_slice.start, 0)
        segment = df.iloc[ctx_start: fold.test_slice.stop]
        res = run_backtest(best.signal(segment), segment, cost_cfg, env_cfg, bpy)
        sub = res.frame.reindex(res.frame.index.intersection(df.index[fold.test_slice]))
        if sub.empty:
            continue
        rets.append(sub["net_return"])
        poss.append(sub["position"])
        sharpes.append(sharpe_ratio(sub["net_return"].to_numpy(), bpy))

    if not rets:
        raise ValueError(f"Aucun fold exploitable pour {cls.__name__}.")

    oos_r = pd.concat(rets).sort_index()
    oos_p = pd.concat(poss).sort_index()
    # n_trials = taille RÉELLE de la grille balayée à chaque fold. Le sous-déclarer
    # rendrait le Deflated Sharpe faussement rassurant.
    report = compute_report(oos_r.to_numpy(), bpy, n_trials=len(candidates),
                            positions=oos_p.to_numpy())
    proto = candidates[0]
    return ScreeningResult(
        name=f"{cls.__name__} [grille de {len(candidates)}]",
        hypothesis=proto.hypothesis, fails_when=proto.fails_when,
        n_trials=len(candidates), report=report, oos_returns=oos_r, oos_positions=oos_p,
        fold_sharpes=np.asarray(sharpes, dtype=float), selected_params=chosen,
    )


# =======================================================================================
def screening_table(results: Sequence[ScreeningResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "stratégie": r.name,
            "sharpe_oos": round(r.report.sharpe, 3),
            "CAGR": round(r.report.cagr, 4),
            "maxDD": round(r.report.max_drawdown, 4),
            "profit_factor": round(r.report.profit_factor, 3),
            "essais": r.n_trials,
            "DSR": round(r.report.deflated_sharpe, 4),
            "folds_positifs": round(r.pct_folds_positive, 2),
            "verdict": r.verdict,
        })
    return pd.DataFrame(rows).sort_values("sharpe_oos", ascending=False)


def family_pbo(results: Sequence[ScreeningResult], n_partitions: int = 10) -> Optional[dict]:
    """PBO et Reality Check sur l'ENSEMBLE des stratégies criblées.

    C'est le bon niveau d'analyse : la question n'est pas « cette stratégie est-elle
    bonne ? » mais « le fait d'avoir choisi la meilleure d'entre elles a-t-il une valeur
    prédictive ? ». Sans cette correction, tester N stratégies au seuil de 5 % produit
    en moyenne 0.05·N fausses découvertes.
    """
    from ..validation import compute_pbo, whites_reality_check

    series = [r.oos_returns for r in results]
    if len(series) < 2:
        return None
    common = series[0].index
    for s in series[1:]:
        common = common.intersection(s.index)
    if len(common) < n_partitions * 4:
        log.warning("Recouvrement temporel insuffisant (%d barres) pour la PBO.", len(common))
        return None

    matrix = np.column_stack([s.loc[common].to_numpy() for s in series])
    out = {"n_strategies": matrix.shape[1], "n_obs": matrix.shape[0]}
    try:
        pbo = compute_pbo(matrix, n_partitions=n_partitions, max_combinations=400)
        out["pbo"] = pbo.pbo
        out["degradation_slope"] = pbo.degradation_slope
        out["prob_oos_loss"] = pbo.prob_oos_loss
    except Exception as exc:                                  # pragma: no cover
        log.warning("PBO non calculable : %s", exc)
    wrc = whites_reality_check(matrix, n_samples=1000)
    out["reality_check_p"] = wrc["p_value"]
    out["best_strategy_index"] = wrc["best_strategy"]
    return out


def print_screening(results: Sequence[ScreeningResult], family: Optional[dict] = None) -> None:  # pragma: no cover
    table = screening_table(results)
    print()
    print(table.to_string(index=False))
    if family:
        print()
        print("┌─ NIVEAU FAMILLE (correction du choix de la meilleure) ───────┐")
        if "pbo" in family:
            print(f"│ PBO                       {family['pbo']:>10.3f}                       │")
            print(f"│ Pente de dégradation      {family['degradation_slope']:>10.3f}                       │")
        print(f"│ Reality Check (p-value)   {family['reality_check_p']:>10.4f}                       │")
        verdict = ("la meilleure bat le hasard" if family["reality_check_p"] < 0.05
                   else "indiscernable du hasard")
        print(f"│ -> {verdict:<57}│")
        print("└──────────────────────────────────────────────────────────────┘")
    retained = [r for r in results if r.verdict == "RETENUE"]
    print(f"\n{len(retained)}/{len(results)} hypothèses survivent au criblage.")
    for r in retained:
        print(f"  • {r.name}")

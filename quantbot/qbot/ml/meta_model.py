"""Méta-modèle : filtrer les signaux d'une stratégie primaire (cahier des charges §9).

Deux principes gouvernent ce module.

**1. L'évaluation est ÉCONOMIQUE avant d'être statistique.** Un modèle qui gagne 0.02 d'AUC
sans améliorer le profit factor n'a aucune valeur : l'AUC pondère tous les trades
également, alors qu'en trading un gros trade gagnant compense vingt petits perdants. On
rapporte donc systématiquement les deux, et le verdict s'appuie sur la métrique
économique.

**2. Le seuil de décision est choisi DANS le fold d'entraînement.** Choisir le seuil qui
maximise la performance hors échantillon est une fuite déguisée — et l'une des plus
fréquentes, parce qu'elle ne ressemble pas à de la triche.

La validation croisée est purgée et embargotée : les labels de la triple barrière se
chevauchent, une K-fold standard ferait fuir le futur dans le passé.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from ..validation import PurgedKFold
from .dataset import MetaDataset
from .models import MODEL_ZOO, build_model, fit_model

log = get_logger("ml.meta_model")


# =======================================================================================
@dataclass
class MetaEvaluation:
    """Résultat d'une validation croisée purgée, volets statistique et économique."""
    model_name: str
    complexity: int
    n_folds: int
    n_samples: int

    auc: float
    accuracy: float
    precision: float
    recall: float
    brier: float

    threshold: float
    base_return_per_trade: float
    filtered_return_per_trade: float
    base_profit_factor: float
    filtered_profit_factor: float
    base_sharpe: float
    filtered_sharpe: float
    n_trades_base: int
    n_trades_filtered: int
    oof_proba: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)

    @property
    def trade_retention(self) -> float:
        return self.n_trades_filtered / max(self.n_trades_base, 1)

    @property
    def economic_gain(self) -> float:
        """Amélioration du rendement moyen par trade — la métrique qui décide."""
        return self.filtered_return_per_trade - self.base_return_per_trade

    @property
    def verdict(self) -> str:
        if self.n_trades_filtered < 30:
            return "INEXPLOITABLE (trop peu de trades retenus)"
        if self.economic_gain <= 0:
            return "INUTILE (n'améliore pas le rendement par trade)"
        if self.filtered_profit_factor <= self.base_profit_factor:
            return "INUTILE (profit factor non amélioré)"
        if self.auc < 0.52:
            return "FRAGILE (gain économique sans pouvoir discriminant)"
        return "UTILE"


# =======================================================================================
def _select_threshold(proba: np.ndarray, returns: np.ndarray,
                      grid: Sequence[float] = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65)) -> float:
    """Seuil maximisant le rendement moyen par trade, avec un plancher de trades.

    Appelé UNIQUEMENT sur le fold d'entraînement. Un seuil qui ne retiendrait qu'une
    poignée de trades obtiendrait un rendement moyen flatteur et ininterprétable, d'où
    la contrainte de conserver au moins 15 % des signaux.
    """
    best, best_score = 0.5, -np.inf
    floor = max(int(0.15 * len(returns)), 20)
    for t in grid:
        mask = proba >= t
        if mask.sum() < floor:
            continue
        score = float(returns[mask].mean())
        if score > best_score:
            best, best_score = float(t), score
    return best


def _trade_sharpe(returns: np.ndarray, index: pd.DatetimeIndex) -> float:
    """Sharpe au niveau TRADE, annualisé par la fréquence réelle des trades.

    Annualiser des rendements de trades avec un facteur de barres serait faux : dix
    trades par an et dix mille n'ont pas la même signification statistique.
    """
    if returns.size < 3 or returns.std(ddof=1) < 1e-14:
        return 0.0
    span_years = max((index[-1] - index[0]).total_seconds() / (365.25 * 86400), 1e-6)
    trades_per_year = returns.size / span_years
    return float(returns.mean() / returns.std(ddof=1) * np.sqrt(trades_per_year))


def _profit_factor(returns: np.ndarray) -> float:
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    return float(gains / losses) if losses > 1e-14 else float("inf")


# =======================================================================================
def cross_validate_meta(
    dataset: MetaDataset,
    model_name: str = "logistic",
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> MetaEvaluation:
    """Validation croisée purgée + embargotée, avec seuil choisi in-sample."""
    from sklearn.metrics import brier_score_loss, roc_auc_score

    X = dataset.X.to_numpy(dtype=float)
    y = dataset.y.to_numpy(dtype=int)
    w = dataset.sample_weight.to_numpy(dtype=float)
    rets = dataset.returns.to_numpy(dtype=float)

    cv = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct, t1=dataset.t1)
    oof = np.full(len(y), np.nan)
    thresholds: List[float] = []
    n_folds = 0

    for train_idx, test_idx in cv.split(dataset.X.index):
        if train_idx.size < 100 or test_idx.size < 20:
            continue
        if len(np.unique(y[train_idx])) < 2:
            continue

        model = build_model(model_name, **(model_kwargs or {}))
        fit_model(model, X[train_idx], y[train_idx], w[train_idx])

        p_train = model.predict_proba(X[train_idx])[:, 1]
        thresholds.append(_select_threshold(p_train, rets[train_idx]))
        oof[test_idx] = model.predict_proba(X[test_idx])[:, 1]
        n_folds += 1

    valid = np.isfinite(oof)
    if n_folds == 0 or valid.sum() < 50:
        raise ValueError(f"Validation croisée impossible pour {model_name} "
                         f"({n_folds} folds, {int(valid.sum())} prédictions).")

    p = oof[valid]
    y_v, r_v, idx_v = y[valid], rets[valid], dataset.X.index[valid]
    threshold = float(np.median(thresholds)) if thresholds else 0.5

    mask = p >= threshold
    filtered = r_v[mask]
    filtered_idx = idx_v[mask]

    auc = float(roc_auc_score(y_v, p)) if len(np.unique(y_v)) > 1 and p.std() > 0 else 0.5
    predicted = (p >= threshold).astype(int)
    tp = int(((predicted == 1) & (y_v == 1)).sum())
    fp = int(((predicted == 1) & (y_v == 0)).sum())
    fn = int(((predicted == 0) & (y_v == 1)).sum())

    return MetaEvaluation(
        model_name=model_name,
        complexity=MODEL_ZOO[model_name].complexity,
        n_folds=n_folds,
        n_samples=int(valid.sum()),
        auc=auc,
        accuracy=float((predicted == y_v).mean()),
        precision=float(tp / (tp + fp)) if (tp + fp) else 0.0,
        recall=float(tp / (tp + fn)) if (tp + fn) else 0.0,
        brier=float(brier_score_loss(y_v, p)),
        threshold=threshold,
        base_return_per_trade=float(r_v.mean()),
        filtered_return_per_trade=float(filtered.mean()) if filtered.size else 0.0,
        base_profit_factor=_profit_factor(r_v),
        filtered_profit_factor=_profit_factor(filtered) if filtered.size else 0.0,
        base_sharpe=_trade_sharpe(r_v, idx_v),
        filtered_sharpe=_trade_sharpe(filtered, filtered_idx) if filtered.size > 3 else 0.0,
        n_trades_base=int(r_v.size),
        n_trades_filtered=int(filtered.size),
        oof_proba=oof,
    )


def compare_models(
    dataset: MetaDataset,
    model_names: Optional[Sequence[str]] = None,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
) -> pd.DataFrame:
    """Compare le zoo par complexité croissante (§9).

    La colonne décisive n'est pas l'AUC mais `gain_par_trade` : c'est elle qui dit si un
    modèle plus complexe est RÉELLEMENT justifié, ou seulement plus flatteur.
    """
    names = list(model_names) if model_names else sorted(
        MODEL_ZOO, key=lambda n: (MODEL_ZOO[n].complexity, n))
    rows = []
    for name in names:
        try:
            ev = cross_validate_meta(dataset, name, n_splits, embargo_pct)
        except Exception as exc:                              # pragma: no cover
            log.warning("%s ignoré : %s", name, exc)
            continue
        rows.append({
            "modèle": name,
            "complexité": ev.complexity,
            "AUC": round(ev.auc, 4),
            "précision": round(ev.precision, 4),
            "seuil": ev.threshold,
            "trades_retenus": round(ev.trade_retention, 3),
            "PF_base": round(ev.base_profit_factor, 3),
            "PF_filtré": round(ev.filtered_profit_factor, 3),
            "gain_par_trade": round(ev.economic_gain, 6),
            "sharpe_filtré": round(ev.filtered_sharpe, 3),
            "verdict": ev.verdict,
        })
    return pd.DataFrame(rows)


def justify_complexity(table: pd.DataFrame, margin: float = 0.15) -> str:
    """Répond littéralement à la question du §9 : le modèle complexe est-il justifié ?

    Règle : un modèle plus complexe n'est retenu que s'il dépasse le meilleur modèle
    simple d'au moins `margin` en gain économique relatif. En dessous, la complexité
    achète du risque de sur-apprentissage sans contrepartie mesurable.
    """
    if table.empty:
        return "Aucun modèle évaluable."
    useful = table[table["verdict"] == "UTILE"]
    if useful.empty:
        return ("Aucun modèle n'améliore le rendement par trade : le méta-modèle "
                "n'apporte rien ici. Suivre tous les signaux primaires.")

    simple = useful[useful["complexité"] <= 1]
    complex_ = useful[useful["complexité"] >= 3]
    if simple.empty:
        best = complex_.loc[complex_["gain_par_trade"].idxmax()]
        return (f"Seuls des modèles complexes fonctionnent ; le meilleur est "
                f"{best['modèle']}. À surveiller : vérifier sa stabilité en walk-forward "
                f"avant de lui faire confiance.")
    best_simple = simple.loc[simple["gain_par_trade"].idxmax()]
    if complex_.empty:
        return (f"Le meilleur modèle est simple ({best_simple['modèle']}, gain "
                f"{best_simple['gain_par_trade']:+.5f}/trade). La complexité "
                f"supplémentaire n'est pas justifiée.")

    best_complex = complex_.loc[complex_["gain_par_trade"].idxmax()]
    ratio = best_complex["gain_par_trade"] / max(abs(best_simple["gain_par_trade"]), 1e-12)
    if ratio > 1.0 + margin:
        return (f"{best_complex['modèle']} dépasse {best_simple['modèle']} de "
                f"{100 * (ratio - 1):.0f}% en gain par trade : la complexité est justifiée, "
                f"sous réserve de confirmation en walk-forward.")
    return (f"{best_complex['modèle']} n'apporte que {100 * (ratio - 1):+.0f}% par rapport à "
            f"{best_simple['modèle']} : rester sur le modèle simple. La complexité "
            f"achèterait du risque de sur-apprentissage sans contrepartie mesurable.")


# =======================================================================================
class MetaModel:
    """Modèle entraîné, prêt à filtrer les signaux d'une stratégie primaire."""

    def __init__(self, model_name: str = "logistic", threshold: float = 0.5,
                 soft_sizing: bool = True, **model_kwargs):
        self.model_name = model_name
        self.threshold = float(threshold)
        # Dimensionnement progressif plutôt que binaire : une probabilité de 0.51 et une
        # de 0.90 ne méritent pas la même taille de position.
        self.soft_sizing = soft_sizing
        self.model_kwargs = model_kwargs
        self.model: Optional[Any] = None
        self.feature_names: List[str] = []

    def fit(self, dataset: MetaDataset) -> "MetaModel":
        self.feature_names = list(dataset.X.columns)
        self.model = build_model(self.model_name, **self.model_kwargs)
        fit_model(self.model, dataset.X.to_numpy(float), dataset.y.to_numpy(int),
                  dataset.sample_weight.to_numpy(float))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("MetaModel non entraîné.")
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes à l'inférence : {missing}")
        return self.model.predict_proba(X[self.feature_names].to_numpy(float))[:, 1]

    def size(self, proba: np.ndarray) -> np.ndarray:
        """Convertit une probabilité en taille de position dans [0, 1]."""
        p = np.asarray(proba, dtype=float)
        if not self.soft_sizing:
            return (p >= self.threshold).astype(float)
        return np.clip((p - self.threshold) / max(1.0 - self.threshold, 1e-9), 0.0, 1.0)

    def filter_signal(self, signal: pd.Series, X: pd.DataFrame) -> pd.Series:
        """Applique le filtre : position finale = signal primaire x confiance du méta-modèle."""
        common = signal.index.intersection(X.index)
        sizes = pd.Series(self.size(self.predict_proba(X.loc[common])), index=common)
        return (signal.loc[common] * sizes).reindex(signal.index).fillna(0.0)

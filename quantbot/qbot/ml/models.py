"""Zoo de modèles, du plus simple au plus complexe (cahier des charges §9).

Le cahier demande explicitement de « comparer des modèles simples et complexes » et
d'expliquer « pourquoi un modèle plus complexe serait réellement justifié ». Le zoo est
ordonné par capacité croissante pour que la comparaison soit lisible, et il commence par
deux références sans lesquelles tout chiffre est ininterprétable :

  * `always_trade` — suivre tous les signaux primaires. C'est le vrai point de comparaison :
    un méta-modèle qui ne bat pas cette référence n'a aucune raison d'exister.
  * `logistic`     — le plancher linéaire. Le battre n'est pas acquis : sur données
    financières, la mesure faite dans ce dépôt montre qu'un réseau non régularisé fait
    nettement PIRE qu'une régression.

Note sur `max_samples` de la forêt : López de Prado (ch. 6) montre qu'un bagging standard
sur des labels chevauchants produit des arbres bien plus corrélés qu'on ne le croit, donc
une variance out-of-sample sous-estimée. On borne donc la taille des tirages à l'unicité
moyenne de l'échantillon.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np


@dataclass
class ModelSpec:
    name: str
    complexity: int              # 0 = référence triviale, 4 = le plus flexible
    build: Callable[..., Any]
    rationale: str
    supports_weights: bool = True


class AlwaysTrade:
    """Référence : suivre tous les signaux, sans filtrage. Pas d'apprentissage."""

    def fit(self, X, y, sample_weight=None):
        self.base_rate_ = float(np.average(y, weights=sample_weight)) if len(y) else 0.5
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.base_rate_)
        return np.column_stack([1.0 - p, p])


def _logistic(**kw):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # Le StandardScaler est ajusté DANS le pipeline, donc uniquement sur le fold
    # d'entraînement : c'est ce qui empêche la fuite des moments du test.
    return make_pipeline(
        StandardScaler(),
        # l1_ratio=0 -> pénalisation purement L2 (API sklearn >= 1.8 ; `penalty` est déprécié).
        LogisticRegression(l1_ratio=0.0, C=kw.get("C", 0.1), max_iter=2000,
                           class_weight="balanced", solver="lbfgs"),
    )


def _elasticnet(**kw):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(l1_ratio=kw.get("l1_ratio", 0.5), C=kw.get("C", 0.1),
                           max_iter=3000, solver="saga", class_weight="balanced"),
    )


def _tree(**kw):
    from sklearn.tree import DecisionTreeClassifier

    return DecisionTreeClassifier(max_depth=kw.get("max_depth", 3),
                                  min_samples_leaf=kw.get("min_samples_leaf", 200),
                                  class_weight="balanced", random_state=0)


def _forest(**kw):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=kw.get("n_estimators", 300),
        max_depth=kw.get("max_depth", 5),
        min_samples_leaf=kw.get("min_samples_leaf", 100),
        max_features=kw.get("max_features", "sqrt"),
        # Borné par l'unicité moyenne : voir la note en tête de module.
        max_samples=kw.get("max_samples", 0.5),
        bootstrap=True, class_weight="balanced_subsample",
        n_jobs=-1, random_state=0,
    )


def _gbm(**kw):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_depth=kw.get("max_depth", 3),
        learning_rate=kw.get("learning_rate", 0.05),
        max_iter=kw.get("max_iter", 200),
        min_samples_leaf=kw.get("min_samples_leaf", 100),
        l2_regularization=kw.get("l2_regularization", 1.0),
        early_stopping=False, random_state=0,
    )


MODEL_ZOO: Dict[str, ModelSpec] = {
    "always_trade": ModelSpec(
        "always_trade", 0, lambda **kw: AlwaysTrade(),
        "Référence obligatoire : suivre tous les signaux. Tout méta-modèle doit la battre."),
    "logistic": ModelSpec(
        "logistic", 1, _logistic,
        "Plancher linéaire, fortement régularisé. Rapide, stable, interprétable. "
        "Sur données à faible rapport signal/bruit, très difficile à battre."),
    "elasticnet": ModelSpec(
        "elasticnet", 1, _elasticnet,
        "Linéaire avec sélection L1 : utile quand beaucoup de features sont redondantes."),
    "tree": ModelSpec(
        "tree", 2, _tree,
        "Arbre peu profond : capture une interaction simple (par ex. signal x régime) "
        "tout en restant lisible."),
    "forest": ModelSpec(
        "forest", 3, _forest,
        "Forêt aléatoire à tirages bornés par l'unicité. Justifiée si les interactions "
        "entre features comptent vraiment."),
    "gbm": ModelSpec(
        "gbm", 4, _gbm,
        "Boosting de gradient. Le plus flexible, donc le plus exposé au sur-apprentissage : "
        "ne le retenir que s'il bat nettement la forêt ET le linéaire."),
}


def fit_model(model: Any, X, y, sample_weight: Optional[np.ndarray] = None) -> Any:
    """Ajuste un modèle en propageant correctement les poids d'échantillons.

    Les poids ne sont pas optionnels ici : les labels issus de la triple barrière se
    chevauchent, donc leur unicité varie fortement d'une observation à l'autre. Les
    ignorer revient à accorder le même crédit à une observation isolée et à dix
    observations qui décrivent le même mouvement de marché.

    Un `Pipeline` sklearn n'accepte pas `sample_weight` directement : il faut nommer
    l'étape finale (`étape__sample_weight`). Cette fonction absorbe cette asymétrie pour
    que le reste du code n'ait pas à connaître le type du modèle.
    """
    if sample_weight is None:
        return model.fit(X, y)

    from sklearn.pipeline import Pipeline

    if isinstance(model, Pipeline):
        final_step = model.steps[-1][0]
        return model.fit(X, y, **{f"{final_step}__sample_weight": sample_weight})
    try:
        return model.fit(X, y, sample_weight=sample_weight)
    except TypeError:                       # modèle sans support des poids
        return model.fit(X, y)


def build_model(name: str, **kwargs) -> Any:
    if name not in MODEL_ZOO:
        raise ValueError(f"Modèle inconnu : {name}. Disponibles : {list(MODEL_ZOO)}")
    return MODEL_ZOO[name].build(**kwargs)


def zoo_by_complexity() -> list[ModelSpec]:
    return sorted(MODEL_ZOO.values(), key=lambda s: (s.complexity, s.name))

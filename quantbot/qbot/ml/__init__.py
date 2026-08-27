"""Couche ML supervisée (méta-modèle). Dépend de scikit-learn."""
from .dataset import MetaDataset, build_meta_dataset
from .models import MODEL_ZOO, ModelSpec, build_model, fit_model, zoo_by_complexity
from .meta_model import (
    MetaModel, MetaEvaluation, cross_validate_meta, compare_models, justify_complexity,
)
from .importance import (
    mdi_importance, mda_importance, clustered_mda, cluster_features, select_features,
)

__all__ = [
    "MetaDataset", "build_meta_dataset",
    "MODEL_ZOO", "ModelSpec", "build_model", "fit_model", "zoo_by_complexity",
    "MetaModel", "MetaEvaluation", "cross_validate_meta", "compare_models",
    "justify_complexity",
    "mdi_importance", "mda_importance", "clustered_mda", "cluster_features",
    "select_features",
]

"""quant-engine -- moteur de backtesting et d'execution algorithmique.

Le projet est construit autour d'une hypothese : la quasi-totalite des
backtests amateurs sont faux, et le sont pour des raisons methodologiques
identifiees et reproductibles -- look-ahead bias, couts ignores, sur-ajustement,
echantillon insuffisant, absence de baseline. Le moteur est concu pour rendre
ces erreurs difficiles a commettre et impossibles a dissimuler.

Il ne predit rien. Il invalide.
"""

from __future__ import annotations

from .errors import (
    AdjustmentError,
    CacheError,
    ConfigError,
    DataError,
    DataQualityError,
    InsufficientHistoryError,
    LookaheadError,
    ProviderError,
    QuantEngineError,
    SchemaError,
)
from .logging_setup import configure_logging, get_logger

__version__ = "0.1.0"

__all__ = [
    "AdjustmentError",
    "CacheError",
    "ConfigError",
    "DataError",
    "DataQualityError",
    "InsufficientHistoryError",
    "LookaheadError",
    "ProviderError",
    "QuantEngineError",
    "SchemaError",
    "__version__",
    "configure_logging",
    "get_logger",
]

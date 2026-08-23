"""Hierarchie d'exceptions du moteur.

Principe : une erreur silencieuse dans un backtest produit un resultat faux mais
credible, ce qui est pire qu'un crash. Toutes les situations ambigues levent.
"""

from __future__ import annotations

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
]


class QuantEngineError(Exception):
    """Racine de toutes les erreurs du moteur."""


class ConfigError(QuantEngineError):
    """Configuration absente, incoherente ou non validee."""


class DataError(QuantEngineError):
    """Racine des erreurs de la couche donnees."""


class SchemaError(DataError):
    """Le payload d'un provider ne respecte pas le contrat OHLCV canonique."""


class DataQualityError(DataError):
    """Une anomalie de qualite classee bloquante a ete detectee."""


class LookaheadError(DataError):
    """Tentative d'acces a une information non disponible a l'instant courant.

    Cette exception ne devrait jamais etre rattrapee en production : elle
    signale un bug de conception, pas une condition transitoire.
    """


class InsufficientHistoryError(DataError):
    """Historique insuffisant pour honorer la fenetre demandee.

    Leve plutot que de tronquer : une moyenne mobile 200 calculee sur 50 barres
    reste une moyenne mobile, mais ce n'est plus la meme strategie -- et elle
    genere des signaux precoces qui n'auraient jamais existe.
    """


class AdjustmentError(DataError):
    """Ajustement des prix impossible ou methodologiquement invalide."""


class ProviderError(DataError):
    """Echec d'une source de donnees externe."""


class CacheError(DataError):
    """Cache corrompu, incoherent ou de version incompatible."""

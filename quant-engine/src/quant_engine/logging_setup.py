"""Logging structure (JSON) pour le moteur.

Un backtest produit des dizaines de milliers d'evenements. Les logs doivent etre
requetables : un format cle/valeur machine-lisible, pas des phrases.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any, Final

_RESERVED: Final[frozenset[str]] = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__
) | {"message", "asctime", "taskName"}

__all__ = ["JsonFormatter", "configure_logging", "get_logger"]


class JsonFormatter(logging.Formatter):
    """Serialise chaque enregistrement en une ligne JSON.

    Les attributs passes via ``extra=`` sont promus au niveau racine de l'objet
    JSON, ce qui rend les logs directement exploitables (jq, Loki, etc.).
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED:
                payload[key] = value
        if record.exc_info is not None:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO", *, structured: bool = True) -> None:
    """Installe le handler racine. Idempotent."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    if structured:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s :: %(message)s")
        )
    root.addHandler(handler)
    root.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"quant_engine.{name}")

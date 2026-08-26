"""Logging structuré, sobre, sans dépendance externe."""
from __future__ import annotations

import logging
import sys
from typing import Optional

_CONFIGURED = False
_FMT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"


def configure_logging(level: int | str = logging.INFO, logfile: Optional[str] = None) -> None:
    global _CONFIGURED
    root = logging.getLogger("qbot")
    root.handlers.clear()
    root.setLevel(level)
    root.propagate = False

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter(_FMT, datefmt="%Y-%m-%d %H:%M:%S"))
    root.addHandler(stream)

    if logfile:
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FMT, datefmt="%Y-%m-%d %H:%M:%S"))
        root.addHandler(fh)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(f"qbot.{name}")

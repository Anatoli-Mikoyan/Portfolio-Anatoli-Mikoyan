"""Logging structure : les logs d'un backtest doivent etre requetables."""

from __future__ import annotations

import json
import logging

import pytest

from quant_engine.logging_setup import JsonFormatter, configure_logging, get_logger


def test_format_json_une_ligne() -> None:
    record = logging.LogRecord(
        name="quant_engine.test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="ordre envoye", args=(), exc_info=None,
    )
    record.symbol = "AAPL"
    record.quantity = 100
    payload = json.loads(JsonFormatter().format(record))
    assert payload["msg"] == "ordre envoye"
    assert payload["level"] == "INFO"
    assert payload["symbol"] == "AAPL"
    assert payload["quantity"] == 100
    assert payload["ts"].endswith("+00:00")


def test_les_champs_extra_remontent_a_la_racine() -> None:
    """Un backtest emet des dizaines de milliers d'evenements : ils doivent
    etre filtrables par cle, pas par expression reguliere sur une phrase."""
    record = logging.LogRecord(
        name="x", level=logging.WARNING, pathname=__file__, lineno=1,
        msg="slippage", args=(), exc_info=None,
    )
    record.bps = 12.5
    payload = json.loads(JsonFormatter().format(record))
    assert payload["bps"] == 12.5
    assert "pathname" not in payload


def test_exception_serialisee() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = logging.LogRecord(
            name="x", level=logging.ERROR, pathname=__file__, lineno=1,
            msg="echec", args=(), exc_info=sys.exc_info(),
        )
    payload = json.loads(JsonFormatter().format(record))
    assert "ValueError: boom" in payload["exc"]


def test_configuration_idempotente() -> None:
    configure_logging("DEBUG")
    configure_logging("WARNING")
    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert root.level == logging.WARNING


def test_mode_texte() -> None:
    configure_logging("INFO", structured=False)
    handler = logging.getLogger().handlers[0]
    assert not isinstance(handler.formatter, JsonFormatter)
    configure_logging("INFO")


def test_prefixe_de_logger() -> None:
    assert get_logger("data.cache").name == "quant_engine.data.cache"


@pytest.fixture(autouse=True)
def _restore() -> object:
    yield None
    configure_logging("INFO")

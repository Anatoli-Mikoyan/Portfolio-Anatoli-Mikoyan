"""Configuration YAML typee et chargement de bout en bout."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest

from quant_engine.config import load_config
from quant_engine.data import AdjustmentPolicy, DataLoader, Frequency
from quant_engine.data.types import UTC
from quant_engine.errors import ConfigError


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "conf.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_acces_types(tmp_path: Path) -> None:
    node = load_config(write(tmp_path, "a: hello\nb: 3\nc: 1.5\nd: true\ne: [x, y]\n"))
    assert node.str_("a") == "hello"
    assert node.int_("b") == 3
    assert node.float_("c") == 1.5
    assert node.float_("b") == 3.0  # int -> float tolere
    assert node.bool_("d") is True
    assert node.str_list("e") == ("x", "y")


def test_type_incorrect_signale_le_chemin(tmp_path: Path) -> None:
    node = load_config(write(tmp_path, "data:\n  frequency: 3\n"))
    with pytest.raises(ConfigError, match=r"data\.frequency : attendu str"):
        node.section("data").str_("frequency")


def test_cle_manquante(tmp_path: Path) -> None:
    node = load_config(write(tmp_path, "a: 1\n"))
    with pytest.raises(ConfigError, match="Cle de configuration manquante : b"):
        node.str_("b")
    assert node.str_("b", "defaut") == "defaut"


def test_cles_inconnues_refusees(tmp_path: Path) -> None:
    """Une faute de frappe dans un YAML est silencieuse et change le
    comportement du backtest : ``adjstment: raw`` laisserait tourner la
    politique par defaut sans que personne ne le remarque."""
    node = load_config(write(tmp_path, "data:\n  adjstment: raw\n"))
    with pytest.raises(ConfigError, match="cles inconnues"):
        node.section("data").reject_unknown(("adjustment", "provider"))


def test_valeur_hors_enumeration(tmp_path: Path) -> None:
    node = load_config(write(tmp_path, "p: postgres\n"))
    with pytest.raises(ConfigError, match="valeur 'postgres' invalide"):
        node.enum_("p", ("yfinance", "csv"))


def test_substitution_denvironnement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QE_ROOT", "/data/market")
    node = load_config(write(tmp_path, "root: ${QE_ROOT}\nttl: ${QE_TTL:-24}\n"))
    assert node.str_("root") == "/data/market"
    assert node.str_("ttl") == "24"


def test_variable_denvironnement_absente(tmp_path: Path) -> None:
    os.environ.pop("QE_ABSENT", None)
    with pytest.raises(ConfigError, match="Variable d'environnement requise"):
        load_config(write(tmp_path, "x: ${QE_ABSENT}\n"))


def test_fichier_absent(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="introuvable"):
        load_config(tmp_path / "nope.yaml")


def test_racine_non_mapping(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="mapping"):
        load_config(write(tmp_path, "- a\n- b\n"))


def test_section_absente(tmp_path: Path) -> None:
    node = load_config(write(tmp_path, "a: 1\n"))
    assert list(node.optional_section("absente").keys()) == []
    with pytest.raises(ConfigError, match="manquante"):
        node.section("absente")


def test_loader_depuis_configuration(tmp_path: Path) -> None:
    config = write(
        tmp_path,
        f"""
data:
  provider: synthetic
  calendar: XNYS
  frequency: 1d
  adjustment: split_pit
  cache:
    enabled: true
    root: {tmp_path / "cache"}
  synthetic:
    seed: 42
    start_price: 250.0
  quality:
    min_bars: 10
  normalization:
    on_nan: drop
    raise_on_blocking: false
""",
    )
    loader = DataLoader.from_config(config)
    assert loader.frequency is Frequency.DAY_1
    assert loader.adjustment is AdjustmentPolicy.SPLIT_PIT

    data = loader.load(
        "SYNTH", datetime(2021, 1, 1, tzinfo=UTC), datetime(2021, 12, 31, tzinfo=UTC),
        now=datetime(2023, 1, 1, tzinfo=UTC),
    )
    assert len(data) == 252
    assert data.symbol == "SYNTH"
    assert (tmp_path / "cache").exists()


def test_loader_refuse_une_politique_contaminee_par_defaut(tmp_path: Path) -> None:
    """Le YAML peut demander une politique retro ; le moteur refuse quand meme
    de l'appliquer sans opt-in explicite au moment de l'usage."""
    config = write(
        tmp_path,
        """
data:
  provider: synthetic
  adjustment: full_retro_total
  cache:
    enabled: false
  quality:
    min_bars: 10
  normalization:
    raise_on_blocking: false
""",
    )
    loader = DataLoader.from_config(config)
    data = loader.load(
        "SYNTH", datetime(2021, 1, 1, tzinfo=UTC), datetime(2021, 12, 31, tzinfo=UTC),
        now=datetime(2023, 1, 1, tzinfo=UTC),
    )
    from quant_engine.errors import AdjustmentError

    with pytest.raises(AdjustmentError, match="look-ahead"):
        data.cursor(loader.adjustment)


def test_source_inconnue(tmp_path: Path) -> None:
    config = write(tmp_path, "data:\n  provider: bloomberg\n")
    with pytest.raises(ConfigError, match="invalide"):
        DataLoader.from_config(config)

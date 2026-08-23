"""Chargement et validation de configuration YAML, typee.

Choix : pas de pydantic. Le moteur reste installable avec quatre dependances
runtime. La validation est explicite et leve ``ConfigError`` en nommant le
chemin complet de la cle fautive, ce qui evite les ``KeyError`` opaques.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, TypeVar, final

import yaml

from .errors import ConfigError

__all__ = ["MISSING", "ConfigNode", "load_config"]

_T = TypeVar("_T")
_ENV_PATTERN: Final = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


@final
class _Missing:
    """Sentinelle : distingue "absent" de "present avec la valeur None"."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<missing>"


MISSING: Final = _Missing()


def _expand_env(value: str) -> str:
    """Substitue ``${VAR}`` et ``${VAR:-defaut}`` depuis l'environnement."""

    def repl(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        found = os.environ.get(name)
        if found is not None:
            return found
        if default is not None:
            return default
        raise ConfigError(f"Variable d'environnement requise et non definie : {name}")

    return _ENV_PATTERN.sub(repl, value)


def _expand(node: object) -> object:
    if isinstance(node, str):
        return _expand_env(node)
    if isinstance(node, Mapping):
        return {str(key): _expand(val) for key, val in node.items()}
    if isinstance(node, list):
        return [_expand(item) for item in node]
    return node


@final
class ConfigNode:
    """Vue typee et validante sur un fragment de configuration."""

    __slots__ = ("_data", "_path")

    def __init__(self, data: Mapping[str, Any], path: str = "") -> None:
        self._data: dict[str, Any] = dict(data)
        self._path = path

    # -- introspection ------------------------------------------------------
    @property
    def path(self) -> str:
        return self._path or "<root>"

    def keys(self) -> Iterable[str]:
        return self._data.keys()

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"ConfigNode(path={self.path!r}, keys={sorted(self._data)})"

    def raw(self) -> dict[str, Any]:
        """Copie brute du fragment, pour tracer la config dans un rapport."""
        return dict(self._data)

    # -- resolution ---------------------------------------------------------
    def _child_path(self, key: str) -> str:
        return f"{self._path}.{key}" if self._path else key

    def _fetch(self, key: str, default: object) -> object:
        if key in self._data:
            return self._data[key]
        if isinstance(default, _Missing):
            raise ConfigError(f"Cle de configuration manquante : {self._child_path(key)}")
        return default

    def _typed(self, key: str, expected: type[_T], default: object) -> _T:
        value = self._fetch(key, default)
        if type(value) is expected or (isinstance(value, expected) and not isinstance(value, bool)):
            return value
        if expected is bool and isinstance(value, bool):
            return value  # type: ignore[return-value]
        # Seule coercition toleree : un entier YAML la ou un flottant est attendu.
        if expected is float and isinstance(value, int) and not isinstance(value, bool):
            return float(value)  # type: ignore[return-value]
        raise ConfigError(
            f"{self._child_path(key)} : attendu {expected.__name__}, "
            f"recu {type(value).__name__} ({value!r})"
        )

    # -- accesseurs types ---------------------------------------------------
    def str_(self, key: str, default: str | _Missing = MISSING) -> str:
        return self._typed(key, str, default)

    def int_(self, key: str, default: int | _Missing = MISSING) -> int:
        return self._typed(key, int, default)

    def float_(self, key: str, default: float | _Missing = MISSING) -> float:
        return self._typed(key, float, default)

    def bool_(self, key: str, default: bool | _Missing = MISSING) -> bool:
        return self._typed(key, bool, default)

    def str_list(self, key: str, default: Sequence[str] | _Missing = MISSING) -> tuple[str, ...]:
        value = self._fetch(key, default)
        if isinstance(value, str) or not isinstance(value, Sequence):
            raise ConfigError(f"{self._child_path(key)} : attendu une liste, recu {value!r}")
        for item in value:
            if not isinstance(item, str):
                raise ConfigError(
                    f"{self._child_path(key)} : element non textuel {item!r} "
                    f"({type(item).__name__})"
                )
        return tuple(str(item) for item in value)

    def enum_(self, key: str, allowed: Sequence[str], default: str | _Missing = MISSING) -> str:
        value = self.str_(key, default)
        if value not in allowed:
            raise ConfigError(
                f"{self._child_path(key)} : valeur {value!r} invalide, "
                f"attendu l'une de {list(allowed)}"
            )
        return value

    def section(self, key: str) -> ConfigNode:
        value = self._fetch(key, MISSING)
        if not isinstance(value, Mapping):
            raise ConfigError(f"{self._child_path(key)} : attendu une section, recu {value!r}")
        return ConfigNode(value, self._child_path(key))

    def optional_section(self, key: str) -> ConfigNode:
        if key not in self._data:
            return ConfigNode({}, self._child_path(key))
        return self.section(key)

    def reject_unknown(self, known: Sequence[str]) -> None:
        """Refuse les cles inconnues : une faute de frappe dans un YAML est
        sinon parfaitement silencieuse et change le comportement du backtest."""
        unknown = sorted(set(self._data) - set(known))
        if unknown:
            raise ConfigError(
                f"{self.path} : cles inconnues {unknown}. Attendu parmi {sorted(known)}"
            )


def load_config(path: str | Path) -> ConfigNode:
    """Charge un YAML, substitue les variables d'environnement, retourne la racine."""
    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigError(f"Fichier de configuration introuvable : {file_path}")
    try:
        loaded = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover
        raise ConfigError(f"YAML invalide dans {file_path} : {exc}") from exc
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, Mapping):
        raise ConfigError(f"{file_path} : la racine doit etre un mapping, recu {type(loaded)}")
    expanded = _expand(loaded)
    assert isinstance(expanded, dict)
    return ConfigNode(expanded, path="")

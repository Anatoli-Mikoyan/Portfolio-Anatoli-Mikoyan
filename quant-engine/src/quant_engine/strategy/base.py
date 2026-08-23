"""Contrat commun a toutes les strategies.

Deux principes portes par ce module.

**Une strategie ne voit que le passe.** Elle recoit un ``StrategyContext`` dont
le seul acces aux donnees est un ``HistoryView`` borne a la barre courante. Elle
ne recoit jamais un ``MarketData``, jamais un index absolu, jamais la barre
suivante. La garantie vient du typage, pas d'une convention.

**Les parametres sont declares, donc comptables.** Chaque parametre reglable est
declare avec son domaine. Ca sert a l'optimisation automatisee, mais surtout a
compter les **degres de liberte** et la **taille de l'espace de recherche** --
les deux nombres qui determinent si un bon resultat est un signal ou un artefact.

Sur ce dernier point : tester 400 combinaisons de parametres et retenir la
meilleure, c'est tirer 400 fois a pile ou face et s'extasier sur la plus longue
serie de piles. Le moteur calcule ce nombre et le fait figurer dans le rapport,
parce qu'un Sharpe de 1,8 selectionne parmi 400 essais ne vaut pas un Sharpe de
1,8 obtenu du premier coup.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, final

from ..data.dataset import HistoryView
from ..errors import ConfigError

__all__ = [
    "ParameterSet",
    "ParameterSpec",
    "Signal",
    "Strategy",
    "StrategyContext",
]

ParameterKind = Literal["int", "float", "bool"]


# ---------------------------------------------------------------------------
# Parametres
# ---------------------------------------------------------------------------
@final
@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """Declaration d'un parametre reglable."""

    name: str
    default: float | int | bool
    description: str
    kind: ParameterKind = "int"
    low: float | None = None
    high: float | None = None
    step: float | None = None
    tunable: bool = True
    """Un parametre non reglable (ex : le capital initial) ne compte pas comme
    un degre de liberte : il n'est pas optimise, donc il ne sur-ajuste pas."""

    def __post_init__(self) -> None:
        if self.tunable and self.kind != "bool" and (self.low is None or self.high is None):
            raise ConfigError(
                f"Parametre {self.name!r} reglable sans domaine. Un parametre dont "
                "on ne connait pas les bornes ne peut etre ni optimise ni audite."
            )
        self.validate(self.default)

    def validate(self, value: object) -> float | int | bool:
        """Verifie et normalise une valeur candidate."""
        if self.kind == "bool":
            if not isinstance(value, bool):
                raise ConfigError(f"{self.name} : booleen attendu, recu {value!r}")
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{self.name} : nombre attendu, recu {value!r}")
        if self.kind == "int":
            if int(value) != value:
                raise ConfigError(f"{self.name} : entier attendu, recu {value!r}")
            value = int(value)
        else:
            value = float(value)
        if self.low is not None and value < self.low:
            raise ConfigError(f"{self.name} = {value} sous la borne basse {self.low}")
        if self.high is not None and value > self.high:
            raise ConfigError(f"{self.name} = {value} au-dessus de la borne haute {self.high}")
        return value

    def candidates(self) -> tuple[float | int | bool, ...]:
        """Valeurs distinctes explorees par une recherche en grille."""
        if self.kind == "bool":
            return (False, True)
        if not self.tunable or self.low is None or self.high is None:
            return (self.default,)
        step = self.step if self.step is not None else (1.0 if self.kind == "int" else 0.1)
        values: list[float | int | bool] = []
        current = float(self.low)
        while current <= self.high + 1e-12:
            values.append(round(current) if self.kind == "int" else round(current, 10))
            current += step
        return tuple(dict.fromkeys(values))


@final
@dataclass(frozen=True, slots=True)
class ParameterSet:
    """Jeu de parametres valide, associe a ses declarations."""

    specs: tuple[ParameterSpec, ...]
    values: Mapping[str, float | int | bool]

    @classmethod
    def build(
        cls, specs: Sequence[ParameterSpec], overrides: Mapping[str, object] | None = None
    ) -> ParameterSet:
        by_name = {spec.name: spec for spec in specs}
        if len(by_name) != len(specs):
            raise ConfigError("Deux parametres portent le meme nom")
        supplied: dict[str, object] = dict(overrides or {})
        unknown = sorted(set(supplied) - set(by_name))
        if unknown:
            raise ConfigError(
                f"Parametres inconnus {unknown}. Attendus : {sorted(by_name)}. "
                "Une faute de frappe laisserait tourner la valeur par defaut en silence."
            )
        values = {
            name: spec.validate(supplied.get(name, spec.default)) for name, spec in by_name.items()
        }
        return cls(specs=tuple(specs), values=values)

    def __getitem__(self, name: str) -> float | int | bool:
        try:
            return self.values[name]
        except KeyError:
            raise ConfigError(f"Parametre non declare : {name!r}") from None

    def int_(self, name: str) -> int:
        return int(self[name])

    def float_(self, name: str) -> float:
        return float(self[name])

    def bool_(self, name: str) -> bool:
        return bool(self[name])

    def with_values(self, **overrides: object) -> ParameterSet:
        merged: dict[str, object] = dict(self.values)
        merged.update(overrides)
        return ParameterSet.build(self.specs, merged)

    # -- mesures de sur-ajustement -------------------------------------------
    @property
    def degrees_of_freedom(self) -> int:
        """Nombre de parametres reellement optimisables.

        Regle empirique repandue en finance quantitative : il faut de l'ordre de
        plusieurs centaines d'observations *independantes* par degre de liberte
        pour qu'un resultat signifie quelque chose. Une strategie a 6 parametres
        calibree sur 5 ans de donnees journalieres n'est pas calibree : elle est
        memorisee.
        """
        return sum(1 for spec in self.specs if spec.tunable)

    @property
    def search_space_size(self) -> int:
        """Nombre de configurations distinctes d'une recherche en grille.

        C'est le nombre d'essais implicites. Retenir le meilleur de N essais
        gonfle mecaniquement le resultat, meme si aucune strategie n'a d'edge.
        """
        total = 1
        for spec in self.specs:
            if spec.tunable:
                total *= max(1, len(spec.candidates()))
        return total

    def describe(self) -> str:
        lines = [f"{name} = {value}" for name, value in sorted(self.values.items())]
        return " | ".join(lines)

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)


# ---------------------------------------------------------------------------
# Contexte et signaux
# ---------------------------------------------------------------------------
@final
@dataclass(frozen=True, slots=True)
class StrategyContext:
    """Tout ce qu'une strategie a le droit de savoir a l'instant de decision."""

    history: HistoryView
    """Historique borne a la cloture de la barre courante. Aucun acces au futur."""
    as_of: datetime
    position_units: float
    """Quantite detenue, en titres. Negative pour une position vendeuse."""
    position_weight: float
    """Fraction de l'equity investie dans l'actif, au dernier prix connu."""
    cash: float
    equity: float
    bar_index: int

    @property
    def is_flat(self) -> bool:
        return self.position_units == 0.0

    @property
    def is_long(self) -> bool:
        return self.position_units > 0.0


@final
@dataclass(frozen=True, slots=True)
class Signal:
    """Intention exprimee par une strategie : une exposition cible.

    Le choix d'exprimer un signal en *poids cible* plutot qu'en ordre a une
    consequence structurante : la strategie decide de la direction, jamais de la
    taille en euros ni du nombre de titres. Le dimensionnement appartient a la
    couche risque (etape 6), ce qui evite qu'une strategie contourne les limites
    de risque en calculant ses quantites elle-meme.
    """

    target_weight: float
    """Fraction de l'equity a exposer. 1.0 = investi a 100 %, 0.0 = liquide."""
    reason: str = ""
    """Justification courte, journalisee avec l'ordre. Utile en post-mortem."""

    def __post_init__(self) -> None:
        if not -10.0 <= self.target_weight <= 10.0:
            raise ValueError(
                f"target_weight={self.target_weight} hors de toute plage sensee. "
                "Un poids de 10 signifie un levier 10x ; au-dela c'est un bug."
            )


# ---------------------------------------------------------------------------
# Strategie
# ---------------------------------------------------------------------------
class Strategy(ABC):
    """Classe de base de toute strategie."""

    #: Nom court, utilise dans les rapports.
    name: str = "unnamed"

    def __init__(self, **overrides: object) -> None:
        self.params = ParameterSet.build(self.specs(), overrides)

    # -- declaration ----------------------------------------------------------
    @classmethod
    @abstractmethod
    def specs(cls) -> tuple[ParameterSpec, ...]:
        """Parametres de la strategie, avec leur domaine."""

    @property
    @abstractmethod
    def warmup_bars(self) -> int:
        """Nombre de barres d'historique necessaires avant le premier signal.

        Le moteur refuse d'appeler la strategie avant. Une fenetre tronquee
        produirait des signaux precoces qui n'auraient jamais existe.
        """

    @property
    def expected_annual_turnover(self) -> float:
        """Nombre d'allers-retours complets attendus par an.

        Declare par la strategie, sert a estimer la friction *avant* de lancer
        le backtest. Une strategie qui tourne 250 fois par an sur un compte ou
        l'aller-retour coute 1 % doit produire 250 % de performance brute pour
        atteindre l'equilibre : autant le savoir avant de la coder.
        """
        return 0.0

    # -- execution ------------------------------------------------------------
    @abstractmethod
    def on_bar(self, context: StrategyContext) -> Signal | None:
        """Appelee a la cloture de chaque barre, apres le warmup.

        Retourne l'exposition cible, ou ``None`` pour ne rien changer.
        """

    def on_start(self, context: StrategyContext) -> None:  # noqa: B027
        """Crochet appele une fois avant la premiere decision.

        Volontairement concret et vide : la majorite des strategies n'en ont pas
        besoin, et le rendre abstrait forcerait chacune a ecrire un corps inutile.
        """

    def reset(self) -> None:  # noqa: B027
        """Remet l'etat interne a zero. Appele par le moteur avant chaque run.

        Indispensable au walk-forward : sans ca, l'etat d'une fenetre fuit dans
        la suivante et le resultat hors-echantillon est contamine.
        """

    # -- introspection --------------------------------------------------------
    @property
    def degrees_of_freedom(self) -> int:
        return self.params.degrees_of_freedom

    @property
    def search_space_size(self) -> int:
        return self.params.search_space_size

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.params.describe()})"

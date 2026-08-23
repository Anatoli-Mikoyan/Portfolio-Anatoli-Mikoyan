"""Structures de donnees du moteur, et prevention structurelle du look-ahead.

Le probleme
-----------
Le look-ahead bias n'arrive quasiment jamais par malveillance : il arrive parce
qu'une API le rend *facile*. Des qu'une strategie recoit un ``DataFrame``
complet, il suffit d'un ``.iloc[i+1]``, d'un ``.shift(-1)``, d'un
``.rolling(...).mean()`` calcule sur toute la serie, ou d'un ``.max()`` global
pour contaminer le backtest -- sans aucun message d'erreur, avec un resultat
d'autant plus credible qu'il est bon.

La reponse retenue : rendre le futur **ininexprimable dans l'API**.

Trois couches
-------------
1. **Segregation des types.** ``MarketData`` (serie complete) n'est jamais remis
   a une strategie. Une strategie ne recoit que des ``HistoryView``. Ce sont
   deux types distincts, et aucune methode de ``HistoryView`` ne renvoie un
   ``MarketData``. La regle est verifiable par lecture des signatures, pas par
   relecture des corps de fonction.

2. **Borne physique, pas conventionnelle.** Un ``HistoryView`` ne stocke pas la
   serie complete accompagnee d'un index a respecter : il stocke des *tranches*
   numpy de longueur exactement egale a la fenetre visible. La borne est portee
   par les objets tableaux eux-memes ; depasser leve ``IndexError`` au niveau de
   numpy, pas au niveau d'une verification qu'on aurait pu oublier d'ecrire.

3. **Adressage relatif uniquement.** On n'indexe pas une vue par un index
   absolu, mais par une *anciennete* : ``view.bar(0)`` est la derniere barre
   close, ``view.bar(1)`` la precedente. Il n'existe aucune facon d'ecrire
   "la barre suivante" : un offset negatif leve ``LookaheadError``. C'est la
   partie du design qui compte le plus, parce qu'elle attaque le reflexe
   ``i + 1`` a la racine.

Limite assumee
--------------
Une tranche numpy conserve une reference vers son tableau parent via
``ndarray.base``. Quelqu'un de determine peut donc remonter a la serie complete.
Ce n'est pas une faille : aucune API Python n'est etanche a un appelant hostile.
L'objectif est d'eliminer le look-ahead *accidentel*, qui represente la
totalite des cas reels. Le look-ahead deliberatoire, lui, est detecte par
empoisonnement du futur (``with_future_poisoned``), utilise par la suite de
tests et par le mode audit du moteur.

Convention temporelle
---------------------
Les barres sont indexees ``0..N-1`` et labellisees par leur **cloture**. Au pas
``t``, la barre ``t`` vient de cloturer : la vue expose les barres ``0..t``
incluses et ``as_of == timestamps[t]``. Tout ordre emis a ce moment est
executable au plus tot sur la barre ``t+1``, donc strictement apres l'instant
de la derniere information visible.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Final, final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..errors import DataError, InsufficientHistoryError, LookaheadError
from .adjustment import AdjustmentPolicy, Multipliers, build_multipliers
from .corporate_actions import CorporateActions
from .types import UTC, Bar, Field, Frequency, ensure_utc

if TYPE_CHECKING:
    from .quality import DataQualityReport

__all__ = ["BarCursor", "DecisionPoint", "HistoryView", "MarketData"]

_NS: Final = 1_000_000_000


def _freeze(array: NDArray[np.float64]) -> NDArray[np.float64]:
    frozen = np.asarray(array, dtype=np.float64)
    if frozen is array:
        frozen = array.copy()
    frozen.flags.writeable = False
    return frozen


def _to_datetime(epoch_ns: int) -> datetime:
    return datetime.fromtimestamp(epoch_ns / _NS, tz=UTC)


def _check_spacing(symbol: str, stamps: NDArray[np.int64], frequency: Frequency) -> None:
    """Verifie que l'espacement median des barres correspond a la frequence.

    Second filet contre les erreurs d'unite temporelle, complementaire du
    controle de plage absolue. Il est plus fin : il ne regarde pas *ou* se
    situent les barres mais *a quelle distance* elles sont les unes des autres.
    Une serie journaliere relue en microsecondes affiche un espacement median de
    86 secondes au lieu de 24 heures -- anomalie flagrante, alors que les dates
    resultantes (janvier 1970) restent superficiellement valides et parfaitement
    ordonnees.

    Le seuil est volontairement lache (moitie du pas nominal) : les seances
    manquantes, les feries et les demi-seances ecartent legitimement l'espacement
    du pas theorique, mais jamais d'un facteur mille.
    """
    if stamps.size < 3:
        return
    median_gap_ns = float(np.median(np.diff(stamps)))
    expected_ns = frequency.delta.total_seconds() * _NS
    if median_gap_ns < 0.5 * expected_ns:
        raise DataError(
            f"{symbol} : espacement median des barres de "
            f"{median_gap_ns / _NS:.1f} s pour une frequence {frequency.value} "
            f"(attendu environ {expected_ns / _NS:.0f} s). Cause la plus probable : "
            "erreur d'unite temporelle (microsecondes lues comme nanosecondes), "
            "ou frequence declaree incorrecte."
        )


# ---------------------------------------------------------------------------
# Vue bornee : le seul type qu'une strategie manipule.
# ---------------------------------------------------------------------------
@final
class HistoryView:
    """Fenetre en lecture seule sur l'historique disponible a un instant donne.

    Toutes les tranches exposees ont pour longueur maximale le nombre de barres
    deja closes. Il n'existe aucun accesseur vers la serie complete.
    """

    __slots__ = (
        "__actions",
        "__cursor",
        "__fields",
        "__frequency",
        "__multipliers",
        "__symbol",
        "__timestamps",
    )

    def __init__(
        self,
        *,
        symbol: str,
        frequency: Frequency,
        timestamps: NDArray[np.int64],
        fields: dict[Field, NDArray[np.float64]],
        multipliers: Multipliers,
        actions: CorporateActions,
        end: int,
    ) -> None:
        total = int(timestamps.size)
        if not 1 <= end <= total:
            raise ValueError(f"Borne de vue invalide : end={end} pour {total} barres")
        self.__symbol = symbol
        self.__frequency = frequency
        # Les tranches portent physiquement la borne : leur longueur EST la limite.
        self.__timestamps: NDArray[np.int64] = timestamps[:end]
        self.__fields: dict[Field, NDArray[np.float64]] = {
            name: values[:end] for name, values in fields.items()
        }
        self.__multipliers = multipliers
        self.__cursor = end - 1
        self.__actions = actions

    # -- identite -----------------------------------------------------------
    @property
    def symbol(self) -> str:
        return self.__symbol

    @property
    def frequency(self) -> Frequency:
        return self.__frequency

    @property
    def adjustment(self) -> AdjustmentPolicy:
        return self.__multipliers.policy

    @property
    def n_bars(self) -> int:
        """Nombre de barres closes visibles."""
        return self.__cursor + 1

    def __len__(self) -> int:
        return self.n_bars

    @property
    def as_of(self) -> datetime:
        """Instant de cloture de la derniere barre visible.

        Toute information exposee par cette vue etait publique a cet instant.
        """
        return _to_datetime(int(self.__timestamps[self.__cursor]))

    def has(self, n_bars: int) -> bool:
        """Y a-t-il au moins ``n_bars`` barres d'historique ?

        A appeler avant toute fenetre glissante : le moteur refuse de tronquer
        silencieusement une fenetre trop courte.
        """
        return self.n_bars >= n_bars

    def __repr__(self) -> str:
        return (
            f"HistoryView({self.__symbol} {self.__frequency.value}, "
            f"{self.n_bars} barres, as_of={self.as_of.isoformat()}, "
            f"adj={self.adjustment.value})"
        )

    # -- resolution de fenetre ---------------------------------------------
    def _window(self, lookback: int | None) -> tuple[int, int]:
        stop = self.n_bars
        if lookback is None:
            return 0, stop
        if lookback <= 0:
            raise ValueError(f"lookback doit etre strictement positif, recu {lookback}")
        if lookback > stop:
            raise InsufficientHistoryError(
                f"{lookback} barres demandees, {stop} disponibles a {self.as_of.date()}. "
                "Teste view.has(n) avant d'emettre un signal : une fenetre tronquee "
                "produit des signaux precoces qui n'ont jamais existe."
            )
        return stop - lookback, stop

    # -- series -------------------------------------------------------------
    def field(self, field: Field, lookback: int | None = None) -> NDArray[np.float64]:
        """Serie d'un champ, ajustee selon la politique en vigueur.

        Ordre chronologique croissant : le dernier element est la barre la plus
        recente.
        """
        start, stop = self._window(lookback)
        raw = self.__fields[field][start:stop]
        if self.__multipliers.is_identity:
            return raw
        if field is Field.VOLUME:
            factor = self.__multipliers.volume_factor(start, stop, self.__cursor)
        else:
            factor = self.__multipliers.price_factor(start, stop, self.__cursor)
        adjusted: NDArray[np.float64] = raw * factor
        adjusted.flags.writeable = False
        return adjusted

    def open(self, lookback: int | None = None) -> NDArray[np.float64]:
        return self.field(Field.OPEN, lookback)

    def high(self, lookback: int | None = None) -> NDArray[np.float64]:
        return self.field(Field.HIGH, lookback)

    def low(self, lookback: int | None = None) -> NDArray[np.float64]:
        return self.field(Field.LOW, lookback)

    def close(self, lookback: int | None = None) -> NDArray[np.float64]:
        return self.field(Field.CLOSE, lookback)

    def volume(self, lookback: int | None = None) -> NDArray[np.float64]:
        return self.field(Field.VOLUME, lookback)

    def timestamps(self, lookback: int | None = None) -> NDArray[np.datetime64]:
        start, stop = self._window(lookback)
        stamps: NDArray[np.datetime64] = self.__timestamps[start:stop].astype("datetime64[ns]")
        stamps.flags.writeable = False
        return stamps

    # -- acces ponctuel -----------------------------------------------------
    def last(self, field: Field = Field.CLOSE) -> float:
        """Valeur du champ sur la derniere barre close."""
        return float(self.field(field, 1)[0])

    def bar(self, offset: int = 0) -> Bar:
        """Barre datant de ``offset`` pas dans le passe.

        ``offset=0`` est la derniere barre close, ``offset=1`` la precedente.
        Il n'existe volontairement aucune facon d'exprimer une barre future :
        un offset negatif designerait le futur et leve ``LookaheadError``.
        """
        if offset < 0:
            raise LookaheadError(
                f"bar(offset={offset}) designe une barre future. Une vue "
                "d'historique s'adresse uniquement par anciennete (offset >= 0). "
                "Si tu cherches le prix d'execution, il appartient au moteur "
                "d'execution, pas a la strategie."
            )
        if offset >= self.n_bars:
            raise InsufficientHistoryError(
                f"bar(offset={offset}) : seulement {self.n_bars} barres disponibles"
            )
        index = self.n_bars - 1 - offset
        window = self.field  # ajustement applique de facon coherente sur la barre
        span = offset + 1
        return Bar(
            timestamp=_to_datetime(int(self.__timestamps[index])),
            open=float(window(Field.OPEN, span)[0]),
            high=float(window(Field.HIGH, span)[0]),
            low=float(window(Field.LOW, span)[0]),
            close=float(window(Field.CLOSE, span)[0]),
            volume=float(window(Field.VOLUME, span)[0]),
        )

    # -- interoperabilite ---------------------------------------------------
    def as_frame(self, lookback: int | None = None) -> pd.DataFrame:
        """Copie pandas de la fenetre visible, index UTC tz-aware.

        Copie defensive : muter le resultat n'affecte pas le jeu de donnees.
        """
        start, stop = self._window(lookback)
        index = pd.DatetimeIndex(
            pd.to_datetime(self.__timestamps[start:stop], unit="ns", utc=True), name="timestamp"
        ).as_unit("ns")
        data = {
            field.value: np.asarray(self.field(field, stop - start), dtype=np.float64)
            for field in self.__fields
        }
        return pd.DataFrame(data, index=index, copy=True)

    def actions_to_date(self) -> CorporateActions:
        """Operations sur titre connues a l'instant courant, et elles seules."""
        return self.__actions.known_at(self.as_of)


# ---------------------------------------------------------------------------
# Point de decision et curseur
# ---------------------------------------------------------------------------
@final
@dataclass(frozen=True, slots=True)
class DecisionPoint:
    """Un instant de decision : la barre ``index`` vient de cloturer."""

    index: int
    as_of: datetime
    history: HistoryView


@final
class BarCursor:
    """Parcours avant-seulement d'un jeu de donnees.

    Deux garanties :

    * **monotonie** : le curseur n'avance jamais qu'en avant, et le parcours
      n'est consommable qu'une fois. Rejouer une periode apres coup est le
      mecanisme par lequel une optimisation se contamine elle-meme ;
    * **borne** : chaque ``DecisionPoint`` porte une vue arretee a sa propre
      barre, construite ici et nulle part ailleurs.
    """

    __slots__ = ("_consumed", "_data", "_multipliers", "_start", "_stop")

    def __init__(
        self,
        data: MarketData,
        multipliers: Multipliers,
        *,
        warmup: int = 0,
        stop: int | None = None,
    ) -> None:
        total = len(data)
        if warmup < 0:
            raise ValueError(f"warmup negatif : {warmup}")
        if warmup >= total:
            raise InsufficientHistoryError(
                f"warmup={warmup} >= {total} barres disponibles : aucune decision possible"
            )
        self._data = data
        self._multipliers = multipliers
        self._start = warmup
        self._stop = total if stop is None else min(stop, total)
        self._consumed = False

    def __len__(self) -> int:
        return max(0, self._stop - self._start)

    def __iter__(self) -> Iterator[DecisionPoint]:
        if self._consumed:
            raise DataError(
                "BarCursor deja consomme. Un parcours est avant-seulement et "
                "a usage unique ; cree un nouveau curseur pour rejouer la periode."
            )
        self._consumed = True
        data = self._data
        for index in range(self._start, self._stop):
            yield DecisionPoint(
                index=index,
                as_of=_to_datetime(int(data.timestamps[index])),
                history=data.view_at(index, self._multipliers),
            )


# ---------------------------------------------------------------------------
# Jeu de donnees complet : reserve au moteur.
# ---------------------------------------------------------------------------
@final
class MarketData:
    """Serie OHLCV complete, validee et immuable.

    **Ne doit jamais etre transmis a une strategie.** Le moteur de backtest en
    est l'unique detenteur : il a besoin des barres futures pour simuler des
    executions differees, ce qui est legitime pour lui et pour lui seul.
    """

    __slots__ = (
        "_actions",
        "_calendar",
        "_fields",
        "_frequency",
        "_provider",
        "_quality",
        "_symbol",
        "_timestamps",
    )

    def __init__(
        self,
        *,
        symbol: str,
        frequency: Frequency,
        timestamps: NDArray[np.int64],
        open_: NDArray[np.float64],
        high: NDArray[np.float64],
        low: NDArray[np.float64],
        close: NDArray[np.float64],
        volume: NDArray[np.float64],
        actions: CorporateActions | None = None,
        quality: DataQualityReport | None = None,
        provider: str = "unknown",
        calendar: str = "XNYS",
    ) -> None:
        stamps = np.asarray(timestamps, dtype=np.int64)
        n = int(stamps.size)
        if n == 0:
            raise DataError(f"{symbol} : serie vide")
        for name, array in (
            ("open", open_),
            ("high", high),
            ("low", low),
            ("close", close),
            ("volume", volume),
        ):
            if array.size != n:
                raise DataError(f"{symbol} : {name} a {array.size} valeurs pour {n} timestamps")
        if n > 1 and not bool(np.all(np.diff(stamps) > 0)):
            raise DataError(
                f"{symbol} : timestamps non strictement croissants. Doublons ou "
                "desordre doivent etre resolus par le normaliseur, pas ici."
            )
        _check_spacing(symbol, stamps, frequency)
        stamps = stamps.copy()
        stamps.flags.writeable = False
        self._timestamps: NDArray[np.int64] = stamps
        self._fields: dict[Field, NDArray[np.float64]] = {
            Field.OPEN: _freeze(open_),
            Field.HIGH: _freeze(high),
            Field.LOW: _freeze(low),
            Field.CLOSE: _freeze(close),
            Field.VOLUME: _freeze(volume),
        }
        self._symbol = symbol
        self._frequency = frequency
        self._actions = actions if actions is not None else CorporateActions()
        self._quality = quality
        self._provider = provider
        self._calendar = calendar

    # -- identite -----------------------------------------------------------
    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def frequency(self) -> Frequency:
        return self._frequency

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def calendar_name(self) -> str:
        return self._calendar

    @property
    def actions(self) -> CorporateActions:
        return self._actions

    @property
    def quality(self) -> DataQualityReport | None:
        return self._quality

    @property
    def timestamps(self) -> NDArray[np.int64]:
        return self._timestamps

    @property
    def start(self) -> datetime:
        return _to_datetime(int(self._timestamps[0]))

    @property
    def end(self) -> datetime:
        return _to_datetime(int(self._timestamps[-1]))

    def __len__(self) -> int:
        return int(self._timestamps.size)

    def __repr__(self) -> str:
        return (
            f"MarketData({self._symbol} {self._frequency.value}, {len(self)} barres, "
            f"{self.start.date()} -> {self.end.date()}, provider={self._provider})"
        )

    def raw(self, field: Field) -> NDArray[np.float64]:
        """Serie brute, non ajustee, en lecture seule. Usage moteur."""
        return self._fields[field]

    # -- construction de vues -----------------------------------------------
    def multipliers(
        self, policy: AdjustmentPolicy, *, allow_lookahead: bool = False
    ) -> Multipliers:
        return build_multipliers(
            self._timestamps,
            self._fields[Field.CLOSE],
            self._actions,
            policy,
            allow_lookahead=allow_lookahead,
        )

    def view_at(self, index: int, multipliers: Multipliers) -> HistoryView:
        """Vue arretee a la cloture de la barre ``index`` (incluse)."""
        if index < 0:
            raise ValueError(f"index negatif : {index}")
        if index >= len(self):
            raise IndexError(f"index {index} hors serie ({len(self)} barres)")
        return HistoryView(
            symbol=self._symbol,
            frequency=self._frequency,
            timestamps=self._timestamps,
            fields=self._fields,
            multipliers=multipliers,
            actions=self._actions,
            end=index + 1,
        )

    def cursor(
        self,
        policy: AdjustmentPolicy = AdjustmentPolicy.SPLIT_PIT,
        *,
        warmup: int = 0,
        stop: int | None = None,
        allow_lookahead: bool = False,
    ) -> BarCursor:
        """Curseur avant-seulement sur la serie."""
        multipliers = self.multipliers(policy, allow_lookahead=allow_lookahead)
        return BarCursor(self, multipliers, warmup=warmup, stop=stop)

    # -- usage moteur --------------------------------------------------------
    def execution_bar(self, index: int) -> Bar:
        """Barre brute a l'index donne, pour la simulation d'execution.

        Reservee au moteur : c'est le seul composant legitimement autorise a
        regarder une barre posterieure au point de decision, puisque c'est
        precisement ce qu'il simule. Les prix sont bruts, jamais ajustes :
        un ordre s'execute au prix reellement cote.
        """
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} hors serie ({len(self)} barres)")
        return Bar(
            timestamp=_to_datetime(int(self._timestamps[index])),
            open=float(self._fields[Field.OPEN][index]),
            high=float(self._fields[Field.HIGH][index]),
            low=float(self._fields[Field.LOW][index]),
            close=float(self._fields[Field.CLOSE][index]),
            volume=float(self._fields[Field.VOLUME][index]),
        )

    def index_at_or_before(self, moment: datetime) -> int:
        """Index de la derniere barre close a ``moment`` inclus, -1 si aucune."""
        target = np.int64(int(ensure_utc(moment).timestamp() * _NS))
        return int(np.searchsorted(self._timestamps, target, side="right")) - 1

    # -- audit ---------------------------------------------------------------
    def with_future_poisoned(self, visible_bars: int) -> MarketData:
        """Copie ou toute barre d'index >= ``visible_bars`` vaut NaN.

        Outil de verification, pas de production. Executer un backtest sur la
        serie empoisonnee doit rendre exactement les memes decisions que sur la
        serie tronquee : toute divergence prouve qu'un composant a lu le futur,
        y compris par un chemin que la borne de vue ne couvre pas (agregat
        pandas, cache partage, etc.).
        """
        if not 1 <= visible_bars <= len(self):
            raise ValueError(f"visible_bars={visible_bars} hors [1, {len(self)}]")
        poisoned: dict[str, NDArray[np.float64]] = {}
        for field, values in self._fields.items():
            copy = values.copy()
            copy[visible_bars:] = np.nan
            poisoned[field.value] = copy
        return MarketData(
            symbol=self._symbol,
            frequency=self._frequency,
            timestamps=self._timestamps,
            open_=poisoned["open"],
            high=poisoned["high"],
            low=poisoned["low"],
            close=poisoned["close"],
            volume=poisoned["volume"],
            actions=self._actions,
            quality=self._quality,
            provider=f"{self._provider}+poisoned",
            calendar=self._calendar,
        )

    def to_frame(self) -> pd.DataFrame:
        """Serie complete en pandas. Reservee au reporting et au debug.

        Ne jamais transmettre le resultat a une strategie : cet objet contient
        le futur.
        """
        index = pd.DatetimeIndex(
            pd.to_datetime(self._timestamps, unit="ns", utc=True), name="timestamp"
        ).as_unit("ns")
        return pd.DataFrame(
            {field.value: values for field, values in self._fields.items()},
            index=index,
            copy=True,
        )

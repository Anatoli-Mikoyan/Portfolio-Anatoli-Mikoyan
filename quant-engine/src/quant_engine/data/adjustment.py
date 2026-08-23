"""Ajustement des prix pour splits et dividendes -- **point-in-time**.

Le probleme que ce module resout
--------------------------------
Toutes les sources grand public (yfinance en tete) livrent par defaut des prix
retro-ajustes : la serie entiere est recalculee avec *toutes* les operations sur
titre connues aujourd'hui. Le close d'Apple au 2 janvier 2015 y apparait a
~24 $ alors qu'il cotait ~109 $ ce jour-la : la serie integre le split 4-pour-1
de 2020.

C'est un look-ahead bias massif et totalement invisible :

* le niveau de prix vu par la strategie en 2015 n'a jamais existe ;
* les seuils absolus, les tailles de position, la granularite du tick sont faux ;
* pire, le *facteur* d'ajustement encode une information de 2020.

Le remede
---------
On conserve les prix bruts et les operations separement, puis on calcule le
facteur d'ajustement **relatif au curseur**. A l'instant ``t``, la serie visible
est ajustee exactement comme la verrait un operateur ce jour-la : normalisee de
sorte que le prix de la barre courante soit le prix brut reellement negociable,
et que l'historique anterieur soit rendu comparable avec les seules operations
deja survenues.

Formulation
-----------
Soit ``m[i]`` le multiplicateur retro classique (normalise a 1 en fin de serie) :

    m[i] = 1 / PROD(ratio_k : split k avec ex-date > date_i)
           * PROD(1 - d_k / close_precedent_k : dividende k avec ex-date > date_i)

Le facteur point-in-time au curseur ``t`` vaut ``m[i] / m[t]``. Il verifie
``m[t]/m[t] = 1`` : le prix courant reste le prix brut, et seules les operations
survenues dans ``]date_i, date_t]`` corrigent le passe. Un split posterieur a
``t`` n'a par construction aucun effet.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, final

import numpy as np
from numpy.typing import NDArray

from ..errors import AdjustmentError
from ..logging_setup import get_logger

if TYPE_CHECKING:
    from .corporate_actions import CorporateActions

__all__ = ["AdjustmentPolicy", "Multipliers", "build_multipliers"]

_LOG = get_logger("data.adjustment")


class AdjustmentPolicy(Enum):
    """Politique d'ajustement des prix.

    ``SPLIT_PIT`` est le defaut du moteur : les splits sont purement mecaniques
    (ils ne creent ni ne detruisent de valeur) tandis que les dividendes sont un
    flux de tresorerie reel, mieux modelise en creditant le compte a l'ex-date
    qu'en deformant la serie de prix.
    """

    RAW = "raw"
    """Aucun ajustement. Les discontinuites de split restent visibles."""

    SPLIT_PIT = "split_pit"
    """Splits uniquement, normalise au curseur. Defaut recommande."""

    TOTAL_RETURN_PIT = "total_return_pit"
    """Splits + dividendes, normalise au curseur. Pour comparer a un indice TR."""

    FULL_RETRO_SPLIT = "full_retro_split"
    """Retro-ajustement classique (splits), normalise en fin de serie.
    CONTAMINE PAR DU LOOK-AHEAD. Reserve a la reconciliation avec un outil tiers."""

    FULL_RETRO_TOTAL = "full_retro_total"
    """Equivalent du ``Adj Close`` de yfinance. CONTAMINE PAR DU LOOK-AHEAD."""

    @property
    def uses_splits(self) -> bool:
        return self is not AdjustmentPolicy.RAW

    @property
    def uses_dividends(self) -> bool:
        return self in (
            AdjustmentPolicy.TOTAL_RETURN_PIT,
            AdjustmentPolicy.FULL_RETRO_TOTAL,
        )

    @property
    def is_point_in_time(self) -> bool:
        return self in (
            AdjustmentPolicy.RAW,
            AdjustmentPolicy.SPLIT_PIT,
            AdjustmentPolicy.TOTAL_RETURN_PIT,
        )

    @property
    def is_lookahead_contaminated(self) -> bool:
        """Vrai si la politique utilise des operations posterieures au curseur."""
        return not self.is_point_in_time


@final
@dataclass(frozen=True, slots=True)
class Multipliers:
    """Facteurs d'ajustement precalcules sur toute la serie.

    ``normalize_index`` encode la difference fondamentale entre les deux familles
    de politiques :

    * ``None`` -> normalisation au curseur (point-in-time, sans look-ahead) ;
    * un entier -> normalisation a un index fixe, generalement ``N-1``, ce qui
      injecte de l'information de fin de serie dans tout le passe.
    """

    policy: AdjustmentPolicy
    price: NDArray[np.float64]
    volume: NDArray[np.float64]
    normalize_index: int | None

    @property
    def is_identity(self) -> bool:
        """Chemin rapide : aucun ajustement a appliquer."""
        return self.policy is AdjustmentPolicy.RAW

    @property
    def is_lookahead_contaminated(self) -> bool:
        return self.normalize_index is not None

    def price_factor(self, start: int, stop: int, cursor: int) -> NDArray[np.float64]:
        """Facteur multiplicatif a appliquer aux prix bruts de ``[start, stop)``.

        ``cursor`` est l'index de la derniere barre visible.
        """
        anchor = self.normalize_index if self.normalize_index is not None else cursor
        denominator = float(self.price[anchor])
        factor: NDArray[np.float64] = self.price[start:stop] / denominator
        return factor

    def volume_factor(self, start: int, stop: int, cursor: int) -> NDArray[np.float64]:
        anchor = self.normalize_index if self.normalize_index is not None else cursor
        denominator = float(self.volume[anchor])
        factor: NDArray[np.float64] = self.volume[start:stop] / denominator
        return factor


def build_multipliers(
    timestamps: NDArray[np.int64],
    close: NDArray[np.float64],
    actions: CorporateActions,
    policy: AdjustmentPolicy,
    *,
    allow_lookahead: bool = False,
) -> Multipliers:
    """Precalcule les multiplicateurs d'ajustement pour toute la serie.

    Cout : O(n + nombre d'operations), une seule fois par jeu de donnees.
    L'application au curseur est ensuite O(taille de la fenetre demandee).
    """
    n = int(timestamps.size)
    if n == 0:
        raise AdjustmentError("Serie vide : ajustement impossible")
    if close.size != n:
        raise AdjustmentError(f"Tailles incoherentes : {n} timestamps, {close.size} closes")

    if policy.is_lookahead_contaminated and not allow_lookahead:
        raise AdjustmentError(
            f"La politique {policy.value!r} retro-ajuste la serie avec des operations "
            "posterieures a chaque barre : c'est du look-ahead bias. Elle n'est "
            "utilisable que pour reconcilier des chiffres avec un outil tiers. "
            "Passe allow_lookahead=True en connaissance de cause, ou utilise "
            f"{AdjustmentPolicy.SPLIT_PIT.value!r}."
        )

    price_mult = np.ones(n, dtype=np.float64)
    volume_mult = np.ones(n, dtype=np.float64)

    if policy.uses_splits and actions.splits:
        split_divisor = np.ones(n, dtype=np.float64)
        for split in actions.splits:
            ex_index = _ex_index(timestamps, split.ex_date.timestamp())
            if ex_index <= 0 or ex_index >= n:
                # Operation hors de la fenetre chargee : sans effet sur la serie.
                continue
            split_divisor[:ex_index] *= split.ratio
        price_mult /= split_divisor
        volume_mult *= split_divisor

    if policy.uses_dividends and actions.dividends:
        div_mult = np.ones(n, dtype=np.float64)
        for dividend in actions.dividends:
            ex_index = _ex_index(timestamps, dividend.ex_date.timestamp())
            if ex_index <= 0 or ex_index >= n:
                continue
            cum_close = float(close[ex_index - 1])
            if not np.isfinite(cum_close) or cum_close <= 0.0:
                raise AdjustmentError(
                    f"Close de reference invalide ({cum_close}) a l'index {ex_index - 1} "
                    f"pour le dividende du {dividend.ex_date.date()}"
                )
            ratio = 1.0 - dividend.amount / cum_close
            if ratio <= 0.0:
                raise AdjustmentError(
                    f"Dividende de {dividend.amount} superieur au cours {cum_close} "
                    f"le {dividend.ex_date.date()} : donnee incoherente, pas un ajustement."
                )
            div_mult[:ex_index] *= ratio
        price_mult *= div_mult

    normalize_index = None if policy.is_point_in_time else n - 1
    if normalize_index is not None:
        _LOG.warning(
            "politique d'ajustement contaminee par du look-ahead",
            extra={"policy": policy.value, "n_bars": n},
        )

    price_mult.flags.writeable = False
    volume_mult.flags.writeable = False
    return Multipliers(
        policy=policy,
        price=price_mult,
        volume=volume_mult,
        normalize_index=normalize_index,
    )


def _ex_index(timestamps: NDArray[np.int64], ex_epoch_seconds: float) -> int:
    """Index de la premiere barre dont la cloture est >= a l'ex-date.

    Les barres d'index strictement inferieur sont pre-operation et doivent etre
    ajustees ; celles a partir de cet index cotent deja post-operation.
    """
    ex_ns = np.int64(round(ex_epoch_seconds * 1_000_000_000))
    return int(np.searchsorted(timestamps, ex_ns, side="left"))

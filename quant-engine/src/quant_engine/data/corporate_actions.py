"""Operations sur titre : splits et dividendes.

Conservees **separement** de la serie de prix, et jamais fusionnees dedans a
l'ingestion. Raison : l'ajustement d'un prix depend de la date a laquelle on
regarde. Ecraser les prix bruts avec un ajustement calcule sur l'historique
complet detruit l'information necessaire pour reconstruire ce qu'un operateur
voyait reellement a une date passee (cf. ``adjustment.py``).
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime
from typing import final

from .types import ensure_utc

__all__ = ["CorporateActions", "Dividend", "Split"]


@dataclass(frozen=True, slots=True, order=True)
class Split:
    """Division/regroupement d'actions.

    ``ratio`` suit la convention "nouvelles pour anciennes" : 2.0 pour un
    2-pour-1 (le prix est divise par 2, le nombre de titres double), 0.1 pour
    un regroupement 1-pour-10.
    """

    ex_date: datetime
    ratio: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "ex_date", ensure_utc(self.ex_date, what="Split.ex_date"))
        if not self.ratio > 0.0:
            raise ValueError(f"Ratio de split invalide : {self.ratio}")


@dataclass(frozen=True, slots=True, order=True)
class Dividend:
    """Dividende en numeraire, date a l'ex-date (pas a la date de paiement).

    L'ex-date est la seule qui compte pour le prix : c'est ce jour-la que le
    cours decroche du montant du coupon.
    """

    ex_date: datetime
    amount: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "ex_date", ensure_utc(self.ex_date, what="Dividend.ex_date"))
        if self.amount < 0.0:
            raise ValueError(f"Dividende negatif : {self.amount}")


@final
@dataclass(frozen=True, slots=True)
class CorporateActions:
    """Collection immuable et triee d'operations sur titre."""

    splits: tuple[Split, ...] = ()
    dividends: tuple[Dividend, ...] = ()
    _split_dates: tuple[datetime, ...] = field(default=(), repr=False, compare=False)
    _dividend_dates: tuple[datetime, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        splits = tuple(sorted(self.splits))
        dividends = tuple(sorted(self.dividends))
        object.__setattr__(self, "splits", splits)
        object.__setattr__(self, "dividends", dividends)
        object.__setattr__(self, "_split_dates", tuple(item.ex_date for item in splits))
        object.__setattr__(self, "_dividend_dates", tuple(item.ex_date for item in dividends))

    @property
    def is_empty(self) -> bool:
        return not self.splits and not self.dividends

    def known_at(self, as_of: datetime) -> CorporateActions:
        """Sous-ensemble des operations dont l'ex-date est <= ``as_of``.

        C'est la seule vue legitime pour un backtest a l'instant ``as_of``.
        """
        moment = ensure_utc(as_of, what="as_of")
        cut_splits = bisect_right(self._split_dates, moment)
        cut_dividends = bisect_right(self._dividend_dates, moment)
        return CorporateActions(
            splits=self.splits[:cut_splits],
            dividends=self.dividends[:cut_dividends],
        )

    def dividends_between(
        self, start: datetime, end: datetime, *, inclusive_end: bool = True
    ) -> tuple[Dividend, ...]:
        """Dividendes dont l'ex-date tombe dans ``]start, end]``.

        Utilise par le moteur pour crediter le cash d'une position detenue :
        modeliser un dividende comme un flux de tresorerie est plus honnete
        que de le noyer dans un prix ajuste.
        """
        low = ensure_utc(start, what="start")
        high = ensure_utc(end, what="end")
        first = bisect_right(self._dividend_dates, low)
        last = (
            bisect_right(self._dividend_dates, high)
            if inclusive_end
            else _bisect_left(self._dividend_dates, high)
        )
        return self.dividends[first:last]

    def splits_between(self, start: datetime, end: datetime) -> tuple[Split, ...]:
        low = ensure_utc(start, what="start")
        high = ensure_utc(end, what="end")
        first = bisect_right(self._split_dates, low)
        last = bisect_right(self._split_dates, high)
        return self.splits[first:last]

    def __repr__(self) -> str:
        return f"CorporateActions(splits={len(self.splits)}, dividends={len(self.dividends)})"


def _bisect_left(items: tuple[datetime, ...], value: datetime) -> int:
    from bisect import bisect_left

    return bisect_left(items, value)

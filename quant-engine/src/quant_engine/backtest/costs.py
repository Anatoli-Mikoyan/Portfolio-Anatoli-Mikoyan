"""Modelisation des frictions d'execution.

Pourquoi ce module est le plus important du moteur
--------------------------------------------------
Un backtest sans couts n'est pas un backtest optimiste : c'est un backtest
**faux**. La difference n'est pas marginale. Sur un compte de 100 EUR chez un
courtier facturant 1 EUR minimum par ordre, un aller-retour coute 2 % du
capital. Une strategie qui traite une fois par semaine detruit donc 104 % du
compte par an en commissions -- avant meme de se demander si elle a raison.

C'est la raison numero un pour laquelle les backtests amateurs paraissent
brillants et les comptes reels se vident. Le moteur refuse donc de demarrer
sans configuration de couts explicite : il n'existe **aucune valeur par defaut
a zero** dans ce module.

Les quatre frictions modelisees
-------------------------------
1. **Commission** -- ce que facture le courtier. Fixe, proportionnelle, ou les
   deux, avec plancher et plafond.
2. **Spread** -- l'ecart entre le prix d'achat et le prix de vente. On paie
   toujours la moitie du spread, dans les deux sens. Invisible sur un releve,
   bien reel dans la performance.
3. **Slippage** -- l'ecart entre le prix espere et le prix obtenu, du fait de
   l'impact de l'ordre sur le marche. Modele en racine carree de la
   participation au volume, forme standard de la litterature sur l'impact.
4. **Latence et execution partielle** -- traites dans le moteur, pas ici.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import final

from ..errors import ConfigError

__all__ = [
    "CommissionModel",
    "CostModel",
    "FillContext",
    "SlippageModel",
    "SpreadModel",
    "SquareRootSlippage",
    "VolatilitySpread",
]


@final
@dataclass(frozen=True, slots=True)
class FillContext:
    """Etat de marche au moment ou un ordre est execute."""

    reference_price: float
    """Prix de reference avant frictions (typiquement l'ouverture de la barre)."""
    quantity: float
    """Quantite signee : positive a l'achat, negative a la vente."""
    bar_volume: float
    """Volume echange sur la barre d'execution."""
    average_volume: float
    """Volume moyen recent, pour normaliser la participation."""
    volatility: float
    """Volatilite recente en rendement quotidien, pour le spread dynamique."""

    @property
    def notional(self) -> float:
        return abs(self.quantity) * self.reference_price

    @property
    def participation(self) -> float:
        """Fraction du volume de la barre que represente l'ordre.

        Au-dela de quelques pour cent, l'ordre deplace le marche contre
        lui-meme : c'est ce que capture le modele de slippage.
        """
        reference = max(self.bar_volume, self.average_volume, 1.0)
        return abs(self.quantity) / reference


# ---------------------------------------------------------------------------
# Commission
# ---------------------------------------------------------------------------
@final
@dataclass(frozen=True, slots=True)
class CommissionModel:
    """Grille tarifaire d'un courtier.

    Les valeurs par defaut reproduisent la grille actions US d'Interactive
    Brokers : 0,005 USD par titre, 1 USD minimum par ordre, plafonne a 1 % du
    montant. C'est le plafond a 1 % qui rend les petits comptes non viables :
    sur un ordre de 100 EUR, la commission est de 1 EUR, soit 1 %.
    """

    per_order: float = 0.0
    """Montant fixe par ordre."""
    per_unit: float = 0.005
    """Montant par titre."""
    bps_of_notional: float = 0.0
    """Points de base du montant traite (1 bp = 0,01 %)."""
    minimum: float = 1.0
    """Plancher par ordre."""
    maximum_pct_of_notional: float = 0.01
    """Plafond, en fraction du montant traite."""

    def __post_init__(self) -> None:
        for name in ("per_order", "per_unit", "bps_of_notional", "minimum"):
            if getattr(self, name) < 0.0:
                raise ConfigError(f"CommissionModel.{name} ne peut pas etre negatif")
        if not 0.0 < self.maximum_pct_of_notional <= 1.0:
            raise ConfigError("maximum_pct_of_notional doit etre dans ]0, 1]")

    def charge(self, context: FillContext) -> float:
        raw = (
            self.per_order
            + self.per_unit * abs(context.quantity)
            + self.bps_of_notional / 10_000.0 * context.notional
        )
        capped = min(max(raw, self.minimum), self.maximum_pct_of_notional * context.notional)
        return max(0.0, capped)

    @classmethod
    def zero(cls) -> CommissionModel:
        """Grille nulle. Reservee aux tests analytiques du moteur.

        Ne jamais utiliser pour evaluer une strategie : c'est precisement
        l'hypothese qui rend faux la quasi-totalite des backtests amateurs.
        """
        return cls(
            per_order=0.0, per_unit=0.0, bps_of_notional=0.0,
            minimum=0.0, maximum_pct_of_notional=1.0,
        )


# ---------------------------------------------------------------------------
# Spread
# ---------------------------------------------------------------------------
class SpreadModel(ABC):
    """Ecart achat/vente. On en paie toujours la moitie, dans chaque sens."""

    @abstractmethod
    def half_spread_bps(self, context: FillContext) -> float:
        """Demi-spread en points de base du prix de reference."""


@final
@dataclass(frozen=True, slots=True)
class FixedSpread(SpreadModel):
    """Spread constant, exprime en points de base."""

    spread_bps: float

    def __post_init__(self) -> None:
        if self.spread_bps < 0.0:
            raise ConfigError("spread_bps ne peut pas etre negatif")

    def half_spread_bps(self, context: FillContext) -> float:  # noqa: ARG002
        return self.spread_bps / 2.0


@final
@dataclass(frozen=True, slots=True)
class VolatilitySpread(SpreadModel):
    """Spread proportionnel a la volatilite recente, avec plancher.

    Plus realiste qu'un spread fixe : les ecarts se creusent exactement quand
    la volatilite monte, c'est-a-dire aux moments ou la plupart des strategies
    veulent traiter. Un spread fixe sous-estime donc le cout la ou il compte.
    """

    coefficient: float = 0.10
    """Fraction de la volatilite quotidienne retenue comme spread complet."""
    floor_bps: float = 1.0
    cap_bps: float = 100.0

    def __post_init__(self) -> None:
        if self.coefficient < 0.0 or self.floor_bps < 0.0:
            raise ConfigError("Parametres de VolatilitySpread negatifs")
        if self.cap_bps < self.floor_bps:
            raise ConfigError("cap_bps doit etre >= floor_bps")

    def half_spread_bps(self, context: FillContext) -> float:
        implied = self.coefficient * context.volatility * 10_000.0
        return min(max(implied, self.floor_bps), self.cap_bps) / 2.0


# ---------------------------------------------------------------------------
# Slippage
# ---------------------------------------------------------------------------
class SlippageModel(ABC):
    """Deplacement du prix cause par l'ordre lui-meme."""

    @abstractmethod
    def slippage_bps(self, context: FillContext) -> float:
        """Cout d'impact en points de base, toujours defavorable."""


@final
@dataclass(frozen=True, slots=True)
class SquareRootSlippage(SlippageModel):
    """Impact en racine carree de la participation au volume.

    ``impact = coefficient x volatilite x sqrt(participation)``

    La forme en racine carree est la specification standard de la litterature
    sur l'impact de marche : doubler la taille d'un ordre ne double pas son
    cout d'impact, il le multiplie par environ 1,41. Le modele lineaire, plus
    simple, surestime fortement le cout des gros ordres et sous-estime celui
    des petits.
    """

    coefficient: float = 1.0
    max_bps: float = 500.0

    def __post_init__(self) -> None:
        if self.coefficient < 0.0:
            raise ConfigError("coefficient de slippage negatif")

    def slippage_bps(self, context: FillContext) -> float:
        impact = self.coefficient * context.volatility * math.sqrt(context.participation)
        return min(impact * 10_000.0, self.max_bps)


@final
@dataclass(frozen=True, slots=True)
class LinearSlippage(SlippageModel):
    """Impact proportionnel a la participation. Plus pessimiste sur les gros ordres."""

    coefficient: float = 10.0
    max_bps: float = 500.0

    def slippage_bps(self, context: FillContext) -> float:
        return min(self.coefficient * context.participation * 10_000.0, self.max_bps)


# ---------------------------------------------------------------------------
# Agregat
# ---------------------------------------------------------------------------
@final
@dataclass(frozen=True, slots=True)
class CostModel:
    """Ensemble des frictions. Aucun composant n'est optionnel."""

    commission: CommissionModel
    spread: SpreadModel
    slippage: SlippageModel

    def execution_price(self, context: FillContext) -> float:
        """Prix reellement obtenu, frictions de marche incluses.

        Le signe est toujours defavorable : on achete plus cher et on vend moins
        cher que le prix de reference. Un modele qui laisserait le slippage
        jouer dans les deux sens produirait une esperance de cout nulle, ce qui
        est faux -- l'impact d'un ordre va par construction contre celui qui le
        passe.
        """
        friction_bps = self.spread.half_spread_bps(context) + self.slippage.slippage_bps(context)
        direction = 1.0 if context.quantity > 0 else -1.0
        return context.reference_price * (1.0 + direction * friction_bps / 10_000.0)

    def commission_for(self, context: FillContext) -> float:
        return self.commission.charge(context)

    def round_trip_cost_pct(self, notional: float, *, volatility: float = 0.015) -> float:
        """Cout estime d'un aller-retour complet, en fraction du montant.

        Sert a l'avertissement prealable du moteur : il calcule ce que la
        rotation declaree d'une strategie va couter *avant* de lancer le
        backtest. Sur un petit compte, ce seul chiffre suffit souvent a conclure.
        """
        if notional <= 0.0:
            raise ConfigError("notional doit etre strictement positif")
        probe = FillContext(
            reference_price=100.0,
            quantity=notional / 100.0,
            bar_volume=1e7,
            average_volume=1e7,
            volatility=volatility,
        )
        friction_bps = 2.0 * (
            self.spread.half_spread_bps(probe) + self.slippage.slippage_bps(probe)
        )
        commissions = 2.0 * self.commission.charge(probe)
        return friction_bps / 10_000.0 + commissions / notional

    @classmethod
    def interactive_brokers_us_equity(cls) -> CostModel:
        """Profil realiste pour une action US liquide chez Interactive Brokers."""
        return cls(
            commission=CommissionModel(),
            spread=VolatilitySpread(coefficient=0.05, floor_bps=1.0),
            slippage=SquareRootSlippage(coefficient=1.0),
        )

    @classmethod
    def frictionless(cls) -> CostModel:
        """Monde sans frictions. **Tests analytiques du moteur uniquement.**

        Utiliser ce profil pour evaluer une strategie revient a mesurer la
        performance d'une voiture en supprimant la resistance de l'air.
        """
        return cls(
            commission=CommissionModel.zero(),
            spread=FixedSpread(0.0),
            slippage=SquareRootSlippage(coefficient=0.0),
        )

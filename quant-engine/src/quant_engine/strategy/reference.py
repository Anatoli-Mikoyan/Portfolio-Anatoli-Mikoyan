"""Trois strategies de reference.

Elles ne sont pas la pour gagner de l'argent. Elles sont la pour **tester le
moteur** : chacune a un comportement connu et verifiable analytiquement, ce qui
permet de detecter une derive du moteur avant qu'elle ne contamine un vrai
resultat.

* ``BuyAndHold`` -- la baseline obligatoire. Zero parametre reglable, donc zero
  degre de liberte, donc aucun risque de sur-ajustement. Toute strategie doit
  la battre *apres couts* pour meriter qu'on s'y interesse. La plupart echouent.
* ``MovingAverageCrossover`` -- le croisement de moyennes mobiles, socle de la
  quasi-totalite des bots vendus sur internet. Publie dans les annees 1970,
  etudie depuis, et documente comme ne survivant generalement pas aux couts de
  transaction sur actions liquides.
* ``BollingerMeanReversion`` -- retour a la moyenne sur bandes de Bollinger.
  Rotation elevee, donc tres sensible aux frictions : un excellent revelateur
  de moteur trop optimiste.

Aucune de ces trois strategies n'a d'edge documente sur actions liquides en
journalier. Si le moteur les fait paraitre rentables, c'est le moteur qu'il faut
corriger, pas la strategie qu'il faut celebrer.
"""

from __future__ import annotations

from typing import final

import numpy as np

from .base import ParameterSpec, Signal, Strategy, StrategyContext

__all__ = ["BollingerMeanReversion", "BuyAndHold", "MovingAverageCrossover"]


@final
class BuyAndHold(Strategy):
    """Achete a la premiere barre, ne touche plus a rien.

    Baseline de reference du moteur. Son interet methodologique est d'avoir
    **zero degre de liberte** : il n'y a rien a optimiser, donc rien a
    sur-ajuster, donc sa performance mesuree est une estimation non biaisee de
    sa performance future -- ce qu'aucune strategie optimisee ne peut affirmer.
    """

    name = "buy_and_hold"

    @classmethod
    def specs(cls) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec(
                name="weight",
                default=1.0,
                kind="float",
                description="Exposition cible constante.",
                low=0.0,
                high=1.0,
                tunable=False,
            ),
        )

    @property
    def warmup_bars(self) -> int:
        return 1

    @property
    def expected_annual_turnover(self) -> float:
        return 0.0

    def on_bar(self, context: StrategyContext) -> Signal | None:
        if context.is_flat:
            return Signal(self.params.float_("weight"), reason="entree initiale")
        return None


@final
class MovingAverageCrossover(Strategy):
    """Long quand la moyenne courte passe au-dessus de la moyenne longue.

    Deux parametres reglables, donc deux degres de liberte. Sur les domaines
    declares, une recherche en grille explore plusieurs milliers de couples :
    retenir le meilleur revient a selectionner l'extreme d'un echantillon de
    plusieurs milliers de tirages. Le moteur affiche ce nombre pour que la
    performance annoncee soit lue a cette aune.
    """

    name = "ma_crossover"

    @classmethod
    def specs(cls) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec(
                name="fast",
                default=50,
                kind="int",
                description="Fenetre de la moyenne mobile courte, en barres.",
                low=5,
                high=100,
                step=5,
            ),
            ParameterSpec(
                name="slow",
                default=200,
                kind="int",
                description="Fenetre de la moyenne mobile longue, en barres.",
                low=20,
                high=300,
                step=10,
            ),
            ParameterSpec(
                name="allow_short",
                default=False,
                kind="bool",
                description="Prendre une position vendeuse quand le signal s'inverse.",
            ),
        )

    def __init__(self, **overrides: object) -> None:
        super().__init__(**overrides)
        self._target: float = 0.0
        if self.params.int_("fast") >= self.params.int_("slow"):
            raise ValueError(
                f"fast={self.params.int_('fast')} doit etre strictement inferieur a "
                f"slow={self.params.int_('slow')} : sinon le croisement n'a pas de sens."
            )

    def reset(self) -> None:
        self._target = 0.0

    @property
    def warmup_bars(self) -> int:
        return self.params.int_("slow")

    @property
    def expected_annual_turnover(self) -> float:
        """Estimation empirique : un croisement 50/200 en journalier produit de
        l'ordre de 2 a 4 allers-retours par an sur actions liquides."""
        slow = self.params.int_("slow")
        return max(1.0, 252.0 / (2.0 * slow) * 4.0)

    def on_bar(self, context: StrategyContext) -> Signal | None:
        slow = self.params.int_("slow")
        if not context.history.has(slow):
            return None
        closes = context.history.close(slow)
        fast_mean = float(closes[-self.params.int_("fast") :].mean())
        slow_mean = float(closes.mean())

        if fast_mean > slow_mean:
            target = 1.0
            reason = "MM courte au-dessus de la MM longue"
        elif self.params.bool_("allow_short"):
            target = -1.0
            reason = "MM courte en dessous : position vendeuse"
        else:
            target = 0.0
            reason = "MM courte en dessous : sortie"

        # On compare a l'exposition *voulue*, pas a l'exposition constatee : le
        # poids reel derive en permanence avec le prix, et s'y comparer
        # declencherait un rebalancement a chaque barre -- donc des frais a
        # chaque barre, pour une strategie censee traiter deux fois par an.
        if abs(target - self._target) < 1e-9:
            return None
        self._target = target
        return Signal(target, reason=reason)


@final
class BollingerMeanReversion(Strategy):
    """Achete sous la bande basse, sort au retour vers la moyenne.

    Rotation nettement plus elevee que le croisement de moyennes, donc bien plus
    exposee aux frictions. C'est precisement ce qui en fait un bon test : une
    strategie a forte rotation est le premier endroit ou un moteur qui
    sous-estime les couts se trahit.
    """

    name = "bollinger_mean_reversion"

    @classmethod
    def specs(cls) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec(
                name="window",
                default=20,
                kind="int",
                description="Fenetre de la moyenne et de l'ecart-type.",
                low=10,
                high=60,
                step=5,
            ),
            ParameterSpec(
                name="entry_sigma",
                default=2.0,
                kind="float",
                description="Nombre d'ecarts-types sous la moyenne declenchant l'achat.",
                low=1.0,
                high=3.5,
                step=0.25,
            ),
            ParameterSpec(
                name="exit_sigma",
                default=0.0,
                kind="float",
                description="Niveau de sortie, en ecarts-types autour de la moyenne.",
                low=-1.0,
                high=2.0,
                step=0.25,
            ),
        )

    @property
    def warmup_bars(self) -> int:
        return self.params.int_("window")

    @property
    def expected_annual_turnover(self) -> float:
        """Une bande a 2 sigma est franchie de l'ordre de 20 a 30 fois par an
        sur une serie journaliere ordinaire."""
        return 25.0

    def on_bar(self, context: StrategyContext) -> Signal | None:
        window = self.params.int_("window")
        if not context.history.has(window):
            return None
        closes = context.history.close(window)
        mean = float(closes.mean())
        # Ecart-type d'echantillon (ddof=1) : avec ddof=0 on sous-estime la
        # dispersion sur fenetre courte, ce qui resserre les bandes et fabrique
        # des signaux qui n'auraient pas eu lieu.
        std = float(np.std(closes, ddof=1))
        if std <= 0.0:
            return None

        last = float(closes[-1])
        z_score = (last - mean) / std

        if context.is_flat and z_score <= -self.params.float_("entry_sigma"):
            return Signal(1.0, reason=f"z={z_score:.2f} sous la bande basse")
        if context.is_long and z_score >= self.params.float_("exit_sigma"):
            return Signal(0.0, reason=f"z={z_score:.2f} : retour a la moyenne")
        return None

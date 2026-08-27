"""Politiques de cotation d'un teneur de marché.

Toutes répondent à la même question — *à quelle distance du prix moyen placer mon achat
et ma vente ?* — mais elles arbitrent différemment entre les trois forces du métier.

    NaiveSymmetric      fourchette fixe, aucune gestion d'inventaire.
                        Le point de comparaison : ce que produit une implémentation
                        naïve, et la démonstration que l'inventaire est le vrai risque.

    LinearSkew          fourchette fixe, décalée linéairement selon l'inventaire.
                        L'heuristique de bon sens, sans optimalité prouvée.

    AvellanedaStoikov   la référence (2008). Résout le contrôle stochastique d'un teneur
                        maximisant l'utilité exponentielle de sa richesse terminale.
                        Horizon FINI : la fourchette se resserre en fin de séance,
                        l'agent liquide son inventaire.

    GueantLehalleFT     forme fermée asymptotique (Guéant, Lehalle & Fernandez-Tapia,
                        2013) pour l'horizon infini avec bornes d'inventaire.
                        C'est celle qu'on déploie : pas de dépendance au temps restant,
                        donc pas de comportement absurde quand la séance se prolonge.

    AlphaAware          A-S augmenté d'un signal directionnel. Le pont entre la tenue de
                        marché et le reste du dépôt : si le modèle RL sait quoi que ce
                        soit, c'est ici qu'il vaut de l'argent — en décalant les
                        cotations plutôt qu'en traversant la fourchette.

Le point commun à toutes : ce sont des politiques de **placement**, pas de prédiction.
Elles gagnent en fournissant un service, pas en devinant la direction.

---

**Une note sur `gamma`, parce que c'est le piège numérique du domaine.**

L'aversion au risque d'Avellaneda-Stoikov n'est pas un nombre sans dimension entre 0 et 1.
Elle a les dimensions de l'inverse d'une richesse, et sa valeur dépend donc entièrement de
l'échelle des prix manipulés. La lecture utile passe par le décalage qu'elle produit :

    décalage du prix de réserve = q · gamma · sigma² · (T − t)

Sur une paire de change à 1.10, avec sigma ≈ 1.4e-5 par √seconde et un horizon de cinq
minutes, l'écart-type du prix moyen sur l'horizon vaut environ 2.4 pips. Pour qu'une unité
d'inventaire coûte un pip de décalage, il faut gamma ≈ 2 000 — et non 0.5, valeur que l'on
recopie souvent depuis les articles écrits sur des actions cotées en dizaines de dollars.

Avec un gamma mille fois trop petit, les modèles « optimaux » produisent des cotations
rigoureusement symétriques : ils dégénèrent silencieusement en la politique naïve, sans
qu'aucune erreur ne soit levée. `describe()` affiche les grandeurs effectives pour rendre
ce défaut visible avant la simulation.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .model import FlowParams, MarketState

__all__ = ["QuotingPolicy", "NaiveSymmetric", "LinearSkew", "AvellanedaStoikov",
           "GueantLehalleFT", "AlphaAware", "POLICIES"]


# =======================================================================================
class QuotingPolicy(ABC):
    """Une politique rend (delta_bid, delta_ask) : distances au prix moyen, positives."""

    name: str = "politique"

    @abstractmethod
    def quotes(self, state: MarketState, flow: FlowParams) -> Tuple[float, float]:
        ...

    def reset(self) -> None:
        """Réinitialise l'état interne éventuel entre deux simulations."""

    def _borner(self, delta: float, flow: FlowParams) -> float:
        """Une cotation ne peut pas être négative ni plus fine que le pas de cotation."""
        return float(max(delta, flow.tick / 2.0))

    def describe(self, flow: FlowParams, inventaires=(0.0, 1.0, 5.0),
                 time_left: float = 300.0) -> str:
        """Cotations effectives, en pips, pour quelques niveaux d'inventaire.

        À lire AVANT toute simulation : si les colonnes sont identiques, la politique
        ne gère pas l'inventaire et le paramétrage est à revoir.
        """
        lignes = [f"{self.name} — cotations en pips depuis le prix moyen"]
        for q in inventaires:
            st = MarketState(t=0.0, time_left=time_left, mid=flow.s0, inventory=q,
                             cash=0.0, sigma=flow.sigma)
            b, a = self.quotes(st, flow)
            lignes.append(f"   q = {q:+5.1f} :  achat {b * 1e4:7.3f}   vente {a * 1e4:7.3f}")
        return "\n".join(lignes)

    def __repr__(self) -> str:  # pragma: no cover - affichage
        return f"{self.name}"


# =======================================================================================
@dataclass
class NaiveSymmetric(QuotingPolicy):
    """Fourchette fixe, centrée sur le prix moyen. Aucune gestion d'inventaire.

    Sert de témoin. Elle capture la fourchette aussi bien que les autres — et se fait
    détruire par l'inventaire, parce que rien ne la pousse à se rééquilibrer. C'est
    l'implémentation qu'on trouve dans la plupart des tutoriels de « market making bot ».
    """
    half_spread_ticks: float = 5.0
    name: str = "Naïve symétrique"

    def quotes(self, state: MarketState, flow: FlowParams) -> Tuple[float, float]:
        d = self.half_spread_ticks * flow.tick
        return self._borner(d, flow), self._borner(d, flow)


# =======================================================================================
@dataclass
class LinearSkew(QuotingPolicy):
    """Fourchette fixe, décalée proportionnellement à l'inventaire.

    L'heuristique évidente : long, on rend l'achat moins attractif et la vente plus
    attractive. Elle marche étonnamment bien et sert de référence honnête face aux
    modèles optimaux — un gain doit se justifier face à ELLE, pas face à la naïve.
    """
    half_spread_ticks: float = 5.0
    skew_ticks_per_unit: float = 1.0
    max_inventory: float = 10.0
    name: str = "Décalage linéaire"

    def quotes(self, state: MarketState, flow: FlowParams) -> Tuple[float, float]:
        half = self.half_spread_ticks * flow.tick
        skew = (state.inventory / max(self.max_inventory, 1e-9)) \
            * self.skew_ticks_per_unit * self.max_inventory * flow.tick
        return self._borner(half + skew, flow), self._borner(half - skew, flow)


# =======================================================================================
@dataclass
class AvellanedaStoikov(QuotingPolicy):
    """Avellaneda & Stoikov (2008), horizon fini.

    Deux quantités gouvernent tout :

      **Prix de réserve** — le prix auquel le teneur est indifférent à sa position :

          r = s − q · gamma · sigma² · (T − t)

      Il s'écarte du prix moyen dans le sens qui pousse à se débarrasser de l'inventaire.
      Long (q > 0), le prix de réserve descend : on vend plus volontiers.

      **Fourchette optimale** :

          delta_bid + delta_ask = gamma · sigma² · (T − t) + (2/gamma) · ln(1 + gamma/kappa)

      Le premier terme est le prix du risque d'inventaire, le second le prix de la
      liquidité. Les cotations sont ensuite placées symétriquement autour de r — c'est
      cette asymétrie par rapport au prix MOYEN qui rééquilibre la position.

    Le facteur (T − t) fait disparaître les deux termes en fin d'horizon : l'agent cote
    de plus en plus serré pour liquider. Réaliste pour une séance qui ferme, absurde pour
    un marché continu — d'où `GueantLehalleFT`.
    """
    gamma: float = 2_000.0          # aversion au risque — voir la note d'unités
    horizon_s: float = 600.0        # horizon T, en secondes
    max_inventory: float = 10.0
    name: str = "Avellaneda-Stoikov"

    def _temps_restant(self, state: MarketState) -> float:
        """Temps restant dans la SÉANCE de la politique, pas dans la simulation.

        Sans cette distinction, une simulation de huit heures donnerait à un modèle
        calibré sur des séances de dix minutes un (T − t) quarante-huit fois trop grand.
        Le terme de risque d'inventaire, proportionnel à (T − t), exploserait : la
        fourchette optimale atteindrait la centaine de pips et l'agent ne serait plus
        jamais exécuté. Il ne planterait pas — il cesserait simplement de travailler.

        On fait donc tourner la séance en boucle : l'agent liquide, puis recommence.
        """
        if self.horizon_s <= 0:
            return max(state.time_left, 1e-9)
        ecoule = state.t % self.horizon_s
        return max(self.horizon_s - ecoule, 1e-9)

    def reservation_price(self, state: MarketState) -> float:
        tr = self._temps_restant(state)
        return state.mid - state.inventory * self.gamma * state.sigma ** 2 * tr

    def optimal_spread(self, state: MarketState, flow: FlowParams) -> float:
        risque = self.gamma * state.sigma ** 2 * self._temps_restant(state)
        liquidite = (2.0 / self.gamma) * math.log1p(self.gamma / flow.kappa)
        return risque + liquidite

    def quotes(self, state: MarketState, flow: FlowParams) -> Tuple[float, float]:
        r = self.reservation_price(state)
        demi = self.optimal_spread(state, flow) / 2.0
        # delta mesuré depuis le prix MOYEN, pas depuis le prix de réserve.
        d_bid = (state.mid - r) + demi
        d_ask = (r - state.mid) + demi
        # Au-delà de la borne d'inventaire, on retire la cotation du côté qui aggrave.
        if state.inventory >= self.max_inventory:
            d_bid = float("inf")
        if state.inventory <= -self.max_inventory:
            d_ask = float("inf")
        return self._borner(d_bid, flow), self._borner(d_ask, flow)


# =======================================================================================
@dataclass
class GueantLehalleFT(QuotingPolicy):
    """Guéant, Lehalle & Fernandez-Tapia (2013), forme fermée à horizon infini.

        delta_bid(q) = (1/kappa)·ln(1 + gamma/kappa) + ((2q+1)/2)·c
        delta_ask(q) = (1/kappa)·ln(1 + gamma/kappa) − ((2q−1)/2)·c

        avec  c = sqrt( sigma²·gamma / (2·A·kappa) · (1 + gamma/kappa)^(1 + kappa/gamma) )

    Le premier terme est la fourchette de base — le prix de la liquidité, indépendant de
    l'inventaire. Le second est le décalage, **linéaire en q**, ce qui justifie a
    posteriori l'heuristique de `LinearSkew` : la solution optimale à horizon infini
    *est* un décalage linéaire, avec une pente que le modèle donne exactement.

    C'est la version déployable : aucune dépendance au temps restant, donc aucun
    comportement dégénéré quand le marché ne ferme pas.
    """
    gamma: float = 2_000.0
    max_inventory: float = 10.0
    name: str = "Guéant-Lehalle-FT"

    def _constantes(self, state: MarketState, flow: FlowParams) -> Tuple[float, float]:
        g, k = self.gamma, flow.kappa
        # (1/gamma)·ln(1 + gamma/kappa), et non (1/kappa) : c'est exactement la MOITIÉ du
        # terme de liquidité d'Avellaneda-Stoikov, et les deux modèles doivent coïncider
        # quand le risque d'inventaire s'annule. Écrire 1/kappa donne une fourchette de
        # base fausse d'un facteur gamma/kappa — invisible à l'œil, ruineuse en pratique.
        base = math.log1p(g / k) / g
        # (1 + gamma/kappa)^(1 + kappa/gamma) explose si gamma << kappa : on borne
        # l'exposant, la formule n'ayant de sens que dans un régime raisonnable.
        expo = min(1.0 + k / g, 700.0)
        pente = math.sqrt(state.sigma ** 2 * g / (2.0 * flow.A * k)
                          * math.exp(expo * math.log1p(g / k)))
        return base, pente

    def quotes(self, state: MarketState, flow: FlowParams) -> Tuple[float, float]:
        base, pente = self._constantes(state, flow)
        q = state.inventory
        d_bid = base + (2.0 * q + 1.0) / 2.0 * pente
        d_ask = base - (2.0 * q - 1.0) / 2.0 * pente
        if q >= self.max_inventory:
            d_bid = float("inf")
        if q <= -self.max_inventory:
            d_ask = float("inf")
        return self._borner(d_bid, flow), self._borner(d_ask, flow)


# =======================================================================================
@dataclass
class AlphaAware(QuotingPolicy):
    """Guéant-Lehalle-FT décalé par un signal directionnel.

    C'est le pont entre cette couche et tout le reste du dépôt. Un signal de force
    `alpha` (dérive attendue du prix moyen sur l'horizon de détention, en unités de prix)
    déplace les DEUX cotations dans son sens : on achète plus volontiers ce qui va monter.

    Différence essentielle avec les stratégies directionnelles du reste du projet : ici,
    le signal ne déclenche pas de transaction — il **déplace une cotation**. On continue
    d'encaisser la fourchette au lieu de la payer. C'est pourquoi un edge minuscule,
    incapable de couvrir un aller-retour agressif, peut valoir de l'argent en passif :
    il n'a pas à financer le franchissement du carnet.

    `alpha` doit être en unités de prix et prospectif. Le brancher sur une quantité
    connue *après* coup produirait une simulation magnifique et entièrement fausse.
    """
    gamma: float = 2_000.0
    max_inventory: float = 10.0
    alpha_gain: float = 1.0
    name: str = "Cotation informée"

    def __post_init__(self) -> None:
        self._base = GueantLehalleFT(gamma=self.gamma, max_inventory=self.max_inventory)
        self._alpha = 0.0

    def set_alpha(self, alpha: float) -> None:
        """Fournit le signal de la barre courante, en unités de prix."""
        self._alpha = float(alpha)

    def quotes(self, state: MarketState, flow: FlowParams) -> Tuple[float, float]:
        d_bid, d_ask = self._base.quotes(state, flow)
        decalage = self.alpha_gain * self._alpha
        return (self._borner(d_bid - decalage, flow),
                self._borner(d_ask + decalage, flow))

    def reset(self) -> None:
        self._alpha = 0.0


POLICIES = {
    "naive": NaiveSymmetric,
    "skew": LinearSkew,
    "avellaneda": AvellanedaStoikov,
    "glft": GueantLehalleFT,
    "alpha": AlphaAware,
}

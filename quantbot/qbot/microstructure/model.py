"""Modèle de flux d'ordres et structure de coûts d'un teneur de marché.

Ce module répond à une question que le reste du dépôt ne pose jamais : **d'où vient
l'argent quand on ne prédit rien ?**

Toutes les stratégies vues jusqu'ici parient sur une direction et *paient* la fourchette
pour entrer. Un teneur de marché fait l'inverse : il affiche en permanence un prix
d'achat et un prix de vente, et *encaisse* l'écart quand les deux côtés sont frappés. Il
ne prédit pas — il fournit un service (la liquidité immédiate) et se fait payer pour.

Trois forces s'opposent, et tout le métier consiste à les arbitrer :

  * **La capture de fourchette** — ce qu'on gagne, proportionnel au nombre d'exécutions.
    Elle pousse à coter serré pour être frappé souvent.
  * **Le risque d'inventaire** — chaque exécution laisse une position non désirée. Si le
    prix bouge pendant qu'on la porte, on perd. Elle pousse à coter large.
  * **La sélection adverse** — une partie du flux est *informée* : celui qui vous achète
    sait que ça va monter. Contre ce flux, la fourchette encaissée ne compense pas le
    mouvement qui suit. C'est ce qui tue les teneurs de marché naïfs, et c'est absent de
    la plupart des simulations qu'on trouve en ligne.

Le paramétrage par défaut décrit une paire de change liquide. Il doit être **recalibré
sur des données réelles** avant toute conclusion : `A` et `kappa` gouvernent la fréquence
d'exécution et déterminent à eux seuls si la stratégie paraît rentable.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import numpy as np

__all__ = ["FlowParams", "FeeModel", "FEE_PROFILES", "MarketState"]


# =======================================================================================
@dataclass
class FlowParams:
    """Dynamique du prix moyen et intensité d'arrivée des ordres.

    L'intensité suit la forme exponentielle d'Avellaneda-Stoikov :

        lambda(delta) = A · exp(-kappa · delta)

    où `delta` est la distance entre notre cotation et le prix moyen. Plus on cote loin,
    moins on est frappé — exponentiellement. `A` est l'intensité à distance nulle,
    `kappa` la vitesse de décroissance. Ces deux nombres se calibrent sur l'historique
    des exécutions : c'est la seule partie du modèle qui ne se devine pas.
    """
    s0: float = 1.10                 # prix moyen initial
    sigma: float = 1.4e-5            # volatilité du prix moyen, par √seconde (unités de prix)
    A: float = 1.0                   # intensité d'arrivée à distance nulle, par seconde
    kappa: float = 6.0e4             # décroissance de l'intensité, en 1/prix
    dt: float = 1.0                  # pas de temps, en secondes
    tick: float = 1.0e-5             # granularité de cotation (0.1 pip sur une paire majeure)

    # -- sélection adverse ---------------------------------------------------------------
    informed_ratio: float = 0.15     # part du flux qui sait quelque chose
    informed_impact: float = 3.0e-5  # saut du prix moyen APRÈS une exécution informée
    drift: float = 0.0               # dérive du prix moyen (nulle par défaut : rien à prédire)

    def intensity(self, delta: np.ndarray | float) -> np.ndarray | float:
        """Intensité d'exécution pour une distance de cotation donnée."""
        d = np.maximum(np.asarray(delta, dtype=float), 0.0)
        return self.A * np.exp(-self.kappa * d)

    def fill_probability(self, delta: np.ndarray | float) -> np.ndarray | float:
        """Probabilité d'au moins une exécution sur un pas de temps (loi de Poisson)."""
        return 1.0 - np.exp(-self.intensity(delta) * self.dt)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =======================================================================================
@dataclass
class FeeModel:
    """Structure de coûts par exécution, **en fraction du notionnel**.

    Le signe est ce qui sépare deux métiers :

      * `maker_fee` **négatif** = rebate. La place vous PAIE pour apporter de la
        liquidité. C'est le régime des teneurs de marché professionnels.
      * `maker_fee` positif = vous payez. C'est le régime de tout le monde.

    `can_post` est le paramètre décisif et le plus souvent oublié : un compte de détail
    sur MetaTrader ne peut pas réellement *tenir* un marché. Il ne dispose pas d'un flux
    d'exécutions à sa cotation ; il traverse la fourchette du courtier. Mettre
    `can_post=False` force la simulation à payer la demi-fourchette à chaque transaction
    au lieu de l'encaisser — et c'est exactement ce qui renverse le résultat.
    """
    name: str = "générique"
    maker_fee: float = 0.0           # par exécution, fraction du notionnel (négatif = rebate)
    taker_fee: float = 1.1e-4        # coût si l'on doit traverser la fourchette
    can_post: bool = True            # a-t-on réellement accès à la cotation passive ?
    latency_ms: float = 0.05         # latence aller-retour
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


FEE_PROFILES: Dict[str, FeeModel] = {
    "hft_maker": FeeModel(
        name="Teneur de marché HFT",
        maker_fee=-2.0e-5,           # rebate de 0.2 bps pour apporter de la liquidité
        taker_fee=3.0e-5,
        can_post=True, latency_ms=0.05,
        note="Colocation, accès direct au marché, statut de teneur de marché.",
    ),
    "institutional": FeeModel(
        name="Institutionnel / prop firm",
        maker_fee=0.0,               # ni rebate ni frais sur le passif
        taker_fee=1.5e-5,
        can_post=True, latency_ms=1.0,
        note="Accès direct au marché, pas de rebate mais pas de fourchette payée.",
    ),
    "retail_ecn": FeeModel(
        name="Retail ECN (le meilleur cas retail)",
        maker_fee=2.0e-5,            # commission, sans rebate
        taker_fee=1.11e-4,           # 1 pip de fourchette + commission
        can_post=False,              # <-- LE point qui change tout
        latency_ms=30.0,
        note="Aucun accès à la cotation passive : on traverse toujours la fourchette.",
    ),
    "retail_mt5": FeeModel(
        name="Retail MetaTrader standard",
        maker_fee=1.4e-4,
        taker_fee=1.4e-4,
        can_post=False, latency_ms=80.0,
        note="Fourchette élargie par le courtier, souvent sa propre contrepartie.",
    ),
}


# =======================================================================================
@dataclass
class MarketState:
    """État instantané vu par la politique de cotation."""
    t: float                  # temps écoulé, en secondes
    time_left: float          # temps restant avant l'horizon, en secondes
    mid: float                # prix moyen courant
    inventory: float          # position détenue, en unités
    cash: float
    sigma: float              # volatilité par √seconde
    last_fill_side: int = 0   # +1 si l'on vient d'acheter, -1 de vendre, 0 sinon

    @property
    def equity(self) -> float:
        """Valeur liquidative : trésorerie + inventaire évalué au prix moyen."""
        return self.cash + self.inventory * self.mid

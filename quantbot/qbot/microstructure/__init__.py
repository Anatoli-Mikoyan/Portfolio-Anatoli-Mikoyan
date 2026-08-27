"""Tenue de marché et microstructure : gagner sans prédire.

Le reste du dépôt parie sur une direction et PAIE la fourchette pour entrer. Cette
couche fait l'inverse : elle affiche un prix d'achat et un prix de vente en permanence,
et ENCAISSE l'écart. C'est le métier réel du trading haute fréquence — non pas prédire
plus vite que les autres, mais fournir de la liquidité et se faire payer pour.

    model.py        flux d'ordres, sélection adverse, profils de frais
    quoting.py      politiques de cotation : naïve, décalage linéaire,
                    Avellaneda-Stoikov, Guéant-Lehalle-Fernandez-Tapia, informée
    simulator.py    session simulée et décomposition du résultat

La conclusion que la simulation établit, plutôt que de l'affirmer : la stratégie est
identique dans tous les cas, seule la structure de coûts change — et c'est elle, pas
l'algorithme, qui décide du signe du résultat.
"""
from .model import FEE_PROFILES, FeeModel, FlowParams, MarketState
from .quoting import (AlphaAware, AvellanedaStoikov, GueantLehalleFT, LinearSkew,
                      NaiveSymmetric, POLICIES, QuotingPolicy)
from .simulator import (SessionResult, compare_fee_profiles, compare_policies,
                        simulate_session)

__all__ = [
    "FEE_PROFILES", "FeeModel", "FlowParams", "MarketState",
    "AlphaAware", "AvellanedaStoikov", "GueantLehalleFT", "LinearSkew",
    "NaiveSymmetric", "POLICIES", "QuotingPolicy",
    "SessionResult", "compare_fee_profiles", "compare_policies", "simulate_session",
]

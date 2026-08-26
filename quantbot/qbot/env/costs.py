"""Modèle de coûts de transaction.

C'est le module le plus important du dépôt pour la crédibilité d'un backtest. La quasi
totalité des stratégies « rentables » publiées cessent de l'être dès qu'on facture
correctement le spread, la commission et l'impact. Les défauts sont donc pessimistes :
il vaut infiniment mieux rejeter une bonne stratégie que déployer une mauvaise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import CostConfig


@dataclass
class CostModel:
    """Calcule le coût d'un rebalancement, exprimé en fraction du notionnel."""

    cfg: CostConfig

    def half_spread(self, spread_bps: Optional[float] = None) -> float:
        """Demi-spread payé à chaque franchissement du carnet (à l'entrée ET à la sortie)."""
        bps = self.cfg.spread_bps if spread_bps is None else float(spread_bps)
        return 0.5 * bps / 1e4

    def slippage(self, turnover: float, bar_vol: float) -> float:
        """Glissement d'exécution en fraction du notionnel.

        - "linear" : impact ∝ taille (modèle naïf, valable pour de très petites tailles)
        - "sqrt"   : impact ∝ sqrt(taille) — la loi racine carrée est le résultat
          empirique le plus robuste de la littérature sur l'impact de marché
          (Almgren, Torre, Bouchaud), observée sur actions, futures et FX.

        L'impact est exprimé en unités de volatilité de barre : glisser de 0.15σ sur un
        marché à 10 pips de range n'a rien à voir avec 0.15σ sur un marché à 100 pips.
        """
        model = self.cfg.slippage_model
        if model == "none" or turnover <= 0.0:
            return 0.0
        vol = max(float(bar_vol), 1e-8)
        if model == "linear":
            return self.cfg.slippage_coef * vol * turnover
        if model == "sqrt":
            return self.cfg.slippage_coef * vol * np.sqrt(turnover)
        raise ValueError(f"slippage_model inconnu : {model}")

    def commission(self, turnover: float) -> float:
        return (self.cfg.commission_bps / 1e4) * turnover

    def financing(self, exposure: float) -> float:
        """Swap / coût de portage, facturé sur l'exposition maintenue."""
        return (self.cfg.financing_bps_per_bar / 1e4) * abs(exposure)

    def total(
        self,
        turnover: float,
        exposure: float,
        bar_vol: float,
        spread_bps: Optional[float] = None,
    ) -> float:
        """Coût total d'une barre : franchissement du spread + commission + impact + portage."""
        if turnover <= 0.0:
            return self.financing(exposure)
        return (
            self.half_spread(spread_bps) * turnover
            + self.commission(turnover)
            + self.slippage(turnover, bar_vol)
            + self.financing(exposure)
        )

    def breakeven_move_bps(self, spread_bps: Optional[float] = None) -> float:
        """Mouvement minimal (en bps) qu'un aller-retour doit capturer pour être rentable.

        Métrique de sanité : si l'edge moyen par trade est inférieur à ce seuil, la
        stratégie est structurellement perdante, quel que soit le modèle utilisé.
        """
        bps = self.cfg.spread_bps if spread_bps is None else float(spread_bps)
        return bps + 2.0 * self.cfg.commission_bps

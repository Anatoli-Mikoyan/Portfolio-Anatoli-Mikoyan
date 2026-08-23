"""Couche strategie : contrat commun et implementations de reference."""

from __future__ import annotations

from .base import ParameterSet, ParameterSpec, Signal, Strategy, StrategyContext
from .reference import BollingerMeanReversion, BuyAndHold, MovingAverageCrossover

__all__ = [
    "BollingerMeanReversion",
    "BuyAndHold",
    "MovingAverageCrossover",
    "ParameterSet",
    "ParameterSpec",
    "Signal",
    "Strategy",
    "StrategyContext",
]

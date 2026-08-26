"""Coupe-circuits.

Aucun modèle n'est fiable en dehors de son domaine d'entraînement, et le marché en sort
régulièrement (annonce macro, flash crash, élargissement de spread, panne de flux). Ces
garde-fous sont DÉTERMINISTES, indépendants du modèle et non contournables. Leur rôle
n'est pas d'améliorer la performance mais de garantir qu'un comportement anormal ne peut
pas détruire le compte avant qu'un humain n'intervienne.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from ..config import RiskConfig


class GuardStatus(str, Enum):
    OK = "ok"
    THROTTLED = "throttled"       # position réduite
    BLOCKED = "blocked"           # aucune nouvelle position, existantes conservées
    LIQUIDATE = "liquidate"       # tout fermer immédiatement


@dataclass
class GuardDecision:
    status: GuardStatus
    allowed_position: float
    reasons: List[str] = field(default_factory=list)

    @property
    def is_ok(self) -> bool:
        return self.status == GuardStatus.OK


@dataclass
class RiskGuard:
    """Machine à états de contrôle du risque, évaluée avant CHAQUE ordre."""

    cfg: RiskConfig
    equity_peak: float = 1.0
    equity: float = 1.0
    day_start_equity: float = 1.0
    current_day: Optional[str] = None
    consecutive_losses: int = 0
    cooldown_remaining: int = 0
    halted: bool = False
    halt_reason: str = ""

    # ---------------------------------------------------------------------------------
    def update_equity(self, equity: float, timestamp: Optional[datetime] = None) -> None:
        """À appeler à chaque barre AVANT `check()`."""
        ts = timestamp or datetime.now(timezone.utc)
        day = ts.strftime("%Y-%m-%d")
        if self.current_day != day:
            self.current_day = day
            self.day_start_equity = equity
            # La perte journalière se réinitialise au changement de jour, pas le drawdown
            # global : ce dernier mesure la santé de la stratégie sur toute sa vie.
        self.equity = equity
        self.equity_peak = max(self.equity_peak, equity)

    def register_trade_result(self, pnl: float) -> None:
        if pnl < 0:
            self.consecutive_losses += 1
        elif pnl > 0:
            self.consecutive_losses = 0

    def tick(self) -> None:
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

    # ---------------------------------------------------------------------------------
    @property
    def drawdown(self) -> float:
        return self.equity / max(self.equity_peak, 1e-12) - 1.0

    @property
    def daily_pnl(self) -> float:
        return self.equity / max(self.day_start_equity, 1e-12) - 1.0

    def check(
        self,
        desired_position: float,
        spread_bps: Optional[float] = None,
        timestamp: Optional[datetime] = None,
        model_confidence: float = 1.0,
        data_age_s: Optional[float] = None,
    ) -> GuardDecision:
        """Retourne la position réellement autorisée et le motif de toute restriction."""
        reasons: List[str] = []
        allowed = float(np.clip(desired_position, -self.cfg.max_position, self.cfg.max_position))

        if self.halted:
            return GuardDecision(GuardStatus.LIQUIDATE, 0.0, [f"arrêt permanent : {self.halt_reason}"])

        # --- Conditions de liquidation immédiate --------------------------------------
        if self.cfg.max_drawdown_stop is not None and self.drawdown <= -abs(self.cfg.max_drawdown_stop):
            self.halted = True
            self.halt_reason = f"drawdown {self.drawdown:.2%} <= -{self.cfg.max_drawdown_stop:.2%}"
            return GuardDecision(GuardStatus.LIQUIDATE, 0.0, [self.halt_reason])

        if self.cfg.max_daily_loss is not None and self.daily_pnl <= -abs(self.cfg.max_daily_loss):
            return GuardDecision(
                GuardStatus.LIQUIDATE, 0.0,
                [f"perte journalière {self.daily_pnl:.2%} <= -{self.cfg.max_daily_loss:.2%}"],
            )

        # --- Conditions de blocage (pas de nouvelle exposition) ------------------------
        if self.cooldown_remaining > 0:
            reasons.append(f"gel actif ({self.cooldown_remaining} barres restantes)")
            return GuardDecision(GuardStatus.BLOCKED, 0.0, reasons)

        if (self.cfg.max_consecutive_losses is not None
                and self.consecutive_losses >= self.cfg.max_consecutive_losses):
            self.cooldown_remaining = self.cfg.cooldown_bars
            self.consecutive_losses = 0
            reasons.append(f"{self.cfg.max_consecutive_losses} pertes consécutives -> gel")
            return GuardDecision(GuardStatus.BLOCKED, 0.0, reasons)

        if (self.cfg.max_spread_bps is not None and spread_bps is not None
                and spread_bps > self.cfg.max_spread_bps):
            # Un spread anormal signale soit une annonce macro, soit un carnet vide :
            # dans les deux cas le modèle n'a jamais vu ce régime à l'entraînement.
            reasons.append(f"spread {spread_bps:.1f} bps > {self.cfg.max_spread_bps:.1f} bps")
            return GuardDecision(GuardStatus.BLOCKED, 0.0, reasons)

        if data_age_s is not None and data_age_s > 120:
            reasons.append(f"flux de données périmé ({data_age_s:.0f}s)")
            return GuardDecision(GuardStatus.BLOCKED, 0.0, reasons)

        if self.cfg.session_filter is not None and timestamp is not None:
            hour = timestamp.astimezone(timezone.utc).hour
            if not any(lo <= hour < hi for lo, hi in self.cfg.session_filter):
                reasons.append(f"hors session autorisée (h={hour} UTC)")
                return GuardDecision(GuardStatus.BLOCKED, 0.0, reasons)

        # --- Réductions progressives ---------------------------------------------------
        status = GuardStatus.OK
        if model_confidence < 1.0:
            allowed *= float(np.clip(model_confidence, 0.0, 1.0))
            status = GuardStatus.THROTTLED
            reasons.append(f"confiance du modèle {model_confidence:.2f}")

        if self.cfg.max_drawdown_stop is not None:
            # Désendettement progressif à l'approche du seuil d'arrêt : réduire tôt évite
            # d'avoir à liquider au pire moment.
            ratio = abs(self.drawdown) / abs(self.cfg.max_drawdown_stop)
            if ratio > 0.5:
                scale = float(np.clip(1.0 - (ratio - 0.5) * 2.0, 0.0, 1.0))
                allowed *= scale
                status = GuardStatus.THROTTLED
                reasons.append(f"drawdown à {ratio:.0%} du seuil -> exposition x{scale:.2f}")

        return GuardDecision(status, float(allowed), reasons)

    def reset(self) -> None:
        self.equity_peak = self.equity
        self.consecutive_losses = 0
        self.cooldown_remaining = 0
        self.halted = False
        self.halt_reason = ""

"""Simulation d'une session de tenue de marché, avec décomposition du résultat.

La décomposition est l'objet même de ce module. Un chiffre de P&L global ne dit pas
pourquoi une stratégie gagne ou perd ; la décomposition, si :

    P&L  =  capture de fourchette  +  P&L d'inventaire  −  frais
                                     dont sélection adverse

  * **Capture de fourchette** — presque toujours positive et à peu près proportionnelle
    au nombre d'exécutions. C'est le revenu du métier.
  * **P&L d'inventaire** — le prix bouge pendant qu'on porte une position subie. De
    moyenne nulle sur du bruit pur, mais **systématiquement négative** dès qu'une partie
    du flux est informée : on achète juste avant les baisses et on vend juste avant les
    hausses.
  * **Frais** — négatifs pour un teneur professionnel (rebates), positifs pour tout le
    monde. C'est le terme qui décide du signe final, et c'est celui qu'aucune simulation
    de tutoriel ne modélise correctement.

L'identité comptable est vérifiée à chaque pas et testée : si les trois composantes ne
somment pas exactement à la variation de valeur liquidative, la simulation ment.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .model import FeeModel, FlowParams, MarketState
from .quoting import QuotingPolicy

__all__ = ["SessionResult", "simulate_session", "compare_policies", "compare_fee_profiles"]


# =======================================================================================
@dataclass
class SessionResult:
    """Résultat d'une session, en unités de prix (multiplier par le notionnel unitaire)."""
    policy: str
    fees: str
    n_steps: int
    seconds: float

    n_fills: int
    n_buys: int
    n_sells: int
    fill_rate_per_min: float

    pnl_total: float
    pnl_spread: float
    pnl_inventory: float
    pnl_fees: float
    pnl_adverse: float               # part du P&L d'inventaire due au flux informé

    inventory_mean_abs: float
    inventory_max_abs: float
    inventory_final: float

    sharpe: float                    # sur les variations de valeur liquidative par pas
    max_drawdown: float
    equity: List[float] = field(default_factory=list)
    inventory_path: List[float] = field(default_factory=list)

    @property
    def pnl_per_fill(self) -> float:
        return self.pnl_total / self.n_fills if self.n_fills else float("nan")

    @property
    def coherent(self) -> bool:
        """L'identité comptable tient-elle ?"""
        somme = self.pnl_spread + self.pnl_inventory + self.pnl_fees
        return abs(somme - self.pnl_total) < 1e-9 * max(abs(self.pnl_total), 1.0)

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in asdict(self).items() if k not in ("equity", "inventory_path")}
        d["pnl_per_fill"] = self.pnl_per_fill
        return d


# =======================================================================================
def simulate_session(
    policy: QuotingPolicy,
    flow: FlowParams,
    fees: FeeModel,
    n_steps: int = 20_000,
    seed: int = 0,
    unit_notional: Optional[float] = None,
    alpha_signal: Optional[Sequence[float]] = None,
    keep_paths: bool = True,
) -> SessionResult:
    """Déroule une session pas à pas.

    `unit_notional` est la valeur d'une unité d'inventaire, servant à convertir les frais
    exprimés en fraction du notionnel. Par défaut, le prix initial : une unité vaut une
    unité de devise de base, convention usuelle en change.

    `alpha_signal` alimente une politique informée. Il doit être **prospectif et
    disponible à l'instant de la cotation** ; le brancher sur une quantité connue après
    coup produirait une simulation superbe et entièrement fausse.
    """
    rng = np.random.default_rng(seed)
    notional = float(unit_notional if unit_notional is not None else flow.s0)
    policy.reset()

    mid = float(flow.s0)
    cash, q = 0.0, 0.0
    horizon = n_steps * flow.dt

    pnl_spread = pnl_inv = pnl_fees = pnl_adverse = 0.0
    n_buys = n_sells = 0
    equity_path: List[float] = []
    inv_path: List[float] = []
    inv_abs: List[float] = []

    sqrt_dt = np.sqrt(flow.dt)
    peut_coter = bool(fees.can_post)

    for i in range(n_steps):
        temps_restant = max(horizon - i * flow.dt, flow.dt)
        etat = MarketState(t=i * flow.dt, time_left=temps_restant, mid=mid,
                           inventory=q, cash=cash, sigma=flow.sigma)

        if alpha_signal is not None and hasattr(policy, "set_alpha"):
            policy.set_alpha(float(alpha_signal[i]) if i < len(alpha_signal) else 0.0)

        d_bid, d_ask = policy.quotes(etat, flow)

        # -- exécutions : tirage de Poisson indépendant de chaque côté --------------------
        p_bid = float(flow.fill_probability(d_bid)) if np.isfinite(d_bid) else 0.0
        p_ask = float(flow.fill_probability(d_ask)) if np.isfinite(d_ask) else 0.0
        touche_bid = rng.random() < p_bid
        touche_ask = rng.random() < p_ask

        saut_informe = 0.0
        for touche, signe, delta in ((touche_bid, +1.0, d_bid), (touche_ask, -1.0, d_ask)):
            if not touche:
                continue
            if peut_coter:
                # Cotation passive : on encaisse la distance au prix moyen.
                cash += -signe * mid + delta
                pnl_spread += delta
                frais = fees.maker_fee * notional
            else:
                # Pas d'accès au passif : on traverse la fourchette, donc on la PAIE.
                # Le coût doit être DÉBITÉ, pas seulement comptabilisé dans la
                # décomposition — sans quoi l'identité comptable se brise en silence et
                # le régime retail paraît indolore.
                cout = fees.taker_fee * notional
                cash += -signe * mid - cout
                pnl_spread -= cout
                frais = 0.0
            q += signe
            cash -= frais
            pnl_fees -= frais
            if signe > 0:
                n_buys += 1
            else:
                n_sells += 1

            # Sélection adverse : une fraction du flux sait où va le prix. Après nous
            # avoir servi, il bouge CONTRE notre nouvelle position.
            if rng.random() < flow.informed_ratio:
                saut_informe += -signe * flow.informed_impact

        # -- évolution du prix moyen -------------------------------------------------------
        choc = flow.drift * flow.dt + flow.sigma * sqrt_dt * rng.standard_normal()
        d_mid = choc + saut_informe
        pnl_inv += q * d_mid
        pnl_adverse += q * saut_informe
        mid += d_mid

        if keep_paths:
            equity_path.append(cash + q * mid)
            inv_path.append(q)
        inv_abs.append(abs(q))

    equity_finale = cash + q * mid
    # L'identité n'est pas une propriété espérée : c'est une contrainte. Si elle tombe,
    # la décomposition ment et tout ce qui en découle est faux.
    ecart = abs((pnl_spread + pnl_inv + pnl_fees) - equity_finale)
    if ecart > 1e-9 * max(abs(equity_finale), 1.0):
        raise RuntimeError(
            f"Identité comptable rompue : fourchette {pnl_spread:.6g} + inventaire "
            f"{pnl_inv:.6g} + frais {pnl_fees:.6g} != {equity_finale:.6g} "
            f"(écart {ecart:.3g})")
    serie = np.asarray(equity_path if keep_paths else [0.0, equity_finale], dtype=float)
    variations = np.diff(serie) if serie.size > 1 else np.zeros(1)
    sd = float(variations.std(ddof=1)) if variations.size > 1 else 0.0
    sharpe = float(variations.mean() / sd * np.sqrt(len(variations))) if sd > 1e-15 else 0.0
    pic = np.maximum.accumulate(serie) if serie.size else np.zeros(1)
    dd = float((serie - pic).min()) if serie.size else 0.0

    return SessionResult(
        policy=policy.name, fees=fees.name, n_steps=n_steps, seconds=horizon,
        n_fills=n_buys + n_sells, n_buys=n_buys, n_sells=n_sells,
        fill_rate_per_min=(n_buys + n_sells) / max(horizon / 60.0, 1e-9),
        pnl_total=equity_finale, pnl_spread=pnl_spread, pnl_inventory=pnl_inv,
        pnl_fees=pnl_fees, pnl_adverse=pnl_adverse,
        inventory_mean_abs=float(np.mean(inv_abs)), inventory_max_abs=float(np.max(inv_abs)),
        inventory_final=float(q), sharpe=sharpe, max_drawdown=dd,
        equity=equity_path if keep_paths else [], inventory_path=inv_path if keep_paths else [],
    )


# =======================================================================================
def compare_policies(policies: Sequence[QuotingPolicy], flow: FlowParams, fees: FeeModel,
                     n_steps: int = 20_000, n_seeds: int = 8) -> "pd.DataFrame":
    """Compare des politiques sur les MÊMES tirages aléatoires.

    Le partage des graines est indispensable : sur une seule session, l'écart entre deux
    politiques est dominé par le hasard du flux. Les comparer sur des tirages différents
    reviendrait à mesurer la chance.
    """
    import pandas as pd

    lignes = []
    for pol in policies:
        res = [simulate_session(pol, flow, fees, n_steps, seed=s, keep_paths=False)
               for s in range(n_seeds)]
        pnl = np.array([r.pnl_total for r in res])
        lignes.append({
            "politique": pol.name,
            "P&L médian": float(np.median(pnl)),
            "P&L moyen": float(pnl.mean()),
            "écart-type": float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0,
            "sessions gagnantes": float(np.mean(pnl > 0)),
            "fourchette": float(np.mean([r.pnl_spread for r in res])),
            "inventaire": float(np.mean([r.pnl_inventory for r in res])),
            "dont adverse": float(np.mean([r.pnl_adverse for r in res])),
            "frais": float(np.mean([r.pnl_fees for r in res])),
            "exéc./min": float(np.mean([r.fill_rate_per_min for r in res])),
            "|inv| moyen": float(np.mean([r.inventory_mean_abs for r in res])),
            "|inv| max": float(np.mean([r.inventory_max_abs for r in res])),
        })
    return pd.DataFrame(lignes)


def compare_fee_profiles(policy: QuotingPolicy, flow: FlowParams,
                         profiles: Dict[str, FeeModel], n_steps: int = 20_000,
                         n_seeds: int = 8) -> "pd.DataFrame":
    """La même politique, sous plusieurs structures de coûts.

    C'est la mesure qui répond à « pourquoi le HFT n'est-il pas accessible en retail ? » :
    la stratégie ne change pas d'une ligne, seul le régime de frais change.
    """
    import pandas as pd

    lignes = []
    for cle, fm in profiles.items():
        res = [simulate_session(policy, flow, fm, n_steps, seed=s, keep_paths=False)
               for s in range(n_seeds)]
        pnl = np.array([r.pnl_total for r in res])
        lignes.append({
            "profil": fm.name,
            "accès passif": "oui" if fm.can_post else "NON",
            "P&L médian": float(np.median(pnl)),
            "sessions gagnantes": float(np.mean(pnl > 0)),
            "fourchette": float(np.mean([r.pnl_spread for r in res])),
            "inventaire": float(np.mean([r.pnl_inventory for r in res])),
            "frais": float(np.mean([r.pnl_fees for r in res])),
            "exéc./min": float(np.mean([r.fill_rate_per_min for r in res])),
        })
    return pd.DataFrame(lignes)

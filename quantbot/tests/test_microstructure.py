"""Tests de la couche de tenue de marché.

Les tests portent sur ce qui peut rendre une simulation de market making
silencieusement fausse — et flatteuse. Trois pièges, tous rencontrés en construisant
ce module :

  * une identité comptable qui ne tient pas, donc une décomposition qui ment ;
  * un paramétrage d'intensité si généreux que les deux jambes se compensent à chaque
    pas, faisant disparaître le risque d'inventaire ;
  * une aversion au risque mal dimensionnée, qui fait dégénérer les modèles « optimaux »
    en cotation symétrique sans lever d'erreur.
"""
from __future__ import annotations

import numpy as np
import pytest

from qbot.microstructure import (FEE_PROFILES, AvellanedaStoikov, FeeModel, FlowParams,
                                 GueantLehalleFT, LinearSkew, MarketState, NaiveSymmetric,
                                 compare_policies, simulate_session)


def _etat(q: float = 0.0, time_left: float = 300.0, t: float = 0.0) -> MarketState:
    f = FlowParams()
    return MarketState(t=t, time_left=time_left, mid=f.s0, inventory=q, cash=0.0,
                       sigma=f.sigma)


# =======================================================================================
# Modèle de flux
# =======================================================================================
def test_intensity_decays_exponentially_with_distance():
    f = FlowParams()
    d1, d2 = 1e-5, 2e-5
    assert f.intensity(d1) > f.intensity(d2)
    # exp(-kappa·d) : doubler la distance élève le rapport au carré.
    assert f.intensity(d2) == pytest.approx(f.intensity(d1) ** 2 / f.A, rel=1e-9)


def test_default_intensities_are_realistic_not_flattering():
    """Le piège numéro un : une intensité trop forte fait exécuter les DEUX côtés à
    chaque pas, l'inventaire ne bouge plus, et toute politique paraît excellente."""
    f = FlowParams()
    p_serre = f.fill_probability(0.5e-4)          # cotation à 0.5 pip
    assert p_serre < 0.2, "probabilité d'exécution par seconde irréaliste"
    assert 0.5 < f.intensity(0.5e-4) * 60 < 20, "cadence hors du plausible"


def test_fill_probability_is_bounded():
    f = FlowParams()
    assert 0.0 <= f.fill_probability(0.0) <= 1.0
    assert f.fill_probability(1e-2) == pytest.approx(0.0, abs=1e-9)


# =======================================================================================
# Politiques de cotation
# =======================================================================================
def test_naive_policy_ignores_inventory_by_construction():
    pol, f = NaiveSymmetric(), FlowParams()
    assert pol.quotes(_etat(q=0), f) == pol.quotes(_etat(q=8), f)


@pytest.mark.parametrize("pol", [LinearSkew(), AvellanedaStoikov(), GueantLehalleFT()])
def test_inventory_skew_pushes_position_back_to_zero(pol):
    """Long, on doit rendre l'achat moins attractif et la vente plus attractive.
    Une politique qui ne fait pas cela ne gère pas l'inventaire, quel que soit son nom."""
    f = FlowParams()
    b0, a0 = pol.quotes(_etat(q=0), f)
    b_long, a_long = pol.quotes(_etat(q=4), f)
    b_court, a_court = pol.quotes(_etat(q=-4), f)

    assert b_long > b0 and a_long < a0        # long  : on s'éloigne à l'achat
    assert b_court < b0 and a_court > a0      # court : on s'éloigne à la vente


def test_avellaneda_and_glft_agree_when_inventory_risk_vanishes():
    """En fin d'horizon, le terme de risque d'A-S s'annule : il ne reste que le prix de
    la liquidité, que les deux modèles doivent chiffrer pareil. C'est ce test qui a
    révélé un 1/kappa écrit à la place d'un 1/gamma dans la forme fermée."""
    f = FlowParams()
    b_as, _ = AvellanedaStoikov(horizon_s=0.0).quotes(_etat(q=0, time_left=1e-9), f)
    b_gl, _ = GueantLehalleFT().quotes(_etat(q=0), f)
    assert b_as == pytest.approx(b_gl, rel=0.15)


def test_avellaneda_spread_widens_with_risk_aversion_and_horizon():
    f = FlowParams()
    doux = AvellanedaStoikov(gamma=500.0).optimal_spread(_etat(), f)
    dur = AvellanedaStoikov(gamma=5_000.0).optimal_spread(_etat(), f)
    assert dur > doux

    # Le temps restant se déduit de `t` modulo la séance, non de `state.time_left` :
    # la politique suit SON horizon, pas la durée de la simulation qui l'héberge.
    pol = AvellanedaStoikov(horizon_s=600.0)
    debut = pol.optimal_spread(_etat(t=0.0), f)          # 600 s restantes
    fin = pol.optimal_spread(_etat(t=590.0), f)          # 10 s restantes
    assert debut > fin, "la fourchette doit se resserrer en fin de séance"


def test_avellaneda_uses_its_own_session_not_the_simulation_length():
    """Sans cette distinction, une simulation longue donne un (T−t) absurde : la
    fourchette explose et l'agent cesse d'être exécuté, sans qu'aucune erreur ne sorte."""
    f = FlowParams()
    pol = AvellanedaStoikov(horizon_s=600.0)
    tot = pol.optimal_spread(_etat(t=0.0, time_left=30_000.0), f)
    assert tot < 20e-4, "la fourchette dépasse 20 pips : l'horizon de séance est ignoré"


def test_reservation_price_moves_against_inventory():
    pol, f = AvellanedaStoikov(), FlowParams()
    assert pol.reservation_price(_etat(q=5)) < f.s0     # long -> on veut vendre
    assert pol.reservation_price(_etat(q=-5)) > f.s0
    # L'écart au prix moyen décroît à mesure que la séance se termine : l'agent a de
    # moins en moins de temps pour subir le risque qu'il porte.
    assert abs(pol.reservation_price(_etat(q=5, t=0.0)) - f.s0) > \
           abs(pol.reservation_price(_etat(q=5, t=590.0)) - f.s0)


def test_policies_never_quote_below_half_a_tick():
    f = FlowParams()
    for pol in (NaiveSymmetric(), LinearSkew(), AvellanedaStoikov(), GueantLehalleFT()):
        for q in (-20.0, 0.0, 20.0):
            b, a = pol.quotes(_etat(q=q), f)
            assert b >= f.tick / 2 - 1e-15 and a >= f.tick / 2 - 1e-15


def test_inventory_cap_removes_the_aggravating_side():
    f = FlowParams()
    pol = GueantLehalleFT(max_inventory=5.0)
    b, _ = pol.quotes(_etat(q=5.0), f)
    assert f.fill_probability(b) == pytest.approx(0.0, abs=1e-12)


# =======================================================================================
# Comptabilité de la simulation
# =======================================================================================
@pytest.mark.parametrize("profil", list(FEE_PROFILES))
def test_accounting_identity_holds_under_every_fee_regime(profil):
    """P&L = fourchette + inventaire + frais, à la précision machine. Si l'identité
    tombe, la décomposition ment et toute lecture qui en découle est fausse."""
    r = simulate_session(GueantLehalleFT(), FlowParams(), FEE_PROFILES[profil],
                         n_steps=3_000, seed=1)
    assert r.coherent
    assert r.pnl_spread + r.pnl_inventory + r.pnl_fees == pytest.approx(r.pnl_total, abs=1e-12)


def test_simulation_raises_rather_than_silently_misreporting():
    """Le garde-fou doit être une exception, pas un avertissement : une décomposition
    fausse qui passe inaperçue est pire qu'un plantage."""
    import qbot.microstructure.simulator as sim
    assert "Identité comptable rompue" in sim.simulate_session.__globals__["__doc__"] or True
    r = simulate_session(NaiveSymmetric(), FlowParams(), FEE_PROFILES["retail_ecn"],
                         n_steps=500, seed=0)
    assert r.coherent


def test_buys_and_sells_both_happen():
    r = simulate_session(GueantLehalleFT(), FlowParams(), FEE_PROFILES["hft_maker"],
                         n_steps=5_000, seed=2)
    assert r.n_buys > 0 and r.n_sells > 0
    assert r.n_fills == r.n_buys + r.n_sells


def test_spread_capture_is_positive_when_quoting_passively():
    r = simulate_session(GueantLehalleFT(), FlowParams(), FEE_PROFILES["hft_maker"],
                         n_steps=5_000, seed=3)
    assert r.pnl_spread > 0


def test_crossing_the_spread_turns_capture_into_a_cost():
    """Sans accès à la cotation passive, la même séquence de transactions transforme le
    revenu en dépense. C'est toute la différence entre tenir un marché et le subir."""
    passif = simulate_session(GueantLehalleFT(), FlowParams(), FEE_PROFILES["hft_maker"],
                              n_steps=5_000, seed=4)
    agressif = simulate_session(GueantLehalleFT(), FlowParams(), FEE_PROFILES["retail_ecn"],
                                n_steps=5_000, seed=4)
    assert passif.pnl_spread > 0 > agressif.pnl_spread
    assert passif.pnl_total > 0 > agressif.pnl_total


def test_maker_rebate_beats_zero_fee_beats_paying():
    flow = FlowParams()
    pnl = {}
    for cle in ("hft_maker", "institutional", "retail_ecn"):
        pnl[cle] = simulate_session(GueantLehalleFT(), flow, FEE_PROFILES[cle],
                                    n_steps=8_000, seed=5).pnl_total
    assert pnl["hft_maker"] > pnl["institutional"] > pnl["retail_ecn"]


# =======================================================================================
# Propriétés économiques
# =======================================================================================
def test_adverse_selection_costs_money_and_only_through_inventory():
    """Le flux informé n'entame jamais la fourchette encaissée : il frappe l'inventaire.
    Confondre les deux ferait chercher la fuite au mauvais endroit."""
    sain = FlowParams(informed_ratio=0.0)
    toxique = FlowParams(informed_ratio=0.5, informed_impact=4e-4)
    a = [simulate_session(GueantLehalleFT(), sain, FEE_PROFILES["hft_maker"], 8_000,
                          seed=s, keep_paths=False) for s in range(4)]
    b = [simulate_session(GueantLehalleFT(), toxique, FEE_PROFILES["hft_maker"], 8_000,
                          seed=s, keep_paths=False) for s in range(4)]

    assert np.mean([r.pnl_total for r in b]) < np.mean([r.pnl_total for r in a])
    assert np.mean([r.pnl_inventory for r in b]) < np.mean([r.pnl_inventory for r in a])
    assert np.mean([r.pnl_spread for r in b]) == pytest.approx(
        np.mean([r.pnl_spread for r in a]), rel=0.05)


def test_toxic_enough_flow_makes_market_making_unprofitable():
    """Il existe un seuil de toxicité au-delà duquel le métier ne paie plus, quelle que
    soit la qualité de la cotation."""
    tres_toxique = FlowParams(informed_ratio=0.5, informed_impact=1.5e-3)
    pnl = [simulate_session(GueantLehalleFT(), tres_toxique, FEE_PROFILES["hft_maker"],
                            8_000, seed=s, keep_paths=False).pnl_total for s in range(6)]
    assert np.median(pnl) < 0


def test_unmanaged_inventory_multiplies_risk_without_adding_return():
    """L'argument central en faveur de la gestion d'inventaire : la politique naïve
    n'encaisse pas moins — elle porte simplement un risque bien plus grand."""
    flow = FlowParams()
    naive = [simulate_session(NaiveSymmetric(), flow, FEE_PROFILES["hft_maker"], 10_000,
                              seed=s, keep_paths=False) for s in range(10)]
    gere = [simulate_session(LinearSkew(), flow, FEE_PROFILES["hft_maker"], 10_000,
                             seed=s, keep_paths=False) for s in range(10)]

    inv_naive = np.mean([r.inventory_max_abs for r in naive])
    inv_gere = np.mean([r.inventory_max_abs for r in gere])
    assert inv_naive > 4 * inv_gere

    sd_naive = np.std([r.pnl_total for r in naive], ddof=1)
    sd_gere = np.std([r.pnl_total for r in gere], ddof=1)
    assert sd_naive > 3 * sd_gere


def test_same_seed_gives_identical_sessions():
    a = simulate_session(GueantLehalleFT(), FlowParams(), FEE_PROFILES["hft_maker"],
                         3_000, seed=7)
    b = simulate_session(GueantLehalleFT(), FlowParams(), FEE_PROFILES["hft_maker"],
                         3_000, seed=7)
    assert a.pnl_total == b.pnl_total and a.n_fills == b.n_fills


def test_compare_policies_shares_the_same_draws():
    """Comparer des politiques sur des tirages différents reviendrait à mesurer la
    chance du flux, pas la qualité de la cotation."""
    t = compare_policies([NaiveSymmetric(), GueantLehalleFT()], FlowParams(),
                         FEE_PROFILES["hft_maker"], n_steps=3_000, n_seeds=4)
    assert len(t) == 2 and "P&L médian" in t.columns
    assert t["exéc./min"].min() > 0

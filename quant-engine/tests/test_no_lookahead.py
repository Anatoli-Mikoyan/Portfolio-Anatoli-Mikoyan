"""Suite anti-look-ahead : le test central du projet.

Un backtest contamine par du look-ahead produit des chiffres excellents et
parfaitement credibles. Aucune relecture de code ne garantit son absence a
grande echelle ; il faut une propriete verifiable mecaniquement.

La propriete retenue est l'**equivalence par troncature** :

    Pour tout instant t, ce que le moteur expose a t doit etre identique,
    bit a bit, a ce qu'il exposerait si les donnees posterieures a t
    n'existaient pas.

Elle est verifiee de trois facons complementaires :

* par troncature reelle -- on reconstruit un jeu de donnees s'arretant a t ;
* par empoisonnement -- on remplace le futur par des NaN, qui contaminent tout
  calcul les touchant, y compris via un chemin que la borne de vue ne couvre pas ;
* par divergence de futur -- deux series identiques jusqu'a t mais differant
  ensuite (y compris par un split, qui modifie l'ajustement) doivent produire
  exactement la meme vue a t.

Un moteur qui passe ces trois tests peut encore etre lent, mal concu ou
inutile. Mais il ne triche pas sur le temps.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest
from conftest import make_data

from quant_engine.data import (
    AdjustmentPolicy,
    CorporateActions,
    Field,
    MarketData,
    Split,
)
from quant_engine.data.types import UTC
from quant_engine.errors import DataError, InsufficientHistoryError, LookaheadError

FIELDS = (Field.OPEN, Field.HIGH, Field.LOW, Field.CLOSE, Field.VOLUME)


# ---------------------------------------------------------------------------
# 1. L'API ne permet pas d'exprimer le futur
# ---------------------------------------------------------------------------
def test_offset_negatif_leve_lookahead(ramp_data: MarketData) -> None:
    """``bar(-1)`` designerait la barre suivante : c'est une erreur, pas un
    indexage python legitime."""
    view = ramp_data.view_at(10, ramp_data.multipliers(AdjustmentPolicy.RAW))
    with pytest.raises(LookaheadError, match="future"):
        view.bar(-1)


def test_offset_hors_historique_leve(ramp_data: MarketData) -> None:
    view = ramp_data.view_at(10, ramp_data.multipliers(AdjustmentPolicy.RAW))
    with pytest.raises(InsufficientHistoryError):
        view.bar(11)


def test_lookback_superieur_a_lhistorique_leve_au_lieu_de_tronquer(
    ramp_data: MarketData,
) -> None:
    """Une fenetre tronquee en silence fabrique des signaux precoces.

    Une moyenne mobile 200 evaluee sur 20 barres reste calculable ; elle produit
    simplement un signal a une date ou la strategie reelle n'en aurait produit
    aucun. C'est du look-ahead deguise en robustesse.
    """
    view = ramp_data.view_at(19, ramp_data.multipliers(AdjustmentPolicy.RAW))
    with pytest.raises(InsufficientHistoryError, match="200 barres demandees"):
        view.close(200)
    assert view.has(20)
    assert not view.has(21)


def test_la_borne_est_physique_pas_conventionnelle(ramp_data: MarketData) -> None:
    """Les tableaux exposes ont pour longueur la fenetre visible.

    La limite est portee par l'objet tableau lui-meme : depasser leve
    ``IndexError`` au niveau de numpy, pas au niveau d'une verification qu'on
    aurait pu oublier d'ecrire.
    """
    view = ramp_data.view_at(9, ramp_data.multipliers(AdjustmentPolicy.RAW))
    closes = view.close()
    assert closes.size == 10
    with pytest.raises(IndexError):
        _ = closes[10]


def test_les_series_exposees_sont_en_lecture_seule(ramp_data: MarketData) -> None:
    """Muter une vue ne doit pas pouvoir corrompre le jeu de donnees partage."""
    view = ramp_data.view_at(9, ramp_data.multipliers(AdjustmentPolicy.RAW))
    with pytest.raises(ValueError, match="read-only"):
        view.close()[0] = 0.0
    assert ramp_data.raw(Field.CLOSE)[0] == 100.0


def test_as_frame_est_une_copie_defensive(ramp_data: MarketData) -> None:
    view = ramp_data.view_at(9, ramp_data.multipliers(AdjustmentPolicy.RAW))
    frame = view.as_frame()
    frame.loc[frame.index[0], "close"] = -999.0
    assert view.close()[0] == 100.0


# ---------------------------------------------------------------------------
# 2. Equivalence par troncature
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("policy", list(AdjustmentPolicy))
def test_equivalence_par_troncature(policy: AdjustmentPolicy) -> None:
    """La vue a t est identique a une vue construite sur les seules donnees <= t.

    Verifie pour toutes les politiques d'ajustement, y compris celles qui sont
    contaminees : le test doit precisement montrer que les politiques retro
    echouent la propriete, ce que ``test_retro_viole_la_troncature`` acte.
    """
    closes = [100.0 + 3.0 * np.sin(i / 4.0) + i * 0.2 for i in range(60)]
    full = make_data(closes)
    multipliers = full.multipliers(policy, allow_lookahead=True)

    for cursor in range(1, len(full)):
        truncated = make_data(closes[: cursor + 1])
        truncated_mult = truncated.multipliers(policy, allow_lookahead=True)
        view_full = full.view_at(cursor, multipliers)
        view_trunc = truncated.view_at(cursor, truncated_mult)
        for field in FIELDS:
            np.testing.assert_array_equal(
                view_full.field(field),
                view_trunc.field(field),
                err_msg=f"{policy.value} / {field.value} diverge au curseur {cursor}",
            )


def test_empoisonnement_du_futur_sans_effet_sur_les_vues() -> None:
    """Remplacer le futur par des NaN ne doit rien changer aux vues passees.

    C'est le detecteur le plus large : un NaN contamine tout calcul qui le
    touche, y compris a travers une agregation pandas ou un cache mal borne
    qu'une simple verification d'index ne verrait pas.
    """
    closes = [100.0 + i * 0.7 for i in range(80)]
    clean = make_data(closes)
    multipliers = clean.multipliers(AdjustmentPolicy.SPLIT_PIT)

    for cursor in (0, 1, 5, 40, 78, 79):
        poisoned = clean.with_future_poisoned(cursor + 1)
        poisoned_mult = poisoned.multipliers(AdjustmentPolicy.SPLIT_PIT)
        reference = clean.view_at(cursor, multipliers)
        suspect = poisoned.view_at(cursor, poisoned_mult)
        for field in FIELDS:
            values = suspect.field(field)
            assert np.isfinite(values).all(), (
                f"NaN remonte du futur dans {field.value} au curseur {cursor} : "
                "un composant lit au-dela de la borne"
            )
            np.testing.assert_array_equal(reference.field(field), values)


def test_lempoisonnement_detecte_bien_un_tricheur() -> None:
    """Contre-epreuve : le detecteur doit echouer sur du code qui triche.

    Un test de securite qui ne se declenche jamais ne prouve rien. On simule un
    composant ayant obtenu le jeu de donnees complet et lisant la barre suivante.
    """
    closes = [100.0 + i for i in range(40)]
    clean = make_data(closes)
    cursor = 20

    def strategie_tricheuse(data: MarketData, index: int) -> float:
        # Exactement le "i + 1" que l'architecture cherche a rendre inexprimable.
        return data.execution_bar(index + 1).close

    assert strategie_tricheuse(clean, cursor) == 121.0

    poisoned = clean.with_future_poisoned(cursor + 1)
    assert np.isnan(strategie_tricheuse(poisoned, cursor)), (
        "L'empoisonnement doit rendre visible toute lecture du futur"
    )


def test_un_split_futur_ninfluence_pas_les_vues_anterieures() -> None:
    """L'ajustement point-in-time est lui aussi exempt de look-ahead.

    Deux series identiques jusqu'a l'index 5, l'une subissant ensuite un split.
    Les vues arretees avant l'operation doivent etre rigoureusement identiques :
    un operateur du 3 janvier ne connait pas le split du 6.
    """
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 53.0, 53.5, 54.0, 54.5]
    ex_date = datetime(2020, 1, 7, 21, tzinfo=UTC)

    sans_split = make_data(closes[:6])
    avec_split = make_data(closes, actions=CorporateActions(splits=(Split(ex_date, 2.0),)))

    mult_sans = sans_split.multipliers(AdjustmentPolicy.SPLIT_PIT)
    mult_avec = avec_split.multipliers(AdjustmentPolicy.SPLIT_PIT)

    for cursor in range(6):
        np.testing.assert_array_equal(
            sans_split.view_at(cursor, mult_sans).close(),
            avec_split.view_at(cursor, mult_avec).close(),
            err_msg=f"le split du futur a fuite dans la vue au curseur {cursor}",
        )

    # Une fois le split survenu, l'historique EST retro-ajuste : c'est le
    # comportement attendu, et c'est ce que voit un operateur ce jour-la.
    apres = avec_split.view_at(6, mult_avec).close()
    assert apres[0] == pytest.approx(50.0)
    assert apres[-1] == pytest.approx(53.0)


def test_retro_viole_la_troncature_et_doit_etre_refuse_par_defaut() -> None:
    """La politique retro classique injecte du futur : demonstration chiffree."""
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 53.0, 53.5]
    ex_date = datetime(2020, 1, 7, 21, tzinfo=UTC)
    data = make_data(closes, actions=CorporateActions(splits=(Split(ex_date, 2.0),)))

    from quant_engine.errors import AdjustmentError

    with pytest.raises(AdjustmentError, match="look-ahead"):
        data.multipliers(AdjustmentPolicy.FULL_RETRO_SPLIT)

    contamine = data.multipliers(AdjustmentPolicy.FULL_RETRO_SPLIT, allow_lookahead=True)
    propre = data.multipliers(AdjustmentPolicy.SPLIT_PIT)

    # Au curseur 2, avant tout split, la politique retro affiche deja 50 :
    # un prix qui n'a jamais cote, deduit d'une operation encore inconnue.
    assert data.view_at(2, contamine).close()[0] == pytest.approx(50.0)
    assert data.view_at(2, propre).close()[0] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# 3. Ordonnancement temporel
# ---------------------------------------------------------------------------
def test_le_curseur_ne_devance_jamais_la_barre_dexecution(ramp_data: MarketData) -> None:
    """Toute barre executable est strictement posterieure a la derniere
    information visible : c'est la definition operationnelle de la latence."""
    cursor = ramp_data.cursor(AdjustmentPolicy.RAW, warmup=1)
    for point in cursor:
        if point.index + 1 >= len(ramp_data):
            break
        execution = ramp_data.execution_bar(point.index + 1)
        assert point.history.as_of < execution.timestamp
        assert point.history.as_of == point.as_of


def test_le_curseur_est_avant_seulement_et_a_usage_unique(ramp_data: MarketData) -> None:
    """Rejouer une periode est le mecanisme par lequel une optimisation se
    contamine elle-meme. Le curseur l'interdit structurellement."""
    cursor = ramp_data.cursor(AdjustmentPolicy.RAW)
    indices = [point.index for point in cursor]
    assert indices == sorted(indices)
    assert indices == list(range(len(ramp_data)))

    with pytest.raises(DataError, match="deja consomme"):
        list(cursor)


def test_le_warmup_decale_le_premier_point_de_decision(ramp_data: MarketData) -> None:
    points = list(ramp_data.cursor(AdjustmentPolicy.RAW, warmup=30))
    assert points[0].index == 30
    assert points[0].history.n_bars == 31
    assert len(points) == len(ramp_data) - 30


def test_actions_to_date_ignore_les_operations_futures(split_data: MarketData) -> None:
    """Les operations sur titre suivent la meme regle que les prix."""
    multipliers = split_data.multipliers(AdjustmentPolicy.SPLIT_PIT)
    assert split_data.view_at(3, multipliers).actions_to_date().splits == ()
    assert len(split_data.view_at(6, multipliers).actions_to_date().splits) == 1


def test_as_of_est_toujours_timezone_aware(clean_data: MarketData) -> None:
    """Un datetime naif se compare mal et decale silencieusement d'une heure
    deux fois par an."""
    multipliers = clean_data.multipliers(AdjustmentPolicy.SPLIT_PIT)
    for cursor in (0, len(clean_data) // 2, len(clean_data) - 1):
        as_of = clean_data.view_at(cursor, multipliers).as_of
        assert as_of.tzinfo is not None
        assert as_of.utcoffset() == timedelta(0)

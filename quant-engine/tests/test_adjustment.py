"""Ajustement point-in-time des prix.

Les valeurs attendues sont calculees a la main dans les commentaires : un test
d'ajustement qui compare le code a lui-meme ne prouve rien.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from quant_engine.data.adjustment import AdjustmentPolicy, build_multipliers
from quant_engine.data.corporate_actions import CorporateActions, Dividend, Split
from quant_engine.data.types import UTC
from quant_engine.errors import AdjustmentError


def stamps(n: int) -> np.ndarray:
    return np.array(
        [int(datetime(2020, 1, 1 + i, 21, tzinfo=UTC).timestamp() * 1e9) for i in range(n)],
        dtype=np.int64,
    )


def test_politique_raw_est_neutre() -> None:
    close = np.array([100.0, 101.0, 102.0])
    multipliers = build_multipliers(stamps(3), close, CorporateActions(), AdjustmentPolicy.RAW)
    assert multipliers.is_identity
    np.testing.assert_array_equal(multipliers.price, np.ones(3))


def test_split_deux_pour_un() -> None:
    """Split 2-pour-1 le 4 janvier (index 3).

    Attendu, curseur en fin de serie : les trois premiers prix divises par 2,
    les suivants inchanges. Le multiplicateur retro vaut 0,5 avant, 1 apres.
    """
    close = np.array([100.0, 102.0, 104.0, 53.0, 54.0])
    actions = CorporateActions(splits=(Split(datetime(2020, 1, 4, tzinfo=UTC), 2.0),))
    multipliers = build_multipliers(stamps(5), close, actions, AdjustmentPolicy.SPLIT_PIT)
    np.testing.assert_allclose(multipliers.price, [0.5, 0.5, 0.5, 1.0, 1.0])
    np.testing.assert_allclose(multipliers.volume, [2.0, 2.0, 2.0, 1.0, 1.0])

    adjusted = close * multipliers.price_factor(0, 5, cursor=4)
    np.testing.assert_allclose(adjusted, [50.0, 51.0, 52.0, 53.0, 54.0])


def test_le_prix_courant_reste_le_prix_brut() -> None:
    """Invariant central du point-in-time : au curseur, facteur = 1.

    Un ordre s'execute au prix reellement cote. Si l'ajustement modifiait le
    prix courant, la taille de position, les seuils absolus et les frais
    proportionnels seraient tous calcules sur un prix fictif.
    """
    close = np.array([100.0, 102.0, 104.0, 53.0, 54.0])
    actions = CorporateActions(splits=(Split(datetime(2020, 1, 4, tzinfo=UTC), 2.0),))
    multipliers = build_multipliers(stamps(5), close, actions, AdjustmentPolicy.SPLIT_PIT)
    for cursor in range(5):
        factor = multipliers.price_factor(cursor, cursor + 1, cursor)
        assert factor[0] == pytest.approx(1.0)


def test_regroupement_dactions() -> None:
    """Regroupement 1-pour-10 : ratio 0,1, le prix est multiplie par 10."""
    close = np.array([2.0, 2.1, 21.0, 22.0])
    actions = CorporateActions(splits=(Split(datetime(2020, 1, 3, tzinfo=UTC), 0.1),))
    multipliers = build_multipliers(stamps(4), close, actions, AdjustmentPolicy.SPLIT_PIT)
    adjusted = close * multipliers.price_factor(0, 4, cursor=3)
    np.testing.assert_allclose(adjusted, [20.0, 21.0, 21.0, 22.0])


def test_splits_multiples_se_composent() -> None:
    """2-pour-1 puis 3-pour-1 : le premier segment est divise par 6."""
    close = np.array([600.0, 300.0, 100.0, 101.0])
    actions = CorporateActions(
        splits=(
            Split(datetime(2020, 1, 2, tzinfo=UTC), 2.0),
            Split(datetime(2020, 1, 3, tzinfo=UTC), 3.0),
        )
    )
    multipliers = build_multipliers(stamps(4), close, actions, AdjustmentPolicy.SPLIT_PIT)
    np.testing.assert_allclose(multipliers.price, [1 / 6, 1 / 3, 1.0, 1.0])


def test_ajustement_dividende() -> None:
    """Dividende de 2 sur un cours de 100 la veille de l'ex-date.

    Facteur applique en amont : 1 - 2/100 = 0,98. Le prix de 100 devient 98.
    """
    close = np.array([100.0, 98.0, 99.0])
    actions = CorporateActions(dividends=(Dividend(datetime(2020, 1, 2, tzinfo=UTC), 2.0),))
    multipliers = build_multipliers(
        stamps(3), close, actions, AdjustmentPolicy.TOTAL_RETURN_PIT
    )
    np.testing.assert_allclose(multipliers.price, [0.98, 1.0, 1.0])


def test_dividende_ignore_en_politique_split_seul() -> None:
    """Par defaut le dividende est un flux de tresorerie, pas une deformation
    du prix : il sera credite au compte par le moteur."""
    close = np.array([100.0, 98.0, 99.0])
    actions = CorporateActions(dividends=(Dividend(datetime(2020, 1, 2, tzinfo=UTC), 2.0),))
    multipliers = build_multipliers(stamps(3), close, actions, AdjustmentPolicy.SPLIT_PIT)
    np.testing.assert_allclose(multipliers.price, np.ones(3))


def test_dividende_superieur_au_cours_est_refuse() -> None:
    close = np.array([1.0, 0.5, 0.6])
    actions = CorporateActions(dividends=(Dividend(datetime(2020, 1, 2, tzinfo=UTC), 5.0),))
    with pytest.raises(AdjustmentError, match="superieur au cours"):
        build_multipliers(stamps(3), close, actions, AdjustmentPolicy.TOTAL_RETURN_PIT)


def test_operation_hors_fenetre_sans_effet() -> None:
    """Un split anterieur a la premiere barre chargee ne doit rien ajuster :
    les prix disponibles sont deja post-operation."""
    close = np.array([50.0, 51.0, 52.0])
    actions = CorporateActions(splits=(Split(datetime(2019, 6, 1, tzinfo=UTC), 2.0),))
    multipliers = build_multipliers(stamps(3), close, actions, AdjustmentPolicy.SPLIT_PIT)
    np.testing.assert_allclose(multipliers.price, np.ones(3))


@pytest.mark.parametrize(
    "policy", [AdjustmentPolicy.FULL_RETRO_SPLIT, AdjustmentPolicy.FULL_RETRO_TOTAL]
)
def test_politiques_retro_refusees_sans_optin(policy: AdjustmentPolicy) -> None:
    close = np.array([100.0, 101.0, 102.0])
    with pytest.raises(AdjustmentError, match="look-ahead"):
        build_multipliers(stamps(3), close, CorporateActions(), policy)


def test_drapeau_de_contamination() -> None:
    close = np.array([100.0, 101.0, 102.0])
    propre = build_multipliers(stamps(3), close, CorporateActions(), AdjustmentPolicy.SPLIT_PIT)
    sale = build_multipliers(
        stamps(3),
        close,
        CorporateActions(),
        AdjustmentPolicy.FULL_RETRO_TOTAL,
        allow_lookahead=True,
    )
    assert not propre.is_lookahead_contaminated
    assert sale.is_lookahead_contaminated
    assert sale.normalize_index == 2


def test_multiplicateurs_immuables() -> None:
    close = np.array([100.0, 101.0, 102.0])
    multipliers = build_multipliers(
        stamps(3), close, CorporateActions(), AdjustmentPolicy.SPLIT_PIT
    )
    with pytest.raises(ValueError, match="read-only"):
        multipliers.price[0] = 42.0


def test_serie_vide_refusee() -> None:
    with pytest.raises(AdjustmentError, match="vide"):
        build_multipliers(
            np.array([], dtype=np.int64), np.array([]), CorporateActions(),
            AdjustmentPolicy.SPLIT_PIT,
        )


def test_operations_connues_a_une_date() -> None:
    actions = CorporateActions(
        splits=(
            Split(datetime(2020, 1, 5, tzinfo=UTC), 2.0),
            Split(datetime(2021, 1, 5, tzinfo=UTC), 3.0),
        ),
        dividends=(Dividend(datetime(2020, 6, 1, tzinfo=UTC), 1.0),),
    )
    connu = actions.known_at(datetime(2020, 7, 1, tzinfo=UTC))
    assert len(connu.splits) == 1
    assert len(connu.dividends) == 1
    assert actions.known_at(datetime(2019, 1, 1, tzinfo=UTC)).is_empty


def test_dividendes_dans_un_intervalle() -> None:
    actions = CorporateActions(
        dividends=(
            Dividend(datetime(2020, 3, 1, tzinfo=UTC), 1.0),
            Dividend(datetime(2020, 6, 1, tzinfo=UTC), 1.1),
            Dividend(datetime(2020, 9, 1, tzinfo=UTC), 1.2),
        )
    )
    window = actions.dividends_between(
        datetime(2020, 3, 1, tzinfo=UTC), datetime(2020, 9, 1, tzinfo=UTC)
    )
    # Borne basse exclue, borne haute incluse : evite de crediter deux fois un
    # dividende a la jonction de deux barres.
    assert [d.amount for d in window] == [1.1, 1.2]


def test_ratio_de_split_invalide() -> None:
    with pytest.raises(ValueError, match="Ratio de split invalide"):
        Split(datetime(2020, 1, 1, tzinfo=UTC), 0.0)


def test_dividende_negatif_refuse() -> None:
    with pytest.raises(ValueError, match="Dividende negatif"):
        Dividend(datetime(2020, 1, 1, tzinfo=UTC), -1.0)


def test_dividendes_borne_haute_exclue() -> None:
    """La variante exclusive evite de crediter deux fois un dividende tombant
    exactement a la jonction de deux barres."""
    actions = CorporateActions(
        dividends=(
            Dividend(datetime(2020, 3, 1, tzinfo=UTC), 1.0),
            Dividend(datetime(2020, 6, 1, tzinfo=UTC), 1.1),
        )
    )
    inclus = actions.dividends_between(
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 6, 1, tzinfo=UTC)
    )
    exclus = actions.dividends_between(
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 6, 1, tzinfo=UTC), inclusive_end=False
    )
    assert [d.amount for d in inclus] == [1.0, 1.1]
    assert [d.amount for d in exclus] == [1.0]


def test_splits_dans_un_intervalle() -> None:
    actions = CorporateActions(
        splits=(
            Split(datetime(2020, 3, 1, tzinfo=UTC), 2.0),
            Split(datetime(2021, 3, 1, tzinfo=UTC), 3.0),
        )
    )
    window = actions.splits_between(
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 12, 31, tzinfo=UTC)
    )
    assert [s.ratio for s in window] == [2.0]


def test_repr_des_operations() -> None:
    actions = CorporateActions(splits=(Split(datetime(2020, 1, 1, tzinfo=UTC), 2.0),))
    assert "splits=1" in repr(actions)
    assert not actions.is_empty
    assert CorporateActions().is_empty

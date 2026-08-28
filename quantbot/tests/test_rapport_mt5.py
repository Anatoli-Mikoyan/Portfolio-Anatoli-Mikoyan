"""Tests du lecteur de rapport MetaTrader 5 et du verdict statistique.

Le lecteur affronte un format que je ne maîtrise pas : MetaTrader change la
structure de son rapport HTML selon la build et la langue de l'interface. Les
tests fixent donc le comportement sur les variantes connues plutôt que sur un
seul exemple, et surtout sur les pièges : opérations de solde mélangées aux
transactions, formats de nombres européens, tableaux empilés aux colonnes
différentes.
"""
from __future__ import annotations

import numpy as np
import pytest

from qbot.live.rapport_mt5 import juger, lire_rapport


def _rapport_html(profits, entete="Profit", avec_solde=True, sep=".") -> str:
    """Fabrique un rapport de la même forme que celui exporté par MetaTrader."""
    def fmt(v):
        s = f"{v:.2f}"
        return s.replace(".", sep) if sep != "." else s

    lignes = [
        "<html><body><table>",
        "<tr><th>Time</th><th>Deal</th><th>Symbol</th><th>Type</th>"
        f"<th>Volume</th><th>Price</th><th>Commission</th><th>Swap</th><th>{entete}</th></tr>",
    ]
    if avec_solde:
        lignes.append("<tr><td>2026.01.01</td><td>1</td><td></td><td>balance</td>"
                      "<td></td><td></td><td></td><td></td><td>1000.00</td></tr>")
    for i, p in enumerate(profits):
        lignes.append(
            f"<tr><td>2026.01.{i % 28 + 1:02d}</td><td>{i + 2}</td><td>EURUSD</td>"
            f"<td>buy</td><td>0.10</td><td>1.0850</td><td>0.00</td><td>0.00</td>"
            f"<td>{fmt(p)}</td></tr>")
    lignes.append("</table></body></html>")
    return "\n".join(lignes)


def _ecrire(tmp_path, contenu, nom="ReportHistory.html"):
    f = tmp_path / nom
    f.write_text(contenu, encoding="utf-8")
    return f


# ---------------------------------------------------------------------------------------
# Lecture
# ---------------------------------------------------------------------------------------
def test_lit_les_transactions_et_ignore_le_depot(tmp_path):
    """La ligne « balance » est un dépôt, pas un trade : la compter fausserait le test."""
    profits = [12.5, -8.0, 3.25, -1.75]
    f = _ecrire(tmp_path, _rapport_html(profits))
    hist = lire_rapport(f)

    assert hist.n == 4, "le dépôt de 1000 a été compté comme une transaction"
    assert hist.total == pytest.approx(sum(profits))
    assert hist.taux_reussite == pytest.approx(0.5)


def test_recupere_le_capital_initial_depuis_la_ligne_de_solde(tmp_path):
    f = _ecrire(tmp_path, _rapport_html([5.0, -2.0]))
    assert lire_rapport(f).capital_initial == pytest.approx(1000.0)


def test_accepte_le_format_de_nombre_europeen(tmp_path):
    """« 12,50 » et « 12.50 » doivent donner le même résultat."""
    f = _ecrire(tmp_path, _rapport_html([12.5, -8.0], sep=","))
    assert lire_rapport(f).total == pytest.approx(4.5)


def test_accepte_un_entete_traduit(tmp_path):
    """MetaTrader traduit ses en-têtes : la colonne se trouve par son nom, pas sa place."""
    f = _ecrire(tmp_path, _rapport_html([10.0, -4.0], entete="Bénéfice"))
    assert lire_rapport(f).total == pytest.approx(6.0)


def test_commission_et_swap_sont_deduits(tmp_path):
    """Un résultat brut n'est pas un résultat : les frais changent souvent le signe."""
    html = (
        "<table>"
        "<tr><th>Type</th><th>Commission</th><th>Swap</th><th>Profit</th></tr>"
        "<tr><td>buy</td><td>-0.70</td><td>-0.30</td><td>2.00</td></tr>"
        "</table>")
    hist = lire_rapport(_ecrire(tmp_path, html))
    assert hist.total == pytest.approx(1.0), "commission et swap non déduits"


def test_suit_les_changements_dentete_entre_tableaux(tmp_path):
    """Un rapport MT5 empile plusieurs tableaux aux colonnes différentes.

    Se caler une fois sur le premier en-tête ferait lire la mauvaise colonne
    ensuite — et produirait un total faux sans lever la moindre erreur.
    """
    html = (
        "<table>"
        "<tr><th>Time</th><th>Order</th><th>Profit</th></tr>"
        "<tr><td>2026.01.01</td><td>1</td><td>10.00</td></tr>"
        "<tr><th>Profit</th><th>Time</th><th>Deal</th></tr>"
        "<tr><td>5.00</td><td>2026.01.02</td><td>2</td></tr>"
        "</table>")
    hist = lire_rapport(_ecrire(tmp_path, html))
    assert hist.n == 2
    assert hist.total == pytest.approx(15.0), "la deuxième colonne a été mal repérée"


def test_fichier_absent_ou_illisible_donne_un_message_utile(tmp_path):
    with pytest.raises(FileNotFoundError):
        lire_rapport(tmp_path / "rien.html")

    f = _ecrire(tmp_path, "bonjour, ceci n'est pas un rapport", nom="autre.html")
    with pytest.raises(ValueError, match="Historique"):
        lire_rapport(f)


def test_rapport_sans_transaction_le_dit(tmp_path):
    f = _ecrire(tmp_path, _rapport_html([]))
    with pytest.raises(ValueError, match="Aucune transaction"):
        lire_rapport(f)


def test_lit_aussi_un_csv_simple(tmp_path):
    f = tmp_path / "resultats.csv"
    f.write_text("12.5\n-8.0\n3.25\n", encoding="utf-8")
    assert lire_rapport(f).total == pytest.approx(7.75)


# ---------------------------------------------------------------------------------------
# Verdict — c'est ici que se joue l'honnêteté de l'outil
# ---------------------------------------------------------------------------------------
def test_une_serie_perdante_est_declaree_perdante(tmp_path):
    rng = np.random.default_rng(1)
    profits = rng.normal(-1.5, 10.0, 200)
    hist = lire_rapport(_ecrire(tmp_path, _rapport_html(profits)))
    assert juger(hist).conclusion == "PERDANT"


def test_un_petit_gain_sur_peu_de_trades_reste_indecidable():
    """Le cas qui compte : 3 semaines de démo, un gain, et rien de démontré.

    C'est exactement la situation où l'on est tenté de passer au réel. Le verdict
    doit résister à la tentation.
    """
    from qbot.live.rapport_mt5 import HistoriqueMT5

    rng = np.random.default_rng(3)
    # 36 transactions, aucun edge réel, mais le tirage finit dans le vert.
    profits = rng.normal(0.0, 10.0, 36)
    profits = profits - profits.mean() + 1.2      # +1,20 € par trade, purement fortuit
    v = juger(HistoriqueMT5(profits, 1000.0, "test"))

    assert v.total > 0, "le scénario doit bien afficher un gain"
    assert v.conclusion == "INDÉCIDABLE"
    assert v.trades_necessaires > v.n, "l'outil doit dire qu'il manque des transactions"


def test_un_edge_franc_et_long_est_declare_significatif():
    from qbot.live.rapport_mt5 import HistoriqueMT5

    rng = np.random.default_rng(5)
    profits = rng.normal(3.0, 5.0, 800)          # edge net, échantillon confortable
    v = juger(HistoriqueMT5(profits, 1000.0, "test"))

    assert v.conclusion == "SIGNIFICATIF"
    assert v.p_value < 0.05
    assert v.ic_bas > 0


def test_lintervalle_de_confiance_encadre_le_total():
    from qbot.live.rapport_mt5 import HistoriqueMT5

    rng = np.random.default_rng(7)
    profits = rng.normal(1.0, 8.0, 120)
    v = juger(HistoriqueMT5(profits, 1000.0, "test"))
    assert v.ic_bas < v.total < v.ic_haut


def test_deux_transactions_ne_permettent_aucun_verdict():
    from qbot.live.rapport_mt5 import HistoriqueMT5

    v = juger(HistoriqueMT5(np.array([5.0]), 1000.0, "test"))
    assert v.conclusion == "INDÉCIDABLE"


def test_le_test_de_signe_est_plus_severe_que_student():
    """Sur des résultats à queues épaisses, Student surestime la significativité.

    Ce test fixe la raison d'être du bootstrap : sur le même échantillon, il doit
    refuser de conclure là où un t-test naïf conclurait.
    """
    from scipy import stats

    from qbot.live.rapport_mt5 import HistoriqueMT5

    rng = np.random.default_rng(11)
    # Beaucoup de petits gains, quelques grosses pertes : la forme typique d'une
    # stratégie à stop large — et le pire cas pour un t-test.
    profits = np.concatenate([rng.normal(2.0, 1.0, 90), rng.normal(-15.0, 5.0, 10)])
    v = juger(HistoriqueMT5(profits, 1000.0, "test"))
    p_student = stats.ttest_1samp(profits, 0.0).pvalue / 2.0

    assert v.p_value > p_student, (
        f"le bootstrap ({v.p_value:.3f}) devrait être plus prudent "
        f"que Student ({p_student:.3f})")

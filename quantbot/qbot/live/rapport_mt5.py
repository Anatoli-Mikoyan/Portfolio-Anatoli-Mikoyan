"""Lecture du rapport d'historique exporté par MetaTrader 5, et verdict honnête.

MetaTrader sait dire « vous avez gagné 43 € ». Il ne sait pas dire si ces 43 € sont
un edge ou une pièce qui est tombée du bon côté. C'est toute la différence entre
un chiffre et une décision, et c'est ce que ce module ajoute.

Le rapport MT5 (clic droit dans l'onglet Historique → Rapport) est un fichier HTML
dont la structure varie selon la build et la langue de l'interface. Le lecteur ici
est donc volontairement tolérant : il cherche la colonne « Profit » par son en-tête
(dans plusieurs langues), pas par sa position, et ignore toute ligne qu'il ne
comprend pas plutôt que d'échouer. Un rapport partiellement lu vaut mieux qu'une
exception devant un utilisateur qui vient de passer trois semaines à collecter
ces données.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

# En-têtes de la colonne de résultat, selon la langue de MetaTrader.
_ENTETES_PROFIT = ("profit", "bénéfice", "benefice", "gewinn", "ganancia", "прибыль")
_ENTETES_COMMISSION = ("commission", "комиссия")
_ENTETES_SWAP = ("swap", "своп")


# =======================================================================================
# Lecture
# =======================================================================================
def _cellules(ligne: str) -> List[str]:
    """Extrait le texte des cellules d'une ligne HTML, balises et entités retirées."""
    brutes = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", ligne, re.S | re.I)
    return [html.unescape(re.sub(r"<[^>]+>", " ", c)).replace("\xa0", " ").strip()
            for c in brutes]


def _nombre(texte: str) -> Optional[float]:
    """Convertit une cellule en nombre, en acceptant les formats européens.

    MetaTrader écrit « 1 234.56 », « 1 234,56 » ou « -12.30 » selon la localisation.
    Les espaces (y compris insécables) séparent les milliers ; la virgule décimale
    n'apparaît que si le point n'est pas déjà utilisé.
    """
    t = texte.replace(" ", "").replace(" ", "").replace("\xa0", "")
    if not t or t in {"-", "—"}:
        return None
    if "," in t and "." not in t:
        t = t.replace(",", ".")
    else:
        t = t.replace(",", "")
    try:
        return float(t)
    except ValueError:
        return None


@dataclass
class HistoriqueMT5:
    """Les résultats par transaction, tels que MetaTrader les a enregistrés."""
    profits: np.ndarray          # résultat net par transaction, frais compris
    capital_initial: float
    source: str

    @property
    def n(self) -> int:
        return int(self.profits.size)

    @property
    def total(self) -> float:
        return float(self.profits.sum())

    @property
    def taux_reussite(self) -> float:
        return float((self.profits > 0).mean()) if self.n else 0.0


def lire_rapport(chemin: str | Path, capital_initial: float = 0.0) -> HistoriqueMT5:
    """Lit un rapport MT5 (HTML) ou un CSV d'une colonne de résultats."""
    chemin = Path(chemin)
    if not chemin.exists():
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")

    texte = chemin.read_text(encoding="utf-8", errors="replace")

    if chemin.suffix.lower() in {".csv", ".txt"} and "<t" not in texte[:2000].lower():
        valeurs = [_nombre(l.split(";")[-1].split(",")[-1]) for l in texte.splitlines()]
        profits = np.array([v for v in valeurs if v is not None], dtype=float)
        if profits.size == 0:
            raise ValueError(f"Aucun nombre lisible dans {chemin}")
        return HistoriqueMT5(profits, capital_initial, str(chemin))

    lignes = re.findall(r"<tr[^>]*>.*?</tr>", texte, re.S | re.I)
    if not lignes:
        raise ValueError(
            f"{chemin.name} ne ressemble pas à un rapport MetaTrader.\n"
            "Dans MetaTrader : onglet « Boîte à outils » en bas → Historique → "
            "clic droit → Rapport → HTML."
        )

    # On suit les changements d'en-tête : un rapport MT5 empile plusieurs tableaux
    # (Ordres, Transactions, Résumé) avec des colonnes différentes. Se caler une
    # fois pour toutes sur le premier en-tête rencontré ferait lire la mauvaise
    # colonne dans les tableaux suivants.
    i_profit = i_comm = i_swap = None
    profits: List[float] = []
    capital = capital_initial

    for ligne in lignes:
        cells = _cellules(ligne)
        if not cells:
            continue
        bas = [c.lower() for c in cells]

        if any(any(e in c for e in _ENTETES_PROFIT) for c in bas):
            i_profit = next((k for k, c in enumerate(bas)
                             if any(e in c for e in _ENTETES_PROFIT)), None)
            i_comm = next((k for k, c in enumerate(bas)
                           if any(e in c for e in _ENTETES_COMMISSION)), None)
            i_swap = next((k for k, c in enumerate(bas)
                           if any(e in c for e in _ENTETES_SWAP)), None)
            continue

        est_solde = any("balance" in c or "credit" in c or "solde" in c for c in bas)
        if est_solde and capital_initial <= 0 and capital <= 0:
            # Le montant du dépôt est dans la colonne de résultat, pas ailleurs :
            # balayer la ligne à la recherche du « premier nombre positif » ramènerait
            # le numéro de transaction (« 1 »), qui est aussi un nombre positif.
            if i_profit is not None and i_profit < len(cells):
                v = _nombre(cells[i_profit])
                if v is not None and v > 0:
                    capital = v

        if i_profit is None or i_profit >= len(cells):
            continue
        # Un dépôt ou un retrait porte un résultat mais n'est pas une transaction :
        # l'inclure gonflerait le nombre d'observations et fausserait le test.
        if est_solde:
            continue

        p = _nombre(cells[i_profit])
        if p is None:
            continue
        for idx in (i_comm, i_swap):
            if idx is not None and idx < len(cells):
                v = _nombre(cells[idx])
                if v is not None:
                    p += v
        profits.append(p)

    if not profits:
        raise ValueError(
            f"Aucune transaction trouvée dans {chemin.name}.\n"
            "Le rapport est peut-être vide (aucun trade clôturé), ou exporté depuis "
            "l'onglet « Trading » au lieu de « Historique »."
        )
    return HistoriqueMT5(np.asarray(profits, dtype=float), capital, str(chemin))


# =======================================================================================
# Verdict
# =======================================================================================
@dataclass
class Verdict:
    n: int
    total: float
    taux_reussite: float
    moyenne: float
    p_value: float
    ic_bas: float
    ic_haut: float
    trades_necessaires: float
    conclusion: str
    explication: str


def juger(hist: HistoriqueMT5, n_bootstrap: int = 20_000, seed: int = 0) -> Verdict:
    """Le résultat observé est-il distinguable de la chance ?

    Test par permutation de signe plutôt que test de Student : les résultats par
    transaction ne sont pas gaussiens (queues épaisses, asymétrie forte due aux
    stops), et un test de Student sur 36 observations de cette forme surestime
    franchement la significativité. Le bootstrap ne suppose rien sur la forme.
    """
    r = hist.profits
    n = r.size
    if n < 2:
        return Verdict(n, hist.total, hist.taux_reussite, float(r.mean()) if n else 0.0,
                       1.0, 0.0, 0.0, float("inf"), "INDÉCIDABLE",
                       "Moins de deux transactions : il n'y a rien à tester.")

    rng = np.random.default_rng(seed)
    moyenne = float(r.mean())

    # Sous H0 « aucun edge », le signe de chaque résultat est équiprobable.
    signes = rng.choice((-1.0, 1.0), size=(n_bootstrap, n))
    moyennes_h0 = (signes * np.abs(r)).mean(axis=1)
    p_value = float((moyennes_h0 >= moyenne).mean()) if moyenne > 0 else 1.0

    # Intervalle de confiance sur le total, par rééchantillonnage.
    tirages = rng.choice(r, size=(n_bootstrap, n), replace=True)
    totaux = tirages.sum(axis=1)
    ic_bas, ic_haut = (float(x) for x in np.percentile(totaux, [5, 95]))

    # Combien de transactions faudrait-il pour trancher, à ce niveau de bruit ?
    ecart = float(r.std(ddof=1))
    if moyenne > 0 and ecart > 1e-12:
        trades_necessaires = float((1.645 * ecart / moyenne) ** 2)
    else:
        trades_necessaires = float("inf")

    if moyenne <= 0:
        conclusion = "PERDANT"
        explication = ("Le résultat moyen par transaction est négatif. Il n'y a pas de "
                       "test à faire : l'outil perd de l'argent sur cet échantillon.")
    elif p_value < 0.05 and n >= trades_necessaires:
        conclusion = "SIGNIFICATIF"
        explication = (f"Un résultat au moins aussi bon n'arriverait par chance que dans "
                       f"{p_value:.1%} des cas, sur un échantillon assez grand pour que "
                       f"ce chiffre veuille dire quelque chose.")
    else:
        conclusion = "INDÉCIDABLE"
        manque = max(trades_necessaires - n, 0.0)
        explication = (
            f"Le gain est réel mais indiscernable de la chance : un résultat au moins "
            f"aussi bon arriverait par hasard dans {p_value:.1%} des cas. "
            + (f"Il faudrait environ {trades_necessaires:,.0f} transactions pour trancher "
               f"({manque:,.0f} de plus qu'aujourd'hui)."
               if np.isfinite(trades_necessaires)
               else "Le bruit domine trop le signal pour estimer un horizon."))

    return Verdict(n, hist.total, hist.taux_reussite, moyenne, p_value,
                   ic_bas, ic_haut, trades_necessaires, conclusion, explication)

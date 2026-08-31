"""Tests du pont MetaTrader 5 : détection du terminal et installation de l'EA.

Ces deux fonctions tournent sur la machine de l'utilisateur, sur une arborescence
que je ne contrôle pas et que je ne peux pas voir. Elles sont donc testées ici
contre de fausses arborescences reproduisant exactement la forme réelle :

    <dossier de données>/MQL5/Experts/

avec les deux pièges qui font échouer une détection naïve — le dossier `Common`,
partagé entre terminaux et sans `MQL5/Experts` utilisable, et la présence
possible de plusieurs terminaux installés côte à côte.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parent.parent


def _charger_module_mt5():
    """Charge scripts/mt5.py comme un module (le dossier n'est pas un paquet)."""
    chemin = RACINE / "scripts" / "mt5.py"
    spec = importlib.util.spec_from_file_location("script_mt5", chemin)
    module = importlib.util.module_from_spec(spec)
    sys.modules["script_mt5"] = module
    spec.loader.exec_module(module)
    return module


mt5 = _charger_module_mt5()


def _faux_terminal(racine: Path, empreinte: str) -> Path:
    dossier = racine / empreinte
    (dossier / "MQL5" / "Experts").mkdir(parents=True)
    return dossier


# ---------------------------------------------------------------------------------------
# Détection
# ---------------------------------------------------------------------------------------
def test_detection_windows_trouve_les_terminaux(tmp_path, monkeypatch):
    terminal = tmp_path / "MetaQuotes" / "Terminal"
    terminal.mkdir(parents=True)
    attendu = _faux_terminal(terminal, "D0E8209F77C8CF37AD8BF550E51FF075")

    monkeypatch.setattr(mt5.platform, "system", lambda: "Windows")
    monkeypatch.setenv("APPDATA", str(tmp_path))

    assert mt5.dossiers_metatrader() == [attendu]


def test_detection_ignore_le_dossier_common(tmp_path, monkeypatch):
    """`Common` est partagé entre terminaux : y copier un EA ne le rend pas compilable."""
    terminal = tmp_path / "MetaQuotes" / "Terminal"
    terminal.mkdir(parents=True)
    vrai = _faux_terminal(terminal, "ABCDEF0123456789ABCDEF0123456789")
    _faux_terminal(terminal, "Common")

    monkeypatch.setattr(mt5.platform, "system", lambda: "Windows")
    monkeypatch.setenv("APPDATA", str(tmp_path))

    assert mt5.dossiers_metatrader() == [vrai]


def test_detection_ignore_un_dossier_sans_mql5(tmp_path, monkeypatch):
    """Un dossier de terminal incomplet (jamais lancé) ne doit pas être proposé."""
    terminal = tmp_path / "MetaQuotes" / "Terminal"
    terminal.mkdir(parents=True)
    (terminal / "0000000000000000AAAAAAAAAAAAAAAA").mkdir()
    vrai = _faux_terminal(terminal, "1111111111111111BBBBBBBBBBBBBBBB")

    monkeypatch.setattr(mt5.platform, "system", lambda: "Windows")
    monkeypatch.setenv("APPDATA", str(tmp_path))

    assert mt5.dossiers_metatrader() == [vrai]


def test_detection_gere_plusieurs_terminaux(tmp_path, monkeypatch):
    terminal = tmp_path / "MetaQuotes" / "Terminal"
    terminal.mkdir(parents=True)
    a = _faux_terminal(terminal, "AAAA000000000000AAAA000000000000")
    b = _faux_terminal(terminal, "BBBB111111111111BBBB111111111111")

    monkeypatch.setattr(mt5.platform, "system", lambda: "Windows")
    monkeypatch.setenv("APPDATA", str(tmp_path))

    assert set(mt5.dossiers_metatrader()) == {a, b}


def test_detection_sans_metatrader_ne_leve_pas(tmp_path, monkeypatch):
    """Aucune installation : liste vide, pas d'exception."""
    monkeypatch.setattr(mt5.platform, "system", lambda: "Windows")
    monkeypatch.setenv("APPDATA", str(tmp_path / "inexistant"))
    assert mt5.dossiers_metatrader() == []


# ---------------------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------------------
def test_installation_copie_lea_dans_chaque_terminal(tmp_path, monkeypatch, capsys):
    terminal = tmp_path / "MetaQuotes" / "Terminal"
    terminal.mkdir(parents=True)
    a = _faux_terminal(terminal, "AAAA000000000000AAAA000000000000")
    b = _faux_terminal(terminal, "BBBB111111111111BBBB111111111111")

    monkeypatch.setattr(mt5.platform, "system", lambda: "Windows")
    monkeypatch.setenv("APPDATA", str(tmp_path))

    assert mt5.installer_ea() == 0
    for dossier in (a, b):
        copie = dossier / "MQL5" / "Experts" / "QBotBridge.mq5"
        assert copie.exists(), f"EA absent de {dossier}"
        assert copie.read_bytes() == mt5.EA_SOURCE.read_bytes()


def test_installation_sur_dossier_explicite(tmp_path):
    """Chemin fourni à la main quand la détection échoue : doit marcher aussi."""
    cible = tmp_path / "MonTerminal"
    (cible / "MQL5" / "Experts").mkdir(parents=True)
    assert mt5.installer_ea(str(cible)) == 0
    assert (cible / "MQL5" / "Experts" / "QBotBridge.mq5").exists()


def test_installation_echoue_proprement_sans_metatrader(tmp_path, monkeypatch, capsys):
    """Sans terminal : code d'erreur 1 et instructions, jamais une trace Python."""
    monkeypatch.setattr(mt5.platform, "system", lambda: "Windows")
    monkeypatch.setenv("APPDATA", str(tmp_path / "vide"))

    assert mt5.installer_ea() == 1
    sortie = capsys.readouterr().out
    assert "Ouvrir le dossier de données" in sortie
    assert "--dossier" in sortie


def test_lea_existe_et_reste_sans_dll():
    """L'EA doit rester compilable sur un compte prop-firm : aucun `#import`.

    Un `#import` de DLL ferait rejeter l'EA par la plupart des prop-firms et par le
    Market MQL5, et imposerait « Autoriser les importations DLL » — une case que
    personne ne devrait cocher pour un programme qui pilote un compte.
    """
    source = mt5.EA_SOURCE.read_text(encoding="utf-8", errors="replace")
    lignes_import = [
        ligne.strip()
        for ligne in source.splitlines()
        if ligne.strip().startswith("#import")
    ]
    assert not lignes_import, f"L'EA importe une DLL : {lignes_import}"
    assert "SocketConnect" in source, "l'EA doit utiliser les sockets natifs MQL5"


def test_le_dry_run_est_le_defaut_dans_lea():
    """Le verrou côté MetaTrader doit être fermé à l'installation, sans exception."""
    source = mt5.EA_SOURCE.read_text(encoding="utf-8", errors="replace")
    ligne = next(l for l in source.splitlines() if "InpDryRun" in l and "input" in l)
    assert "= true" in ligne.replace(" ", " "), f"InpDryRun n'est pas à true : {ligne}"


# ---------------------------------------------------------------------------------------
# Tolérance sur le chemin fourni à la main
#
# L'utilisateur arrive ici après « Fichier > Ouvrir le dossier de données » et colle ce
# que l'explorateur lui montre. Selon l'endroit où il s'est arrêté de naviguer, ce sera
# la racine du terminal, le dossier MQL5, ou déjà Experts. Exiger la racine créerait
# MQL5/MQL5/Experts sans un mot : l'installeur dirait « ok », et MetaEditor ne verrait
# jamais le fichier.
# ---------------------------------------------------------------------------------------
def test_les_trois_profondeurs_de_chemin_convergent(tmp_path):
    racine = tmp_path / "D0E8209F77C8CF37AD8BF550E51FF075"
    attendu = racine / "MQL5" / "Experts"

    for essai in (racine, racine / "MQL5", racine / "MQL5" / "Experts"):
        assert mt5.dossier_experts(essai) == attendu, f"chemin mal ramené : {essai}"


def test_la_casse_du_dossier_est_toleree(tmp_path):
    """Windows ne distingue pas la casse ; le chemin collé peut arriver en minuscules."""
    racine = tmp_path / "terminal"
    assert mt5.dossier_experts(racine / "mql5").name == "Experts"
    assert mt5.dossier_experts(racine / "MQL5" / "experts").name == "experts"


def test_installation_sur_un_chemin_mql5(tmp_path):
    """Le cas réel : l'utilisateur colle le chemin qui se termine par MQL5."""
    racine = tmp_path / "D0E8209F77C8CF37AD8BF550E51FF075"
    (racine / "MQL5" / "Experts").mkdir(parents=True)

    assert mt5.installer_ea(str(racine / "MQL5")) == 0
    assert (racine / "MQL5" / "Experts" / "QBotBridge.mq5").exists()
    assert not (racine / "MQL5" / "MQL5").exists(), (
        "un dossier MQL5/MQL5 a été créé : l'EA serait invisible pour MetaEditor")


def test_installation_sur_un_chemin_experts(tmp_path):
    racine = tmp_path / "terminal"
    experts = racine / "MQL5" / "Experts"
    experts.mkdir(parents=True)

    assert mt5.installer_ea(str(experts)) == 0
    assert (experts / "QBotBridge.mq5").exists()
    assert not (experts / "MQL5").exists()


def test_les_libelles_des_parametres_portent_leur_nom():
    """MetaTrader affiche le COMMENTAIRE d'un `input`, jamais le nom de la variable.

    Une consigne écrite ailleurs — « mettez InpDryRun à false » — devient alors
    introuvable dans la fenêtre « Paramètres d'entrée » : l'utilisateur y cherche
    « InpDryRun » et n'y lit qu'une phrase en français. Préfixer le commentaire par
    le nom fait coïncider les deux.
    """
    import re

    source = mt5.EA_SOURCE.read_text(encoding="utf-8", errors="replace")
    motif = re.compile(r"^input\s+\w+\s+(\w+)\s*=\s*[^;]+;\s*//\s*(.*)$", re.M)

    declarations = motif.findall(source)
    assert declarations, "aucun paramètre d'entrée trouvé dans l'EA"

    sans_nom = [nom for nom, commentaire in declarations
                if not commentaire.strip().startswith(nom)]
    assert not sans_nom, (
        "Ces paramètres s'afficheront dans MetaTrader sans leur nom, "
        "et seront introuvables pour qui suit une consigne écrite :\n  "
        + "\n  ".join(sans_nom))

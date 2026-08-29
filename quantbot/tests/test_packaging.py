"""Tests d'intégrité du dépôt.

Ces tests ne vérifient pas une formule mathématique mais quelque chose de plus
bête et de plus dangereux : que le code présent sur *ma* machine est bien celui
que l'utilisateur télécharge. Un fichier oublié par `.gitignore` passe tous les
tests en local (le fichier est là) et casse à la première installation propre.

C'est exactement ce qui est arrivé : le motif `data/` de `.gitignore`, écrit
sans barre oblique initiale, correspond à *tout* dossier nommé `data`, à
n'importe quelle profondeur. Il devait masquer `quantbot/data/` (les cours
téléchargés, plusieurs mégaoctets) ; il masquait aussi `quantbot/qbot/data/`,
c'est-à-dire le module qui charge ces cours. Résultat : `ModuleNotFoundError:
No module named 'qbot.data'` chez l'utilisateur, et zéro erreur chez moi.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str:
    """Lance une commande git depuis la racine du projet."""
    return subprocess.run(
        ["git", *args],
        cwd=RACINE,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _depot_disponible() -> bool:
    try:
        _git("rev-parse", "--git-dir")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return True


besoin_git = pytest.mark.skipif(
    not _depot_disponible(), reason="hors d'un dépôt git (archive téléchargée)"
)


@besoin_git
def test_tout_le_code_source_est_suivi_par_git():
    """Chaque .py du paquet `qbot` et des tests doit être dans le dépôt.

    Sinon il n'arrive jamais chez l'utilisateur.
    """
    suivis = set(_git("ls-files").splitlines())
    manquants = []
    for dossier in ("qbot", "tests", "scripts"):
        for fichier in sorted((RACINE / dossier).rglob("*.py")):
            if "__pycache__" in fichier.parts:
                continue
            relatif = fichier.relative_to(RACINE).as_posix()
            if relatif not in suivis:
                manquants.append(relatif)

    assert not manquants, (
        "Ces fichiers existent sur le disque mais ne sont PAS dans le dépôt.\n"
        "L'utilisateur qui télécharge le projet ne les aura pas :\n  "
        + "\n  ".join(manquants)
        + "\n\nCause la plus probable : un motif de .gitignore sans barre oblique "
        "initiale (`data/` au lieu de `/data/`) qui attrape un sous-dossier."
    )


@besoin_git
def test_aucun_motif_gitignore_ne_masque_le_paquet():
    """Vérification directe : `git check-ignore` sur tout le paquet.

    Complémentaire du test précédent — celui-ci attrape aussi le cas d'un
    fichier déjà suivi mais qu'un nouveau motif rendrait invisible pour un
    contributeur qui repartirait de zéro.
    """
    fichiers = [
        f.relative_to(RACINE).as_posix()
        for f in sorted((RACINE / "qbot").rglob("*.py"))
        if "__pycache__" not in f.parts
    ]
    assert fichiers, "aucun fichier trouvé : chemin du dépôt incorrect ?"

    # check-ignore renvoie 1 quand rien n'est ignoré : ce n'est pas une erreur.
    res = subprocess.run(
        ["git", "check-ignore", "--verbose", *fichiers],
        cwd=RACINE,
        capture_output=True,
        text=True,
    )
    ignores = [ligne for ligne in res.stdout.splitlines() if ligne.strip()]
    assert not ignores, (
        "Des fichiers du paquet qbot sont ignorés par .gitignore :\n  "
        + "\n  ".join(ignores)
    )


def test_tous_les_sous_paquets_sont_importables():
    """Chaque dossier de `qbot` contenant un `__init__.py` doit s'importer.

    Ne dépend pas de git : ce test protège aussi contre un `__init__.py`
    manquant ou une dépendance circulaire.
    """
    import importlib

    paquet = RACINE / "qbot"
    noms = sorted(
        ".".join(init.parent.relative_to(RACINE).parts)
        for init in paquet.rglob("__init__.py")
        if "__pycache__" not in init.parts
    )
    assert "qbot.data" in noms, "le sous-paquet qbot.data a disparu du disque"

    erreurs = []
    for nom in noms:
        try:
            importlib.import_module(nom)
        except Exception as exc:  # noqa: BLE001 - on veut le rapport complet
            erreurs.append(f"{nom}: {type(exc).__name__}: {exc}")
    assert not erreurs, "Sous-paquets non importables :\n  " + "\n  ".join(erreurs)


# ---------------------------------------------------------------------------------------
# Vérificateur d'environnement
#
# Il tourne sur la machine de l'utilisateur au moment où quelque chose est déjà cassé :
# c'est le seul message qu'il verra. Il doit donc nommer le paquet fautif, montrer
# l'erreur réelle, et sortir avec un code non nul — sans jamais lever lui-même.
# ---------------------------------------------------------------------------------------
def _lancer_verificateur(pythonpath: str | None = None):
    env = dict(**__import__("os").environ)
    if pythonpath:
        env["PYTHONPATH"] = pythonpath
    return subprocess.run(
        [__import__("sys").executable, "scripts/verifier.py"],
        cwd=RACINE, capture_output=True, text=True, env=env,
    )


def test_le_verificateur_valide_un_environnement_sain():
    res = _lancer_verificateur()
    assert res.returncode == 0, res.stdout + res.stderr
    assert "Environnement complet" in res.stdout
    for paquet in ("numpy", "pandas", "scipy", "scikit-learn", "torch"):
        assert paquet in res.stdout, f"{paquet} absent du rapport"


def test_le_verificateur_nomme_le_paquet_fautif(tmp_path):
    """Un import cassé doit produire un diagnostic exploitable, pas une trace brute."""
    (tmp_path / "torch.py").write_text(
        'raise ImportError("DLL load failed while importing _C")\n', encoding="utf-8")

    res = _lancer_verificateur(pythonpath=str(tmp_path))
    assert res.returncode == 1, "un import cassé doit donner un code de retour non nul"
    assert "torch" in res.stdout
    assert "DLL load failed" in res.stdout, "l'erreur réelle doit être montrée"
    assert "pip install" in res.stdout, "la commande de réparation doit être proposée"
    assert "--force-reinstall" in res.stdout
    assert not res.stderr.strip(), (
        "le vérificateur ne doit rien écrire sur stderr : sous PowerShell avec "
        "$ErrorActionPreference='Stop', la moindre ligne y tuerait l'installeur.\n"
        + res.stderr)


def test_un_paquet_optionnel_absent_nest_pas_bloquant(tmp_path):
    """hmmlearn manque souvent (pas de roue pour les Python récents) : non bloquant."""
    (tmp_path / "hmmlearn.py").write_text(
        'raise ImportError("pas de roue pour cette version")\n', encoding="utf-8")

    res = _lancer_verificateur(pythonpath=str(tmp_path))
    assert res.returncode == 0, "un paquet optionnel ne doit pas faire échouer"
    assert "non bloquant" in res.stdout


def test_linstalleur_ne_redirige_plus_stderr_sans_protection():
    """Le défaut qui a masqué le diagnostic chez l'utilisateur ne doit pas revenir.

    `$ErrorActionPreference = "Stop"` transforme toute ligne écrite sur stderr par un
    programme externe en erreur fatale. Un `2>&1` non protégé tue donc l'installeur
    avant qu'il puisse afficher quoi que ce soit. Seul l'intérieur de la fonction
    `Executer`, qui suspend ce comportement, a le droit d'en contenir un.
    """
    lignes = (RACINE / "install.ps1").read_text(encoding="utf-8").splitlines()
    dans_executer = False
    fautives = []
    for i, ligne in enumerate(lignes, 1):
        if ligne.startswith("function Executer"):
            dans_executer = True
        elif dans_executer and ligne.startswith("}"):
            dans_executer = False
        elif ("2>&1" in ligne or "2>$null" in ligne) and not dans_executer:
            fautives.append(f"{i}: {ligne.strip()}")

    assert not fautives, (
        "Redirection de stderr hors de la fonction Executer :\n  " + "\n  ".join(fautives))

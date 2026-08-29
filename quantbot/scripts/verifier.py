#!/usr/bin/env python3
"""Vérifie que l'environnement Python est utilisable, et dit précisément ce qui manque.

    python scripts/verifier.py

Sort avec le code 0 si tout est importable, 1 sinon.

Pourquoi un script plutôt qu'un `python -c "import a,b,c"` dans l'installeur : un
import groupé s'arrête au premier échec et ne dit pas lequel des cinq paquets est en
cause. Pire, sa trace remonte à PowerShell sous forme d'erreur fatale et l'installeur
meurt avant d'avoir pu l'afficher — l'utilisateur voit une trace tronquée et aucune
piste. Ici chaque import est isolé, l'erreur est capturée, et le diagnostic est écrit
sur la sortie standard avec la commande exacte qui répare.
"""
from __future__ import annotations

import importlib
import platform
import sys

# (nom du paquet à installer, nom du module à importer, à quoi il sert)
NOYAU = [
    ("numpy", "numpy", "calcul numérique"),
    ("pandas", "pandas", "séries temporelles"),
    ("scipy", "scipy", "statistiques"),
    ("scikit-learn", "sklearn", "modèles classiques"),
    ("torch", "torch", "réseaux de neurones"),
    ("PyYAML", "yaml", "fichiers de configuration"),
]
OPTIONNELS = [
    ("hmmlearn", "hmmlearn", "détecteur de régime par HMM"),
]


# Windows peut bloquer une bibliothèque native APRÈS son installation. L'installation
# reussit, le fichier est bien sur le disque, et c'est le chargement qui est refusé —
# par Smart App Control (Windows 11), WDAC ou une stratégie AppLocker. Réinstaller n'y
# change rien : le même fichier sera reposé et rebloqué. Distinguer ce cas d'un paquet
# réellement manquant est indispensable, sinon on envoie l'utilisateur réinstaller en
# boucle une bibliothèque qui est déjà là.
_SIGNES_BLOCAGE_WINDOWS = (
    "contrôle d'application",       # « Une stratégie de contrôle d'application a bloqué… »
    "controle d'application",
    "application control",
    "blocked by group policy",
    "stratégie de groupe",
    "strategie de groupe",
    "0x800704ec",
)


def _normaliser(texte: str) -> str:
    """Minuscules, apostrophes et accents ramenés à une forme unique.

    Windows écrit « d’application » avec l’apostrophe typographique U+2019, pas
    l’apostrophe ASCII — une comparaison littérale échoue silencieusement. Les
    accents subissent le même sort selon l’encodage de la console, d’où leur
    suppression : on compare des chaînes réduites au plus petit dénominateur.
    """
    import unicodedata

    texte = texte.lower()
    for guillemet in ("’", "‘", "´", "ʼ"):
        texte = texte.replace(guillemet, "'")
    decompose = unicodedata.normalize("NFD", texte)
    return "".join(c for c in decompose if unicodedata.category(c) != "Mn")


def _est_bloque_par_windows(erreur: str) -> bool:
    """Un DLL refusé par une stratégie de sécurité, et non un paquet absent."""
    bas = _normaliser(erreur)
    if "dll load failed" not in bas:
        return False
    return any(_normaliser(signe) in bas for signe in _SIGNES_BLOCAGE_WINDOWS)


def _conseil_blocage_windows(paquets: list[str]) -> None:
    print("  Ce n'est PAS une installation ratée : le fichier est bien présent sur")
    print("  le disque. C'est Windows qui refuse de le charger — Smart App Control,")
    print("  WDAC ou une stratégie AppLocker bloque les bibliothèques natives peu")
    print("  répandues, et les versions récentes de scipy ou numpy en font partie.")
    print()
    print("  Réinstaller ne sert à rien : le même fichier sera reposé et rebloqué.")
    print()
    print("  Trois pistes, de la moins définitive à la plus définitive :")
    print()
    print("  1. Essayer une version plus ancienne, mieux établie et donc reconnue :")
    print(f"        python -m pip install \"scipy==1.16.3\"")
    print("     Deux minutes, rien de cassé si ça ne marche pas.")
    print()
    print("  2. Installer Python 3.12 en parallèle. Les bibliothèques y sont")
    print("     déployées depuis des années et Windows les connaît :")
    print("        https://www.python.org/downloads/release/python-3128/")
    print("     (cocher « Add Python to PATH »)")
    print()
    print("  3. Désactiver Smart App Control :")
    print("        Sécurité Windows > Contrôle des applications et du navigateur")
    print("        > Paramètres de Smart App Control > Désactivé")
    print("     ATTENTION : une fois désactivé, il ne peut PLUS être réactivé sans")
    print("     réinstaller Windows. À ne faire qu'en dernier recours.")
    print()
    print("  Pour confirmer que c'est bien lui : Observateur d'événements >")
    print("  Journaux des applications et des services > Microsoft > Windows >")
    print("  CodeIntegrity > Operational.")
    print()


def _version(module) -> str:
    return str(getattr(module, "__version__", "?"))


def main() -> int:
    print(f"Python {platform.python_version()} ({platform.machine()})")
    print()

    echecs: list[tuple[str, str, str]] = []
    for paquet, nom_module, role in NOYAU:
        try:
            mod = importlib.import_module(nom_module)
        except Exception as exc:  # noqa: BLE001 — on veut le diagnostic complet
            echecs.append((paquet, role, f"{type(exc).__name__}: {exc}"))
            print(f"  [X]  {paquet:<14} {role}")
        else:
            print(f"  [ok] {paquet:<14} {_version(mod)}")

    for paquet, nom_module, role in OPTIONNELS:
        try:
            mod = importlib.import_module(nom_module)
        except Exception:  # noqa: BLE001
            print(f"  [-]  {paquet:<14} absent — {role} inactif, non bloquant")
        else:
            print(f"  [ok] {paquet:<14} {_version(mod)}")

    if not echecs:
        print()
        print("Environnement complet.")
        return 0

    print()
    print("=" * 68)
    print("  CE QUI NE VA PAS")
    print("=" * 68)
    for paquet, role, erreur in echecs:
        print(f"\n  {paquet} ({role})")
        for ligne in str(erreur).splitlines():
            print(f"      {ligne}")

    print()
    bloques = [p for p, _, err in echecs if _est_bloque_par_windows(err)]
    if bloques:
        _conseil_blocage_windows(bloques)
        return 1

    print("  Réparation — copiez cette commande :")
    print()
    manquants = " ".join(p for p, _, _ in echecs)
    print(f"      python -m pip install --force-reinstall --no-cache-dir {manquants}")
    print()

    # Une version de Python parue depuis peu est la cause la plus fréquente : les
    # paquets à extension C (numpy, scipy, torch) publient leurs versions
    # précompilées avec plusieurs mois de retard, et pip se rabat alors sur une
    # compilation depuis les sources qui échoue faute d'outils.
    majeur, mineur = sys.version_info[:2]
    if (majeur, mineur) >= (3, 14):
        print("  Si la réparation échoue aussi : votre Python est très récent")
        print(f"  ({majeur}.{mineur}). Les paquets précompilés suivent avec quelques")
        print("  mois. Installer Python 3.12 en parallèle règle le problème :")
        print("      https://www.python.org/downloads/release/python-3128/")
        print("  (cocher « Add Python to PATH »)")
        print()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

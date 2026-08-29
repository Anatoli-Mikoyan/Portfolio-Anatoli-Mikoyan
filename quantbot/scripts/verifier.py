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

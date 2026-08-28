#!/usr/bin/env bash
# ======================================================================
#  QBot — installation et premier lancement, sur macOS et Linux.
#      bash demarrer.sh
# ======================================================================
set -euo pipefail
cd "$(dirname "$0")"

echo
echo "======================================================================"
echo "  QBOT — INSTALLATION ET PREMIER LANCEMENT"
echo "======================================================================"
echo

PY=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "[X] Python n'est pas installé."
  echo "    macOS  : brew install python3"
  echo "    Ubuntu : sudo apt install python3 python3-pip"
  exit 1
fi
echo "[1/3] $($PY --version) détecté."

echo "[2/3] Installation des bibliothèques (quelques minutes la première fois)…"
SILENCE="--quiet --no-warn-script-location"
"$PY" -m pip install $SILENCE --upgrade pip

# En deux temps : le noyau doit réussir, hmmlearn est optionnel. C'est une extension C
# dont les paquets précompilés arrivent tard après chaque version de Python, et il ne
# sert qu'au détecteur de régime par HMM.
if ! "$PY" -m pip install $SILENCE numpy'>=1.24' pandas'>=2.0' scipy'>=1.10' \
        torch'>=2.0' scikit-learn'>=1.3' PyYAML'>=6.0' pytest'>=7.4'; then
  echo "[X] L'installation des bibliothèques essentielles a échoué."
  echo "    Si torch refuse de s'installer, votre version de Python est peut-être trop"
  echo "    récente : les paquets précompilés de PyTorch suivent avec quelques mois."
  exit 1
fi

if ! "$PY" -m pip install $SILENCE hmmlearn'>=0.3' >/dev/null 2>&1; then
  echo "      [!] hmmlearn indisponible — non bloquant (seul le HMM de régime est inactif)."
fi

if ! "$PY" -c "import numpy,pandas,scipy,sklearn,torch" 2>/dev/null; then
  echo "[X] Les bibliothèques ne s'importent pas. Installation incomplète."
  exit 1
fi
echo "      Terminé et vérifié."

echo "[3/3] Lancement de l'analyse complète."
echo "      Comptez 10 à 20 minutes. Le rapport s'ouvrira tout seul."
echo
"$PY" scripts/start.py "$@"

echo
echo "======================================================================"
echo "  Terminé. Le rapport HTML : runs/start/rapport.html"
echo "======================================================================"

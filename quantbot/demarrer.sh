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
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet -r requirements.txt
echo "      Terminé."

echo "[3/3] Lancement de l'analyse complète."
echo "      Comptez 10 à 20 minutes. Le rapport s'ouvrira tout seul."
echo
"$PY" scripts/start.py "$@"

echo
echo "======================================================================"
echo "  Terminé. Le rapport HTML : runs/start/rapport.html"
echo "======================================================================"

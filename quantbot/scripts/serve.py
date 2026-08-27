#!/usr/bin/env python3
"""Démarre le serveur d'inférence pour l'Expert Advisor MetaTrader 5.

    python scripts/serve.py --model runs/eurusd_v1 --port 8912
    python scripts/serve.py --model runs/eurusd_v1 --live      # désactive le dry-run

Par défaut le serveur est en DRY-RUN : il répond normalement mais n'autorise jamais
l'ouverture ni le renforcement d'une position (la réduction et la fermeture restent
permises, pour ne pas piéger un compte déjà exposé). Passer en réel demande un
`--live` explicite : aucune configuration ne doit pouvoir armer le trading par accident.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qbot.config import LiveConfig
from qbot.live import serve
from qbot.utils.logging import configure_logging, get_logger

log = get_logger("scripts.serve")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=str, required=True, help="Répertoire du modèle exporté")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8912)
    p.add_argument("--live", action="store_true", help="Désactive le dry-run (TRADING RÉEL)")
    p.add_argument("--cvar", type=float, default=None,
                   help="Politique averse au risque : maximise le CVaR à ce niveau (ex. 0.1)")
    p.add_argument("--replay", action="store_true",
                   help="Répétition générale sur barres passées : neutralise le SEUL "
                        "contrôle de fraîcheur du flux. Jamais sur un compte réel.")
    p.add_argument("--log", type=str, default=None, help="Fichier de journal")
    args = p.parse_args()

    configure_logging(logging.INFO, logfile=args.log)

    if args.live:
        log.warning("=" * 66)
        log.warning("MODE RÉEL ACTIVÉ — le serveur autorisera de vraies ouvertures.")
        log.warning("Vérifier : compte de démonstration ? capital engagé acceptable ?")
        log.warning("=" * 66)
    if args.host not in ("127.0.0.1", "localhost"):
        log.warning("Le serveur écoute sur %s : n'exposez JAMAIS ce port sur Internet, "
                    "il pilote directement les positions du compte.", args.host)

    cfg = LiveConfig(host=args.host, port=args.port, model_path=args.model, dry_run=not args.live)
    serve(args.model, cfg, cvar_alpha=args.cvar, block=True, replay=args.replay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

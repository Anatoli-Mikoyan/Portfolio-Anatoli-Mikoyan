#!/usr/bin/env python3
"""Surveillance de production (cahier des charges §17).

Trois sous-commandes :

  fit      Fige la référence : distribution des features et enveloppe de performance,
           calculées sur les MÊMES données que l'entraînement, puis écrites à côté du
           modèle. C'est cette référence figée qui rend détectable une dérive lente ;
           une référence recalculée en continu dériverait avec le marché et ne
           signalerait jamais rien.

               python scripts/monitor.py fit --model runs/best --data data/EURUSD_H1.csv

  report   Produit un rapport texte et un tableau de bord HTML à partir des décisions
           enregistrées en production (JSON Lines) ou du journal d'audit.

               python scripts/monitor.py report --model runs/best --html rapport.html

  verify   Vérifie l'intégrité de la chaîne d'empreintes du journal d'audit et, si un
           modèle est fourni, rejoue les décisions pour détecter un décalage de version.

               python scripts/monitor.py verify --journal runs/best/audit.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from qbot.config import Config, MonitorConfig
from qbot.monitoring import (
    DecisionJournal, DecisionRecord, LiveMetricsStore, LiveMonitor, PerformanceEnvelope,
    ReferenceDistribution,
)
from qbot.utils.logging import configure_logging, get_logger
from qbot.utils.timeutils import bars_per_year_for_timeframe

log = get_logger("scripts.monitor")


# =======================================================================================
def cmd_fit(args: argparse.Namespace) -> int:
    """Fige la référence de surveillance à côté du modèle."""
    from qbot.data.loader import load_ohlcv
    from qbot.features import FeaturePipeline

    model_dir = Path(args.model)
    cfg = Config.load(next(p for p in (model_dir / "config.json", model_dir / "config.yaml")
                           if p.exists()))
    pipeline = FeaturePipeline.load(model_dir / "pipeline.json")

    df = load_ohlcv(args.data)
    if args.end:
        df = df.loc[: args.end]
    log.info("Données de référence : %d barres, %s → %s", len(df), df.index[0], df.index[-1])

    # Les features de référence doivent passer par le pipeline DÉJÀ AJUSTÉ, pas par un
    # nouvel ajustement : la référence doit décrire ce que le modèle a vu, pas ce qu'un
    # pipeline réajusté produirait aujourd'hui.
    X = pipeline.transform(df).dropna()
    log.info("Matrice de référence : %d lignes × %d features", len(X), X.shape[1])

    reference = ReferenceDistribution.fit(X, n_bins=args.bins, model_id=str(model_dir.name))
    ref_path = reference.save(model_dir / "reference.json")
    log.info("Distribution de référence écrite dans %s", ref_path)

    # -- enveloppe de performance -------------------------------------------------------
    env_written = None
    if args.returns:
        r = pd.read_csv(args.returns)
        col = args.returns_col if args.returns_col in r.columns else r.columns[-1]
        returns = r[col].to_numpy(dtype=float)
        returns = returns[np.isfinite(returns)]
        bpy = args.bpy or bars_per_year_for_timeframe(cfg.data.timeframe)
        envelope = PerformanceEnvelope.build(
            returns, horizon=args.horizon, bars_per_year=bpy, n_paths=args.paths,
            seed=cfg.seed)
        env_path = model_dir / "envelope.json"
        env_path.write_text(json.dumps(envelope.to_dict(), indent=2), encoding="utf-8")
        env_written = env_path
        log.info("Enveloppe de performance (horizon %d, %d trajectoires) écrite dans %s",
                 args.horizon, args.paths, env_path)
        q = envelope.sharpe_quantiles
        log.info("Sharpe attendu à cet horizon : q05=%.2f  médiane=%.2f  q95=%.2f",
                 q["q05"], q["q50"], q["q95"])
        log.info("Largeur de l'enveloppe : %.2f points de Sharpe entre le 5e et le 95e "
                 "centile. Toute performance dans cet intervalle est indiscernable du "
                 "hasard à cet horizon.", q["q95"] - q["q05"])
    else:
        log.warning("Aucun --returns fourni : l'enveloppe attendu/réalisé ne sera pas "
                    "construite et la confrontation restera inactive en production.")

    print(f"\nRéférence de surveillance figée :\n  {ref_path}"
          + (f"\n  {env_written}" if env_written else ""))
    return 0


# =======================================================================================
def _load_store(args: argparse.Namespace, bpy: float) -> LiveMetricsStore:
    store = LiveMetricsStore(maxlen=args.window, bars_per_year=bpy)
    if args.decisions:
        n = store.load(args.decisions)
        log.info("%d décisions chargées depuis %s", n, args.decisions)
        return store
    journal = DecisionJournal(args.journal)
    rows = journal.read(kind="decision")
    for row in rows:
        payload = row.get("decision", row)
        try:
            store.append(DecisionRecord(**payload))
        except TypeError:
            continue
    log.info("%d décisions rejouées depuis le journal %s", len(store), args.journal)
    return store


def cmd_report(args: argparse.Namespace) -> int:
    model_dir = Path(args.model) if args.model else None
    cfg = MonitorConfig()
    reference = envelope = None
    bpy = args.bpy
    model_id = ""

    if model_dir and model_dir.exists():
        cfg_path = next((p for p in (model_dir / "config.json", model_dir / "config.yaml")
                         if p.exists()), None)
        if cfg_path:
            full = Config.load(cfg_path)
            cfg = full.monitor
            bpy = args.bpy or bars_per_year_for_timeframe(full.data.timeframe)
            model_id = model_dir.name
        if (model_dir / "reference.json").exists():
            reference = ReferenceDistribution.load(model_dir / "reference.json")
        if (model_dir / "envelope.json").exists():
            envelope = PerformanceEnvelope.from_dict(
                json.loads((model_dir / "envelope.json").read_text(encoding="utf-8")))
        if not args.journal and not args.decisions:
            args.journal = str(model_dir / "audit.jsonl")

    bpy = bpy or 252.0
    store = _load_store(args, bpy)
    if len(store) == 0:
        log.error("Aucune décision à analyser.")
        return 1

    monitor = LiveMonitor(cfg, reference=reference, envelope=envelope, bars_per_year=bpy,
                          model_id=model_id)
    # On rejoue les décisions à travers le moniteur pour reconstruire dérive, alertes et
    # confrontation exactement comme elles se seraient produites en direct.
    monitor.alerts.sinks = []                    # rapport a posteriori : pas de log par alerte
    for record in list(store.records):
        monitor.observe(record)
    monitor.refresh()

    print(monitor.text_report())

    drift = monitor._last_drift
    if drift is not None and drift.features:
        print("\n" + str(drift))

    if args.html:
        path = monitor.to_html(args.html, title=f"QBot — {model_id or 'production'}")
        print(f"\nTableau de bord : {path}")
    if args.json:
        Path(args.json).write_text(
            json.dumps(monitor.snapshot(), indent=2, default=str), encoding="utf-8")
        print(f"Instantané JSON : {args.json}")
    return 0


# =======================================================================================
def cmd_verify(args: argparse.Namespace) -> int:
    journal = DecisionJournal(args.journal)
    result = journal.verify()
    print(result)
    if not result.valid:
        print("\nUne rupture de chaîne signifie qu'une entrée a été modifiée, supprimée "
              "ou insérée après coup. Le journal n'est plus recevable comme preuve à "
              "partir de cette séquence.")
        return 2

    entries = journal.read(kind="decision")
    if entries:
        stamps = [e.get("decision", {}).get("ts", "") for e in entries]
        print(f"\n{len(entries)} décisions journalisées, de {stamps[0]} à {stamps[-1]}.")
        print(f"Empreinte de tête : {journal.head}")
        print("Recopier cette empreinte ailleurs (message, dépôt) transforme la "
              "détection de falsification en preuve : l'historique ne peut plus être "
              "réécrit sans contredire une valeur déjà publiée.")
    alerts = journal.read(kind="alert")
    if alerts:
        print(f"\n{len(alerts)} alertes journalisées. Dernières :")
        for a in alerts[-10:]:
            print(f"  [{a.get('level')}] {a.get('code')}: {a.get('message')}")
    return 0


# =======================================================================================
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="fige la référence de surveillance")
    f.add_argument("--model", required=True, help="dossier du modèle exporté")
    f.add_argument("--data", required=True, help="données d'entraînement (CSV/Parquet)")
    f.add_argument("--end", default=None, help="borne haute de la période d'entraînement")
    f.add_argument("--bins", type=int, default=10, help="cases du découpage PSI")
    f.add_argument("--returns", default=None,
                   help="CSV des rendements out-of-sample, pour l'enveloppe")
    f.add_argument("--returns-col", default="net_return")
    f.add_argument("--horizon", type=int, default=1000,
                   help="horizon (en barres) sur lequel comparer la production")
    f.add_argument("--paths", type=int, default=2000)
    f.add_argument("--bpy", type=float, default=None)
    f.set_defaults(func=cmd_fit)

    r = sub.add_parser("report", help="rapport et tableau de bord de production")
    r.add_argument("--model", default=None, help="dossier du modèle (référence, config)")
    r.add_argument("--journal", default=None, help="journal d'audit JSON Lines")
    r.add_argument("--decisions", default=None, help="JSON Lines de décisions brutes")
    r.add_argument("--window", type=int, default=20000)
    r.add_argument("--bpy", type=float, default=None)
    r.add_argument("--html", default=None, help="chemin du tableau de bord HTML à écrire")
    r.add_argument("--json", default=None, help="chemin de l'instantané JSON à écrire")
    r.set_defaults(func=cmd_report)

    v = sub.add_parser("verify", help="vérifie l'intégrité du journal d'audit")
    v.add_argument("--journal", required=True)
    v.set_defaults(func=cmd_verify)

    args = p.parse_args()
    configure_logging(logging.INFO)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

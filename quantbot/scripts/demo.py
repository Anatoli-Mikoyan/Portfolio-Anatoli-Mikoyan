#!/usr/bin/env python3
"""Démonstration de bout en bout sur marché synthétique (quelques minutes).

Enchaîne : génération de données -> features -> entraînement RL -> backtest ->
validation statistique -> pont live. Sert de test d'intégration exécutable et de
point d'entrée pour comprendre le système.

    python scripts/demo.py               # rapide (~3 min)
    python scripts/demo.py --full        # complet avec walk-forward (~20 min)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from qbot.backtest import momentum_positions, random_positions, run_backtest, sharpe_ratio
from qbot.config import (
    AgentConfig, Config, CostConfig, EnvConfig, FeatureConfig, TrainConfig,
)
from qbot.data.synthetic import RegimeSwitchingGBM, generate_synthetic_ohlcv
from qbot.diagnostics import signal_report
from qbot.experiment import backtest_model, train_model
from qbot.features import FeaturePipeline
from qbot.utils.logging import configure_logging, get_logger
from qbot.validation import bootstrap_metric, monte_carlo_drawdown, train_valid_test_split

log = get_logger("demo")
BPY = 6240.0


def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--full", action="store_true")
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--bars", type=int, default=30_000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=str, default="runs/demo")
    args = p.parse_args()
    configure_logging(logging.INFO)

    # ---------------------------------------------------------------------------------
    banner("1/7  DONNÉES — marché synthétique à régimes, avec signal exploitable")
    market = RegimeSwitchingGBM(
        mu=(0.10, -0.08, 0.0), sigma=(0.06, 0.18, 0.10),
        persistence=0.997, autocorr=0.25, t_df=5.0,
    )
    df = generate_synthetic_ohlcv(n=args.bars, seed=101, model=market,
                                  spread_bps=0.6).drop(columns=["regime"])
    rets = np.log(df["close"]).diff()
    print(f"  {len(df)} barres H1 | volatilité annualisée {rets.std() * np.sqrt(BPY):.1%}")
    print(f"  autocorrélation lag-1 = {rets.autocorr(1):+.3f}  <- le signal est là PAR CONSTRUCTION")
    print(f"  kurtosis = {rets.kurtosis():.2f} (queues épaisses, comme un vrai marché)")

    # ---------------------------------------------------------------------------------
    banner("2/7  BORNE SUPÉRIEURE — que rapporterait un oracle qui connaît le signal ?")
    tr, va, te = train_valid_test_split(df.index, 0.6, 0.2, 0.01)
    train_df, valid_df, test_df = df.iloc[tr], df.iloc[va], df.iloc[te]
    costs = CostConfig(spread_bps=0.6, commission_bps=0.1, slippage_coef=0.05, min_trade_size=0.05)
    env_eval = EnvConfig(vol_target=None)

    oracle = np.sign(np.log(test_df["close"]).diff()).fillna(0.0).to_numpy()
    r_free = run_backtest(oracle, test_df, CostConfig(spread_bps=0, commission_bps=0,
                                                     slippage_model="none", min_trade_size=0.0),
                          env_eval, BPY)
    r_cost = run_backtest(oracle, test_df, costs, env_eval, BPY)
    print(f"  oracle sans coûts   : sharpe {r_free.report.sharpe:+.2f}")
    print(f"  oracle AVEC coûts   : sharpe {r_cost.report.sharpe:+.2f}  <- plafond réaliste")

    # ---------------------------------------------------------------------------------
    banner("3/7  SONDE DE SIGNAL — à faire AVANT de lancer un entraînement RL")
    probe_cfg = FeatureConfig(returns_windows=(1, 2, 5, 10, 20), vol_windows=(10, 20),
                              ema_windows=(10, 30), use_microstructure=False,
                              use_calendar=False, scaler_window=300)
    probe_features = FeaturePipeline(probe_cfg).fit_transform(train_df)
    print("  Plancher = régression linéaire ; plafond = le réseau du Rainbow en supervisé.")
    print("  Si le réseau passe sous le plancher, c'est la capacité du modèle qu'il faut")
    print("  réduire — pas l'algorithme qu'il faut changer.")
    signal_report(probe_features, train_df.loc[probe_features.index],
                  windows=(4, 16), steps=2_000)

    # ---------------------------------------------------------------------------------
    banner("4/7  ENTRAÎNEMENT — Rainbow QR-DQN + Munchausen")
    cfg = Config()
    cfg.seed = args.seed
    cfg.costs = costs
    cfg.env = EnvConfig(window=16, positions=(-1.0, 0.0, 1.0), reward="dsr",
                        episode_length=2048, vol_target=None, max_drawdown_stop=0.35)
    cfg.features = FeatureConfig(returns_windows=(1, 2, 5, 10, 20), vol_windows=(10, 20),
                                 ema_windows=(10, 30), use_microstructure=False,
                                 use_calendar=False, scaler_window=300)
    # Réseau petit et régularisé : voir la sonde ci-dessus et docs/METHODOLOGIE.md §6.
    cfg.agent = AgentConfig(hidden_sizes=(64, 64), n_quantiles=32, buffer_size=100_000,
                            learn_start=2_000, batch_size=64, target_update_interval=1_000,
                            lr=3e-4, n_step=3, weight_decay=1e-4)
    cfg.train = TrainConfig(total_steps=args.steps, eval_every=max(args.steps // 8, 1_000),
                            early_stop_patience=None, log_every=max(args.steps // 4, 1_000))

    t0 = time.time()
    model = train_model(cfg, train_df, valid_df, BPY)
    print(f"  entraîné en {time.time() - t0:.0f}s | sharpe validation = {model.valid_sharpe:+.3f}")
    model.save(args.out)

    # ---------------------------------------------------------------------------------
    banner("5/7  TEST OUT-OF-SAMPLE — segment jamais vu, touché une seule fois")
    ctx = df.loc[: test_df.index[0]].iloc[:-1]
    result, positions = backtest_model(model, test_df, context_df=ctx, bpy=BPY, n_trials=1)
    print(result.report)

    print("\n  Références obligatoires sur le même segment :")
    for name, pos in [
        ("buy & hold", np.ones(len(test_df))),
        ("aléatoire", random_positions(len(test_df), args.seed)),
        ("momentum 20", momentum_positions(test_df["close"], 20)),
    ]:
        r = run_backtest(pos, test_df, costs, cfg.env, BPY)
        print(f"    {name:<14} sharpe {r.report.sharpe:+7.3f} | CAGR {100 * r.report.cagr:+7.2f}%")

    # ---------------------------------------------------------------------------------
    banner("6/7  ROBUSTESSE — la performance survit-elle au rééchantillonnage ?")
    oos = result.returns
    boot = bootstrap_metric(oos, lambda x: sharpe_ratio(x, BPY), 1_000, 20, 0, "sharpe")
    print(f"  {boot}")
    mc = monte_carlo_drawdown(oos, 1_000, 20, 0)
    print(f"  drawdown : observé {mc['observed']:.2%} | pire 5% des scénarios {mc['p95_worst']:.2%}")
    print(f"  Deflated Sharpe = {result.report.deflated_sharpe:.4f} "
          f"({'crédible' if result.report.deflated_sharpe > 0.95 else 'NON crédible'})")

    # ---------------------------------------------------------------------------------
    banner("7/7  PONT LIVE — le modèle exporté répond-il comme en backtest ?")
    _demo_live(args.out, df)

    banner("TERMINÉ")
    print(f"  Modèle exporté dans : {args.out}")
    print( "  Étapes suivantes    : scripts/walkforward.py puis scripts/validate.py")
    print( "  Ne JAMAIS déployer sur la seule base d'un backtest sur données synthétiques.")
    return 0


def _demo_live(model_dir: str, df: pd.DataFrame) -> None:
    """Démarre le serveur, envoie une requête réelle, affiche la décision."""
    from qbot.config import LiveConfig
    from qbot.live import SimpleClient, serve

    cfg = LiveConfig(host="127.0.0.1", port=8931, dry_run=True)
    server = serve(model_dir, cfg, block=False)
    try:
        with SimpleClient("127.0.0.1", 8931) as client:
            info = client.request({"type": "info"})
            print(f"  serveur prêt : {info['model_id']}, {info['n_features']} features, "
                  f"{info['min_bars']} barres requises")

            tail = df.iloc[-info["min_bars"]:]
            bars = [[int(ts.timestamp()), float(r.open), float(r.high), float(r.low),
                     float(r.close), float(r.volume), float(r.spread)]
                    for ts, r in tail.iterrows()]
            resp = client.request({
                "type": "predict", "symbol": "EURUSD", "timeframe": "H1", "bars": bars,
                "equity": 10_000.0, "balance": 10_000.0, "current_exposure": 0.0,
            })
            print(f"  décision   : exposition={resp['target_exposure']:+.3f} | "
                  f"action={resp['action']} | confiance={resp['confidence']:.2f} | "
                  f"statut={resp['status']} | latence={resp['latency_ms']:.1f} ms")
            if resp.get("reasons"):
                print(f"  motifs     : {', '.join(resp['reasons'])}")
            print(f"  Q-valeurs  : {resp['q_values']}")
            print(f"  CVaR 10%   : {resp['cvar']}")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())

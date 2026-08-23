"""Etape 1 : chargement, ajustement point-in-time, protection anti-look-ahead.

Execution :

    python examples/step1_data.py                      # source synthetique, hors ligne
    python examples/step1_data.py --config configs/data.yaml --symbol AAPL

Le script est volontairement bavard : il montre ce que le moteur voit a
chaque instant, et pourquoi cela differe de ce qu'un backtest naif verrait.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from quant_engine import configure_logging
from quant_engine.data import AdjustmentPolicy, DataLoader, Field
from quant_engine.data.types import UTC

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "data.offline.yaml"))
    parser.add_argument("--symbol", default="SYNTH")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2023-12-31")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    configure_logging(args.log_level, structured=False)

    loader = DataLoader.from_config(args.config)
    data = loader.load(
        args.symbol,
        datetime.fromisoformat(args.start).replace(tzinfo=UTC),
        datetime.fromisoformat(args.end).replace(tzinfo=UTC),
    )

    print("=" * 78)
    print(f"  {data}")
    print("=" * 78)

    # -- 1. Rapport qualite ------------------------------------------------
    print("\n[1] CONTROLE QUALITE")
    if data.quality is None:
        print("    aucun rapport")
    else:
        print("    " + data.quality.summary().replace("\n", "\n    "))

    # -- 2. Operations sur titre -------------------------------------------
    print("\n[2] OPERATIONS SUR TITRE")
    actions = data.actions
    print(f"    splits     : {len(actions.splits)}")
    print(f"    dividendes : {len(actions.dividends)}")
    for split in actions.splits[:5]:
        print(f"      split {split.ratio:g}-pour-1 le {split.ex_date.date()}")

    # -- 3. Ajustement point-in-time vs retro-ajustement -------------------
    print("\n[3] AJUSTEMENT : POINT-IN-TIME vs RETRO")
    pit = data.multipliers(AdjustmentPolicy.SPLIT_PIT)
    retro = data.multipliers(AdjustmentPolicy.FULL_RETRO_SPLIT, allow_lookahead=True)
    milestones = [0, len(data) // 4, len(data) // 2, len(data) - 1]
    print(f"    {'date':<12} {'prix cote':>12} {'vu en PIT':>12} {'vu en retro':>12}")
    for index in milestones:
        brut = float(data.raw(Field.CLOSE)[index])
        vue_pit = float(data.view_at(index, pit).last())
        vue_retro = float(data.view_at(index, retro).last())
        stamp = data.execution_bar(index).timestamp.date()
        print(f"    {stamp!s:<12} {brut:>12.2f} {vue_pit:>12.2f} {vue_retro:>12.2f}")
    if not np.allclose(
        [float(data.view_at(i, pit).last()) for i in milestones],
        [float(data.view_at(i, retro).last()) for i in milestones],
    ):
        print("    -> le retro-ajustement montre des prix qui n'ont jamais cote")

    # -- 4. Parcours evenementiel ------------------------------------------
    print("\n[4] PARCOURS EVENEMENTIEL (5 derniers points de decision)")
    points = list(data.cursor(AdjustmentPolicy.SPLIT_PIT, warmup=20))
    print(f"    {len(points)} points de decision (20 barres de warmup)")
    for point in points[-5:]:
        view = point.history
        moyenne = float(view.close(20).mean())
        print(
            f"    {view.as_of.date()} | {view.n_bars:>5} barres visibles "
            f"| close {view.last():8.2f} | MM20 {moyenne:8.2f}"
        )

    # -- 5. La borne est physique ------------------------------------------
    print("\n[5] TENTATIVES D'ACCES AU FUTUR")
    view = points[-1].history
    for label, action in (
        ("view.bar(-1)", lambda: view.bar(-1)),
        ("view.close(999999)", lambda: view.close(999_999)),
        ("view.close()[n_bars]", lambda: view.close()[view.n_bars]),
    ):
        try:
            action()
        except Exception as exc:  # on veut precisement montrer le type leve
            print(f"    {label:<24} -> {type(exc).__name__}: {str(exc).splitlines()[0][:60]}")
        else:
            print(f"    {label:<24} -> AUCUNE ERREUR (probleme !)")

    # -- 6. Equivalence par troncature -------------------------------------
    print("\n[6] EQUIVALENCE PAR TRONCATURE")
    cursor_index = len(data) // 2
    poisoned = data.with_future_poisoned(cursor_index + 1)
    reference = data.view_at(cursor_index, pit).close()
    suspect = poisoned.view_at(cursor_index, poisoned.multipliers(AdjustmentPolicy.SPLIT_PIT))
    identical = np.array_equal(reference, suspect.close())
    finite = bool(np.isfinite(suspect.close()).all())
    print(f"    futur remplace par des NaN a partir de l'index {cursor_index + 1}")
    print(f"    vue identique   : {identical}")
    print(f"    aucun NaN infiltre : {finite}")
    print("\n" + "=" * 78)
    return 0 if identical and finite else 1


if __name__ == "__main__":
    raise SystemExit(main())

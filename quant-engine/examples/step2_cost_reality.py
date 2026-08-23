"""Etape 2 : ce que les frais font reellement a un bot de trading.

Deux demonstrations, toutes deux chiffrees par le moteur, aucune n'etant une
opinion.

A. **La taille du compte.** La meme strategie, sur la meme serie de prix, avec
   100 EUR puis avec 100 000 EUR. Seule la taille du compte change.

B. **Un marche sans predictibilite.** Une marche aleatoire sans tendance --
   ce a quoi ressemble un marche pour quelqu'un qui n'a aucun avantage
   informationnel. On repete sur 20 series differentes pour que le resultat ne
   depende pas d'un tirage chanceux.

Execution :
    PYTHONPATH=src python examples/step2_cost_reality.py
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from quant_engine import configure_logging
from quant_engine.backtest import BacktestEngine, CostModel, ExecutionConfig
from quant_engine.data import (
    MarketData,
    NormalizationPolicy,
    QualityPolicy,
    SyntheticProvider,
    SyntheticSpec,
    normalize,
)
from quant_engine.data.providers.base import DataRequest
from quant_engine.data.types import UTC, Frequency
from quant_engine.strategy import (
    BollingerMeanReversion,
    BuyAndHold,
    MovingAverageCrossover,
    Strategy,
)

START = datetime(2015, 1, 1, tzinfo=UTC)
END = datetime(2023, 12, 31, tzinfo=UTC)
LENIENT = NormalizationPolicy(raise_on_blocking=False, quality=QualityPolicy(min_bars=1))
RULE = "-" * 78


def build_series(spec: SyntheticSpec) -> MarketData:
    raw = SyntheticProvider(spec).fetch(
        DataRequest(symbol="MARCHE", frequency=Frequency.DAY_1, start=START, end=END)
    )
    return normalize(raw, LENIENT, now=datetime(2030, 1, 1, tzinfo=UTC))


def strategies() -> list[Strategy]:
    return [BuyAndHold(), MovingAverageCrossover(fast=50, slow=200), BollingerMeanReversion()]


def demo_taille_du_compte() -> None:
    print(RULE)
    print("A. LA MEME STRATEGIE, DEUX TAILLES DE COMPTE")
    print(RULE)
    print(
        "Meme serie de prix, memes regles, memes decisions. Seul le capital\n"
        "de depart change. Courtier : Interactive Brokers, actions US.\n"
    )
    # Titre a 20 EUR : avec 100 EUR on peut acheter 5 actions. Avec un titre a
    # 110 EUR on ne pourrait en acheter aucune -- le bot ne passerait jamais le
    # moindre ordre, ce qui est deja une reponse en soi.
    data = build_series(
        SyntheticSpec(seed=1234, annual_drift=0.07, annual_volatility=0.18, start_price=20.0)
    )
    costs = CostModel.interactive_brokers_us_equity()

    for capital in (100.0, 1_000.0, 10_000.0, 100_000.0):
        round_trip = costs.round_trip_cost_pct(capital)
        print(f"  Capital {capital:>10,.0f} EUR   "
              f"| cout d'un aller-retour : {round_trip:>6.2%}")
    print("\n  La courbe est en U : sur les petits comptes la commission plancher")
    print("  ecrase tout ; sur les tres gros, c'est l'ordre lui-meme qui deplace le")
    print("  marche contre soi. L'optimum se situe entre les deux.\n")

    header = (f"  {'strategie':<26} {'capital':>10} {'final':>12} "
              f"{'perf':>9} {'frais':>9} {'trades':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for capital in (100.0, 100_000.0):
        engine = BacktestEngine(
            costs,
            initial_capital=capital,
            execution=ExecutionConfig(allow_fractional_units=False),
        )
        for strategy in strategies():
            result = engine.run(strategy, data, start_index=200)
            print(
                f"  {strategy.name:<26} {capital:>10,.0f} {result.final_equity:>12,.2f} "
                f"{result.total_return:>+9.1%} {result.cost_drag_pct:>9.1%} "
                f"{result.n_trades:>7}"
            )
        print()

    print("  Lecture : a 100 EUR, la commission plancher de 1 EUR represente 1 % du")
    print("  compte a chaque ordre. Le meme bot, avec exactement les memes signaux,")
    print("  detruit le capital d'un cote et survit de l'autre. La strategie n'y est")
    print("  pour rien : c'est de l'arithmetique.")
    print()


def demo_marche_sans_edge(n_seeds: int = 20) -> None:
    print(RULE)
    print(f"B. UN MARCHE SANS PREDICTIBILITE ({n_seeds} SERIES DIFFERENTES)")
    print(RULE)
    print(
        "Marche aleatoire sans tendance : chaque jour monte ou descend au hasard.\n"
        "C'est ce a quoi ressemble un marche pour qui n'a aucun avantage\n"
        "informationnel. On repete sur plusieurs series pour que le resultat ne\n"
        "depende pas d'un tirage chanceux.\n"
    )
    costs = CostModel.interactive_brokers_us_equity()
    engine = BacktestEngine(
        costs, initial_capital=100_000.0,
        execution=ExecutionConfig(allow_fractional_units=False),
    )

    scores: dict[str, list[float]] = {}
    drags: dict[str, list[float]] = {}
    beats: dict[str, int] = {}

    for seed in range(n_seeds):
        data = build_series(
            SyntheticSpec(seed=5000 + seed, annual_drift=0.0, annual_volatility=0.20)
        )
        for strategy in strategies():
            result = engine.run(strategy, data, start_index=200)
            scores.setdefault(strategy.name, []).append(result.excess_return)
            drags.setdefault(strategy.name, []).append(result.cost_drag_pct)
            beats[strategy.name] = beats.get(strategy.name, 0) + int(result.beats_benchmark)

    header = (f"  {'strategie':<26} {'ecart moyen':>13} {'mediane':>10} "
              f"{'frais moy.':>11} {'bat B&H':>9}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, values in scores.items():
        array = np.asarray(values)
        print(
            f"  {name:<26} {array.mean():>+13.1%} {float(np.median(array)):>+10.1%} "
            f"{np.mean(drags[name]):>11.1%} {beats[name]:>4}/{n_seeds:<4}"
        )
    print()
    print("  'ecart' = performance de la strategie moins celle du buy & hold, sur la")
    print("  meme periode. Un ecart negatif signifie qu'acheter et ne rien faire")
    print("  aurait mieux valu.")
    print()
    print("  Le point essentiel n'est pas la moyenne : c'est la colonne 'bat B&H'.")
    print("  Le croisement de moyennes mobiles gagne environ une fois sur deux --")
    print("  un pile ou face -- alors que sur UNE serie prise isolement il peut")
    print("  afficher +44 % et donner l'impression d'un systeme qui marche.")
    print("  C'est exactement le piege : un backtest unique ne prouve rien.")
    print()


def demo_degres_de_liberte() -> None:
    print(RULE)
    print("C. COMBIEN DE FOIS FAUDRAIT-IL ESSAYER POUR TROUVER LA BONNE COMBINAISON")
    print(RULE)
    for strategy in strategies():
        print(
            f"  {strategy.name:<26} {strategy.degrees_of_freedom} parametre(s) reglable(s), "
            f"{strategy.search_space_size:,} combinaisons possibles"
        )
    print()
    print("  Tester 1 160 combinaisons et garder la meilleure, ce n'est pas trouver")
    print("  une strategie : c'est tirer 1 160 fois a pile ou face et s'extasier sur")
    print("  la plus longue serie de piles. Le moteur affiche ce nombre pour que la")
    print("  performance annoncee soit lue a cette aune.")
    print()


def main() -> int:
    configure_logging("ERROR", structured=False)
    print()
    demo_taille_du_compte()
    demo_marche_sans_edge()
    demo_degres_de_liberte()
    print(RULE)
    print(
        "Ces chiffres viennent de series synthetiques : cet environnement n'a pas\n"
        "acces a Yahoo Finance. Sur ta machine, remplace configs/data.offline.yaml\n"
        "par configs/data.yaml et le meme code tournera sur de vraies cotations."
    )
    print(RULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

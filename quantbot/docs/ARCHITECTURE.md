# Architecture

## Vue d'ensemble

```
qbot/
├── config.py            Configuration typée — un run = un fichier, donc rejouable
├── experiment.py        Orchestration : données -> features -> agent -> backtest
│
├── data/                numpy/pandas uniquement
│   ├── loader.py        Chargement OHLCV + validation stricte des invariants
│   ├── bars.py          Barres tick / volume / dollar / imbalance
│   └── synthetic.py     Marchés simulés (tests + contrôles négatifs)
│
├── features/            numpy/pandas uniquement — TOUT est causal
│   ├── technical.py     36 indicateurs, sans TA-Lib
│   ├── microstructure.py Roll, Corwin-Schultz, Amihud, Kyle λ, VPIN, Yang-Zhang
│   ├── regime.py        Variance ratio, Hurst, entropie plug-in, force de tendance
│   ├── fracdiff.py      Différenciation fractionnaire + test ADF maison
│   └── pipeline.py      Assemblage, normalisation causale, parité live
│
├── labeling/            numpy/pandas uniquement
│   ├── triple_barrier.py Triple barrière, meta-labeling, filtre CUSUM
│   └── weights.py       Unicité, attribution de rendement, bootstrap séquentiel
│
├── env/                 numpy/pandas uniquement
│   ├── trading_env.py   Environnement RL (API Gymnasium sans la dépendance)
│   ├── costs.py         Spread, commission, impact en racine, portage
│   └── rewards.py       DSR, log-PnL, vol-scaled, drawdown-penalized
│
├── agents/              ← SEUL sous-paquet dépendant de PyTorch
│   ├── networks.py      NoisyLinear, MLP/GRU/TCN causal, dueling, QR/C51
│   ├── replay.py        SumTree, replay priorisé, n-step, stockage compact
│   ├── rainbow.py       L'agent (7 extensions, chacune désactivable)
│   ├── trainer.py       Boucle + sélection de checkpoint sur validation
│   └── ensemble.py      Multi-graines + gel sur désaccord
│
├── risk/                numpy uniquement
│   ├── sizing.py        Kelly fractionnaire, vol targeting, risk parity (CCD)
│   └── guards.py        Coupe-circuits déterministes
│
├── backtest/            numpy/pandas/scipy uniquement
│   ├── engine.py        Moteur vectorisé + benchmarks obligatoires
│   └── metrics.py       Métriques descriptives ET inférentielles (PSR, DSR, MinTRL)
│
├── validation/          numpy/pandas/scipy uniquement
│   ├── cv.py            PurgedKFold, CPCV
│   ├── walkforward.py   Walk-forward avec ré-entraînement
│   ├── pbo.py           CSCV / probabilité de sur-apprentissage
│   └── monte_carlo.py   Bootstrap stationnaire, Reality Check, permutation de trades
│
└── live/
    ├── protocol.py      JSON délimité + cadrage TCP
    ├── engine.py        Inférence réutilisant EXACTEMENT le pipeline d'entraînement
    └── server.py        Serveur TCP multi-thread
```

**Règle de dépendance** : seul `agents/` importe PyTorch. Backtest, validation, features
et labeling fonctionnent sans. On peut donc valider une stratégie, calculer une PBO ou
un Deflated Sharpe sur une machine sans GPU ni torch installé.

---

## Trois décisions structurantes

### 1. L'environnement contient la couche de risque

Le vol targeting et le coupe-circuit de drawdown sont appliqués **pendant l'entraînement**,
pas seulement au déploiement. Sinon l'agent apprend une politique pour une échelle de
risque et en exécute une autre en production — mismatch classique et coûteux.

### 2. Le tampon de rejeu ne stocke pas les observations

Une observation vaut `window × n_features` flottants, soit 2 000 à 4 000 valeurs. Pour
300 000 transitions avec `obs` **et** `next_obs`, cela ferait **5.2 Go**.

Or toute observation est entièrement déterminée par `(indice temporel, état de
portefeuille)`. On stocke donc ce couple (~65 octets) et on reconstruit la fenêtre à la
volée : **48 Mo**, soit un facteur 107.

### 3. Un seul chemin de code pour les features

`FeaturePipeline` sert au backtest **et** au live. `InferenceEngine` ne réimplémente rien.
Le test `test_live_features_match_backtest_features` vérifie que les features live
reproduisent celles du backtest à 1e-6 près.

Ce test a déjà attrapé un bug réel : `min_history` était calculé comme un *maximum*
alors qu'il doit être *additif* (warm-up des features **+** fenêtre du z-score glissant).
La conséquence était un écart de 0.46 sur une feature normalisée — invisible, silencieux,
et directement traduisible en pertes.

---

## Flux d'une décision, du prix au lot

```
 OHLCV brut
    │  build_bars()                     barres à information constante (option)
    ▼
 FeaturePipeline.build_raw()            67 features causales
    │  _scale()                         z-score glissant + winsorisation
    ▼
 matrice (T, F)
    │  TradingEnv._observation()        fenêtre (W, F) + 6 variables de portefeuille
    ▼
 observation (W·F + 6,)
    │  QNetwork                         encodeur -> dueling -> quantiles
    ▼
 (n_actions, n_quantiles)
    │  q_values() ou risk_measure()     espérance, ou CVaR si politique averse au risque
    ▼
 argmax -> position brute ∈ {-1, -0.5, 0, 0.5, 1}
    │  vol targeting                    × min(σ_cible / σ_réalisée, levier_max)
    │  RiskGuard.check()                drawdown, perte du jour, spread, session, gel
    ▼
 exposition autorisée
    │  ExposureToLots()  [MQL5]         arrondi INFÉRIEUR au pas de lot
    ▼
 ordre CTrade
```

---

## Conventions temporelles

C'est ici que se jouent la majorité des fuites de données.

| Moment | Ce qui est disponible |
|---|---|
| `t` | Features calculées avec l'information ≤ **clôture** de `t` |
| action `a_t` | Position cible pour la barre suivante |
| exécution | `close` → à la clôture de `t` ; `next_open` → à l'ouverture de `t+1` |
| rendement | Celui de la barre `t+1`. **Jamais** celui de `t` |

Le décalage est matérialisé **une seule fois** dans le dépôt, dans `run_backtest` et
`TradingEnv._precompute`. Centraliser cette convention évite la classe de bugs la plus
insidieuse : le double décalage (qui détruit le signal sans erreur visible) et le
décalage manquant (qui fabrique un signal inexistant).

---

## Extension

| Objectif | Point d'entrée |
|---|---|
| Nouvelle feature | `features/technical.py`, puis l'ajouter dans `build_technical_features` |
| Nouvelle récompense | Sous-classer `RewardFunction`, l'enregistrer dans `build_reward` |
| Nouvelle architecture | Nouvel encodeur dans `networks.py`, branché via `AgentConfig.encoder` |
| Autre algorithme (PPO, SAC) | Réutiliser `TradingEnv` tel quel ; l'API est celle de Gymnasium |
| Multi-actifs | Généraliser `TradingEnv` à des positions vectorielles ; `risk_parity_weights` est prêt |
| Autre courtier | Réimplémenter uniquement l'EA ; le protocole JSON est stable et documenté |

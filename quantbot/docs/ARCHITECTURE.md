# Architecture

## Vue d'ensemble

```
qbot/
├── config.py            Configuration typée — un run = un fichier, donc rejouable
├── experiment.py        Orchestration : données -> features -> agent -> backtest
├── diagnostics.py       Sondes de signal : plancher linéaire vs plafond réseau
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
├── live/
│   ├── protocol.py      JSON délimité + cadrage TCP
│   ├── engine.py        Inférence réutilisant EXACTEMENT le pipeline d'entraînement
│   └── server.py        Serveur TCP multi-thread
│
└── monitoring/          numpy/pandas/scipy uniquement
    ├── drift.py         PSI, KL, JS, KS corrigé, Page-Hinkley calibré
    ├── tca.py           Implementation shortfall décomposé, rabais de Sharpe
    ├── store.py         Mémoire de production et indicateurs
    ├── reconciliation.py Enveloppe bootstrap, test séquentiel, rejeu des décisions
    ├── journal.py       Trace d'audit chaînée par SHA-256
    ├── alerts.py        Règles déterministes, niveaux, temporisation croissante
    ├── monitor.py       Orchestrateur appelé une fois par barre
    └── dashboard.py     Tableau de bord HTML autonome (SVG à la main)
```

**Règle de dépendance** : seul `agents/` importe PyTorch. Backtest, validation, features
et labeling fonctionnent sans. On peut donc valider une stratégie, calculer une PBO ou
un Deflated Sharpe sur une machine sans GPU ni torch installé.

---

## Cinq décisions structurantes

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

Le même principe s'applique à l'**état de portefeuille** : deux de ses six composantes
(volatilité de la stratégie, intensité de trading) ne sont pas observables depuis l'EA.
Plutôt que de les remplir de zéros, `InferenceEngine` les reconstruit à partir de la suite
des requêtes. `test_live_portfolio_state_matches_environment` vérifie la parité
composante par composante en régime établi.

### 4. Bruit d'exploration et dropout sont séparés

Deux stochasticités qu'on confond souvent :

- **NoisyNet** est la politique d'exploration *apprise*. Elle doit être active pendant la
  collecte d'expérience et coupée à l'évaluation comme en production.
- **Le dropout** est un régularisateur de la *passe d'apprentissage*. Le laisser actif au
  moment de choisir une action ajoute un aléa non maîtrisé au comportement et fait
  diverger la politique exécutée de celle qui a été évaluée.

`NoisyLinear.noise_override` découple les deux : le réseau reste en mode `eval` pour
choisir une action (dropout inactif) tandis que le bruit d'exploration est réactivé
explicitement.

---

### 5. Une sonde avant l'entraînement

`diagnostics.py` encadre le problème : régression linéaire (plancher de ce qui est
extractible) et réseau réel entraîné en supervisé (plafond de ce que le RL peut espérer,
puisque le RL résout un problème strictement plus dur). L'écart IC train / IC test rend
le sur-apprentissage visible immédiatement.

Sans cette sonde, un run RL qui échoue laisse trois causes indiscernables : features sans
signal, architecture incapable de l'extraire, ou boucle RL défaillante. Avec elle, la
question se tranche en quelques secondes.

---

### 6. La surveillance observe, elle ne décide pas

`LiveMonitor` est branché en **aval** de la décision et ne ferme aucune position. Il
expose `should_halt`, que le serveur est libre de consulter et que la configuration relie
ou non au coupe-circuit (`halt_on_critical`, faux par défaut).

Trois conséquences concrètes :

- `observe()` **ne lève jamais** : toute exception est capturée, comptée dans `n_errors`
  et journalisée. Le pire scénario acceptable est un tableau de bord dégradé, jamais une
  session de trading interrompue par son propre observateur.
- Un moniteur absent ou en panne ne change **rien** au comportement de trading. Les
  fichiers `reference.json` / `envelope.json` manquants dégradent proprement la
  surveillance, et le serveur le dit au démarrage.
- Les couches coûteuses (dérive, coûts, réconciliation) sont **cadencées**
  (`drift_every`, 25 barres par défaut) pour que le coût par barre reste borné.

La raison de fond : une couche d'observation capable de liquider un portefeuille devient
elle-même un risque opérationnel. Le premier seuil mal réglé coûterait un compte, et il
n'y aurait rien pour l'arrêter — puisque c'est précisément la couche censée surveiller.

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

## Extension

| Objectif | Point d'entrée |
|---|---|
| Nouvelle feature | `features/technical.py`, puis l'ajouter dans `build_technical_features` |
| Nouvelle récompense | Sous-classer `RewardFunction`, l'enregistrer dans `build_reward` |
| Nouvelle architecture | Nouvel encodeur dans `networks.py`, branché via `AgentConfig.encoder` |
| Autre algorithme (PPO, SAC) | Réutiliser `TradingEnv` tel quel ; l'API est celle de Gymnasium |
| Multi-actifs | Généraliser `TradingEnv` à des positions vectorielles ; `risk_parity_weights` est prêt |
| Autre courtier | Réimplémenter uniquement l'EA ; le protocole JSON est stable et documenté |

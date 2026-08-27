# qbot — trading algorithmique quantitatif par apprentissage par renforcement

Système complet de recherche, validation et exécution pour stratégies de trading
systématiques : **features causales**, **agent Rainbow distributionnel**, **backtest
événementiel avec coûts réalistes**, **batterie anti-sur-apprentissage** et **pont
d'exécution MetaTrader 5**.

---

## Position de départ, en une phrase

Construire un bot qui trade est facile. Construire un bot dont on peut **démontrer** qu'il
possède un edge, et non un artefact de recherche, est difficile — et c'est le seul travail
qui a de la valeur.

Ce dépôt est organisé autour de cette asymétrie. La partie modélisation (Rainbow DQN,
QR-DQN, Munchausen) est de l'ingénierie standard. La partie validation (purge, embargo,
CPCV, PBO, Deflated Sharpe, bootstrap par blocs, Reality Check) est ce qui sépare un
projet sérieux d'une courbe d'équité pour capture d'écran.

**Un chiffre pour fixer les idées** : en testant 1 000 stratégies **sans aucun edge**,
la meilleure affichera un Sharpe de **3.26** en backtest. C'est un résultat mathématique,
pas une hypothèse — il est vérifié par simulation dans les tests de ce dépôt.

---

## Démarrage

```bash
pip install -r requirements.txt

# Démonstration complète (données synthétiques, ~5 min) :
# données -> features -> entraînement RL -> backtest -> validation -> pont live
python scripts/demo.py

# Sur vos propres données — dans cet ordre
python scripts/probe.py       --csv data/EURUSD_H1.csv   # 1. y a-t-il seulement du signal ?
python scripts/screen.py      --csv data/EURUSD_H1.csv   # 2. quelles hypothèses survivent ?
python scripts/regime.py      --csv data/EURUSD_H1.csv   # 3. quelle stratégie dans quel régime ?
python scripts/meta.py        --csv data/EURUSD_H1.csv --importance   # 4. filtrer les signaux
python scripts/allocate.py    --csv data/EURUSD_H1.csv   # 5. allouer le capital (RL)
python scripts/walkforward.py --csv data/EURUSD_H1.csv   # 6. protocole de référence
python scripts/validate.py    --returns runs/walkforward/oos_returns.csv --trials 40
python scripts/serve.py       --model runs/v1            # dry-run par défaut
```

**Commencez toujours par `probe.py`.** Il répond en quelques secondes à ce qu'un
entraînement RL de plusieurs heures laisse indécidable : les features contiennent-elles
un signal, et l'architecture choisie peut-elle l'extraire ? Si la sonde réseau fait moins
bien que la sonde linéaire, lancer le RL est une perte de temps — c'est la représentation
ou la capacité du modèle qu'il faut corriger.

---

## Ce que contient le système

### Données et features

- Barres **tick / volume / dollar / imbalance** (López de Prado ch. 2). Sur les données de
  test, passer en barres dollar fait chuter l'aplatissement des rendements de **43.8 à 1.9** —
  c'est-à-dire d'une distribution ingérable à une distribution presque gaussienne.
- **Différenciation fractionnaire** : rend la série stationnaire tout en conservant sa
  mémoire. Mesuré sur les données de test : `d* = 0.15` donne un ADF de −2.97
  (stationnaire) avec **97.7 %** de corrélation conservée avec le log-prix.
- **67 features** en trois familles : technique (36), microstructure (11 — Roll,
  Corwin-Schultz, Amihud, Kyle λ, VPIN, Yang-Zhang), régime (10 — variance ratio, Hurst,
  entropie plug-in, force de tendance) et calendaire (10, encodage cyclique).
- **Normalisation strictement causale**. Aucun `fit` global, aucune fenêtre centrée.

### Labeling

Triple barrière (take-profit / stop-loss / expiration), **meta-labeling**, filtre CUSUM,
et la correction des labels chevauchants : unicité moyenne, attribution de rendement,
décroissance temporelle, bootstrap séquentiel.

### Agent

Rainbow complet, chaque extension activable indépendamment pour permettre une véritable
ablation : Double Q, Dueling, Prioritized Replay, retours n-step, NoisyNet,
**distributionnel (QR-DQN ou C51)** et **Munchausen**.

Encodeurs interchangeables : MLP, GRU, TCN à convolutions **strictement causales**.

La tête distributionnelle expose le **CVaR par action** : `serve.py --cvar 0.1` bascule la
politique sur « maximiser le pire décile » plutôt que la moyenne — le critère qu'applique
réellement une table de trading sous contrainte de risque.

### Coûts et risque

Spread, commission, **impact en racine carrée**, portage, bande de non-négociation.
Ordre de grandeur mesuré : le trading aléatoire passe de **+0.23 à −14.63 de Sharpe**
lorsqu'on facture correctement les coûts.

Kelly fractionnaire, vol targeting, risk parity (descente coordonnée cyclique,
contributions égalisées à 1e-15), et des coupe-circuits déterministes **dupliqués** côté
Python et côté MQL5.

### Diagnostic préalable

`scripts/probe.py` encadre le problème avant tout entraînement : une régression linéaire
donne le **plancher** de ce qui est extractible, le réseau réellement utilisé — entraîné
en supervisé — donne le **plafond** de ce que le RL peut espérer. Il affiche aussi l'écart
IC train / IC test, qui rend le sur-apprentissage immédiatement visible.

Mesure obtenue avec cet outil sur un marché synthétique à R² connu de 0.06 :

| Pas d'entraînement | IC out-of-sample |
|---|---|
| 2 000 | **+0.109** |
| 6 000 | +0.038 |
| 20 000 | **−0.003** |
| 50 000 | −0.004 |

La performance **décroît** avec l'entraînement : le réseau mémorise le bruit avant
d'épuiser le signal, pendant qu'une régression linéaire atteint un IC de +0.216 sur les
mêmes données.

Correctement régularisé, il rattrape presque son retard — à 20 000 pas, l'IC passe de
**−0.003** (`weight_decay=0`) à **+0.070** (1e-4) puis **+0.179** (1e-3). C'est pourquoi
les valeurs par défaut de ce dépôt sont volontairement petites : réseau 64×64, fenêtre
d'observation de 16 barres, `weight_decay=1e-3`, évaluation fréquente et patience courte.
En finance, la capacité du modèle est une contrainte, pas une ressource.

### Stratégies (§8 du cahier des charges)

Cinq familles — suivi de tendance, momentum absolu, retour à la moyenne, cassure de
Donchian, compression de volatilité. Chacune **doit** déclarer son hypothèse et le régime
dans lequel elle est censée échouer : l'exigence est portée par le type, pas par un
commentaire, donc on ne peut pas ajouter une stratégie sans avoir formulé ce qui la
réfuterait.

Le banc de criblage mesure deux choses distinctes : l'edge de l'hypothèse à paramètres
figés, et ce qu'un praticien obtient après avoir optimisé les paramètres à chaque fold.
**L'écart entre les deux est la mesure directe du data-snooping.**

Validation du banc lui-même : **0/5 hypothèses retenues** sur une marche aléatoire pure
(Reality Check p = 0.999), **2/5** sur un marché à momentum réel (p = 0.034, PBO = 0.071).
Un banc incapable de dire non n'a aucune valeur ; un banc qui rejette tout non plus.

### Méta-modèle ML (§9)

Le modèle ne prédit pas la direction — il répond à « ce signal-là vaut-il la peine d'être
suivi ? ». L'évaluation est **économique avant d'être statistique** : un modèle qui gagne
0.02 d'AUC sans améliorer le profit factor n'a aucune valeur.

| Modèle | Profit factor |
|---|---|
| suivre tous les signaux (référence) | 1.06 |
| régression logistique | 1.31 |
| forêt aléatoire | 1.46 |
| boosting de gradient | 1.49 |

`justify_complexity()` tranche automatiquement : ici +70 % de gain par trade sur le
linéaire, donc la complexité est justifiée. Sur d'autres données, la réponse est souvent
l'inverse — et c'est un résultat, pas un échec.

### Détection de régime (§7)

Trois approches comparables (règles, clustering, HMM), avec un critère unique : **un
détecteur n'a de valeur que si la performance des stratégies diffère réellement entre ses
régimes**, établi par permutation par blocs des étiquettes.

Un défaut de conception corrigé en cours de route, mesuré : les features du modèle
prédictif sont z-scorées sur 300 barres, or un régime dure ~833 barres, si bien que la
normalisation efface le NIVEAU — qui est l'information de régime. Sur un marché à deux
régimes de volatilité 2 % et 40 %, trivialement séparables : **ARI 0.011 (le hasard) avec
les features z-scorées, 0.947 avec des features de niveau.**

Ce que la couche sait faire, et ce qu'elle ne sait pas :

| Marché | ARI |
|---|---|
| vol 2 % vs 40 % | 0.93 – 0.95 |
| vol 10 % vs 15 % | 0.61 – 0.72 (le HMM en tête, grâce à la persistance) |
| **dérive seule, vol identique** | **≈ 0 — indétectable** |

On détecte la volatilité, jamais la dérive : à l'échelle de la barre, celle-ci est deux
ordres de grandeur sous le bruit. Toute couche de régime qui prétendrait le contraire ment.

Le biais du lissage non causal est lui aussi contre-intuitif : négligeable quand la
détection est facile (0 % de désaccord sur 2 %/40 %), maximal quand elle est difficile
(2.7 % sur 10 %/13 %, dispersion apparente du Sharpe passant de 0.51 à 1.05). Il trompe
donc le plus précisément dans le cas réaliste.

### Allocateur RL (§10)

Le RL ne prédit plus la direction : il **répartit le capital entre les stratégies
validées** selon le régime. C'est le bon usage du Deep Q-Learning ici, parce que l'espace
d'états est bien plus petit, que les stratégies portent déjà l'hypothèse économique, et
que l'échec est gracieux — un allocateur qui n'apprend rien converge vers l'équipondéré
ou vers le plat.

Mesure sur un marché alternant blocs momentum et blocs de retour à la moyenne :

| | Sharpe | Max drawdown |
|---|---|---|
| équipondéré constant | −2.02 | −15.4 % |
| meilleure référence fixe | +0.54 | −15.9 % |
| **allocateur RL** | **+1.23** | **−6.8 %** |

L'agent choisit de rester **hors marché 28 % du temps** — l'exigence du §2 « savoir ne pas
trader », satisfaite concrètement et non déclarativement.

Les coûts sont facturés sur la position **nette**, jamais stratégie par stratégie :
sommer des rendements déjà nets double-compterait les frais et ignorerait la compensation
entre signaux opposés.

### Validation

| Outil | Question à laquelle il répond |
|---|---|
| Purged K-Fold + embargo | Le modèle a-t-il vu le futur via des labels chevauchants ? |
| CPCV | Le résultat tient-il sur *plusieurs* chemins historiques, ou sur un seul ? |
| Walk-forward | Que donne un ré-entraînement périodique, comme en production ? |
| Deflated Sharpe | Le Sharpe survit-il à la correction du nombre d'essais ? |
| PBO (CSCV) | Le meilleur in-sample finit-il sous la médiane out-of-sample ? |
| Bootstrap par blocs | Quel est l'intervalle de confiance du Sharpe ? |
| Monte-Carlo du drawdown | Quel drawdown provisionner, au-delà du seul chemin observé ? |
| Reality Check de White | La meilleure de N stratégies bat-elle le hasard, N pris en compte ? |

### Exécution

Expert Advisor MQL5 (`mql5/QBotBridge.mq5`) communiquant en **TCP + JSON** via les sockets
natifs de MetaTrader — **aucune DLL**, donc compatible prop-firms et Market MQL5.

---

## Ce qui est vérifié, et comment

Le dépôt contient **193 tests**. Les plus importants ne testent pas l'absence d'exception
mais des **propriétés mathématiques connues** :

| Vérification | Résultat mesuré |
|---|---|
| Position longue constante == achat/conservation | écart **1e-15** |
| Moteur de backtest == environnement RL | écart **0.00e+00** |
| Features live == features backtest | écart **1.19e-06** |
| Perturber le futur ne change pas le passé | 7 tests de causalité, tous verts |
| Détecteur de look-ahead réellement sensible | vision du futur → Sharpe > 20 (contrôle inverse) |
| E[max Sharpe] sous H₀ (théorie vs simulation) | 1.652 vs **1.663** |
| E[PBO] sous H₀ (théorie 0.50) | **0.494** à **0.509** pour N ≥ 20 |
| PBO en présence d'un vrai edge | tombe à **0.000** |
| Bootstrap par blocs préserve l'autocorrélation | 0.498 → **0.484** (i.i.d. : −0.031) |
| L'agent apprend un signal trivial | converge, sinon la boucle serait cassée |
| Reality Check : faux positifs contrôlés | p = 0.09 sans edge, p < 0.001 avec |

```bash
python -m pytest tests/ -q
```

Cinq bugs réels ont été trouvés **par ces tests** pendant le développement, et sont
documentés dans le code à l'endroit du correctif :

1. `evaluate()` n'évaluait qu'une fenêtre **aléatoire** de 1024 barres au lieu du segment
   complet — la sélection de checkpoint se faisait donc sur du bruit d'échantillonnage.
2. `min_history` était un *maximum* au lieu d'être *additif* (warm-up des features **+**
   fenêtre du z-score) — écart entraînement/service de 0.46 sur une feature normalisée.
3. L'inférence en lot alimentait l'agent avec un état de portefeuille nul, alors qu'il
   avait été entraîné avec — l'agent était interrogé sur des états jamais rencontrés.
4. Le retour n-step déduisait sa longueur d'un logarithme du facteur d'actualisation :
   division par zéro à γ=1, `log(0)` à γ=0.
5. Le dropout restait actif à la sélection d'action, ajoutant un aléa non maîtrisé
   par-dessus l'exploration NoisyNet — la politique exécutée différait de celle évaluée.

Aucun de ces bugs ne provoquait d'erreur. Tous dégradaient silencieusement la
performance. C'est exactement la classe de défauts que ce type de tests existe pour
attraper.

---

## Documentation

- **[docs/METHODOLOGIE.md](docs/METHODOLOGIE.md)** — pourquoi la plupart des bots échouent,
  et les défenses implémentées ici. **À lire en premier.**
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — structure du code et décisions de conception.
- **[docs/INTEGRATION_MT5.md](docs/INTEGRATION_MT5.md)** — installation, protocole,
  diagnostic, et le protocole de mise en production.

---

## Ce que ce système ne fait pas

- **Il ne garantit aucun profit.** Il fournit l'outillage pour mesurer honnêtement s'il
  existe un edge. Le plus souvent, la réponse mesurée est « non » — et c'est la valeur
  de l'outil, pas son échec.
- **Un backtest sur données synthétiques ne valide que la plomberie**, jamais une performance.
- Pas de modélisation des événements macro (d'où les coupe-circuits déterministes).
- Mono-actif (le squelette multi-actifs existe : `risk_parity_weights`).
- Pas de simulation de carnet d'ordres — l'impact est modélisé, pas reproduit.

---

## Avertissement

Le trading à effet de levier fait perdre de l'argent à la majorité des comptes retail.
Ce dépôt est un outil de recherche : il rend le sur-apprentissage **mesurable**, il ne le
supprime pas. Un résultat qui ne passe pas la batterie de `scripts/validate.py` ne doit
pas être déployé, même s'il est beau sur un graphique.

Le protocole de mise en production (dry-run → démo → réel minimal → montée en taille) est
détaillé dans [docs/INTEGRATION_MT5.md](docs/INTEGRATION_MT5.md). Sauter une étape ne fait
pas gagner du temps : cela déplace la découverte du problème vers le moment où il coûte
de l'argent réel.

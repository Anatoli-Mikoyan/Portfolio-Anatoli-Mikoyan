# Méthodologie : pourquoi la plupart des bots de trading échouent

Ce document explique les décisions de conception du dépôt. Il est plus important que le
code : n'importe qui peut copier un Rainbow DQN depuis GitHub, presque personne ne le
valide correctement.

---

## 1. Le problème central : le rapport signal/bruit

En vision par ordinateur, un modèle atteint 99 % de précision. En trading, un modèle
qui a raison **52 %** du temps est excellent. Cette différence change tout.

Sur EURUSD H1, le rendement horaire a un écart-type d'environ 0.1 %. Un edge réaliste
vaut peut-être 0.005 % par barre. Le rapport signal/bruit est donc de l'ordre de **1/20**.

Conséquence directe et non négociable : avec un tel rapport, il faut des dizaines de
milliers d'observations pour distinguer une compétence de la chance. Et comme un modèle
suffisamment flexible peut ajuster n'importe quel bruit, **tout backtest non contrôlé
produit un résultat magnifique**. Le travail sérieux consiste à démontrer que le résultat
n'est pas un artefact — pas à l'obtenir.

---

## 2. Le sur-apprentissage de backtest, chiffré

### Le résultat qu'il faut connaître par cœur

Testez N stratégies **sans aucun edge**, dont les Sharpes ont un écart-type de 1 :

| Stratégies testées | Sharpe du meilleur backtest |
|---|---|
| 1 | 0.00 |
| 10 | **1.57** |
| 100 | **2.53** |
| 1 000 | **3.26** |
| 10 000 | **3.86** |

*(Formule de Bailey & López de Prado, vérifiée par simulation dans
`tests/test_validation.py::test_expected_max_sharpe_matches_simulation`.)*

Un Sharpe de 3 obtenu après avoir essayé mille configurations n'est donc **pas** un
signal : c'est la valeur attendue du meilleur tirage sous l'hypothèse nulle. Et « essayer
mille configurations » arrive plus vite qu'on ne croit : dix fenêtres d'indicateur ×
cinq seuils × quatre paires × cinq graines = 1 000.

### Trois défenses implémentées

**Deflated Sharpe Ratio** (`qbot/backtest/metrics.py`) — le PSR évalué contre le Sharpe
maximal attendu au lieu de zéro. Il exige de déclarer honnêtement le nombre d'essais
(`validation.n_trials_for_dsr`). C'est le paramètre le plus facile à sous-déclarer et le
plus coûteux à ignorer.

**Probability of Backtest Overfitting** (`qbot/validation/pbo.py`) — par validation croisée
combinatoirement symétrique. Répond à : « si je sélectionne la meilleure configuration
in-sample, quelle est la probabilité qu'elle finisse sous la médiane out-of-sample ? »
Sous l'hypothèse nulle, l'estimateur vaut bien 0.50 (vérifié par simulation, N ≥ 20).

**Reality Check de White** (`qbot/validation/monte_carlo.py`) — teste la *meilleure* des N
stratégies en tenant compte du fait qu'on a regardé les N. Sans cette correction, tester
100 stratégies au seuil de 5 % produit en moyenne 5 « découvertes » purement fortuites.

---

## 3. Les fuites de données

Une fuite ne provoque jamais d'erreur. Elle produit un backtest superbe, et rien d'autre.

| Fuite | Manifestation | Défense dans ce dépôt |
|---|---|---|
| Normalisation globale | `StandardScaler().fit(X)` sur tout l'historique fait fuir la moyenne du futur | `scaler: rolling_zscore`, fenêtre causale |
| Extrême non décalé | `high.rolling(20).max()` inclut la barre courante → breakout trivial | `donchian()` applique `shift(1)` |
| Labels chevauchants | Un label à 24 barres d'horizon partage ses rendements avec le fold suivant | Purge (`PurgedKFold`) |
| Autocorrélation résiduelle | Les features restent corrélées au-delà de la zone de chevauchement | Embargo (`embargo_pct`) |
| Décalage manquant | Positionnement sur le rendement de la barre courante | `run_backtest` décale d'une barre, une seule fois |
| Historique tronqué en live | Les EMA calculées sur un historique court sont fausses, pas NaN | `min_history` additif |

**Le test décisif** : perturber le futur et vérifier que le passé ne bouge pas
(`tests/test_causality.py`). Toute dépendance au futur, même indirecte, échoue.
Ce fichier contient aussi le contrôle inverse — avec une *vraie* vision du futur, le
Sharpe doit exploser — car un détecteur qu'on ne teste pas est un détecteur qu'on ne
sait pas fonctionnel.

---

## 4. Les coûts de transaction décident de tout

Sur 10 000 barres H1 de données synthétiques, avec un modèle de coûts standard
(spread 1.5 bp, commission 0.35 bp, impact en racine) :

| Stratégie | Sharpe **sans** coûts | Sharpe **avec** coûts |
|---|---|---|
| Achat et conservation | +1.35 | +1.35 |
| Aléatoire | +0.23 | **−14.63** |
| Momentum 20 | −0.56 | −2.96 |
| Retour à la moyenne 20 | −0.04 | −5.56 |

Le trading aléatoire passe de neutre à catastrophique. Ce n'est pas anecdotique : c'est
la raison pour laquelle une stratégie haute fréquence « rentable en backtest » cesse
presque toujours de l'être en réel. Le modèle de coûts de ce dépôt est délibérément
pessimiste : rejeter une bonne stratégie coûte moins cher que déployer une mauvaise.

L'impact suit une **loi en racine carrée** — résultat empirique robuste observé sur
actions, futures et FX (Almgren, Torre, Bouchaud). Un modèle linéaire sous-estime le coût
des petits ordres et surestime celui des gros.

**Métrique de sanité** : `CostModel.breakeven_move_bps()` donne le mouvement minimal qu'un
aller-retour doit capturer. Si l'edge moyen par trade lui est inférieur, la stratégie est
structurellement perdante — aucun modèle, aussi sophistiqué soit-il, ne peut corriger cela.

---

## 5. Pourquoi l'apprentissage par renforcement, et sous quelle forme

### Ce que le RL apporte réellement

Un modèle supervisé prédit un rendement. Il ne sait rien des coûts de transaction, ne
tient pas compte de la position déjà détenue, et ignore que la décision d'aujourd'hui
contraint celle de demain. Le RL optimise directement la **séquence de décisions**, coûts
inclus. C'est son seul avantage décisif — mais il est structurel.

### Rainbow, extension par extension

| Extension | Ce qu'elle change ici |
|---|---|
| Double Q | Supprime le biais d'optimisme de `max_a Q`, qui en trading se traduit par une sur-exposition systématique |
| Dueling | Sépare « ce marché est-il porteur ? » de « quelle action prendre ? » |
| Prioritized replay | 95 % des barres sont sans intérêt ; on échantillonne les 5 % informatives |
| Retours n-step | Propage le crédit sur l'horizon réel d'un trade, pas sur une barre |
| NoisyNet | Exploration apprise et spécifique à l'état ; l'ε-greedy explore aussi là où l'agent sait déjà, et chaque exploration inutile coûte le spread |
| **Distributionnel (QR)** | Apprend la **loi** du retour, pas sa moyenne |
| Munchausen | Bonus d'entropie implicite ; accélération nette pour zéro paramètre supplémentaire |

Le point distributionnel mérite qu'on s'y arrête. Deux actions de même espérance mais de
distributions différentes — l'une régulière, l'autre à queue gauche épaisse — ne sont
**pas** interchangeables pour un gérant. Un agent qui n'apprend que `E[Q]` est
structurellement aveugle au risque de queue.

QR-DQN (quantiles) est préféré à C51 (support fixe) parce qu'il n'exige pas de borner
`v_min`/`v_max` a priori — or l'échelle des retours financiers est inconnue et dérive.
Les deux restent implémentés et interchangeables.

Conséquence pratique : `QNetwork.risk_measure()` expose le **CVaR** par action.
`serve.py --cvar 0.1` bascule la politique sur « maximiser le pire décile » au lieu de
« maximiser la moyenne » — le critère qu'applique réellement une table de trading
sous contrainte de risque.

### Le choix de la récompense DÉFINIT la stratégie

Maximiser le PnL brut produit un agent à levier maximal qui se fait détruire au premier
régime défavorable. Ce n'est pas un bug : c'est mathématiquement le comportement optimal
pour cet objectif.

Le **Differential Sharpe Ratio** (Moody & Saffell, 1998) est la dérivée du Sharpe,
calculable en ligne. Propriété vérifiée dans `qbot/env/rewards.py` : à gain égal, un
même +0.5 % est récompensé **dix fois moins** en régime volatil qu'en régime calme.
L'agent est donc structurellement incité à la régularité.

---

## 6. Capacité du modèle : le piège le plus contre-intuitif

Ce point mérite sa propre section parce qu'il contredit frontalement l'intuition acquise
en apprentissage profond « classique », où plus de paramètres et plus d'époques donnent
presque toujours de meilleurs résultats.

### Une mesure faite sur ce dépôt

Sonde supervisée (`scripts/probe.py`) sur un marché synthétique dont le signal est connu
par construction — autocorrélation lag-1 de +0.25, soit un R² théorique de 0.06 :

| Pas d'entraînement | IC out-of-sample |
|---|---|
| 2 000 | **+0.109** |
| 6 000 | +0.038 |
| 20 000 | **−0.003** |
| 50 000 | −0.004 |

La performance **décroît avec l'entraînement**. Le réseau (128×128, ~100 000 paramètres)
mémorise le bruit de 11 600 observations bien avant d'avoir épuisé le signal. Pendant ce
temps, une simple régression linéaire sur les mêmes features atteint un IC de **+0.216**
et un Sharpe out-of-sample de **+9.5**, coûts inclus.

La régularisation change complètement la conclusion. À 20 000 pas, fenêtre de 16 barres :

| `weight_decay` | IC out-of-sample | Précision du signe |
|---|---|---|
| 0 | −0.003 | 0.504 |
| 1e-4 | +0.070 | 0.506 |
| **1e-3** | **+0.179** | **0.568** |
| *référence linéaire* | *+0.216* | *0.579* |

Autrement dit : correctement régularisé, le réseau retrouve l'essentiel de ce qu'une
simple régression linéaire obtenait d'emblée. Ce n'est pas un réglage cosmétique — c'est
la différence entre un modèle inutilisable et un modèle exploitable. C'est aussi un
rappel utile : sur ce type de données, battre le linéaire demande beaucoup de travail
pour un gain modeste.

Élargir la fenêtre d'observation n'aide pas non plus : de 1 à 32 barres, l'IC reste dans
la même plage médiocre. Les features encodent déjà l'historique (rendements à 2, 5, 20
barres, EMA, volatilités) ; empiler des copies décalées ne fait qu'augmenter la dimension
d'entrée, donc la capacité à mémoriser.

### Pourquoi c'est structurel, pas anecdotique

Le nombre d'observations nécessaires pour estimer un coefficient à un rapport
signal/bruit donné croît comme l'inverse du carré de ce rapport. Avec un R² de 0.06, il
faut des ordres de grandeur plus de données qu'en vision ou en langage pour le même
nombre de paramètres. Un réseau de 100 000 paramètres sur 12 000 barres est dans un
régime où la mémorisation est strictement plus facile que la généralisation.

### Ce qu'il faut en tirer

1. **Commencer petit.** Un modèle linéaire est la bonne référence, pas un point de départ
   à dépasser d'office. S'il n'est pas battu, le modèle complexe n'apporte rien.
2. **Lancer `scripts/probe.py` AVANT tout entraînement RL.** Il répond en quelques
   secondes à ce qu'un run RL de plusieurs heures laisse indécidable : y a-t-il du signal,
   et l'architecture peut-elle l'extraire ?
3. **Surveiller l'écart IC train / IC test**, pas seulement l'IC test. Un écart élevé
   signale une capacité excédentaire — le remède est de réduire le modèle, pas de
   l'entraîner davantage.
4. **L'arrêt anticipé sur validation n'est pas une option.** C'est ce qui empêche le
   modèle de dépasser son point optimal, qui arrive beaucoup plus tôt qu'on ne le croit.

Les valeurs par défaut de ce dépôt reflètent cette contrainte : réseaux volontairement
petits, `weight_decay` non nul, évaluation fréquente et patience courte.

---

## 7. Le protocole de validation

```
   Données complètes
   ├── train  (60 %)  -> apprentissage des poids
   ├── [embargo]
   ├── valid  (20 %)  -> sélection du checkpoint, arrêt anticipé
   ├── [embargo]
   └── test   (20 %)  -> touché UNE FOIS, à la fin, jamais pour décider
```

Puis, dans l'ordre :

1. **Walk-forward** (`scripts/walkforward.py`) — ré-entraînement à chaque fold. Chaque
   barre de la courbe d'équité produite a été générée par un modèle qui ne l'avait
   jamais vue. C'est le seul chiffre de ce projet qu'il soit raisonnable de citer.
2. **Régularité inter-folds** — plus informative que le Sharpe agrégé. Une stratégie à
   Sharpe 1.5 portée par 1 fold sur 8 sera abandonnée avant que le fold gagnant n'arrive.
3. **CPCV** — le walk-forward ne donne **qu'un** chemin historique. La validation croisée
   purgée combinatoire en reconstruit plusieurs dizaines et fournit une *distribution*
   de Sharpe au lieu d'un point.
4. **Bootstrap par blocs** — intervalle de confiance du Sharpe. Les blocs sont
   indispensables : rééchantillonner barre par barre détruit le clustering de volatilité
   et produit des intervalles beaucoup trop étroits, donc faussement rassurants.
5. **Monte-Carlo du drawdown** — le drawdown observé est celui d'un seul chemin, donc
   optimiste. Le quantile 5 % est l'estimation à provisionner.
6. **PBO, Deflated Sharpe, Reality Check** — corrections du biais de sélection.

`scripts/validate.py` produit un verdict binaire volontairement sévère : le coût d'un
faux positif (déployer une stratégie perdante) dépasse largement celui d'un faux négatif
(jeter une bonne stratégie).

---

## 8. La surveillance arrive toujours trop tard — sauf si on surveille la cause

Une fois le modèle en production, la question devient : **comment sais-je que ça marche
encore ?** La réponse naïve — « je regarde le Sharpe » — se heurte au même rapport
signal/bruit que la section 1, mais en pire, parce qu'on n'a cette fois qu'une seule
trajectoire et qu'elle avance en temps réel.

### Le chiffre à retenir

Mesure sur ce dépôt, barres horaires, stratégie passant d'un Sharpe de 1.2 à −1.5 :

| Durée observée | Détection par la performance |
|---|---|
| 300 barres (~2 semaines) | 10 % |
| 3 000 barres (~5 mois) | 68 % |
| 6 240 barres (**1 an**) | 93 % |

**Un an pour prouver l'effondrement.** Et l'enveloppe de ce qui est « normal » est
gigantesque à court terme : pour un backtest à Sharpe 1.2, l'intervalle attendu à
300 barres va de **−5.8 à +9.4**. Un Sharpe live de −2 y est parfaitement compatible.

Deux conséquences pratiques :

1. **Couper une stratégie après deux semaines, c'est décider sur du bruit.** Dans les deux
   sens : la garder parce qu'elle gagne aussi.
2. **Il faut surveiller autre chose que le résultat.** Non pas parce que le résultat ne
   compte pas — c'est le seul juge — mais parce qu'il répond trop tard.

### Surveiller la cause, pas seulement l'effet

| Ce qu'on surveille | Délai de détection | Nature |
|---|---|---|
| comportement du bot | immédiat | il ne fait plus ce qu'il faisait |
| coûts d'exécution réels | ~50 exécutions | le backtest était optimiste |
| dérive des features | ~250 barres | le marché n'est plus le même |
| performance | ~1 an | confirmation finale |

Les trois premières lignes préviennent pendant que la quatrième accumule des données.
C'est l'architecture de `qbot/monitoring/`, détaillée dans
**[SUPERVISION.md](SUPERVISION.md)**.

### Trois pièges qui annulent la surveillance sans rien casser

Ils méritent d'être nommés, parce qu'ils produisent un dispositif *qui a l'air de
fonctionner* :

- **Ignorer l'autocorrélation.** 250 barres horaires ne valent que ~13 observations
  indépendantes quand ρ₁ = 0.9. Un test de Kolmogorov-Smirnov brut déclare alors une
  dérive significative en permanence. Conséquence réelle : l'équipe coupe les alertes, y
  compris celles qui comptaient.
- **Un détecteur séquentiel qui réapprend sa moyenne.** Le Page-Hinkley adaptatif *suit*
  la dégradation au lieu de la signaler : sa référence glisse avec la série. Il reste muet
  pendant l'effondrement, en toute logique et en toute inutilité. Pour la performance, la
  référence doit venir du backtest et rester fixe.
- **Une référence de dérive recalculée en continu.** Comparer la fenêtre live aux 250
  barres précédentes ne détecte qu'un choc brutal ; une dérive lente sur six mois passe
  inaperçue puisque la référence dérive avec elle. La référence se fige à l'entraînement
  et se version avec le modèle.

### Calibrer, pas deviner

Un détecteur dont on n'a pas mesuré le taux de fausses alarmes n'est pas un détecteur.
Le seuil λ du Page-Hinkley est ici résolu numériquement à partir du budget de fausses
alarmes souhaité (approximation de Siegmund du temps moyen avant alarme), et la valeur
par défaut sort d'une mesure du compromis puissance / fausses alarmes — pas d'une
convention recopiée.

Corollaire utile : `expected_delay()` dit combien de barres il faudra en moyenne. Si ce
délai dépasse votre horizon de décision, ce n'est pas ce détecteur qui protégera le
compte, et il vaut mieux le savoir avant que pendant.

---

## 9. Ce que ce système ne fait pas

Une liste honnête vaut mieux qu'une promesse.

- **Il ne garantit aucun profit.** Il fournit l'outillage pour mesurer honnêtement s'il
  existe un edge. Le plus souvent, la réponse mesurée est « non » — et c'est précisément
  la valeur de l'outil.
- **Il ne modélise pas les événements macro.** Une annonce de banque centrale sort de la
  distribution d'entraînement. D'où les coupe-circuits déterministes, indépendants du modèle.
- **Il ne gère qu'un actif à la fois.** Le squelette multi-actifs existe
  (`risk_parity_weights`) mais l'environnement est mono-actif.
- **Il n'a pas de carnet d'ordres.** L'impact est modélisé par une loi en racine, pas simulé.
  Pour du market-making ou de la vraie haute fréquence, il faudrait un simulateur de carnet.
- **Il ne remplace pas l'exécution.** Slippage réel, requotes, gaps de week-end et
  élargissements de spread sur annonce sont approximés, pas reproduits.

## 10. Les erreurs qui coûtent le plus cher

1. **Regarder le test plus d'une fois.** Chaque consultation le transforme en jeu de
   validation, et le chiffre rapporté devient un estimateur biaisé.
2. **Sous-déclarer le nombre d'essais.** Le Deflated Sharpe ne protège que si `n_trials`
   est honnête. Compter *toutes* les variantes, y compris celles abandonnées.
3. **Optimiser les coûts vers le bas** pour « voir ce que ça donne ». Ça donne toujours
   un meilleur résultat, et toujours faux.
4. **Passer en réel après un seul backtest réussi.** Le protocole de mise en production du
   document d'intégration existe pour cette raison.
5. **Augmenter la taille après une bonne série.** L'impact croît en racine de la taille ;
   la capacité se mesure, elle ne se suppose pas.

---

## Références

- López de Prado, *Advances in Financial Machine Learning* (2018) — fracdiff, triple
  barrière, meta-labeling, purge/embargo, CPCV, PBO.
- Bailey, Borwein, López de Prado & Zhu, *The Probability of Backtest Overfitting* (2014).
- Bailey & López de Prado, *The Deflated Sharpe Ratio* (2014).
- Hessel et al., *Rainbow: Combining Improvements in Deep RL* (2018).
- Dabney et al., *Distributional RL with Quantile Regression* (2018).
- Vieillard, Pietquin & Geist, *Munchausen Reinforcement Learning* (2020).
- Moody & Saffell, *Learning to Trade via Direct Reinforcement* (1998, 2001).
- Schaul et al., *Prioritized Experience Replay* (2016).
- Fortunato et al., *Noisy Networks for Exploration* (2017).
- Politis & Romano, *The Stationary Bootstrap* (1994).
- White, *A Reality Check for Data Snooping* (2000).
- Griveau-Billion, Richard & Roncalli, *A Fast Algorithm for Computing High-Dimensional
  Risk Parity Portfolios* (2013) — descente par coordonnées cycliques.

**Surveillance de production (§17)**

- Page, *Continuous Inspection Schemes* (1954) — le test séquentiel.
- Siegmund, *Sequential Analysis* (1985) — approximation de l'ARL, utilisée ici pour
  calibrer λ à partir du budget de fausses alarmes.
- Perold, *The Implementation Shortfall: Paper versus Reality* (1988) — la mesure de coût
  d'exécution non manipulable.
- Bayley & Hammersley, *The "Effective" Number of Independent Observations* (1946) —
  correction d'autocorrélation des tests à deux échantillons.
- Kullback & Leibler (1951) ; Lin, *Divergence Measures Based on the Shannon Entropy*
  (1991) — KL et Jensen-Shannon, dont dérivent le PSI et le score global de dérive.
- ESMA, *Règlement délégué (UE) 2017/589 (MiFID II RTS 6)* — obligations de traçabilité
  et de conservation applicables au trading algorithmique.

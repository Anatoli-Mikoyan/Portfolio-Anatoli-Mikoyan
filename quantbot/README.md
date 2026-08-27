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

### La façon la plus simple : une seule commande

**Windows** — double-cliquez sur `DEMARRER.bat`.

**macOS / Linux** :

```bash
bash demarrer.sh
```

Ça installe les bibliothèques, télécharge un historique EURUSD horaire si vous n'en avez
pas, mesure s'il existe un signal, crible les stratégies classiques, entraîne l'agent,
l'évalue sur une période qu'il n'a jamais vue, puis **ouvre un rapport HTML dans votre
navigateur**. Comptez 10 à 20 minutes.

```bash
bash demarrer.sh --rapide                    # entraînement écourté, ~5 min
bash demarrer.sh --csv mes_donnees.csv       # vos propres données
bash demarrer.sh --capital 5000              # capital simulé
```

Le rapport répond à une seule question, chiffres à l'appui : **si j'avais placé cette
somme sur la période de test, qu'est-ce que ça aurait donné — et cet écart est-il
distinguable du hasard ?**

### En détail, étape par étape

```bash
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
python scripts/monitor.py fit --model runs/v1 --data data/EURUSD_H1.csv \
                              --returns runs/walkforward/oos_returns.csv   # 7. figer la référence
python scripts/serve.py       --model runs/v1            # dry-run par défaut
python scripts/monitor.py report --model runs/v1 --html supervision.html   # 8. surveiller
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

### Tenue de marché : gagner sans prédire

Tout le reste du dépôt parie sur une direction et **paie** la fourchette pour entrer.
Cette couche fait l'inverse : elle cote un prix d'achat et un prix de vente en
permanence et **encaisse** l'écart. C'est le métier réel du trading haute fréquence —
non pas prédire plus vite, mais fournir de la liquidité et se faire payer pour.

```bash
python scripts/market_making.py
```

Quatre politiques de cotation, de la plus naïve à la solution optimale :

| Politique | P&L médian | Écart-type | \|inv\| max | P&L / risque |
|---|---|---|---|---|
| Naïve symétrique | 0.218 | 0.051 | **67.6** | 4.2 |
| Décalage linéaire | 0.216 | 0.005 | 3.3 | 45.6 |
| Avellaneda-Stoikov (2008) | 0.166 | 0.003 | 4.8 | 50.3 |
| **Guéant-Lehalle-Fernandez-Tapia (2013)** | **0.660** | 0.005 | 6.8 | **147.0** |

La naïve encaisse autant en moyenne — avec un inventaire dix fois plus gros et une
variance dix fois plus forte. **L'inventaire est le risque du métier**, pas la
prédiction.

Puis la mesure qui répond à « pourquoi le HFT n'est-il pas accessible en retail ? ».
Même politique, même flux, **pas une ligne de code changée** — seuls les frais et
l'accès à la cotation passive varient :

| Profil d'exécution | Accès passif | P&L médian | Sessions gagnantes |
|---|---|---|---|
| Teneur de marché HFT (rebates) | oui | **+0.660** | 100 % |
| Institutionnel / prop firm | oui | +0.269 | 100 % |
| Retail ECN (meilleur cas retail) | **NON** | **−2.189** | **0 %** |
| Retail MetaTrader standard | **NON** | **−2.753** | **0 %** |

Sans accès à la cotation passive, on ne tient pas un marché : on le traverse. On devient
le client, pas le teneur.

Enfin, la **sélection adverse** — la part du flux qui sait où va le prix, absente de la
plupart des simulations qu'on trouve en ligne :

| Impact du flux informé | P&L | Fourchette encaissée | P&L d'inventaire |
|---|---|---|---|
| 0 pip | +0.687 | 0.298 | −0.000 |
| 1 pip | +0.495 | 0.298 | −0.192 |
| 2 pips | +0.306 | 0.298 | −0.383 |
| **4 pips** | **−0.072** | 0.298 | −0.766 |
| 8 pips | −0.829 | 0.298 | −1.532 |

La fourchette encaissée ne bouge pas d'un centième : **c'est l'inventaire qui bascule**.
Le point de rupture se situe entre 2 et 4 pips d'impact — au-delà, aucun réglage de
cotation ne rattrape le mouvement qui suit chaque exécution.

Trois conditions rendent ce métier possible, et **aucune ne relève de l'algorithme** :
l'accès à la cotation passive, des frais négatifs (rebates), et la vitesse — non pour
prédire, mais pour *annuler* une cotation avant qu'un flux informé ne la frappe, ce qui
se joue en microsecondes contre 30 à 200 ms depuis un terminal de détail.

### Supervision de production (§17)

C'est la couche qui répond à « comment sais-je que ça marche encore ? ». Elle part d'un
constat mesuré et désagréable :

| Durée en production | Probabilité de détecter une chute de 1.2 → −1.5 de Sharpe |
|---|---|
| 300 barres (~2 semaines) | **10 %** |
| 1 000 barres | 33 % |
| 3 000 barres | 68 % |
| 6 240 barres (**1 an**) | **93 %** |

Il faut donc environ **un an** pour prouver par la performance qu'une stratégie horaire
s'est effondrée. D'où le principe de la couche : **surveiller la cause avant l'effet.**

| Couche | Délai de détection | Ce qu'elle voit |
|---|---|---|
| comportement | immédiat | modèle bloqué, garde-fous qui écrasent tout, latence |
| coûts d'exécution | ~50 exécutions | le backtest sous-estimait le slippage |
| dérive des features | ~250 barres | le marché n'est plus celui de l'entraînement |
| performance | ~1 an | le seul juge qui compte, et le plus lent |

Mesures de la détection de dérive (300 barres live contre 5 000 de référence) :

| Situation | PSI | Verdict |
|---|---|---|
| rien ne change | 0.03 – 0.05 | stable |
| moyenne décalée de 1.2 σ | 1.52 | critique |
| volatilité × 3, **moyenne inchangée** | 1.07 | critique |
| feature gelée qui redémarre | 15.3 | critique |

Le troisième cas est celui qu'un test de moyenne manquerait entièrement.

Trois pièges traités explicitement, parce qu'ils annulent la surveillance sans rien
casser de visible :

- **Autocorrélation.** 250 barres horaires ne valent que ~13 observations indépendantes
  pour ρ₁ = 0.9. Sans correction, le Kolmogorov-Smirnov crie en permanence et l'équipe
  coupe les alertes.
- **Détecteur qui réapprend sa moyenne.** Un Page-Hinkley adaptatif *suit* la dégradation
  au lieu de la signaler. Le mode à référence fixe est obligatoire pour la performance.
- **Alertes répétitives.** Temporisation doublée à chaque répétition : sur le scénario de
  démonstration, **99 alertes → 20** pour la même information.

Le tout est calibré, pas deviné : λ du Page-Hinkley est résolu numériquement à partir du
budget de fausses alarmes (approximation de Siegmund), et l'`arl0` par défaut sort d'une
mesure de compromis puissance / fausses alarmes.

S'y ajoutent une **analyse des coûts de transaction** (implementation shortfall de Perold,
décomposé en délai / spread / commission / impact) qui traduit le slippage réel en points
de Sharpe perdus, un **journal d'audit chaîné par SHA-256** (esprit MiFID II RTS 6 :
toute modification ou suppression a posteriori est localisée), et un **tableau de bord
HTML autonome** — un seul fichier, aucune ressource externe, aucune donnée qui sort de la
machine.

Le moniteur **ne ferme aucune position** : il expose `should_halt`, relié au coupe-circuit
seulement si on le demande (`halt_on_critical`, faux par défaut). Une couche d'observation
qui peut liquider un portefeuille devient elle-même un risque.

→ **[docs/SUPERVISION.md](docs/SUPERVISION.md)**

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

Le dépôt contient **268 tests**. Les plus importants ne testent pas l'absence d'exception
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
| ARL₀ du Page-Hinkley : théorie vs simulation | 10 000 visé, **~16 000** mesuré |
| Délai de détection : théorie vs simulation | 58 barres prévues, **51** mesurées |
| Détecteur adaptatif aveugle à la dérive qu'il apprend | figé par un test dédié |
| Décomposition des coûts == shortfall total | somme **exacte** (0.8+1.0+2.0 = 3.8 bps) |
| Journal falsifié : rupture localisée | modification **et** suppression détectées |
| Tableau de bord : aucune ressource externe | 0 occurrence de `http`, `<script`, `src=` |

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
6. Le détecteur séquentiel de dégradation recentrait les rendements sur **sa propre**
   moyenne courante, laquelle absorbait exactement la dégradation cherchée. La détection
   d'un effondrement à un an passait de 22 % à **92 %** une fois la référence figée sur
   les moments du backtest.
7. Le découpage PSI d'une feature constante plaçait référence et production dans la même
   case : une feature gelée qui redémarrait affichait un PSI de 0.004. Après encadrement
   de la valeur, **15.3**.
8. **Le facteur d'annualisation était faux d'un facteur 1 000.** `infer_bars_per_year`
   lisait les entiers de la résolution sous-jacente de l'index en supposant des
   nanosecondes ; depuis pandas 3, `date_range` et `read_csv` produisent par défaut du
   `datetime64[us]`. Le pas mesuré était donc mille fois trop petit, le nombre de barres
   par an mille fois trop grand, et **tout Sharpe annualisé multiplié par √1000 ≈ 31.6**.
   En production, l'effet était pire encore : la volatilité annualisée servant au
   vol-targeting était gonflée du même facteur, donc les positions divisées par ~31 —
   le bot restait à plat sans qu'aucune erreur ne soit levée.
9. Les **seuils du PSI n'étaient pas calibrés**. Les valeurs industrielles (0.10 / 0.25)
   viennent du scoring de crédit, où l'on compare deux grandes populations stables.
   Appliquées à une fenêtre de 250 barres face à une référence groupée qui mélange des
   dizaines de régimes, elles déclaraient **27 features sur 61 « en dérive critique » sur
   les données d'entraînement elles-mêmes**. Les seuils sont désormais calibrés par
   feature au 99ᵉ centile in-sample : 26.6 → **1.2** fausses alertes par fenêtre.

Aucun de ces bugs ne provoquait d'erreur. Tous dégradaient silencieusement la performance
— ou la capacité à *constater* cette dégradation. C'est exactement la classe de défauts
que ce type de tests existe pour attraper.

Les deux derniers méritent une note : ils n'ont été trouvés qu'en faisant **réellement
tourner** la chaîne complète — modèle entraîné, serveur TCP démarré, barres envoyées une
par une par le protocole. Ni les 246 tests ni les backtests ne les voyaient, parce que
les tests passent explicitement `bars_per_year=6240` et que la dérive n'y était mesurée
que sur des distributions construites pour diverger. La leçon est générale : **une suite
de tests verte ne remplace pas une répétition générale sur le chemin d'exécution réel.**
D'où le mode `replay` du serveur, qui rend cette répétition possible.

---

## Documentation

- **[docs/GUIDE.md](docs/GUIDE.md)** — **le mode d'emploi complet** : les 13 commandes une
  par une, le parcours de bout en bout sur vos données, comment lire les résultats sans
  se tromper, et les problèmes fréquents. **Commencez par là pour vous en servir.**
- **[docs/METHODOLOGIE.md](docs/METHODOLOGIE.md)** — pourquoi la plupart des bots échouent,
  et les défenses implémentées ici. **À lire en premier pour comprendre.**
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — structure du code et décisions de conception.
- **[docs/INTEGRATION_MT5.md](docs/INTEGRATION_MT5.md)** — installation, protocole,
  diagnostic, et le protocole de mise en production.
- **[docs/SUPERVISION.md](docs/SUPERVISION.md)** — surveillance en production : dérive,
  coûts réels, attendu vs réalisé, journal d'audit, alertes, tableau de bord.

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

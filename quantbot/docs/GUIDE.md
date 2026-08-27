# Guide d'utilisation — de zéro à la production

> Ce document explique **quoi faire, dans quel ordre, et pourquoi**. Il ne suppose aucune
> connaissance préalable du dépôt. Si vous ne devez lire qu'un seul fichier pour vous en
> servir, c'est celui-ci.

---

## Table des matières

1. [Ce que fait ce projet (et ce qu'il ne fait pas)](#1-ce-que-fait-ce-projet)
2. [Installation](#2-installation)
3. [Le tour du propriétaire en 5 minutes](#3-le-tour-du-propriétaire-en-5-minutes)
4. [Les 13 commandes, une par une](#4-les-13-commandes-une-par-une)
5. [Le parcours complet sur vos données](#5-le-parcours-complet-sur-vos-données)
6. [Brancher sur MetaTrader 5](#6-brancher-sur-metatrader-5)
7. [Surveiller une fois en production](#7-surveiller-une-fois-en-production)
8. [Lire les résultats sans se tromper](#8-lire-les-résultats-sans-se-tromper)
9. [Problèmes fréquents](#9-problèmes-fréquents)

---

## 1. Ce que fait ce projet

### En une phrase

C'est un **laboratoire de trading quantitatif** : il construit des stratégies, les teste
honnêtement, et vous dit — chiffres à l'appui — si elles valent quelque chose ou non.

### Ce qu'il fait vraiment

| Il fait | Il ne fait pas |
|---|---|
| mesurer s'il existe un signal exploitable | garantir un profit |
| détecter le sur-apprentissage | supprimer le sur-apprentissage |
| chiffrer les coûts réels d'exécution | négocier vos frais de courtage |
| surveiller la dégradation en production | l'empêcher |
| dire « cette stratégie ne tient pas » | trouver la stratégie magique |

**Le point le plus important à comprendre.** La valeur de cet outil n'est pas de produire
de belles courbes — n'importe qui en produit. Elle est de **réfuter**. Une démonstration
chiffrée, faite dans ce dépôt :

- Un trading complètement **aléatoire** affiche un Sharpe de **+0.23** sans les frais…
  et **−14.63** une fois les frais réels appliqués.
- Une recherche sur **1 000 stratégies sans aucun edge** produit tout de même un
  « meilleur » backtest à Sharpe **3.26**.

Autrement dit : un beau backtest ne prouve rien. Tout l'outillage sert à distinguer un
vrai signal du hasard bien habillé.

### Comment c'est organisé

```
quantbot/
├── qbot/          le moteur (72 modules)
├── scripts/       les 13 commandes que vous lancez
├── tests/         268 tests qui vérifient les propriétés mathématiques
├── configs/       vos réglages, en YAML
├── mql5/          l'Expert Advisor MetaTrader 5
└── docs/          la documentation
```

---

## 2. Installation

```bash
cd quantbot
pip install -r requirements.txt
```

**Vérifier que tout va bien :**

```bash
python -m pytest tests/ -q
```

Comptez ~7 minutes. Tout doit passer. Si quelque chose échoue, ne continuez pas : le
problème est dans l'installation, pas dans vos données.

> **PyTorch est-il obligatoire ?** Non. Seul le sous-paquet `agents/` en dépend. Vous
> pouvez faire tourner backtest, validation, criblage de stratégies, détection de régime
> et supervision sur une machine sans GPU ni torch.

---

## 3. Le tour du propriétaire en 5 minutes

```bash
python scripts/demo.py
```

Cette commande enchaîne tout le pipeline sur des données **synthétiques** : génération de
prix → features → entraînement RL → backtest → validation → pont live.

> ⚠️ Une démonstration sur données synthétiques valide **la plomberie**, jamais une
> performance. Le marché simulé est bien plus docile que le vrai.

---

## 4. Les 13 commandes, une par une

Voici chaque script, ce qu'il répond, et quand s'en servir.

### `probe.py` — « y a-t-il seulement du signal ? »

```bash
python scripts/probe.py --csv data/EURUSD_H1.csv
```

**Lancez toujours celle-ci en premier.** En quelques secondes, elle répond à ce qu'un
entraînement RL de plusieurs heures laisse indécidable :

- une **sonde linéaire** donne le plancher : ce qu'une simple régression extrait ;
- une **sonde réseau** donne le plafond : ce que la capacité choisie peut extraire.

Trois lectures possibles :

| Résultat | Ce que ça veut dire | Quoi faire |
|---|---|---|
| les deux sondes ≈ 0 | aucun signal dans vos features | changer les features, pas le modèle |
| réseau **<** linéaire | le modèle sur-apprend | réduire la capacité |
| réseau **>** linéaire | il y a du non-linéaire à capter | continuer vers le RL |

C'est ainsi qu'a été trouvé le vrai problème de ce dépôt : le Rainbow DQN faisait moins
bien qu'une régression OLS. La sonde a montré du sur-apprentissage — corrélation de
+0.109 à 2 000 pas, **−0.004** à 50 000. Les réglages par défaut en ont été changés.

### `screen.py` — « quelles hypothèses survivent ? »

```bash
python scripts/screen.py --csv data/EURUSD_H1.csv
```

Passe au crible 5 familles de stratégies classiques (suivi de tendance, momentum, retour
à la moyenne, cassure de Donchian, compression de volatilité) — 34 combinaisons de
paramètres — et rend un verdict par famille, **frais compris**.

Chaque stratégie doit déclarer son **hypothèse économique** et **quand elle échoue**.
C'est une contrainte du code, pas une convention : une stratégie qui ne sait pas dire
pourquoi elle devrait marcher est un ajustement de courbe.

### `regime.py` — « quelle stratégie dans quel régime ? »

```bash
python scripts/regime.py --csv data/EURUSD_H1.csv
```

Détecte les régimes de marché (HMM, clustering, règles) et mesure lesquels comptent.

**Une vérité utile trouvée ici :** la détection de régime voit très bien la
**volatilité** (indice de Rand ajusté de **0.947** contre la vérité terrain), et
**absolument pas la direction** (ARI ≈ 0). Personne ne devine la tendance à venir. Sachez
ce que vous achetez.

Attention aussi au **lissage** : un HMM interrogé après coup place les transitions
parfaitement — parce qu'il regarde le futur. Le code refuse d'utiliser une série lissée
dans un backtest (`LookaheadError`).

### `meta.py` — « faut-il vraiment prendre ce trade ? »

```bash
python scripts/meta.py --csv data/EURUSD_H1.csv --importance
```

Le **meta-labeling** de López de Prado : une stratégie primaire donne la direction, un
modèle ML dit s'il faut la suivre. Il ne crée pas de signal, il **filtre**.

`--importance` ajoute l'analyse d'importance des features par MDA clusterisée — utile
parce que deux features corrélées se volent mutuellement leur importance quand on les
teste isolément.

### `allocate.py` — « comment répartir le capital ? »

```bash
python scripts/allocate.py --csv data/EURUSD_H1.csv
```

Ici, le RL ne prédit plus la direction : il **répartit le capital entre les stratégies
validées** selon le régime. C'est le bon usage du Deep Q-Learning dans ce contexte —
espace d'états bien plus petit, hypothèses économiques déjà portées par les stratégies,
et échec gracieux (un allocateur qui n'apprend rien converge vers l'équipondéré).

Mesure sur un marché alternant momentum et retour à la moyenne :

| | Sharpe | Drawdown max |
|---|---|---|
| équipondéré constant | −2.02 | −15.4 % |
| meilleure stratégie fixe | +0.54 | −15.9 % |
| **allocateur RL** | **+1.23** | **−6.8 %** |

L'agent reste **hors marché 28 % du temps**. Savoir ne pas trader est une compétence.

### `train.py` — entraîner l'agent

```bash
python scripts/train.py --config configs/eurusd_h1.yaml
```

Entraîne le Rainbow DQN et exporte le modèle dans `runs/<nom>/`.

### `backtest.py` — rejouer un modèle sur une période

```bash
python scripts/backtest.py --model runs/v1 --csv data/EURUSD_H1.csv
```

### `walkforward.py` — le protocole de référence

```bash
python scripts/walkforward.py --csv data/EURUSD_H1.csv
```

Ré-entraîne périodiquement, comme en production. **C'est ce résultat qui compte**, pas
celui d'un backtest en une passe.

### `validate.py` — « est-ce distinguable du hasard ? »

```bash
python scripts/validate.py --returns runs/walkforward/oos_returns.csv --trials 40
```

> ⚠️ `--trials` est le nombre de configurations que vous avez **réellement essayées**.
> Le sous-déclarer rend le Deflated Sharpe faussement rassurant. C'est le paramètre le
> plus facile à tricher et le plus coûteux à ignorer.

| Outil | Question |
|---|---|
| Deflated Sharpe | le Sharpe survit-il à la correction du nombre d'essais ? |
| PBO (CSCV) | le meilleur in-sample finit-il sous la médiane out-of-sample ? |
| Bootstrap par blocs | quel est l'intervalle de confiance du Sharpe ? |
| Monte-Carlo du drawdown | quel drawdown provisionner ? |
| Reality Check de White | la meilleure de N stratégies bat-elle le hasard ? |

### `sweep.py` — explorer des hyperparamètres

Chaque configuration essayée **compte** dans `--trials` de `validate.py`. Notez-les.

### `monitor.py fit` — figer la référence de surveillance

```bash
python scripts/monitor.py fit \
    --model runs/v1 \
    --data data/EURUSD_H1.csv --end 2024-12-31 \
    --returns runs/walkforward/oos_returns.csv --horizon 1500
```

À lancer **une fois**, après l'entraînement. Écrit `reference.json` (distribution des
features) et `envelope.json` (performance attendue) à côté du modèle.

### `serve.py` — le serveur d'inférence

```bash
python scripts/serve.py --model runs/v1          # dry-run par défaut
```

### `monitor.py report` / `verify` — surveiller

```bash
python scripts/monitor.py report --model runs/v1 --html supervision.html
python scripts/monitor.py verify --journal runs/v1/audit.jsonl
```

---

## 5. Le parcours complet sur vos données

### Étape 0 — obtenir les données

Depuis MetaTrader 5 : **Outils → Centre d'historique**, choisir le symbole et l'unité de
temps, exporter en CSV. Prenez au moins **3 ans**, idéalement 10.

Format attendu :

```csv
time,open,high,low,close,volume,spread
2020-01-02 00:00:00,1.12105,1.12140,1.12080,1.12122,1250,8
```

### Étape 1 à 8

```bash
# 1. Y a-t-il du signal ?                    → si non, STOP, changez les features
python scripts/probe.py --csv data/EURUSD_H1.csv

# 2. Quelles hypothèses survivent aux frais ?
python scripts/screen.py --csv data/EURUSD_H1.csv

# 3. Quel régime ? quelle stratégie dedans ?
python scripts/regime.py --csv data/EURUSD_H1.csv

# 4. Filtrer les signaux (meta-labeling)
python scripts/meta.py --csv data/EURUSD_H1.csv --importance

# 5. Répartir le capital (RL)
python scripts/allocate.py --csv data/EURUSD_H1.csv

# 6. Protocole de référence : walk-forward
python scripts/walkforward.py --csv data/EURUSD_H1.csv

# 7. Est-ce distinguable du hasard ?         → si non, STOP, ne déployez pas
python scripts/validate.py --returns runs/walkforward/oos_returns.csv --trials 40

# 8. Figer la référence de surveillance
python scripts/monitor.py fit --model runs/v1 --data data/EURUSD_H1.csv \
                              --returns runs/walkforward/oos_returns.csv
```

**Les deux points d'arrêt sont réels.** Si l'étape 1 ne trouve pas de signal, ou si
l'étape 7 ne rejette pas le hasard, le résultat de l'outil est « non ». C'est sa valeur,
pas son échec.

---

## 6. Brancher sur MetaTrader 5

### Comment ça communique

```
   MetaTrader 5                            Python
   ┌──────────────┐    TCP + JSON      ┌──────────────┐
   │ QBotBridge   │ ─── {"predict"} ─► │  serve.py    │
   │     .mq5     │ ◄── {exposition} ──│              │
   └──────────────┘                    └──────────────┘
     sockets natifs — AUCUNE DLL
```

**Pourquoi pas de DLL ?** Une DLL impose `#import` dans l'EA : incompatible prop-firms,
refusé sur le Market MQL5, déploiement compliqué. MQL5 a des sockets natifs depuis la
build 2085. Et à l'échelle d'une barre H1, la latence réseau locale (< 1 ms) est sans
commune mesure avec celle du courtier (30–200 ms).

### Installation

1. Copier `mql5/QBotBridge.mq5` dans `MQL5/Experts/` du dossier de données MetaTrader.
2. Compiler dans MetaEditor (F7).
3. **Outils → Options → Expert Advisors** : autoriser `127.0.0.1` dans les WebRequest.
4. Lancer `python scripts/serve.py --model runs/v1`.
5. Glisser l'EA sur le graphique.

### Les réglages qui comptent

| Paramètre | Défaut | À savoir |
|---|---|---|
| `InpDryRun` | **true** | aucun ordre réel. Le laisser vrai longtemps. |
| `InpHistoryBars` | 1200 | doit être ≥ `min_bars` du serveur (message `info`) |
| `InpMaxExposure` | 1.0 | exposition maximale |
| `InpTradeOnNewBarOnly` | true | décider à la clôture de barre, pas à chaque tick |
| `InpStatusEveryBars` | 24 | cadence de la supervision |
| `InpBlockOnCritical` | **false** | bloquer le renforcement sur alerte critique |

> **Pourquoi 1 200 barres ?** Les features ont besoin d'un échauffement (variance ratio,
> rang de volatilité sur 1 000 barres) **plus** la fenêtre du z-score. C'est additif, pas
> un maximum — une erreur qui a réellement coûté un écart de 0.46 sur une feature
> normalisée avant d'être corrigée.

### D'abord : la répétition générale

```bash
python scripts/serve.py --model runs/v1 --replay
```

Fait passer votre historique par le **chemin d'exécution réel** — le serveur, le
protocole, les features, le réseau, les garde-fous. Ce n'est pas un backtest de plus :
c'est le seul moyen de vérifier que la chaîne complète fait ce qu'on croit, avant d'y
engager quoi que ce soit.

Le mode rejeu neutralise uniquement le contrôle de fraîcheur du flux (sinon toute barre
passée serait rejetée). Tous les autres garde-fous restent actifs. **Ne l'activez jamais
sur un compte réel.**

### La mise en production, dans cet ordre

```
1. DRY-RUN         plusieurs semaines. Aucun ordre. On regarde.
2. COMPTE DÉMO     ordres réels sur argent fictif. On compare au backtest.
3. RÉEL MINIMAL    la plus petite taille possible. Des mois.
4. MONTÉE EN TAILLE  progressive, et seulement si les 3 étapes ont tenu.
```

Sauter une étape ne fait pas gagner du temps : cela déplace la découverte du problème
vers le moment où il coûte de l'argent réel.

---

## 7. Surveiller une fois en production

Une fois branché, la question devient : **comment sais-je que ça marche encore ?**

### Le chiffre qui change tout

| Durée en production | Chance de détecter une chute de Sharpe 1.2 → −1.5 |
|---|---|
| 2 semaines | **10 %** |
| 5 mois | 68 % |
| **1 an** | **93 %** |

Il faut **environ un an** pour prouver par la performance qu'une stratégie horaire s'est
effondrée. Corollaire immédiat : **couper une stratégie après deux semaines, c'est
décider sur du bruit.** Dans les deux sens.

D'où le principe : **surveiller la cause, pas seulement l'effet.**

| Couche | Vous prévient en | Ce qu'elle voit |
|---|---|---|
| comportement | immédiat | le bot ne fait plus ce qu'il faisait |
| coûts d'exécution | ~50 trades | le backtest sous-estimait les frais |
| dérive des features | ~250 barres | le marché n'est plus le même |
| performance | ~1 an | confirmation finale |

### Regarder le tableau de bord

```bash
python scripts/monitor.py report --model runs/v1 --html supervision.html
```

Ouvrez le fichier dans un navigateur. **Un seul fichier, aucune dépendance réseau** — il
fonctionne hors ligne, s'archive, et aucune donnée de compte ne sort de la machine.

Ce qu'il montre, de haut en bas : un bandeau d'état · les indicateurs clés · les alertes ·
équité, drawdown, exposition · la distribution des rendements · la dérive feature par
feature · attendu vs réalisé · les coûts réels · l'intégrité du journal.

Détail complet dans **[SUPERVISION.md](SUPERVISION.md)**.

---

## 8. Lire les résultats sans se tromper

### Les cinq réflexes

**1. Le Sharpe seul ne veut rien dire.** Regardez le **Deflated Sharpe** : il corrige du
nombre d'essais. Un Sharpe de 2.5 après 500 configurations testées est moins crédible
qu'un Sharpe de 1.1 obtenu du premier coup.

**2. Regardez la PBO avant tout le reste.** Au-dessus de 0.5, le meilleur choix
in-sample finit sous la médiane out-of-sample : votre sélection est du bruit.

**3. Les frais décident.** Toujours lire les résultats **nets**. Rappel : un trading
aléatoire passe de +0.23 à −14.63 de Sharpe une fois les frais appliqués.

**4. Un intervalle de confiance, pas un chiffre.** Le bootstrap par blocs vous donne la
fourchette. À 300 barres, pour un backtest à Sharpe 1.2, elle va de **−5.8 à +9.4**.

**5. Le walk-forward prime sur le backtest.** Un backtest en une passe suppose que vous
connaissiez les bons paramètres dès le départ. Vous ne les connaissiez pas.

### Un tableau de décision

| Ce que vous voyez | Conclusion |
|---|---|
| PBO > 0.5 | ❌ ne pas déployer — la sélection est du bruit |
| Deflated Sharpe < 0.95 | ❌ ne pas déployer — indistinguable du hasard |
| Sharpe brut élevé, net négatif | ❌ les frais mangent tout |
| Sonde réseau < sonde linéaire | ⚠️ sur-apprentissage — réduire la capacité |
| PSI > 0.25 sur plusieurs features | ⚠️ le marché a changé — envisager de réentraîner |
| Coûts réels > 2× le modèle | ⚠️ recalibrer le modèle de coûts, revoir le courtier |
| Tout est vert | ✅ dry-run, puis démo, puis réel minimal |

---

## 9. Problèmes fréquents

**« L'EA n'arrive pas à se connecter »**
Vérifier que `serve.py` tourne, que le port correspond (8912 par défaut), et que
`127.0.0.1` est autorisé dans **Outils → Options → Expert Advisors**.

**« historique insuffisant : N barres reçues, M requises »**
Augmenter `InpHistoryBars` au-delà du `min_bars` renvoyé par le message `info`.

**« reference.json absent : détection de dérive INACTIVE »**
Normal si vous n'avez pas lancé `monitor.py fit`. Le serveur fonctionne, mais sans
détection de dérive ni confrontation attendu/réalisé.

**« Le RL fait moins bien qu'une régression linéaire »**
Lancer `probe.py`. C'est presque toujours du sur-apprentissage : réduire `hidden_sizes`,
augmenter `weight_decay`, réduire `window`. Mesuré sur ce dépôt : `weight_decay` de 0 à
1e-3 fait passer la corrélation de −0.003 à **+0.179**.

**« Toutes mes stratégies ont un Sharpe négatif »**
C'est le résultat le plus fréquent, et il est probablement juste. Le marché ne doit rien
à personne. Vérifiez d'abord que ce n'est pas un problème de coûts sur-estimés, puis
acceptez le verdict.

**« Le tableau de bord affiche n/a partout »**
Pas assez de barres. Les métriques de performance demandent au moins 30 barres, la
dérive au moins 100. C'est délibéré : un Sharpe calculé sur 10 points n'informe sur rien.

---

## Pour aller plus loin

- **[METHODOLOGIE.md](METHODOLOGIE.md)** — pourquoi la plupart des bots échouent. À lire
  en premier si vous voulez comprendre plutôt qu'exécuter.
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — structure du code et décisions de conception.
- **[INTEGRATION_MT5.md](INTEGRATION_MT5.md)** — protocole détaillé et diagnostic.
- **[SUPERVISION.md](SUPERVISION.md)** — la surveillance de production en détail.

---

## Le mot de la fin

Le trading à effet de levier fait perdre de l'argent à la majorité des comptes retail.
Ce dépôt est un **outil de recherche** : il rend le sur-apprentissage *mesurable*, il ne
le supprime pas.

Un résultat qui ne passe pas `scripts/validate.py` ne doit pas être déployé, même s'il
est magnifique sur un graphique. **Surtout** s'il est magnifique sur un graphique.

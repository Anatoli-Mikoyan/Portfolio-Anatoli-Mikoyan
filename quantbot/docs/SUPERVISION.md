# Supervision de production (§17)

> **En une phrase.** Un modèle ne tombe jamais en panne : il continue de répondre, avec
> la même assurance, alors que le marché qu'on lui présente n'est plus celui sur lequel
> il a appris. Cette couche existe pour que vous l'appreniez avant votre relevé de compte.

---

## Le problème, chiffré

C'est le résultat le plus important de tout ce document, et il est contre-intuitif.

**Combien de temps faut-il pour prouver qu'une stratégie s'est effondrée ?**

Mesure faite sur ce dépôt (`tests/test_monitoring.py`), en barres horaires, pour une
stratégie qui passe d'un Sharpe de 1.2 à un Sharpe de −1.5 :

| Durée observée | Détection par la performance |
|---|---|
| 300 barres (~2 semaines) | **10 %** |
| 1 000 barres (~7 semaines) | 33 % |
| 3 000 barres (~5 mois) | 68 % |
| 6 240 barres (**1 an**) | **93 %** |

Autrement dit : **il faut environ un an de production pour établir statistiquement qu'une
stratégie horaire a perdu deux points et demi de Sharpe.** Ce n'est pas un défaut de
l'outil, c'est le rapport signal/bruit des rendements financiers. Aucune astuce ne le
contourne.

D'où l'architecture de cette couche : **surveiller la cause, pas seulement l'effet.**

```
            RAPIDE                                                    LENT
    ┌──────────────────┬──────────────────┬─────────────────┬──────────────────┐
    │  comportement    │  coûts réels     │  dérive des     │  performance     │
    │  (immédiat)      │  (~50 exécutions)│  features       │  (~1 an)         │
    │                  │                  │  (~250 barres)  │                  │
    ├──────────────────┼──────────────────┼─────────────────┼──────────────────┤
    │ modèle bloqué,   │ le backtest      │ le marché n'est │ le seul juge     │
    │ garde-fous qui   │ sous-estimait    │ plus celui de   │ qui compte —     │
    │ écrasent tout,   │ le slippage      │ l'entraînement  │ et le plus lent  │
    │ latence          │                  │                 │                  │
    └──────────────────┴──────────────────┴─────────────────┴──────────────────┘
         alerte en          alerte en           alerte en          confirmation
         quelques           quelques            quelques           finale
         barres             jours               semaines
```

Les trois premières colonnes vous préviennent pendant que la quatrième accumule encore
des données. C'est tout l'intérêt.

---

## Les six briques

| Module | Ce qu'il surveille | La question à laquelle il répond |
|---|---|---|
| `drift.py` | les **entrées** du modèle | « Le marché ressemble-t-il encore à celui de l'entraînement ? » |
| `tca.py` | l'**exécution** | « Les frais réels correspondent-ils au modèle du backtest ? » |
| `store.py` | le **comportement** | « Le bot fait-il ce qu'il faisait au backtest ? » |
| `reconciliation.py` | la **performance** | « Le résultat est-il dans l'enveloppe attendue ? » |
| `journal.py` | la **traçabilité** | « Pourquoi cette position, ce jour-là ? Le registre est-il intact ? » |
| `alerts.py` | la **restitution** | « Qu'est-ce qui mérite qu'on interrompe ce qu'on fait ? » |

---

## 1. Dérive des features — `drift.py`

### Le principe

À l'entraînement, on fige une **photographie** de la distribution de chaque feature :
le découpage en 10 cases de même effectif, et le comptage de référence. En production,
on compte à nouveau sur une fenêtre glissante et on compare.

**La référence est figée et versionnée avec le modèle.** Ce point est central : comparer
la fenêtre live aux 250 barres précédentes ne détecterait qu'un changement brutal. Une
dérive lente — celle qui tue un modèle en six mois — passerait inaperçue, puisque la
référence dériverait avec elle.

### Les mesures

| Mesure | Lecture | Seuils |
|---|---|---|
| **PSI** | divergence de Jeffreys entre les deux histogrammes | < 0.10 stable · 0.10–0.25 modéré · > 0.25 **critique** |
| **Jensen-Shannon** | même idée, mais **bornée dans [0, 1]** | sert de score global agrégeable |
| **KS (1 échantillon)** | test contre la répartition de référence connue | p-value |
| **Δ (σ)** | décalage de moyenne, en écarts-types de référence | lecture directe |

Trois détails qui font la différence entre un détecteur utile et un détecteur ignoré :

1. **Lissage de Jeffreys (0.5 par case).** Sans lui, une case vide côté production rend
   le PSI infini et le tableau de bord illisible.

2. **Correction d'autocorrélation.** Les features financières sont fortement
   autocorrélées : 250 barres horaires ne contiennent pas 250 informations
   indépendantes. Pour ρ₁ = 0.9 — banal pour une volatilité glissante — elles n'en valent
   que **13**. Un Kolmogorov-Smirnov brut déclarerait donc une dérive significative en
   permanence, et l'équipe couperait les alertes. On corrige par la taille effective
   `n·(1−ρ)/(1+ρ)`.

3. **Features constantes encadrées.** Une feature figée à l'entraînement qui se remet à
   bouger est le cas que les découpages naïfs manquent — toute la masse tombe dans la
   même case et le PSI reste nul. Ici, la valeur constante est encadrée : « en dessous »,
   « à la valeur » et « au-dessus » sont distingués. Mesuré : **PSI = 15.3** quand la
   feature revient à la vie.

### Ce que ça donne

Mesures sur 300 barres de production contre 5 000 barres de référence :

| Situation en production | PSI de la feature touchée | Verdict |
|---|---|---|
| rien ne change | 0.03 – 0.05 | stable |
| décalage de moyenne de 1.2 σ | **1.52** | critique |
| volatilité × 3 (moyenne inchangée) | **1.07** | critique |
| feature gelée qui redémarre | **15.3** | critique |
| drapeau rare passant de 3 % à 40 % | **1.14** | critique |

Le deuxième cas mérite d'être noté : la moyenne ne bouge pas. Un simple test de moyenne
manquerait complètement une volatilité qui triple.

---

## 2. Détecteur séquentiel de Page-Hinkley

Le test de Page-Hinkley détecte **en ligne** un décrochage de moyenne. C'est le test
séquentiel du rapport de vraisemblance appliqué à un saut : à taux de fausse alarme égal,
il détecte plus vite que n'importe quel seuil sur moyenne glissante. Et sur un compte
réel, le délai de détection est exactement ce qu'on paie.

### Il est calibré, pas deviné

Un détecteur non calibré est pire que pas de détecteur : soit il produit du bruit qu'on
apprend à ignorer, soit un silence rassurant et faux.

- δ (tolérance) = **moitié de la dérive à détecter** — règle classique du CUSUM.
- λ (seuil) se déduit du budget de fausses alarmes via l'approximation de Siegmund :

```
ARL₀ ≈ [exp(2δ(λ + 1.166)) − 2δ(λ + 1.166) − 1] / (2δ²)
```

`PageHinkley.calibrate(magnitude, arl0)` résout λ numériquement. Vérification :

| Amplitude visée | ARL₀ demandé | δ obtenu | λ obtenu | ARL₀ mesuré | Délai mesuré |
|---|---|---|---|---|---|
| 0.5 σ | 10 000 | 0.250 | 14.5 | ~16 000 | 51 barres (théorie 58) |
| 0.025 σ | 12 480 | 0.0127 | 95.1 | — | 7 518 barres |

La deuxième ligne est la traduction honnête d'« une chute de 2 points de Sharpe en barres
horaires ». Elle vaut 0.025 σ de dérive **par barre**. C'est minuscule, et c'est pourquoi
la détection prend un an.

### Le piège qu'il fallait éviter

Un Page-Hinkley qui réapprend sa propre moyenne **suit la dégradation au lieu de la
signaler**. Il reste muet pendant que la stratégie s'effondre — en toute logique, et en
toute inutilité.

D'où deux modes explicitement séparés :

- **adaptatif** — pour ce qui n'a pas de référence (latence, spread, PSI) ;
- **référence fixe** (`ref_mean`, `ref_std`) — **obligatoire pour la performance**, où la
  référence vient du backtest.

Le test `test_adaptive_page_hinkley_is_blind_to_the_drift_it_learns` fige ce comportement.

---

## 3. Coûts d'exécution — `tca.py`

Le backtest **suppose** un modèle de coûts. La production révèle les vrais. L'écart est la
première cause de mort d'une stratégie « qui marchait sur historique » : le Sharpe ne
s'effondre pas d'un coup, il fond de quelques dixièmes, silencieusement.

La mesure de référence est l'**implementation shortfall** de Perold (1988) : l'écart entre
le portefeuille papier — exécuté instantanément au prix du moment de la décision — et le
portefeuille réel. Avantage décisif sur la comparaison au VWAP : elle **n'est pas
manipulable**. On peut battre le VWAP en retardant ses ordres ; on ne peut pas battre le
prix qui existait au moment de la décision.

```
IS  =  coût de délai  +  coût de spread  +  commission  +  impact/résidu
       ↑                 ↑                  ↑              ↑
       infrastructure    courtier /         courtier       taille de l'ordre
       lente             type d'ordre                      (et bruit)
```

Vérification de la décomposition sur des exécutions construites (0.8 / 1.0 / 2.0 bps) :

```
Délai (infrastructure)      0.80 bps
Spread (franchissement)     1.00 bps
Commission                  0.00 bps
Impact / résidu             2.00 bps
                          ─────────
Implementation shortfall    3.80 bps      ← somme exacte
```

### Les deux chiffres à lire

- **Coût excédentaire annuel** — turnover annuel × excès par exécution, en % du capital.
  Se compare directement au rendement espéré.
- **Rabais de Sharpe** — le même excès, rapporté à la volatilité annuelle : combien de
  points de Sharpe le backtest surestime.

Exemple mesuré : coûts réels 2.5 × le modèle, turnover 0.157/barre → **15.0 % du capital
par an** en frais non prévus, soit **1.29 point de Sharpe** évaporé. Le test de Student
apparié sur les écarts (réalisé − modélisé) dit si c'est systématique ou du bruit.

---

## 4. Attendu vs réalisé — `reconciliation.py`

### L'enveloppe bootstrap

On ne compare **pas** le Sharpe live à celui du backtest : ce serait comparer une
réalisation à une moyenne. On simule par bootstrap par blocs (Politis & Romano) des
milliers de trajectoires **de la même longueur que la période live**, et on regarde dans
quel centile tombe le résultat réel.

Le résultat est brutal et instructif. Pour un backtest à Sharpe 1.2, en barres horaires :

| Horizon | Enveloppe du Sharpe (5e → 95e centile) |
|---|---|
| 300 barres | **−5.8 → +9.4** |
| 6 240 barres (1 an) | −0.31 → +2.7 |

**À 300 barres, un Sharpe live de −2 est parfaitement compatible avec un backtest à
+1.2.** Toute personne qui coupe une stratégie après deux semaines décide sur du bruit.
Le test `test_envelope_is_wide_at_short_horizons` fige ce fait pour qu'aucune évolution
du code ne laisse croire à une détection rapide.

### Le rejeu des décisions

`replay_mismatch` réexécute les entrées journalisées à travers le modèle courant et compte
les désaccords. Il répond à une question à laquelle rien d'autre ne répond : **« le
serveur qui tourne est-il bien le modèle qu'on a validé ? »**

Un déploiement partiel, un fichier de features d'une version antérieure, une bibliothèque
mise à jour sous le pied du modèle — aucun de ces incidents ne produit d'erreur. Ils
produisent des décisions différentes, silencieusement.

**L'attente est zéro désaccord.** Pas « peu » : zéro. Un modèle déterministe rejoué sur
ses propres entrées doit reproduire ses sorties à l'identique. Toute valeur non nulle est
un incident, pas une statistique.

---

## 5. Journal d'audit — `journal.py`

Toute table de trading régulée doit pouvoir répondre, des mois après, à « pourquoi cette
position à cette seconde-là ? ». Le règlement délégué **MiFID II RTS 6** l'impose aux
entreprises pratiquant le trading algorithmique. Le besoin existe sans aucun régulateur :
quand un modèle perd de l'argent, la première chose à établir est si le code a fait ce
qu'il croyait faire.

Deux propriétés séparent un journal d'audit d'un fichier de logs :

**Chaînage cryptographique.** Chaque entrée porte l'empreinte SHA-256 de la précédente.

```
entrée 0 ── hash(GENESIS | 0 | ts | contenu) ──┐
entrée 1 ── hash(     ↑    | 1 | ts | contenu) ─┐
entrée 2 ── hash(          ↑ | 2 | ts | contenu)
```

Modifier ou supprimer une ligne casse la chaîne **à cet endroit précis**, et `verify()` le
dit. On ne rend pas la falsification impossible — on la rend **détectable**, ce qui suffit
à ce qu'elle ne passe pas inaperçue.

```
Journal intègre : 20 entrées, chaîne continue.
JOURNAL COMPROMIS à l'entrée seq=4 — contenu modifié après écriture
JOURNAL COMPROMIS à l'entrée seq=6 — chaînage rompu : entrée supprimée ou insérée
```

L'**empreinte de tête** recopiée ailleurs (un message, un dépôt) transforme la détection
en preuve : l'historique ne peut plus être réécrit sans contredire une valeur publiée.

**Rejouabilité.** L'entrée contient l'**entrée** du modèle, pas seulement sa sortie — c'est
ce qui rend `replay_mismatch` possible.

Format : JSON Lines. Une ligne tronquée par un arrêt brutal n'invalide que sa propre
entrée. Un redémarrage **reprend** la chaîne (il ne repart pas de la genèse, ce qui
signalerait une fausse falsification).

---

## 6. Alertes — `alerts.py`

Un système d'alerte se juge à une seule chose : **est-il encore lu au bout de six mois ?**

**Trois niveaux, trois conduites.** INFO se consulte · WARN se regarde dans la journée ·
CRITIQUE interrompt ce qu'on fait. Plus fin, on n'utilise pas les niveaux ; plus
grossier, une latence en hausse et un drawdown hors limite se retrouvent au même rang.

**Temporisation croissante.** Une condition vraie pendant 2 000 barres doit produire
quelques alertes, pas quarante. Chaque répétition **double** la temporisation (50, 100,
200, 400…) : une condition persistante coûte un nombre d'alertes **logarithmique** en sa
durée. Mesuré sur le scénario de démonstration : **99 alertes → 20**, pour la même
information. Un long silence remet la temporisation à sa valeur de base — une alerte
revenue après des mois est une nouvelle alerte.

**Règles déterministes.** Aucun seuil appris, aucun modèle dans la couche de surveillance.
Le jour où le modèle surveillé se trompe, on a besoin d'un juge dont le comportement est
prévisible et lisible dans le code.

### Le catalogue

| Code | Niveau | Déclencheur |
|---|---|---|
| `latence_critique` | ‼ | ≥ 1 % des réponses hors délai, ou p99 au-delà du seuil |
| `flux_prix_perime` | ‼ | dernière barre trop ancienne — flux probablement mort |
| `journal_compromis` | ‼ | la chaîne d'empreintes ne vérifie plus |
| `drawdown_critique` | ‼ | drawdown au-delà de la limite |
| `derive_generalisee` | ‼ | plus de N features au-delà de PSI 0.25 |
| `degradation_confirmee` | ‼ | enveloppe **et** détecteur séquentiel concordent |
| `couts_execution_critiques` | ‼ | coûts réels ≥ 2.5 × le modèle, significatif |
| `derive_feature` | ! | au moins une feature en dérive critique |
| `sous_performance` | ! | Sharpe sous le 5e centile de l'enveloppe |
| `decrochage_sequentiel` | ! | Page-Hinkley déclenché à la baisse |
| `sharpe_glissant_bas` | ! | Sharpe glissant sous le plancher |
| `modele_sur_contraint` | ! | les garde-fous modifient ≥ 40 % des décisions |
| `modele_inactif` | ! | aucune exposition sur ≥ 98 % des barres |
| `changement_regime` | · | nouveau régime confirmé sur N barres |

La règle de latence mérite un mot : elle se juge sur un **taux de dépassement**, pas
seulement sur le p99. Avec 0.8 % de réponses à 900 ms, le p99 reste bas et ne dit rien —
alors qu'une réponse sur cent arrivée après la barre est un incident bien réel.

### Le moniteur ne coupe rien

`LiveMonitor` **ne ferme aucune position.** Il expose `should_halt`, que le serveur est
libre de consulter, et que la configuration relie ou non au coupe-circuit
(`halt_on_critical`, **faux par défaut**).

C'est délibéré : une couche d'observation qui peut liquider un portefeuille devient
elle-même un risque opérationnel, et le premier seuil mal réglé coûterait un compte. On
l'active après avoir observé le comportement des alertes sur un compte réel.

Même logique côté EA (`InpBlockOnCritical`, faux par défaut) : quand il est actif, il
bloque le **renforcement**, jamais la réduction. Un mode de sécurité qui empêcherait aussi
de fermer serait plus dangereux que le problème qu'il traite.

---

## Mode d'emploi

### Étape 1 — figer la référence (une fois, après l'entraînement)

```bash
python scripts/monitor.py fit \
    --model runs/best \
    --data data/EURUSD_H1.csv \
    --end 2024-12-31 \
    --returns runs/walkforward/oos_returns.csv \
    --horizon 1500
```

Écrit `runs/best/reference.json` et `runs/best/envelope.json` **à côté du modèle**.

> ⚠️ `--data` doit être **exactement la période d'entraînement**, et `--returns` les
> rendements **out-of-sample** du walk-forward. Utiliser les rendements in-sample
> donnerait une enveloppe trop optimiste : toute production semblerait décevante.

### Étape 2 — servir avec supervision

```bash
python scripts/serve.py --model runs/best --dry-run
```

La supervision s'active **toute seule** si les deux fichiers sont là. Sinon le serveur
le dit explicitement plutôt que de laisser croire à une surveillance complète :

```
AVERTISSEMENT | reference.json absent de runs/best : détection de dérive INACTIVE
```

### Étape 3 — regarder

Trois façons, selon où vous êtes.

**Depuis MetaTrader** — l'EA interroge le serveur toutes les `InpStatusEveryBars` barres
(24 par défaut) et affiche un panneau sur le graphique :

```
QBot — supervision
Barres observées : 2 000
Sharpe glissant  : -0.55
Drawdown         : -2.87 %
Latence p99      : 945 ms
Dérive features  : critique (4 critiques)
Attendu/réalisé  : conforme à l'attendu
Coûts exécution  : COÛTS RÉELS ≫ MODÈLE (2.54x)
Alertes          : 20 — pire : critical
Journal d'audit  : intègre
```

**En Python** — `{"type": "status"}` et `{"type": "alerts"}` sur le socket.

**Hors ligne** — le tableau de bord HTML :

```bash
python scripts/monitor.py report --model runs/best --html supervision.html
```

Un seul fichier, **aucune dépendance** : ni CDN, ni serveur, ni bibliothèque de
graphiques. Les courbes sont du SVG produit à la main. Trois raisons, toutes
opérationnelles : un tableau de bord qui dépend du réseau est indisponible exactement le
jour où quelque chose ne va pas · un fichier unique s'archive et se relit dans deux ans ·
aucune donnée de compte ne sort de la machine.

### Étape 4 — vérifier le registre

```bash
python scripts/monitor.py verify --journal runs/best/audit.jsonl
```

---

## Réglages

| Paramètre | Défaut | Ce qu'il fait |
|---|---|---|
| `window` | 500 | barres retenues pour les métriques glissantes |
| `drift_window` | 250 | fenêtre de comparaison de distribution |
| `drift_min_samples` | 100 | en deçà, **aucun** verdict rendu |
| `drift_every` | 25 | cadence de recalcul du verdict |
| `psi_warn` / `psi_critical` | 0.10 / 0.25 | seuils industriels du PSI |
| `max_drifted_features` | 3 | au-delà, la dérive devient « généralisée » |
| `delta_sharpe` | 2.0 | chute de Sharpe que la détection doit voir |
| `arl0` | 31 200 | budget de fausses alarmes (≈ 5 ans en H1) |
| `alert_cooldown_bars` | 30 | temporisation de base, doublée à chaque répétition |
| `halt_on_critical` | **false** | relie ou non le moniteur au coupe-circuit |
| `journal_path` | `<modèle>/audit.jsonl` | trace d'audit chaînée |

Le choix d'`arl0` vient d'une **mesure**, pas d'une convention (150 tirages, backtest à
Sharpe 1.2, production à −1.5, barres horaires) :

| ARL₀ | Fausses alarmes / an | Détection à 1 an |
|---|---|---|
| 12 480 | 18 % | 93 % |
| **31 200** | **8 %** | **86 %** ← retenu |
| 62 400 | 3 % | 60 % |

---

## Ce que cette couche ne fait pas

- **Elle ne prédit rien.** Elle constate, et elle constate en retard — c'est la nature
  du problème, pas une limite d'implémentation.
- **Elle ne remplace pas le coupe-circuit** (`qbot/risk/guards.py`), qui agit dans la
  boucle de décision, barre par barre. Le moniteur observe et signale.
- **Elle ne détecte pas ce qui n'est pas mesuré.** Un choc macro, un changement de
  courtier, une modification du contrat du symbole : rien de tout cela n'apparaît dans
  les features.
- **Elle ne dit pas quoi faire.** Un PSI de 0.4 sur la volatilité peut vouloir dire
  « réentraîner », « réduire la taille » ou « ne rien faire, c'est un mois d'août ».
  L'arbitrage reste humain, et c'est volontaire.

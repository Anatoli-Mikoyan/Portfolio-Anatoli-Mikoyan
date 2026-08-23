# ADR-001 — Prévention structurelle du look-ahead bias

**Statut :** accepté (étape 1)
**Portée :** couche `data/`, contrat imposé aux couches `strategy/` et `backtest/`

---

## Contexte

Le look-ahead bias — utiliser à l'instant *T* une information qui n'était pas
disponible à *T* — est la première cause de backtests faux. Il a trois
propriétés qui le rendent particulièrement dangereux :

1. **Il est silencieux.** Aucune exception, aucun avertissement. Le backtest
   tourne, produit des courbes lisses et des métriques flatteuses.
2. **Il améliore les résultats.** Contrairement à un bug ordinaire, il ne se
   manifeste pas par un crash ou un chiffre absurde, mais par une performance
   supérieure — c'est-à-dire par exactement ce qu'on espérait trouver. Le biais
   de confirmation fait le reste.
3. **Il se réintroduit.** Même dans une base saine, une seule ligne suffit :
   `df.rolling(20).mean()` calculé sur la série complète, un `.shift(-1)`, un
   `.max()` global pour normaliser, un `train_test_split` sans `shuffle=False`.

Une convention d'équipe (« ne regardez pas le futur ») ne tient pas à l'échelle.
Une revue de code non plus : le biais est indétectable à la lecture d'un diff
isolé. Il faut une contrainte que l'architecture applique elle-même.

## Décision

Le futur n'est pas *interdit* : il est rendu **inexprimable dans l'API** offerte
aux stratégies. Quatre mécanismes, du plus fort au plus faible.

### 1. Ségrégation des types

Deux types distincts, et un seul chemin entre eux.

| Type | Contient | Détenteur légitime |
|---|---|---|
| `MarketData` | la série complète, futur inclus | le moteur de backtest, exclusivement |
| `HistoryView` | l'historique borné à un instant | les stratégies |

Aucune méthode de `HistoryView` ne renvoie un `MarketData`. La règle se vérifie
en lisant les **signatures**, pas les corps de fonction — ce qui la rend
vérifiable automatiquement et robuste à une revue distraite.

Le moteur, lui, a besoin des barres futures : il simule des exécutions
différées. C'est légitime, et cantonné à `MarketData.execution_bar(index)`, dont
la docstring établit qui a le droit de l'appeler.

### 2. Adressage relatif — le mécanisme qui compte le plus

On n'indexe pas une vue par une position absolue, mais par une **ancienneté** :

```python
view.bar(0)     # dernière barre close
view.bar(1)     # la précédente
view.bar(-1)    # LookaheadError
view.close(20)  # les 20 derniers closes
```

Il n'existe aucune façon d'écrire « la barre suivante ». Le réflexe `i + 1`,
qui est le vecteur d'introduction du biais dans le code écrit vite, n'a pas de
traduction dans cette API. Une tentative lève `LookaheadError` avec un message
qui explique où se trouve réellement le prix d'exécution.

### 3. Borne physique plutôt que conventionnelle

Un `HistoryView` ne stocke pas la série complète accompagnée d'un index qu'il
faudrait penser à respecter. Il stocke des **tranches numpy** dont la longueur
*est* la fenêtre visible :

```python
self.__close = store.close[:end]   # len == end, par construction
```

Dépasser la borne lève `IndexError` au niveau de numpy, sans qu'aucune
vérification n'ait eu à être écrite — donc sans qu'on ait pu oublier de
l'écrire. Le coût est nul : une tranche numpy ne copie rien.

Les tableaux sont en outre marqués non modifiables : une stratégie ne peut pas
corrompre les données partagées par les autres.

### 4. Ajustement des prix point-in-time

C'est le vecteur de biais le plus discret, et celui que presque aucun projet
amateur ne traite.

`yfinance` renvoie par défaut des prix rétro-ajustés avec **toutes** les
opérations sur titre connues aujourd'hui. Le close d'Apple au 2 janvier 2015
y apparaît vers 24 $, alors qu'il cotait 109 $ : la série intègre le split
4-pour-1 de 2020. Un backtest « au 2015 » raisonne donc sur un niveau de prix
qui n'a jamais existé, et dont le facteur d'ajustement encode une information
de 2020.

Le moteur conserve les prix bruts et les opérations séparément, puis calcule le
facteur d'ajustement **relatif au curseur** :

```
prix_ajusté(i | curseur t) = prix_brut(i) × m[i] / m[t]
```

où `m` est le multiplicateur rétro classique. Deux conséquences :

- au curseur, le facteur vaut exactement 1 : le prix courant reste le prix
  réellement négociable, ce dont dépendent la taille de position, les seuils
  absolus et les frais proportionnels ;
- l'historique est corrigé uniquement par les opérations déjà survenues — soit
  précisément ce qu'un opérateur voyait sur son écran ce jour-là.

Les politiques rétro restent disponibles, mais exigent `allow_lookahead=True`,
émettent un avertissement, et portent un drapeau
`is_lookahead_contaminated` que le rapport de backtest affichera.

### Décision annexe : labellisation temporelle

Toute barre est étiquetée par sa **clôture réelle** (16:00 America/New_York,
13:00 les demi-séances), en UTC. `yfinance` date ses barres journalières à
minuit heure de place ; conserver ce label revient à connaître le close du soir
dès minuit — une séance entière de futur, offerte à toutes les stratégies, sans
la moindre erreur levée.

## Vérification

Une architecture qui *prétend* empêcher le look-ahead sans le prouver ne vaut
pas mieux qu'une convention. La propriété retenue est **l'équivalence par
troncature** :

> Pour tout instant *t*, ce que le moteur expose à *t* doit être identique, bit
> à bit, à ce qu'il exposerait si les données postérieures à *t* n'existaient pas.

Elle est vérifiée de trois façons complémentaires dans `tests/test_no_lookahead.py` :

| Méthode | Ce qu'elle attrape |
|---|---|
| **Troncature réelle** — reconstruire un jeu de données s'arrêtant à *t* | toute dépendance au contenu futur, quelle qu'en soit la voie |
| **Empoisonnement** — remplacer le futur par des `NaN` | les accès passant par un chemin non couvert par la borne (agrégat pandas, cache mal borné) : un `NaN` contamine tout calcul qui le touche |
| **Divergence de futur** — deux séries identiques jusqu'à *t*, différant ensuite par un split | le biais introduit par l'ajustement des prix lui-même |

Le test `test_lempoisonnement_detecte_bien_un_tricheur` est la contre-épreuve :
il vérifie que le détecteur **échoue** sur du code qui triche délibérément. Un
test de sécurité qui ne se déclenche jamais ne prouve rien.

## Conséquences

**Acquis**

- Le look-ahead accidentel — la totalité des cas réels — devient très difficile
  à introduire, et détectable mécaniquement s'il l'est.
- Les niveaux de prix vus par une stratégie sont ceux qui ont réellement coté.
- La propriété est testable en continu, donc elle ne se dégrade pas.

**Coûts assumés**

- Une stratégie ne peut pas recevoir un `DataFrame` complet et faire du pandas
  vectorisé sur toute la série. C'est délibéré : c'est exactement le geste qui
  introduit le biais. Les fenêtres glissantes se calculent sur `view.close(n)`.
- Le parcours événementiel est plus lent qu'un backtest vectorisé, de l'ordre
  d'un facteur 10 à 100. Acceptable : l'objectif est de savoir si une stratégie
  est mauvaise, pas d'itérer vite sur mille variantes — cette dernière activité
  étant elle-même une source de sur-ajustement.
- L'ajustement point-in-time coûte une multiplication par fenêtre demandée.
  Négligeable dès lors que les stratégies utilisent `lookback`.

**Limite explicitement reconnue**

Une tranche numpy conserve une référence vers son tableau parent via
`ndarray.base`. Un appelant déterminé peut donc remonter à la série complète.
Ce n'est pas une faille à corriger : aucune API Python n'est étanche à un
appelant hostile, et prétendre le contraire serait malhonnête. L'objectif est
d'éliminer le biais *accidentel* ; le biais délibéré est du ressort de
l'empoisonnement, appliqué en mode audit.

## Ce que cette décision ne résout pas

Le look-ahead au niveau des *données* n'est qu'un des biais qui invalident un
backtest. Restent, à traiter dans les étapes suivantes :

- **le biais du survivant** — `yfinance` n'expose que les titres encore cotés ;
- **le biais de sélection sur l'univers** — reconstituer la composition
  historique d'un indice est hors de portée de cette source ;
- **le sur-ajustement par tests multiples** — le biais que vous introduisez
  vous-même en essayant vingt variantes. Aucune contrainte de typage n'en
  protège : il faudra un journal de recherche et une correction de type
  *deflated Sharpe ratio* (étape 4) ;
- **les révisions rétroactives des sources** — atténuées par l'empreinte
  SHA-256 du cache, qui rend au moins l'écart détectable.

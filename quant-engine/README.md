# quant-engine

Moteur de backtesting et d'exécution algorithmique pour stratégies systématiques.

**Ce projet ne prédit pas les marchés.** C'est une infrastructure de validation,
conçue autour d'une hypothèse : la quasi-totalité des backtests amateurs sont
faux, et ils le sont pour des raisons méthodologiques identifiées et
reproductibles. L'outil est fait pour rendre ces erreurs difficiles à commettre
et impossibles à dissimuler.

Sa valeur se mesure à sa capacité à dire qu'une stratégie est mauvaise.

```
Python 3.11+  ·  mypy --strict  ·  ruff  ·  205 tests  ·  91 % de couverture
```

---

## État d'avancement

Le projet se construit par paliers. Chaque palier est testé et fonctionnel avant
le suivant.

| # | Palier | État |
|---|---|---|
| 1 | Structure, couche données, protection anti-look-ahead | **livré** |
| 2 | Interface stratégie, buy & hold, moteur de backtest avec coûts | à venir |
| 3 | Métriques et rapport HTML | à venir |
| 4 | Walk-forward et out-of-sample sanctuarisé | à venir |
| 5 | Monte Carlo et robustesse paramétrique | à venir |
| 6 | Gestion du risque et kill-switch | à venir |
| 7 | Exécution paper | à venir |
| 8 | Adaptateur Interactive Brokers | à venir |

---

## Démarrage

```bash
cd quant-engine
pip install -e ".[dev]"
make check      # ruff + mypy strict + pytest + couverture
make example    # démonstration reproductible, hors ligne
```

Avec Docker :

```bash
docker compose run --rm example
docker compose run --rm ci
```

## Exemple reproductible

`examples/step1_data.py` tourne sans réseau, sur une source synthétique
déterministe. Extrait de sortie :

```
[3] AJUSTEMENT : POINT-IN-TIME vs RETRO
    date            prix cote    vu en PIT  vu en retro
    2019-01-02         100.00       100.00        25.00
    2020-04-01         139.67       139.67        34.92
    2021-07-01          54.16        54.16        54.16
    2023-12-29          88.63        88.63        88.63
    -> le retro-ajustement montre des prix qui n'ont jamais cote

[5] TENTATIVES D'ACCES AU FUTUR
    view.bar(-1)             -> LookaheadError: bar(offset=-1) designe une barre future
    view.close(999999)       -> InsufficientHistoryError: 999999 barres demandees, 1258 disponibles
    view.close()[n_bars]     -> IndexError: index 1258 is out of bounds for axis 0 with size 1258

[6] EQUIVALENCE PAR TRONCATURE
    futur remplace par des NaN a partir de l'index 630
    vue identique      : True
    aucun NaN infiltre : True
```

Le tableau de la section 3 est le cœur du sujet. Au 2 janvier 2019 le titre
cotait **100,00**. La vue point-in-time montre 100,00. Le rétro-ajustement
classique — celui que renvoie `yfinance` par défaut — montre **25,00** : un prix
qui n'a jamais existé, obtenu en appliquant rétroactivement un split de 2021.

## Utilisation

```python
from datetime import datetime, timezone

from quant_engine.data import AdjustmentPolicy, DataLoader

loader = DataLoader.from_config("configs/data.yaml")
data = loader.load(
    "AAPL",
    datetime(2015, 1, 1, tzinfo=timezone.utc),
    datetime(2024, 1, 1, tzinfo=timezone.utc),
)

print(data.quality.summary())     # anomalies détectées, classées par gravité

for point in data.cursor(AdjustmentPolicy.SPLIT_PIT, warmup=200):
    view = point.history          # borné à la clôture de point.index
    if view.has(200):
        signal = view.close(50).mean() > view.close(200).mean()
```

`view` est un `HistoryView` : il ne contient physiquement rien d'autre que le
passé. `point.history.bar(-1)` lève `LookaheadError`.

---

## Architecture

```
quant-engine/
├── configs/                    configuration YAML (aucun réglage en dur)
├── docs/ADR-001-anti-look-ahead.md
├── examples/step1_data.py
├── src/quant_engine/
│   ├── config.py               chargement YAML typé et validant
│   ├── errors.py               hiérarchie d'exceptions
│   ├── logging_setup.py        logs JSON structurés
│   └── data/
│       ├── types.py            vocabulaire canonique, invariants temporels
│       ├── calendar.py         calendriers de séances (NYSE sans dépendance)
│       ├── corporate_actions.py
│       ├── adjustment.py       ajustement point-in-time  ← cœur méthodologique
│       ├── dataset.py          MarketData / HistoryView / BarCursor
│       ├── quality.py          détecteurs d'anomalies
│       ├── normalize.py        RawSeries → MarketData
│       ├── cache.py            cache Parquet + empreinte SHA-256
│       ├── loader.py           assemblage depuis la configuration
│       └── providers/          yfinance · CSV · synthétique
└── tests/                      205 tests, aucun accès réseau
```

### Prévention du look-ahead bias

Le sujet est traité en détail dans
[`docs/ADR-001-anti-look-ahead.md`](docs/ADR-001-anti-look-ahead.md). En résumé,
le futur n'est pas *interdit* : il est rendu **inexprimable**.

1. **Ségrégation des types.** `MarketData` contient le futur et appartient au
   moteur ; une stratégie ne reçoit que des `HistoryView`. Aucune méthode de
   `HistoryView` ne renvoie un `MarketData` — la règle se vérifie sur les
   signatures, pas sur les corps de fonction.

2. **Adressage relatif.** On adresse une barre par son *ancienneté*, jamais par
   sa position : `view.bar(0)` est la dernière barre close, `view.bar(1)` la
   précédente. Il n'existe aucune façon d'écrire « la barre suivante ». C'est le
   mécanisme le plus important, parce qu'il attaque le réflexe `i + 1` à la
   racine.

3. **Borne physique.** Une vue stocke des tranches numpy dont la longueur *est*
   la fenêtre visible. Dépasser lève `IndexError` au niveau de numpy, sans
   qu'une vérification ait eu à être écrite — donc sans qu'on ait pu oublier de
   l'écrire.

4. **Ajustement point-in-time.** Le facteur d'ajustement est calculé
   relativement au curseur : au prix courant il vaut exactement 1, et
   l'historique n'est corrigé que par les opérations déjà survenues.

**Vérification.** La propriété *équivalence par troncature* — « ce que le moteur
expose à *t* est identique à ce qu'il exposerait si le futur n'existait pas » —
est testée par troncature réelle, par empoisonnement du futur avec des `NaN`, et
par divergence de futur (deux séries identiques jusqu'à *t*, l'une subissant
ensuite un split). Un test de contre-épreuve vérifie que le détecteur **échoue**
bien sur du code qui triche : un garde-fou qui ne se déclenche jamais ne prouve
rien.

### Autres décisions notables

**Étiquetage à la clôture réelle.** Une barre journalière porte le timestamp de
la clôture de séance (21:00 UTC l'hiver, 20:00 l'été, 18:00 les demi-séances),
pas minuit. `yfinance` date ses barres journalières à minuit heure de place ;
conserver ce label donne une séance entière d'avance à toute stratégie, sans
qu'aucune erreur ne soit levée. Le calendrier NYSE est implémenté sans
dépendance et validé contre les décomptes officiels de séances (252 en 2024,
250 en 2023).

**Aucune correction silencieuse.** Chaque anomalie — doublon, `NaN`, OHLC
incohérent, séance manquante, barre figée, saut de prix inexpliqué — est
détectée, classée par gravité et transportée avec le jeu de données jusqu'au
rapport final. Le forward-fill reste possible, mais émet un avertissement
explicite : combler un trou de prix crée une barre à rendement nul, ce qui réduit
la volatilité mesurée et gonfle mécaniquement tout Sharpe calculé dessus.

**Splits non déclarés.** Un cours passant de 400 à 100 sans opération déclarée
n'est pas un krach de −75 % : c'est une série incohérente. Le détecteur bloque
sur les ratios non ambigus (au moins un doublement ou une division par deux) et
se contente d'avertir sur les cas ambigus — un rapport de 0,67 est tout aussi
compatible avec un 3-pour-2 qu'avec une séance à −33 % parfaitement réelle.

**Empreinte de reproductibilité.** Le cache Parquet calcule un SHA-256 de chaque
série servie. Les sources grand public révisent leur historique sans prévenir :
sans cette empreinte, « j'ai obtenu un Sharpe de 1,3 » n'est pas une affirmation
vérifiable.

---

## Limites connues

Elles sont documentées ici plutôt que découvertes plus tard.

**Le biais du survivant n'est pas corrigé, et ne peut pas l'être avec `yfinance`.**
L'API n'expose que les titres encore cotés. Un backtest sur un panier choisi
aujourd'hui exclut mécaniquement les faillites et les radiations : le rendement
en ressort surestimé, souvent de plusieurs points par an. C'est acceptable pour
prototyper, jamais pour engager du capital.

**Aucun univers point-in-time.** Reconstituer la composition historique d'un
indice est hors de portée de cette source. Tout backtest multi-actifs sur une
liste fixe hérite d'un biais de sélection.

**Le sur-ajustement par tests multiples n'est pas traité à ce stade.** Aucune
contrainte de typage ne protège du biais que vous introduisez vous-même en
essayant vingt variantes. C'est l'objet de l'étape 4 (out-of-sample sanctuarisé,
journal de recherche, correction du Sharpe).

**Une tranche numpy garde une référence vers son tableau parent** via
`ndarray.base`. Un appelant déterminé peut donc remonter à la série complète.
L'architecture élimine le look-ahead *accidentel* — la totalité des cas réels ;
le look-ahead délibéré relève de l'empoisonnement du futur, appliqué en audit.

---

## Développement

```bash
make lint     # ruff
make type     # mypy --strict, sans exception pour le code du projet
make test     # pytest
make cov      # couverture, rapport HTML dans htmlcov/
make check    # la chaîne complète
```

Les tests marqués `network` sont exclus par défaut (`make network-test` pour les
jouer). Un test qui dépend de Yahoo Finance n'est pas un test, c'est un
détecteur de panne réseau : toutes les séries de la suite proviennent d'un
générateur déterministe hors ligne, capable d'injecter à la demande splits,
dividendes, trous, valeurs aberrantes et barres figées.

## Licence

MIT.

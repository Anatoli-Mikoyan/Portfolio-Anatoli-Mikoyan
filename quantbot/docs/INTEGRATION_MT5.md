# Intégration MetaTrader 5

## Architecture retenue

```
   MetaTrader 5 (Windows)                    Python (même machine ou VPS)
   ┌────────────────────────┐                ┌───────────────────────────┐
   │  QBotBridge.mq5        │                │  scripts/serve.py         │
   │                        │   TCP + JSON   │                           │
   │  • collecte 1200 barres│ ──────────────►│  FeaturePipeline (le MÊME │
   │  • envoie état compte  │                │    qu'à l'entraînement)   │
   │                        │                │  RainbowAgent / Ensemble  │
   │  • reçoit exposition   │ ◄──────────────│  RiskGuard                │
   │  • convertit en lots   │                │                           │
   │  • exécute via CTrade  │                └───────────────────────────┘
   │  • garde-fous locaux   │
   └────────────────────────┘
```

**Pourquoi séparer ?** Parce que MQL5 ne peut pas exécuter un réseau de neurones
distributionnel, et surtout parce que le code qui décide en réel doit être *exactement*
celui qui a été backtesté. Réimplémenter les features en MQL5 introduirait un écart
entraînement/service invisible, qui ne se manifeste que par des pertes inexpliquées.

**Pourquoi TCP nu et pas ZeroMQ ?** ZeroMQ exige une DLL côté MT5, ce qui bloque le
déploiement chez la plupart des prop-firms et sur le Market MQL5. Les sockets natifs
(`SocketCreate`, disponibles depuis la build 2085) suffisent : sur une barre M15 ou H1,
la latence réseau locale (< 1 ms) est négligeable devant la latence d'exécution du
courtier (30 à 200 ms).

---

## Installation

### 0. Le chemin court

Trois commandes couvrent tout ce que décrivent les sections 1 et 2 ci-dessous :

```bash
python scripts/mt5.py installer    # trouve le dossier de données MetaTrader, copie l'EA
python scripts/mt5.py tester       # rejoue le dialogue de l'EA depuis Python
python scripts/mt5.py demarrer     # lance le serveur (dry-run)
```

`installer` explore `%APPDATA%\MetaQuotes\Terminal\<empreinte>\MQL5\Experts` —
l'empreinte de 32 caractères ne peut pas être devinée, seulement trouvée — et copie l'EA
dans chaque terminal détecté. Il ne peut pas compiler à votre place ni cocher
l'autorisation réseau : MetaTrader n'expose aucune interface pour cela, et ces deux
étapes restent manuelles.

`tester` sert au diagnostic. Il monte le serveur, ouvre une connexion, envoie un
historique de la bonne taille et vérifie la réponse — exactement ce que fait l'EA. Il
départage la seule question utile quand rien ne marche : **le problème est-il côté
Python ou côté MetaTrader ?**

Les sections suivantes détaillent ce que ces commandes font, pour qui veut le faire
à la main ou comprendre ce qui se passe.


### 1. Côté Python

```bash
pip install -r requirements.txt
python scripts/train.py --config configs/eurusd_h1.yaml --csv data/EURUSD_H1.csv --out runs/eurusd_v1
python scripts/serve.py --model runs/eurusd_v1 --port 8912       # dry-run par défaut
```

Le serveur affiche au démarrage le nombre de barres qu'il attend :

```
Serveur d'inférence sur 127.0.0.1:8912 — DRY-RUN (aucune ouverture de position)
L'EA doit envoyer au moins 1070 barres par requête.
```

### 2. Côté MetaTrader 5

1. Copier `mql5/QBotBridge.mq5` dans `MQL5/Experts/` (via *Fichier → Ouvrir le dossier de données*).
2. Compiler avec MetaEditor (F7).
3. **Autoriser les sockets** — sans cette étape, `SocketConnect` échoue avec l'erreur **5273** :
   *Outils → Options → Expert Advisors → cocher « Autoriser WebRequest pour les URL listées »*
   puis ajouter `127.0.0.1` à la liste.
4. Glisser l'EA sur un graphique, régler `InpHistoryBars` ≥ la valeur annoncée par le serveur.

### 3. Exporter l'historique depuis MetaTrader

*Outils → Fenêtre de données historiques → sélectionner le symbole et la période → Exporter.*
Le format `<DATE>\t<TIME>\t<OPEN>…` est reconnu directement par `qbot.data.load_ohlcv`.

> Utilisez au minimum 3 à 5 ans de données horaires. En dessous, le
> `min_track_record_length` calculé par `scripts/validate.py` dépassera la taille de votre
> échantillon, et aucune conclusion statistique ne sera possible.

---

## Protocole

Toutes les trames sont du JSON UTF-8 terminé par `\n` (TCP est un flux d'octets : sans
délimiteur, un `SocketRead` peut renvoyer un demi-message ou deux messages collés).

### Requête `predict`

```json
{
  "type": "predict",
  "symbol": "EURUSD",
  "timeframe": "PERIOD_H1",
  "equity": 10000.00,
  "balance": 10000.00,
  "peak_equity": 10450.00,
  "current_exposure": 0.5,
  "bars_in_position": 12,
  "entry_price": 1.08420,
  "bars": [[1735689600, 1.0840, 1.0855, 1.0838, 1.0851, 1420, 0.00012], "..."]
}
```

Chaque barre est `[time_epoch, open, high, low, close, volume, spread]`, de la plus
ancienne à la plus récente, **barres clôturées uniquement**.

### Réponse

```json
{
  "ok": true,
  "target_exposure": 0.5,
  "action": 3,
  "confidence": 0.82,
  "status": "ok",
  "reasons": [],
  "sl_distance": 0.00164,
  "tp_distance": 0.00246,
  "q_values": [-0.12, 0.03, 0.41],
  "cvar": [-0.88, -0.31, -0.15],
  "latency_ms": 7.4
}
```

| Champ | Signification |
|---|---|
| `target_exposure` | Exposition cible signée, en fraction du capital. `0.5` = 50 % de l'équité en notionnel. |
| `confidence` | Consensus de l'ensemble (ou marge Q pour un agent seul). Pilote la réduction de taille. |
| `status` | `ok` / `throttled` (réduit) / `blocked` (pas de nouvelle position) / `liquidate` (tout fermer). |
| `cvar` | CVaR 10 % du retour par action — visibilité directe sur le risque de queue. |

### Messages de supervision

| Message | Réponse | Coût |
|---|---|---|
| `{"type":"status"}` | instantané complet : métriques, dérive, coûts, alertes, journal | quelques ms |
| `{"type":"alerts"}` | historique et résumé des alertes seulement | négligeable |

La séparation d'avec `info` est volontaire : `info` décrit la **configuration** (stable,
interrogée une fois à la connexion), `status` décrit l'**état vivant** et coûte plus cher
à produire. L'EA interroge `status` toutes les `InpStatusEveryBars` barres — 24 par
défaut, soit une fois par jour en H1 — et jamais à chaque tick.

La réponse `status` contient un **résumé à plat, à clés uniques**, destiné à l'analyseur
JSON minimal de l'EA :

```json
{"ok": true, "type": "status",
 "drift_status": "critique", "drift_critical": 4, "drift_worst": "vol_ratio_short_long",
 "alert_count": 20, "alert_worst": "critical",
 "recon_verdict": "conforme à l'attendu",
 "tca_verdict": "COÛTS RÉELS ≫ MODÈLE (significatif)", "tca_ratio": 2.54,
 "journal_ok": true, "sharpe_rolling": -0.55, "drawdown": -0.0287, ...}
```

Ces clés existent parce que l'EA n'embarque ni bibliothèque JSON ni DLL : son analyseur
cherche une clé n'importe où dans la chaîne. Une clé comme `status`, présente à la fois
dans le bloc dérive et dans le bloc réconciliation, lui renverrait la première trouvée —
c'est-à-dire au hasard. On expose donc des noms uniques plutôt que de complexifier l'EA.

Paramètres correspondants dans l'EA :

| Paramètre | Défaut | Effet |
|---|---|---|
| `InpStatusEveryBars` | 24 | cadence d'interrogation (0 = jamais) |
| `InpShowPanel` | true | panneau de supervision affiché sur le graphique |
| `InpBlockOnCritical` | **false** | bloquer le renforcement sur alerte critique serveur |

`InpBlockOnCritical` bloque le **renforcement**, jamais la réduction : un mode de sécurité
qui empêcherait aussi de fermer serait plus dangereux que le problème qu'il traite. Faux
par défaut — on l'active après avoir observé le comportement des alertes en dry-run.

Autres types de message : `ping`, `info` (métadonnées du modèle), `reset_guard`
(réarmement manuel du coupe-circuit après intervention).

---

## Défense en profondeur

Les garde-fous existent **en double**, côté Python et côté MQL5, volontairement.

| Contrôle | Python (`RiskGuard`) | MQL5 (`CheckLocalGuards`) |
|---|---|---|
| Drawdown maximal | `risk.max_drawdown_stop` | `InpMaxDrawdownPct` |
| Perte journalière | `risk.max_daily_loss` | `InpMaxDailyLossPct` |
| Spread anormal | `risk.max_spread_bps` | `InpMaxSpreadPoints` |
| Pertes consécutives | `risk.max_consecutive_losses` | — |
| Filtre de session | `risk.session_filter` | — |
| Flux de données périmé | `data_age_s > 120` | — |
| Équité plancher | — | `InpMinEquity` |

Si le processus Python plante, se fige ou renvoie une valeur aberrante, les contrôles
MQL5 restent actifs dans le terminal. C'est le seul moyen de garantir qu'un bug logiciel
ne peut pas vider un compte pendant la nuit.

En cas d'absence de réponse du serveur, l'EA suit `InpCloseOnDisconnect` : soit il conserve
la position (défaut), soit il ferme. Le serveur, lui, renvoie **toujours** une réponse
exploitable — un échec donne `target_exposure: 0.0`, jamais un silence. Un EA qui ne reçoit
rien ne sait pas s'il doit fermer ou attendre ; cette ambiguïté est supprimée par construction.

---

## Répétition générale : le mode rejeu

Avant la première étape de la mise en production, il existe une vérification que rien
d'autre ne remplace : **faire passer de l'historique par le chemin d'exécution réel** —
protocole TCP, calcul des features, réseau de neurones, garde-fous, supervision — et
regarder ce qui sort.

```bash
python scripts/serve.py --model runs/best --replay
```

Ce n'est pas un backtest. Le backtest court dans le moteur vectorisé ; le rejeu court
dans le serveur, avec le même code que la production, y compris la reconstruction de
l'état de portefeuille requête après requête.

Le mode rejeu neutralise **uniquement** le contrôle de fraîcheur du flux (120 s), qui
bloquerait évidemment toute barre passée. Drawdown, perte du jour, spread, séries de
pertes, plafond d'exposition : tout le reste s'applique — sans quoi la répétition ne
dirait rien de la production. Le serveur l'annonce au démarrage et chaque réponse porte
un motif `replay`, pour que personne ne puisse l'activer sans s'en apercevoir.

> **À n'utiliser jamais sur un compte réel.** Un serveur en mode rejeu accepterait de
> trader sur un flux de prix mort.

Ce mode existe parce que son absence a coûté cher : deux défauts réels de ce dépôt —
un facteur d'annualisation faux d'un facteur 1 000, et des seuils de dérive non calibrés
— n'ont été trouvés qu'en faisant réellement tourner la chaîne. Ni les tests ni les
backtests ne les voyaient.

---

## Mise en production — ordre non négociable

1. **Backtest** — `scripts/train.py` puis `scripts/walkforward.py`.
2. **Validation statistique** — `scripts/validate.py`. Si le Deflated Sharpe est < 0.95
   ou la PBO > 0.35, **s'arrêter là**. Le reste est une perte de temps et d'argent.
3. **Dry-run** — `scripts/serve.py` sans `--live`, EA avec `InpDryRun=true`, pendant au
   moins un mois. On vérifie ici la plomberie (latence, reconnexions, cohérence des
   décisions), pas la performance.
4. **Compte de démonstration** — `--live` + `InpDryRun=false` sur un compte démo, capital
   fictif identique au capital réel envisagé. Minimum trois mois.
5. **Réel, capital minimal** — le plus petit montant que le courtier accepte. Comparer les
   rendements réels aux rendements du dry-run sur la même période : tout écart supérieur
   à quelques points de base par trade signale un problème d'exécution non modélisé.
6. **Montée en taille** — seulement après plusieurs mois de concordance, et jamais plus
   vite que la capacité mesurée (l'impact de marché croît en racine de la taille).

Sauter une étape ne fait pas gagner du temps : cela déplace simplement la découverte du
problème vers le moment où il coûte de l'argent réel.

---

## Diagnostic

| Symptôme | Cause probable |
|---|---|
| `SocketConnect` erreur **5273** | Adresse absente de la liste autorisée (étape 3 ci-dessus). |
| `historique insuffisant : N barres reçues` | `InpHistoryBars` < `min_bars`. Interroger `{"type":"info"}`. |
| `les barres doivent être triées` | `ArraySetAsSeries(rates, false)` manquant, ou `CopyRates` mal orienté. |
| Exposition toujours nulle | Mode dry-run actif (normal), ou coupe-circuit déclenché — lire `reasons`. |
| Décisions instables dans une barre | `InpTradeOnNewBarOnly=false` : le modèle voit une barre incomplète. |
| `0.00 lot` alors que l'exposition est non nulle | Capital trop faible pour le lot minimal : exposition inexprimable. |
| Écart réel / backtest | Comparer `spread` réel et `costs.spread_bps` du backtest. C'est presque toujours ça. |

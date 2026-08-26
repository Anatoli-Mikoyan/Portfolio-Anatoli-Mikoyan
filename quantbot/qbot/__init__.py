"""qbot — moteur de trading quantitatif : features, labels, RL distributionnel, backtest et validation anti-overfitting.

Le paquet est volontairement découpé en couches indépendantes :

    data       -> chargement et échantillonnage des barres (temps, tick, volume, dollar, imbalance)
    features   -> construction causale de la matrice de features (aucune fuite du futur)
    labeling   -> triple barrière, meta-labeling, poids d'échantillons
    env        -> environnement de trading (coûts réalistes, récompenses risk-adjusted)
    agents     -> Rainbow / QR-DQN / Munchausen, réseaux, replay priorisé
    risk       -> dimensionnement de position et coupe-circuits
    backtest   -> moteur événementiel et métriques
    validation -> Purged K-Fold, CPCV, walk-forward, PBO, Deflated Sharpe, bootstrap
    live       -> serveur d'inférence pour le pont MetaTrader 5

Les modules `data`, `features`, `labeling`, `backtest` et `validation` ne dépendent que de
numpy/pandas/scipy. PyTorch n'est requis que pour `agents` (et donc l'entraînement RL).
"""

__version__ = "1.0.0"

__all__ = ["__version__"]

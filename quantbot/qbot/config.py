"""Configuration typée du système.

Tout est piloté par des dataclasses sérialisables : un run = un fichier YAML/JSON, ce qui
rend chaque backtest rejouable et auditable (condition nécessaire pour distinguer un
résultat réel d'un artefact de tuning).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------------------
# Données / features
# --------------------------------------------------------------------------------------
@dataclass
class DataConfig:
    symbol: str = "EURUSD"
    timeframe: str = "H1"
    csv_path: Optional[str] = None
    asset_class: str = "fx"                    # "fx" | "equity" | "crypto"
    bar_type: str = "time"                     # "time" | "tick" | "volume" | "dollar" | "imbalance"
    bar_threshold: Optional[float] = None      # seuil pour les barres non temporelles
    start: Optional[str] = None
    end: Optional[str] = None
    price_col: str = "close"


@dataclass
class FeatureConfig:
    # Fenêtres des indicateurs. Volontairement peu nombreuses : chaque paramètre supplémentaire
    # augmente le nombre d'essais N et donc dégrade le Deflated Sharpe.
    returns_windows: Tuple[int, ...] = (1, 2, 5, 10, 20, 60)
    vol_windows: Tuple[int, ...] = (10, 20, 60)
    rsi_windows: Tuple[int, ...] = (14,)
    ema_windows: Tuple[int, ...] = (10, 30, 100)
    atr_window: int = 14
    adx_window: int = 14
    bb_window: int = 20
    donchian_window: int = 20
    use_fracdiff: bool = True
    fracdiff_d: Optional[float] = None         # None => recherche du d minimal stationnaire
    fracdiff_thresh: float = 1e-4
    use_microstructure: bool = True
    use_regime: bool = True
    use_calendar: bool = True
    winsorize_sigma: float = 8.0               # écrêtage des valeurs aberrantes après scaling
    scaler: str = "rolling_zscore"             # "rolling_zscore" | "expanding_zscore" | "rank" | "none"
    scaler_window: int = 500                   # fenêtre causale du z-score glissant
    dropna: bool = True


# --------------------------------------------------------------------------------------
# Coûts / environnement
# --------------------------------------------------------------------------------------
@dataclass
class CostConfig:
    """Modèle de coûts. Les valeurs par défaut sont volontairement pessimistes.

    Un backtest optimiste sur les coûts est la façon la plus rapide de se mentir à soi-même :
    la majorité des stratégies HF « rentables » meurent en ajoutant le vrai spread.
    """
    spread_bps: float = 1.5              # spread moyen en points de base du notionnel
    spread_col: Optional[str] = None     # colonne de spread réel si disponible (prioritaire)
    commission_bps: float = 0.35         # commission aller-retour, en bps du notionnel
    slippage_model: str = "sqrt"         # "none" | "linear" | "sqrt"
    slippage_coef: float = 0.15          # coefficient d'impact (en unités de vol de barre)
    financing_bps_per_bar: float = 0.0   # swap/carry par barre et par unité d'exposition
    min_trade_size: float = 0.05         # turnover en-dessous duquel on ne rebalance pas (no-trade band)


@dataclass
class EnvConfig:
    # Fenêtre volontairement courte. Elle pilote directement la dimension d'entrée, donc
    # le nombre de paramètres de la première couche — le poste dominant du réseau.
    # Mesure : de 1 à 32 barres, l'IC out-of-sample reste dans la même plage. Les features
    # encodent déjà l'historique (rendements à 2/5/20 barres, EMA, volatilités) ; empiler
    # des copies décalées n'ajoute pas d'information, seulement de la capacité à mémoriser.
    window: int = 16
    positions: Tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)   # espace d'actions discret
    execution: str = "close"             # "close" (rebalance à la clôture) | "next_open"
    reward: str = "dsr"                  # "pnl" | "log_pnl" | "dsr" | "vol_scaled" | "dd_penalized"
    reward_scale: float = 100.0
    dsr_eta: float = 0.01                # constante de décroissance du Differential Sharpe Ratio
    turnover_penalty: float = 0.0        # pénalité additionnelle sur le turnover (au-delà des coûts)
    drawdown_penalty: float = 0.25       # utilisé par reward="dd_penalized"
    holding_penalty: float = 0.0         # pénalise l'exposition passive
    episode_length: Optional[int] = 2048  # None => un épisode = tout le segment
    random_start: bool = True
    include_position_in_obs: bool = True
    max_drawdown_stop: Optional[float] = 0.25   # arrêt d'épisode (et coupe-circuit live)
    vol_target: Optional[float] = 0.10   # volatilité annualisée cible (None = pas de scaling)
    vol_target_window: int = 60
    max_leverage: float = 1.0


# --------------------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------------------
@dataclass
class AgentConfig:
    # Capacité volontairement FAIBLE. Mesure faite sur ce dépôt (scripts/probe.py, marché
    # synthétique à R² connu de 0.06) : un réseau 128x128 passe d'un IC out-of-sample de
    # +0.109 à 2 000 pas, à +0.038 à 6 000 pas, puis -0.003 à 20 000 pas. Il mémorise le
    # bruit bien avant d'épuiser le signal. Sur des données financières, la capacité est
    # une contrainte, pas une ressource : voir docs/METHODOLOGIE.md §6.
    algo: str = "rainbow"                # "rainbow" | "dqn" (rainbow = toutes les extensions)
    encoder: str = "mlp"                 # "mlp" | "gru" | "tcn"
    hidden_sizes: Tuple[int, ...] = (64, 64)
    encoder_hidden: int = 128
    tcn_channels: Tuple[int, ...] = (64, 64)
    tcn_kernel: int = 3
    dropout: float = 0.1
    layer_norm: bool = True

    # Extensions Rainbow (chacune activable indépendamment pour l'ablation)
    double_q: bool = True
    dueling: bool = True
    noisy: bool = True
    noisy_sigma0: float = 0.5
    prioritized: bool = True
    per_alpha: float = 0.6
    per_beta0: float = 0.4
    per_beta_steps: int = 200_000
    per_eps: float = 1e-6
    n_step: int = 5
    distributional: str = "qr"           # "qr" (quantile) | "c51" | "none"
    n_quantiles: int = 51
    n_atoms: int = 51
    v_min: float = -10.0                 # utilisé uniquement par C51
    v_max: float = 10.0
    munchausen: bool = True
    m_alpha: float = 0.9
    m_tau: float = 0.03
    m_clip: float = -1.0

    gamma: float = 0.99
    lr: float = 1e-4
    adam_eps: float = 1.5e-4
    weight_decay: float = 1e-3           # mesuré : IC out-of-sample -0.003 (wd=0) -> +0.070
                                         # (1e-4) -> +0.179 (1e-3), contre +0.216 pour une régression
                                         # linéaire. La régularisation n'est pas un réglage fin ici.
    batch_size: int = 128
    buffer_size: int = 300_000
    learn_start: int = 5_000
    train_freq: int = 1
    target_update_interval: int = 2_000
    target_soft_tau: Optional[float] = None   # si défini, mise à jour douce au lieu de dure
    grad_clip: float = 10.0
    huber_kappa: float = 1.0

    # Exploration epsilon-greedy (résiduelle : NoisyNet fait l'essentiel du travail)
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_steps: int = 50_000

    device: str = "auto"                 # "auto" | "cpu" | "cuda"


@dataclass
class TrainConfig:
    total_steps: int = 200_000
    # Évaluation fréquente et patience courte : l'optimum de généralisation arrive
    # beaucoup plus tôt qu'en apprentissage profond classique.
    eval_every: int = 5_000
    eval_episodes: int = 1
    seeds: Tuple[int, ...] = (0,)        # >1 seed => ensemble
    early_stop_patience: Optional[int] = 6   # en nombre d'évaluations
    early_stop_metric: str = "sharpe"
    checkpoint_dir: str = "runs"
    log_every: int = 1_000


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------
@dataclass
class ValidationConfig:
    scheme: str = "walkforward"          # "walkforward" | "cpcv" | "purged_kfold" | "holdout"
    n_splits: int = 6
    n_test_groups: int = 2               # CPCV : taille du groupe de test
    embargo_pct: float = 0.01            # embargo López de Prado (fraction de l'échantillon)
    train_bars: int = 20_000             # walk-forward : taille de la fenêtre d'entraînement
    test_bars: int = 5_000
    anchored: bool = False               # True => fenêtre d'entraînement ancrée (expanding)
    n_trials_for_dsr: int = 1            # nombre d'essais réellement testés (pour le Deflated Sharpe)
    bootstrap_samples: int = 2_000
    block_size: int = 20                 # bootstrap par blocs stationnaires


@dataclass
class RiskConfig:
    max_position: float = 1.0
    vol_target: Optional[float] = 0.10
    kelly_fraction: float = 0.25         # fraction de Kelly (jamais 1.0 : variance ruineuse)
    max_daily_loss: Optional[float] = 0.03
    max_drawdown_stop: Optional[float] = 0.20
    max_consecutive_losses: Optional[int] = 8
    cooldown_bars: int = 20              # gel après déclenchement d'un coupe-circuit
    max_spread_bps: Optional[float] = 5.0   # refuse de trader si le spread explose
    session_filter: Optional[List[Tuple[int, int]]] = None  # heures UTC autorisées


@dataclass
class LiveConfig:
    host: str = "127.0.0.1"
    port: int = 8912
    protocol: str = "tcp_json"           # "tcp_json" | "zmq"
    model_path: str = "runs/best"
    max_latency_ms: int = 500
    heartbeat_s: int = 15
    fail_safe: str = "flat"              # "flat" | "hold" — comportement si le modèle ne répond pas
    dry_run: bool = True                 # True => aucune décision d'ouverture n'est envoyée


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    live: LiveConfig = field(default_factory=LiveConfig)
    seed: int = 42
    run_name: str = "qbot"

    # -- (dé)sérialisation -------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, default=list)
        if path.suffix in {".yaml", ".yml"}:
            try:
                import yaml

                path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")
                return path
            except ImportError:
                path = path.with_suffix(".json")
        path.write_text(payload, encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix in {".yaml", ".yml"}:
            import yaml

            raw = yaml.safe_load(text)
        else:
            raw = json.loads(text)
        return cls.from_dict(raw or {})

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Config":
        return _build(cls, raw)

    def merge(self, overrides: Dict[str, Any]) -> "Config":
        """Applique des surcharges pointées : {"agent.lr": 3e-4, "env.window": 32}."""
        raw = self.to_dict()
        for key, value in overrides.items():
            node = raw
            parts = key.split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = value
        return Config.from_dict(raw)


def _build(dc_type, raw: Any):
    """Construit récursivement une dataclass depuis un dict, en respectant les types tuple."""
    if not is_dataclass(dc_type) or raw is None:
        return raw
    if not isinstance(raw, dict):
        return raw
    kwargs: Dict[str, Any] = {}
    for f in fields(dc_type):
        if f.name not in raw:
            continue
        value = raw[f.name]
        if is_dataclass(f.type) or (isinstance(f.type, type) and is_dataclass(f.type)):
            kwargs[f.name] = _build(f.type, value)
        elif isinstance(value, list) and "Tuple" in str(f.type):
            kwargs[f.name] = tuple(value)
        else:
            kwargs[f.name] = value
    # sous-dataclasses résolues par annotation textuelle (from __future__ import annotations)
    type_map = {f.name: f for f in fields(dc_type)}
    for name, sub in list(kwargs.items()):
        annotation = str(type_map[name].type)
        for sub_cls in (DataConfig, FeatureConfig, CostConfig, EnvConfig, AgentConfig,
                        TrainConfig, ValidationConfig, RiskConfig, LiveConfig):
            if sub_cls.__name__ in annotation and isinstance(sub, dict):
                kwargs[name] = _build(sub_cls, sub)
    return dc_type(**kwargs)


__all__ = [
    "Config", "DataConfig", "FeatureConfig", "CostConfig", "EnvConfig",
    "AgentConfig", "TrainConfig", "ValidationConfig", "RiskConfig", "LiveConfig",
]

"""Moteur d'inférence live.

Point crucial de tout le dépôt : ce module **réutilise exactement** le `FeaturePipeline`
et le `RainbowAgent` du backtest. Aucune ré-implémentation, aucune approximation « plus
rapide » côté production. Le jour où l'on réécrit le calcul des features pour le live,
on introduit un écart entraînement/service (training-serving skew) qui ne se manifeste
que par des pertes inexpliquées — le bug le plus coûteux et le plus difficile à voir de
tout le ML appliqué.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import Config, EnvConfig, RiskConfig
from ..env.trading_env import N_PORTFOLIO_FEATURES
from ..features import FeaturePipeline
from ..risk import GuardStatus, RiskGuard
from ..utils.logging import get_logger
from .protocol import PredictResponse, error_response, validate_predict_request

log = get_logger("live.engine")

_BAR_COLUMNS = ["time", "open", "high", "low", "close", "volume", "spread"]


@dataclass
class ModelBundle:
    """Tout ce qu'il faut pour produire une décision, chargé depuis le disque."""
    pipeline: FeaturePipeline
    agent: Any                       # RainbowAgent ou EnsembleAgent
    config: Config
    model_id: str

    @property
    def positions(self) -> np.ndarray:
        return np.asarray(self.config.env.positions, dtype=float)


def load_bundle(model_dir: str | Path, device: Optional[str] = None) -> ModelBundle:
    """Charge un modèle exporté par `scripts/train.py`."""
    model_dir = Path(model_dir)
    cfg_path = next((p for p in (model_dir / "config.json", model_dir / "config.yaml") if p.exists()), None)
    if cfg_path is None:
        raise FileNotFoundError(f"config.json introuvable dans {model_dir}")
    config = Config.load(cfg_path)

    pipe_path = model_dir / "pipeline.json"
    if not pipe_path.exists():
        raise FileNotFoundError(f"pipeline.json introuvable dans {model_dir}")
    pipeline = FeaturePipeline.load(pipe_path)

    from ..agents import EnsembleAgent, RainbowAgent   # import tardif : torch optionnel

    ensemble_files = sorted(model_dir.glob("agent_*.pt"))
    if ensemble_files:
        agent = EnsembleAgent.load(model_dir, device=device, agreement_threshold=0.6)
        model_id = f"ensemble[{len(ensemble_files)}]"
    else:
        agent = RainbowAgent.load(model_dir / "agent.pt", device=device)
        model_id = "single"

    log.info("Modèle chargé depuis %s (%s, %d features)", model_dir, model_id, len(pipeline.feature_names))
    return ModelBundle(pipeline=pipeline, agent=agent, config=config, model_id=model_id)


class InferenceEngine:
    """Transforme une requête de l'EA en décision d'exposition, garde-fous compris."""

    def __init__(self, bundle: ModelBundle, risk_cfg: Optional[RiskConfig] = None,
                 cvar_alpha: Optional[float] = None, dry_run: bool = True):
        self.bundle = bundle
        self.env_cfg: EnvConfig = bundle.config.env
        self.guard = RiskGuard(risk_cfg or bundle.config.risk)
        self.cvar_alpha = cvar_alpha
        self.dry_run = dry_run
        self.n_requests = 0
        self.last_decision: Optional[PredictResponse] = None

        # Historique interne : deux composantes de l'état de portefeuille (volatilité de la
        # stratégie, intensité de trading) ne sont pas observables depuis l'EA. Les mettre
        # à zéro serait un écart entraînement/service — le modèle serait interrogé sur des
        # états qu'il n'a jamais rencontrés. On les reconstruit ici à partir de la suite
        # des requêtes, comme l'environnement le fait à partir de la suite des pas.
        #
        # Démarrage : le serveur ignore l'équité antérieure à sa première requête, il lui
        # manque donc un rendement dans la fenêtre glissante de 60. La parité avec
        # l'environnement devient exacte une fois cette fenêtre entièrement remplie
        # (~60 barres), ce que vérifie test_live_portfolio_state_matches_environment.
        # Conséquence pratique : ne pas juger le modèle sur sa première heure de service.
        self._equity_history: List[float] = []
        self._strategy_returns: List[float] = []
        self._turnover_history: List[float] = []
        self._last_exposure: float = 0.0

    # ---------------------------------------------------------------------------------
    @property
    def min_bars(self) -> int:
        """Nombre de barres que l'EA doit fournir pour que TOUTES les features soient définies."""
        return self.bundle.pipeline.min_history + self.env_cfg.window + 5

    # ---------------------------------------------------------------------------------
    @staticmethod
    def _to_frame(bars: List[List[float]]) -> pd.DataFrame:
        arr = np.asarray(bars, dtype=float)
        if arr.shape[1] == 6:
            arr = np.column_stack([arr, np.zeros(len(arr))])
        df = pd.DataFrame(arr[:, :7], columns=_BAR_COLUMNS)
        df["time"] = pd.to_datetime(df["time"].astype("int64"), unit="s", utc=True)
        return df.set_index("time").sort_index()

    def _update_history(self, equity: float, exposure: float) -> None:
        """Met à jour l'historique interne à chaque requête (une requête = une barre)."""
        if self._equity_history:
            prev = self._equity_history[-1]
            if prev > 0:
                self._strategy_returns.append(equity / prev - 1.0)
        self._equity_history.append(equity)
        self._turnover_history.append(abs(exposure - self._last_exposure))
        self._last_exposure = exposure

        # Bornage : seules les 200 dernières barres servent aux fenêtres glissantes.
        for buf in (self._equity_history, self._strategy_returns, self._turnover_history):
            if len(buf) > 200:
                del buf[:-200]

    def _portfolio_state(self, msg: Dict[str, Any], df: pd.DataFrame, bar_vol: float) -> np.ndarray:
        """Reconstruit le vecteur d'état de portefeuille tel que vu à l'entraînement.

        Ces 6 composantes doivent correspondre EXACTEMENT, dans le même ordre et avec les
        mêmes transformations, à celles produites par `TradingEnv._observation()`. Toute
        divergence — y compris remplir une composante de zéros « faute de mieux » — place
        le modèle hors de la distribution sur laquelle il a été entraîné.
        """
        exposure = float(msg.get("current_exposure", 0.0))
        equity = float(msg.get("equity", 1.0))
        peak = float(msg.get("peak_equity") or equity)
        drawdown = equity / max(peak, 1e-12) - 1.0

        bars_in_pos = int(msg.get("bars_in_position", 0))
        entry = msg.get("entry_price")
        close = float(df["close"].iloc[-1])
        unrealized = ((close / float(entry) - 1.0) * np.sign(exposure)
                      if entry and float(entry) > 0 and exposure != 0 else 0.0)
        vol = max(bar_vol, 1e-8)

        self._update_history(equity, exposure)

        recent = np.asarray(self._strategy_returns[-60:], dtype=float)
        strat_vol = float(recent.std()) if recent.size >= 10 else 0.0
        mean_turnover = (float(np.mean(self._turnover_history)) if self._turnover_history else 0.0)

        return np.array(
            [
                exposure,
                float(np.clip(drawdown, -1.0, 0.0)),
                float(np.tanh(bars_in_pos / 50.0)),
                float(np.clip(unrealized / (vol * 10.0), -3.0, 3.0)),
                float(np.clip(strat_vol / vol, 0.0, 5.0) - 1.0),
                float(np.tanh(mean_turnover * 10.0)),
            ],
            dtype=np.float32,
        )

    # ---------------------------------------------------------------------------------
    def predict(self, msg: Dict[str, Any]) -> PredictResponse:
        t0 = time.perf_counter()
        self.n_requests += 1

        err = validate_predict_request(msg, self.min_bars)
        if err:
            return error_response(err, status="blocked")

        try:
            df = self._to_frame(msg["bars"])
            window = int(self.env_cfg.window)

            # Même code de features qu'à l'entraînement — c'est le point non négociable.
            feats = self.bundle.pipeline.transform_latest(df, n_rows=window)

            bar_ret = np.diff(np.log(df["close"].to_numpy(float)))
            bar_vol = float(np.std(bar_ret[-self.env_cfg.vol_target_window:])) if bar_ret.size > 5 else 1e-4
            portfolio = self._portfolio_state(msg, df, bar_vol)

            obs = np.concatenate([feats.ravel(), portfolio]).astype(np.float32)[None, :]

            self.bundle.agent.bind_features(feats)   # requis par le décodeur d'observations
            action = int(self.bundle.agent.act_batch(obs, cvar_alpha=self.cvar_alpha)[0])
            raw_exposure = float(self.bundle.positions[action])

            q_values, cvar_values, confidence = self._diagnostics(obs, action)

            # --- vol targeting, identique à l'environnement ----------------------------
            if self.env_cfg.vol_target:
                ann_vol = bar_vol * np.sqrt(self._bars_per_year(df))
                scalar = float(np.clip(self.env_cfg.vol_target / max(ann_vol, 1e-6), 0.0,
                                       self.env_cfg.max_leverage))
            else:
                scalar = 1.0
            desired = float(np.clip(raw_exposure * scalar, -self.env_cfg.max_leverage,
                                    self.env_cfg.max_leverage))

            # --- garde-fous ------------------------------------------------------------
            ts = df.index[-1].to_pydatetime()
            equity = float(msg.get("equity", 1.0))
            self.guard.update_equity(equity, ts)
            self.guard.tick()
            spread_bps = (float(df["spread"].iloc[-1]) / float(df["close"].iloc[-1]) * 1e4
                          if float(df["spread"].iloc[-1]) > 0 else None)
            data_age = max((datetime.now(timezone.utc) - df.index[-1].to_pydatetime()).total_seconds(), 0.0)

            decision = self.guard.check(
                desired, spread_bps=spread_bps, timestamp=ts,
                model_confidence=confidence, data_age_s=data_age,
            )

            exposure = decision.allowed_position
            reasons = list(decision.reasons)
            if self.dry_run and abs(exposure) > abs(float(msg.get("current_exposure", 0.0))):
                # En dry-run on autorise la RÉDUCTION d'exposition mais jamais l'ouverture :
                # un mode simulation qui empêcherait aussi de fermer serait dangereux.
                exposure = float(msg.get("current_exposure", 0.0))
                reasons.append("dry_run : ouverture/renforcement bloqué")

            atr = self._atr(df)
            resp = PredictResponse(
                ok=True,
                target_exposure=round(float(exposure), 6),
                action=action,
                confidence=round(float(confidence), 4),
                status=decision.status.value,
                reasons=reasons,
                sl_distance=round(2.0 * atr, 6),
                tp_distance=round(3.0 * atr, 6),
                q_values=[round(float(v), 6) for v in q_values],
                cvar=[round(float(v), 6) for v in cvar_values],
                model_id=self.bundle.model_id,
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
            )
            self.last_decision = resp
            return resp

        except Exception as exc:  # pragma: no cover - filet de sécurité
            log.exception("Erreur d'inférence")
            return error_response(f"{type(exc).__name__}: {exc}", status="blocked")

    # ---------------------------------------------------------------------------------
    def _diagnostics(self, obs: np.ndarray, action: int) -> tuple[np.ndarray, np.ndarray, float]:
        """Q, CVaR et confiance. La confiance pilote directement la taille de position."""
        import torch

        agent = self.bundle.agent
        nets = getattr(agent, "agents", [agent])
        q_all, cvar_all = [], []
        with torch.no_grad():
            for a in nets:
                x = torch.as_tensor(obs, dtype=torch.float32, device=a.device)
                a.online.eval()
                q_all.append(a.online.q_values(x).cpu().numpy()[0])
                cvar_all.append(a.online.risk_measure(x, 0.1).cpu().numpy()[0])
        q = np.mean(q_all, axis=0)
        cvar = np.mean(cvar_all, axis=0)

        if len(nets) > 1:
            # Confiance = fraction d'agents qui votent la même action que l'ensemble.
            votes = [int(np.argmax(v)) for v in q_all]
            confidence = float(np.mean([v == action for v in votes]))
        else:
            # Agent unique : écart relatif entre la meilleure action et la deuxième.
            order = np.sort(q)[::-1]
            gap = float(order[0] - order[1]) if q.size > 1 else 1.0
            spread = float(order[0] - order[-1]) if q.size > 1 else 1.0
            confidence = float(np.clip(gap / spread, 0.0, 1.0)) if spread > 1e-9 else 0.0
        return q, cvar, confidence

    @staticmethod
    def _atr(df: pd.DataFrame, window: int = 14) -> float:
        h, l, c = df["high"], df["low"], df["close"]
        pc = c.shift(1)
        tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
        return float(tr.ewm(alpha=1.0 / window, adjust=False).mean().iloc[-1])

    @staticmethod
    def _bars_per_year(df: pd.DataFrame) -> float:
        from ..utils.timeutils import infer_bars_per_year

        return infer_bars_per_year(df.index)

    def info(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "model_id": self.bundle.model_id,
            "n_features": len(self.bundle.pipeline.feature_names),
            "window": int(self.env_cfg.window),
            "min_bars": self.min_bars,
            "positions": self.bundle.positions.tolist(),
            "dry_run": self.dry_run,
            "requests_served": self.n_requests,
            "halted": self.guard.halted,
            "halt_reason": self.guard.halt_reason,
        }

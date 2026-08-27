"""Serveur d'inférence TCP pour l'Expert Advisor MetaTrader 5.

Un thread par connexion (`ThreadingTCPServer`) : l'EA n'ouvre qu'une seule connexion
persistante, et même une poignée de symboles reste très en deçà de ce que ce modèle
supporte. Pas d'asyncio ici — la complexité ne serait justifiée qu'au-delà de plusieurs
centaines de connexions simultanées.

Sécurité : le serveur écoute par défaut sur 127.0.0.1 uniquement. L'exposer sur une
interface publique donnerait à quiconque le pouvoir de piloter les positions du compte.
"""
from __future__ import annotations

import json
import socket
import socketserver
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import LiveConfig, RiskConfig
from ..utils.logging import get_logger
from ..monitoring.journal import DecisionJournal
from .engine import InferenceEngine, ModelBundle, load_bundle
from .protocol import PROTOCOL_VERSION, LineFramer, error_response

log = get_logger("live.server")


class _Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        engine: InferenceEngine = self.server.engine          # type: ignore[attr-defined]
        framer = LineFramer()
        peer = f"{self.client_address[0]}:{self.client_address[1]}"
        log.info("Connexion EA %s", peer)
        self.request.settimeout(300.0)

        try:
            while not self.server.stopping:                   # type: ignore[attr-defined]
                try:
                    chunk = self.request.recv(65536)
                except socket.timeout:
                    continue
                if not chunk:
                    break

                try:
                    messages = framer.feed(chunk)
                except ValueError as exc:
                    self.request.sendall(error_response(str(exc)).to_json())
                    continue

                for msg in messages:
                    self.request.sendall(self._dispatch(engine, msg))
        except (ConnectionResetError, BrokenPipeError):
            log.warning("Connexion %s interrompue par le client", peer)
        except Exception:  # pragma: no cover - filet
            log.exception("Erreur inattendue sur %s", peer)
        finally:
            log.info("Déconnexion EA %s", peer)

    @staticmethod
    def _dispatch(engine: InferenceEngine, msg: Dict[str, Any]) -> bytes:
        kind = msg.get("type")
        if kind == "ping":
            return (json.dumps({"ok": True, "type": "pong", "t": int(time.time()),
                                "version": PROTOCOL_VERSION}) + "\n").encode("utf-8")
        if kind == "info":
            return (json.dumps(engine.info()) + "\n").encode("utf-8")
        if kind == "status":
            # Instantané de supervision. Volontairement distinct de `info` : `info` décrit
            # la configuration (stable, interrogée à la connexion), `status` décrit l'état
            # vivant (métriques, dérive, alertes) et coûte plus cher à produire.
            return (json.dumps(engine.status(), default=str) + "\n").encode("utf-8")
        if kind == "alerts":
            if engine.monitor is None:
                return error_response("surveillance désactivée sur ce serveur").to_json()
            payload = engine.monitor.alerts.summary()
            payload.update({"ok": True, "type": "alerts"})
            return (json.dumps(payload, default=str) + "\n").encode("utf-8")
        if kind == "reset_guard":
            engine.guard.reset()
            log.warning("Coupe-circuit réarmé manuellement.")
            return (json.dumps({"ok": True, "type": "guard_reset"}) + "\n").encode("utf-8")
        if kind == "predict":
            return engine.predict(msg).to_json()
        if kind == "__parse_error__":
            return error_response(f"JSON invalide : {msg.get('error')}").to_json()
        return error_response(f"type de message inconnu : {kind!r}").to_json()


class InferenceServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, engine: InferenceEngine, host: str = "127.0.0.1", port: int = 8912):
        self.engine = engine
        self.stopping = False
        super().__init__((host, port), _Handler)

    def shutdown(self) -> None:  # pragma: no cover - arrêt
        self.stopping = True
        super().shutdown()


def build_monitor(model_dir: str | Path, bundle: ModelBundle,
                  cfg: Optional[Any] = None) -> Optional[Any]:
    """Construit le moniteur d'un modèle exporté, si les fichiers de référence existent.

    `reference.json` et `envelope.json` sont produits par `scripts/monitor.py fit` à
    partir des MÊMES données que l'entraînement. Leur absence n'est pas une erreur : la
    surveillance se dégrade proprement — pas de détection de dérive, pas de confrontation
    attendu/réalisé — et les métriques de production restent collectées. Le serveur le
    dit explicitement au démarrage plutôt que de laisser croire à une surveillance
    complète.
    """
    from ..monitoring import LiveMonitor, PerformanceEnvelope, ReferenceDistribution
    from ..utils.timeutils import bars_per_year_for_timeframe

    model_dir = Path(model_dir)
    cfg = cfg if cfg is not None else getattr(bundle.config, "monitor", None)
    if cfg is None or not getattr(cfg, "enabled", True):
        return None

    ref_path = model_dir / "reference.json"
    env_path = model_dir / "envelope.json"
    reference = ReferenceDistribution.load(ref_path) if ref_path.exists() else None
    envelope = None
    if env_path.exists():
        envelope = PerformanceEnvelope.from_dict(
            json.loads(env_path.read_text(encoding="utf-8")))

    if reference is None:
        log.warning("reference.json absent de %s : détection de dérive INACTIVE "
                    "(la produire avec « python scripts/monitor.py fit »).", model_dir)
    if envelope is None:
        log.warning("envelope.json absent de %s : confrontation attendu/réalisé INACTIVE.",
                    model_dir)

    bpy = bars_per_year_for_timeframe(bundle.config.data.timeframe)
    journal_path = getattr(cfg, "journal_path", None) or str(model_dir / "audit.jsonl")
    return LiveMonitor(cfg, reference=reference, envelope=envelope, bars_per_year=bpy,
                       model_id=bundle.model_id, journal=DecisionJournal(journal_path))


def serve(
    model_dir: str | Path,
    live_cfg: Optional[LiveConfig] = None,
    risk_cfg: Optional[RiskConfig] = None,
    cvar_alpha: Optional[float] = None,
    block: bool = True,
    monitor: Optional[Any] = None,
    replay: bool = False,
) -> InferenceServer:
    """Démarre le serveur d'inférence."""
    live_cfg = live_cfg or LiveConfig()
    bundle = load_bundle(model_dir)
    if monitor is None:
        monitor = build_monitor(model_dir, bundle)
    engine = InferenceEngine(bundle, risk_cfg or bundle.config.risk,
                             cvar_alpha=cvar_alpha, dry_run=live_cfg.dry_run,
                             monitor=monitor, replay=replay)

    server = InferenceServer(engine, live_cfg.host, live_cfg.port)
    mode = "DRY-RUN (aucune ouverture de position)" if live_cfg.dry_run else "*** TRADING RÉEL ***"
    log.info("Serveur d'inférence sur %s:%d — %s", live_cfg.host, live_cfg.port, mode)
    log.info("L'EA doit envoyer au moins %d barres par requête.", engine.min_bars)
    if monitor is not None:
        log.info("Supervision active — messages `status` et `alerts` disponibles.")
    if replay:
        log.warning("*** MODE REJEU *** : répétition générale sur barres passées. "
                    "Le contrôle de fraîcheur du flux est neutralisé.")

    if not block:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Arrêt demandé.")
    finally:
        server.shutdown()
        server.server_close()
    return server


class SimpleClient:
    """Client TCP minimal — sert aux tests et au diagnostic depuis Python."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8912, timeout: float = 10.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.framer = LineFramer()

    def request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.sock.sendall(json.dumps(payload).encode("utf-8") + b"\n")
        while True:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("Connexion fermée par le serveur")
            msgs = self.framer.feed(chunk)
            if msgs:
                return msgs[0]

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()

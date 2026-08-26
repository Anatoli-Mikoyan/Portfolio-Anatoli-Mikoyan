"""Protocole de communication Python <-> MetaTrader 5.

Choix technique : **TCP + JSON délimité par saut de ligne**.

Pourquoi pas une DLL ? Parce qu'une DLL impose `#import` dans l'EA, ce qui casse la
compatibilité avec les comptes prop-firm et le Market MQL5, complique le déploiement et
crée un couplage binaire. MQL5 dispose de sockets natifs (`SocketCreate`, `SocketSend`,
`SocketRead`) depuis la build 2085 : aucune dépendance externe n'est nécessaire.

Pourquoi pas ZeroMQ ? Il reste supporté ici en option, mais il exige une DLL côté MT5.
Le TCP nu suffit largement : à l'échelle d'une barre H1 ou M15, la latence réseau locale
(< 1 ms) est sans commune mesure avec la latence d'exécution du courtier (30-200 ms).

Toutes les trames sont du JSON UTF-8 terminé par '\\n'. Le délimiteur est indispensable :
TCP est un flux d'octets sans notion de message, et un `SocketRead` peut très bien
retourner un demi-message ou deux messages collés.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

PROTOCOL_VERSION = "1.0"
DELIMITER = b"\n"
MAX_MESSAGE_BYTES = 4 * 1024 * 1024


# =======================================================================================
# Requêtes
# =======================================================================================
@dataclass
class BarPayload:
    """Une barre OHLCV telle qu'envoyée par l'EA."""
    time: int          # epoch UTC en secondes
    open: float
    high: float
    low: float
    close: float
    volume: float
    spread: float = 0.0


@dataclass
class PredictRequest:
    symbol: str
    timeframe: str
    bars: List[List[float]]                 # [[time, o, h, l, c, v, spread], ...] du plus ancien au plus récent
    equity: float = 10_000.0
    balance: float = 10_000.0
    peak_equity: Optional[float] = None
    current_exposure: float = 0.0
    bars_in_position: int = 0
    entry_price: Optional[float] = None
    magic: int = 0
    type: str = "predict"
    version: str = PROTOCOL_VERSION


# =======================================================================================
# Réponses
# =======================================================================================
@dataclass
class PredictResponse:
    ok: bool
    target_exposure: float = 0.0            # fraction du capital, signée, dans [-1, 1]
    action: int = 0                         # index de l'action discrète choisie
    confidence: float = 0.0                 # [0,1] : consensus d'ensemble / certitude du modèle
    status: str = "ok"                      # ok | throttled | blocked | liquidate | error
    reasons: List[str] = field(default_factory=list)
    sl_distance: float = 0.0                # distance de stop en unités de prix (0 = pas de SL serveur)
    tp_distance: float = 0.0
    q_values: List[float] = field(default_factory=list)
    cvar: List[float] = field(default_factory=list)
    model_id: str = ""
    latency_ms: float = 0.0
    server_time: int = field(default_factory=lambda: int(time.time()))
    error: str = ""
    version: str = PROTOCOL_VERSION

    def to_json(self) -> bytes:
        return json.dumps(asdict(self), separators=(",", ":")).encode("utf-8") + DELIMITER


def error_response(message: str, status: str = "error") -> PredictResponse:
    """En cas d'erreur, on renvoie TOUJOURS une réponse exploitable plutôt que rien.

    Un EA qui n'obtient pas de réponse ne sait pas s'il doit fermer ou attendre. Une
    réponse explicite `target_exposure = 0` supprime cette ambiguïté : le mode dégradé
    est de ne pas être exposé.
    """
    return PredictResponse(ok=False, target_exposure=0.0, status=status,
                           reasons=[message], error=message)


# =======================================================================================
# Cadrage des messages
# =======================================================================================
class LineFramer:
    """Réassemble des messages JSON complets depuis un flux TCP fragmenté."""

    def __init__(self, max_bytes: int = MAX_MESSAGE_BYTES):
        self._buf = bytearray()
        self.max_bytes = max_bytes

    def feed(self, chunk: bytes) -> List[Dict[str, Any]]:
        self._buf.extend(chunk)
        if len(self._buf) > self.max_bytes:
            self._buf.clear()
            raise ValueError(f"Message dépassant {self.max_bytes} octets : tampon vidé.")

        messages: List[Dict[str, Any]] = []
        while True:
            i = self._buf.find(DELIMITER)
            if i < 0:
                break
            raw = bytes(self._buf[:i])
            del self._buf[: i + 1]
            if not raw.strip():
                continue
            try:
                messages.append(json.loads(raw.decode("utf-8")))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                messages.append({"type": "__parse_error__", "error": str(exc)})
        return messages

    def reset(self) -> None:
        self._buf.clear()


def validate_predict_request(msg: Dict[str, Any], min_bars: int) -> Optional[str]:
    """Valide une requête ; retourne un message d'erreur ou None si tout est correct."""
    if msg.get("type") != "predict":
        return f"type inattendu : {msg.get('type')!r}"
    bars = msg.get("bars")
    if not isinstance(bars, list) or not bars:
        return "champ 'bars' absent ou vide"
    if len(bars) < min_bars:
        return f"historique insuffisant : {len(bars)} barres reçues, {min_bars} requises"
    first = bars[0]
    if not isinstance(first, (list, tuple)) or len(first) < 6:
        return "chaque barre doit être [time, open, high, low, close, volume, (spread)]"
    times = [b[0] for b in bars]
    if any(times[i] >= times[i + 1] for i in range(len(times) - 1)):
        # Un EA qui envoie les barres à l'envers produirait des features calculées sur une
        # série inversée — sans erreur visible, mais avec des prédictions absurdes.
        return "les barres doivent être triées par temps croissant et sans doublon"
    return None

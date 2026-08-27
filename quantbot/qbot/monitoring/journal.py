"""Journal d'audit chaîné des décisions (cahier des charges §17).

Toute table de trading régulée doit pouvoir répondre, des mois après les faits, à la
question : « pourquoi cette position a-t-elle été prise à cette seconde-là ? ». Le
règlement délégué MiFID II RTS 6 impose aux entreprises pratiquant le trading
algorithmique de conserver un enregistrement des décisions, reconstituable et non
altérable. Le même besoin existe sans aucun régulateur : quand un modèle perd de
l'argent, la première chose à établir est si le code a fait ce qu'il croyait faire.

Deux propriétés font la différence entre un fichier de logs et un journal d'audit :

  * **Chaînage cryptographique.** Chaque entrée porte l'empreinte SHA-256 de la
    précédente. Modifier ou supprimer une ligne a posteriori casse la chaîne à cet
    endroit précis, et `verify()` le dit. On ne rend pas la falsification impossible —
    quiconque a le fichier peut le réécrire entièrement — on la rend *détectable*, ce
    qui suffit à ce qu'elle ne passe pas inaperçue.
  * **Rejouabilité.** L'entrée contient l'ENTRÉE du modèle, pas seulement sa sortie. On
    peut donc réexécuter la décision plus tard avec le même modèle et vérifier qu'on
    retrouve la même action. C'est le seul moyen de détecter un décalage de version
    (`replay_mismatch` dans `reconciliation.py`) — la panne la plus silencieuse qui
    soit : le serveur tourne, répond, et n'est plus le modèle qu'on a validé.

Format : JSON Lines. Un enregistrement par ligne, lisible par n'importe quel outil, et
robuste à une coupure — une ligne tronquée par un arrêt brutal n'invalide que sa propre
entrée, pas le fichier entier.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

__all__ = ["JournalEntry", "DecisionJournal", "ChainVerification"]

GENESIS = "0" * 64


def _canonical(payload: Dict[str, Any]) -> str:
    """Sérialisation déterministe : clés triées, séparateurs figés, ASCII échappé.

    Le hachage n'a de sens que si deux relectures du même contenu produisent exactement
    la même chaîne d'octets. `sort_keys=True` et des séparateurs explicites l'imposent,
    quelle que soit la version de Python qui a écrit le fichier.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      default=str)


def _digest(prev_hash: str, seq: int, ts: str, payload: Dict[str, Any]) -> str:
    blob = f"{prev_hash}|{seq}|{ts}|{_canonical(payload)}".encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@dataclass
class JournalEntry:
    seq: int
    ts: str
    kind: str
    payload: Dict[str, Any]
    prev_hash: str
    hash: str

    def to_line(self) -> str:
        return json.dumps(
            {"seq": self.seq, "ts": self.ts, "kind": self.kind, "payload": self.payload,
             "prev_hash": self.prev_hash, "hash": self.hash},
            sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str,
        )

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "JournalEntry":
        return cls(seq=int(raw["seq"]), ts=str(raw["ts"]), kind=str(raw.get("kind", "")),
                   payload=raw.get("payload", {}), prev_hash=str(raw["prev_hash"]),
                   hash=str(raw["hash"]))

    def recompute(self) -> str:
        return _digest(self.prev_hash, self.seq, self.ts, self.payload)


@dataclass
class ChainVerification:
    """Résultat d'une vérification d'intégrité."""
    n_entries: int
    valid: bool
    first_broken_seq: Optional[int] = None
    reason: str = ""
    n_malformed: int = 0

    def __str__(self) -> str:  # pragma: no cover - affichage
        if self.valid:
            return f"Journal intègre : {self.n_entries} entrées, chaîne continue."
        return (f"JOURNAL COMPROMIS à l'entrée seq={self.first_broken_seq} — "
                f"{self.reason} ({self.n_entries} entrées lues, "
                f"{self.n_malformed} illisibles)")


class DecisionJournal:
    """Journal append-only, chaîné par empreinte, sûr entre threads.

    Le serveur d'inférence est multi-thread (un thread par connexion d'EA). Un verrou
    protège l'écriture : deux entrées qui s'entrelaceraient produiraient une ligne
    corrompue et, pire, un chaînage incohérent qui ferait croire à une falsification.
    """

    def __init__(self, path: str | Path, fsync: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fsync = bool(fsync)
        self._lock = threading.Lock()
        self._seq, self._last_hash = self._resume()

    # -- reprise ------------------------------------------------------------------------
    def _resume(self) -> tuple[int, str]:
        """Reprend la chaîne là où un précédent processus l'a laissée.

        Un redémarrage ne doit pas repartir de la genèse : la chaîne serait rompue au
        point de reprise et `verify()` signalerait à tort une falsification.
        """
        if not self.path.exists():
            return 0, GENESIS
        last: Optional[Dict[str, Any]] = None
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    last = json.loads(line)
                except json.JSONDecodeError:
                    # Ligne tronquée par un arrêt brutal : on la conserve dans le fichier
                    # (elle fait partie de la preuve) et on repart de la dernière valide.
                    continue
        if not last:
            return 0, GENESIS
        return int(last["seq"]) + 1, str(last["hash"])

    # -- écriture -----------------------------------------------------------------------
    def append(self, kind: str, payload: Dict[str, Any],
               ts: Optional[str] = None) -> JournalEntry:
        from datetime import datetime, timezone

        stamp = ts or datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        with self._lock:
            seq = self._seq
            prev = self._last_hash
            digest = _digest(prev, seq, stamp, payload)
            entry = JournalEntry(seq=seq, ts=stamp, kind=kind, payload=payload,
                                 prev_hash=prev, hash=digest)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(entry.to_line() + "\n")
                if self.fsync:
                    # Coûteux (~1 ms par écriture). À n'activer qu'en production réelle,
                    # où perdre les dernières décisions lors d'un arrêt machine coûte
                    # plus cher que la latence : sans fsync, le tampon de l'OS peut
                    # contenir plusieurs secondes de décisions non écrites sur le disque.
                    fh.flush()
                    os.fsync(fh.fileno())
            self._seq = seq + 1
            self._last_hash = digest
        return entry

    # -- lecture ------------------------------------------------------------------------
    def __len__(self) -> int:
        return self._seq

    @property
    def head(self) -> str:
        """Empreinte de tête : à publier ailleurs (message, dépôt) pour horodater l'état.

        Une empreinte de tête recopiée quelque part d'inaltérable transforme la
        détection de falsification en preuve : réécrire l'historique devient impossible
        sans contredire une valeur déjà publiée.
        """
        return self._last_hash

    def entries(self) -> Iterator[JournalEntry]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield JournalEntry.from_dict(json.loads(line))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

    def read(self, kind: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for e in self.entries():
            if kind is None or e.kind == kind:
                out.append(e.payload)
        return out[-limit:] if limit else out

    # -- intégrité ----------------------------------------------------------------------
    def verify(self) -> ChainVerification:
        """Rejoue la chaîne d'empreintes et localise la première rupture."""
        prev = GENESIS
        expected_seq = 0
        n = 0
        malformed = 0

        if not self.path.exists():
            return ChainVerification(0, True)

        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = JournalEntry.from_dict(json.loads(line))
                except (json.JSONDecodeError, KeyError, ValueError):
                    malformed += 1
                    continue

                if entry.seq != expected_seq:
                    return ChainVerification(n, False, entry.seq,
                                             f"numéro de séquence {entry.seq} attendu {expected_seq}",
                                             malformed)
                if entry.prev_hash != prev:
                    return ChainVerification(n, False, entry.seq,
                                             "chaînage rompu : entrée supprimée ou insérée",
                                             malformed)
                if entry.recompute() != entry.hash:
                    return ChainVerification(n, False, entry.seq,
                                             "contenu modifié après écriture", malformed)
                prev = entry.hash
                expected_seq += 1
                n += 1

        return ChainVerification(n, True, None, "", malformed)

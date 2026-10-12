"""Per-session prefix ledger: exact emitted strings, replayed verbatim; epochs re-route."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Emitted:
    segment_id: str
    key: str
    lane: str
    ratio: float
    text: str
    tokens: int
    source: str = ""
    compressor_fallback: bool = False
    verifier: Optional[Dict[str, Any]] = None
    backref: Optional[str] = None
    pinned: bool = False
    dropped: bool = False


@dataclass
class LedgerEntry(Emitted):
    epoch_id: int = 0


class Ledger:
    def __init__(self):
        self.entries: List[LedgerEntry] = []
        self.by_key: Dict[str, LedgerEntry] = {}
        self.epoch_id = -1
        self.last_messages: Optional[List[Dict[str, Any]]] = None
        self.last_tools_on = False
        self.history: List[Dict[str, Any]] = []   # per-turn: replayed / new / epoch

    def keys(self):
        return self.by_key.keys()

    def append(self, e: Emitted) -> LedgerEntry:
        le = LedgerEntry(**asdict(e), epoch_id=self.epoch_id)
        self.entries.append(le)
        self.by_key[le.key] = le
        return le

    def reset(self, emitted: List[Emitted]):
        self.epoch_id += 1
        self.entries, self.by_key = [], {}
        for e in emitted:
            self.append(e)

    def replay(self) -> List[LedgerEntry]:
        return list(self.entries)


def first_divergence(old: Optional[List[Dict[str, Any]]], new: List[Dict[str, Any]]) -> int:
    """Index of the first message whose content differs (len(old) if new extends old)."""
    if not old:
        return 0
    for i, (a, b) in enumerate(zip(old, new)):
        if a.get("role") != b.get("role") or a.get("content") != b.get("content"):
            return i
    return min(len(old), len(new))


def common_prefix_chars(old: Optional[List[Dict[str, Any]]], new: List[Dict[str, Any]]) -> int:
    if not old:
        return 0
    n = 0
    for a, b in zip(old, new):
        if a.get("content") == b.get("content") and a.get("role") == b.get("role"):
            n += len(a.get("content") or "")
            continue
        x, y = a.get("content") or "", b.get("content") or ""
        k = 0
        while k < min(len(x), len(y)) and x[k] == y[k]:
            k += 1
        return n + k
    return n

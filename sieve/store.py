"""Externalize store: a session-scoped Python dict (optionally persisted to JSON) plus
deterministic, memoized stubs."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from sieve.verifier import extract_atoms

STOP = set("""the a an of to in on for and or is are was were be been by with at from as that this these those it
its our we us you your they their them which who what when where how will would should could can may might must
not no than then there here also more most less least any some all each every other such into over under about
after before while whenever although remains still needs need rarely only own does do did has have had being""".split())
KEY_PRIORITY = ["id", "date", "version", "number", "entity", "name", "quoted"]


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace('"', "'").replace("|", "/").replace("[", "(").replace("]", ")")).strip()


def extractive_summary(text: str, max_words: int = 25) -> str:
    sents = [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    lead = _clean(sents[0]) if sents else ""
    lead_words = lead.split()[:14]
    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text)]
    lead_set = {w.lower().strip(".,;:") for w in lead_words}
    tf = Counter(w for w in words if w not in STOP and w not in lead_set)
    # idf across sentences (deterministic tf-idf; ties broken alphabetically)
    df = Counter()
    for s in sents:
        for w in set(re.findall(r"[a-z][a-z\-]{3,}", s.lower())):
            df[w] += 1
    import math
    score = {w: c * math.log(1 + len(sents) / (1 + df[w])) for w, c in tf.items()}
    top = [w for w, _ in sorted(score.items(), key=lambda x: (-x[1], x[0]))][: max(0, max_words - len(lead_words))]
    top = top[:8]
    out = " ".join(lead_words) + (" ..." if len(lead.split()) > 14 else "")
    if top:
        out += " / topics: " + ", ".join(top)
    return " ".join(out.split()[:max_words + 3])


def salient_keys(text: str, k: int = 5) -> List[str]:
    atoms = extract_atoms(text)
    found = []
    for kind in KEY_PRIORITY:
        for a in atoms.get(kind, ()):
            found.append((KEY_PRIORITY.index(kind), text.find(a), a))
    found.sort()
    out: List[str] = []
    for _, _, a in found:
        a = _clean(a).strip("'")
        if a and a not in out:
            out.append(a)
        if len(out) >= k:
            break
    return out


class Store:
    def __init__(self, cfg, tokenizer, session_id: str, persist_path: Optional[Path] = None,
                 llm_summarize: Optional[Callable[[str], str]] = None):
        self.cfg = cfg
        self.tok = tokenizer
        self.session_id = session_id
        self.data: Dict[str, Dict[str, Any]] = {}
        self.persist_path = Path(persist_path) if persist_path else None
        self.llm_summarize = llm_summarize
        self._stub_memo: Dict[str, str] = {}
        self._summary_memo: Dict[str, str] = {}
        self.request_expanded_tokens = 0
        self.expand_log: List[Dict[str, Any]] = []
        if self.persist_path and self.persist_path.exists():
            self.data = {k.split(":", 1)[1]: v for k, v in json.loads(self.persist_path.read_text()).items()
                         if v.get("session_id") == session_id}

    def summary(self, seg_id: str, text: str) -> str:
        if seg_id not in self._summary_memo:
            if self.cfg.summary_mode == "llm" and self.llm_summarize:
                s = _clean(self.llm_summarize(text))
                self._summary_memo[seg_id] = " ".join(s.split()[:25])
            else:
                self._summary_memo[seg_id] = extractive_summary(text)
        return self._summary_memo[seg_id]

    def stub_for(self, seg) -> str:
        """[ref:<id> | <kind> | <N>t | "<summary>" | keys: ... | use expand_context("<id>") to read]"""
        if seg.id not in self._stub_memo:
            keys = salient_keys(seg.content)
            self._stub_memo[seg.id] = (
                f'[ref:{seg.id} | {seg.kind} | {seg.tokens}t | "{self.summary(seg.id, seg.content)}" | '
                f'keys: {", ".join(keys) if keys else "-"} | use expand_context("{seg.id}") to read]')
        return self._stub_memo[seg.id]

    def put(self, seg, turn: int) -> str:
        if seg.id not in self.data:
            self.data[seg.id] = dict(content=seg.content, kind=seg.kind, tokens=seg.tokens,
                                     summary=self.summary(seg.id, seg.content), keys=salient_keys(seg.content),
                                     session_id=self.session_id, created_turn=turn, expand_count=0,
                                     last_expanded=None)
            self._persist()
        return self.stub_for(seg)

    def put_group(self, content: str, kind: str, turn: int):
        """Merged stub for a message whose chunks are all externalized (deterministic id)."""
        import hashlib
        from types import SimpleNamespace
        gid = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
        g = SimpleNamespace(id=gid, kind=kind, tokens=self.tok.count(content), content=content)
        self.put(g, turn)
        return g

    def begin_request(self):
        self.request_expanded_tokens = 0

    def expand(self, seg_id: str, session_id: Optional[str] = None, turn: Optional[int] = None) -> str:
        """Full text, or an explicit error string (never empty, so the model cannot invent content)."""
        sid = session_id or self.session_id
        rec = self.data.get(seg_id)
        if rec is None or rec.get("session_id") != sid:
            return (f"[expand_context error: no stored segment with id '{seg_id}' in this session. "
                    f"Do not guess its contents.]")
        remaining = self.cfg.expansion_budget_tokens - self.request_expanded_tokens
        if remaining <= 0:
            return (f"[expand_context error: expansion budget of {self.cfg.expansion_budget_tokens} tokens "
                    f"for this request is exhausted. Answer from what is available.]")
        text = rec["content"]
        n = rec["tokens"]
        if n > remaining:
            text = self.tok.truncate_to(text, remaining) + "\n[... truncated: expansion budget reached]"
            n = remaining
        self.request_expanded_tokens += n
        rec["expand_count"] += 1
        rec["last_expanded"] = turn
        self.expand_log.append({"seg": seg_id, "tokens": n, "turn": turn})
        self._persist()
        return f"[expanded ref:{seg_id}]\n{text}"

    @property
    def expand_counts(self) -> Dict[str, int]:
        return {k: v["expand_count"] for k, v in self.data.items()}

    def _persist(self):
        if self.persist_path:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            allrec = {}
            if self.persist_path.exists():
                try:
                    allrec = json.loads(self.persist_path.read_text())
                except Exception:
                    allrec = {}
            allrec.update({f"{self.session_id}:{k}": v for k, v in self.data.items()})
            self.persist_path.write_text(json.dumps(allrec, sort_keys=True))

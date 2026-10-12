"""messages -> Segment objects, stable chunking, exact/near-duplicate tracking (per session)."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from sieve.canonicalizer import canonicalize

KINDS = ["system", "instruction", "doc", "chat", "tool_result", "attachment", "query"]
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sha12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


@dataclass
class Segment:
    id: str
    key: str                 # unique position key "<msg_index>.<part>"
    index: int               # position among all segments of the conversation
    msg_index: int
    part: int
    role: str
    kind: str
    content: str
    tokens: int
    sep: str                 # separator that followed this chunk in the message
    tags: Dict[str, bool] = field(default_factory=dict)
    turn_index: int = 0
    mid: str = ""
    features: Dict[str, float] = field(default_factory=dict)
    dup_of: Optional[str] = None       # key of the first identical / near-identical segment
    dup_type: Optional[str] = None     # exact | near
    repeat_count: int = 0
    near_dup_score: float = 0.0
    canon_tokens_before: int = 0       # tokens before canonicalization (message-level, on part 0)


def shingles(text: str, k: int = 5) -> set:
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def chunk_text(text: str, count, lo: int, hi: int) -> List[Tuple[str, str]]:
    """Split into (chunk, separator) on paragraph boundaries; oversize paragraphs split on
    lines, then sentences. Deterministic: depends only on the text."""
    if count(text) <= hi:
        return [(text, "")]
    units: List[Tuple[str, str]] = []
    paras = text.split("\n\n")
    for pi, p in enumerate(paras):
        sep = "\n\n" if pi < len(paras) - 1 else ""
        if count(p) <= hi:
            units.append((p, sep))
            continue
        lines = p.split("\n")
        sub: List[Tuple[str, str]] = []
        if len(lines) > 1:
            for li, l in enumerate(lines):
                sub.append((l, "\n" if li < len(lines) - 1 else sep))
        else:
            sents = SENT_SPLIT.split(p)
            for si, s in enumerate(sents):
                sub.append((s, " " if si < len(sents) - 1 else sep))
        units.extend(sub)
    chunks: List[Tuple[str, str]] = []
    cur, cur_sep, cur_n = "", "", 0
    for u, sep in units:
        n = count(u)
        if cur and cur_n + n > hi and cur_n >= lo:
            chunks.append((cur, cur_sep))
            cur, cur_n = "", 0
        elif cur and cur_n + n > hi * 1.25:
            chunks.append((cur, cur_sep))
            cur, cur_n = "", 0
        cur = cur + cur_sep + u if cur else u
        cur_sep, cur_n = sep, cur_n + n
    if cur:
        chunks.append((cur, cur_sep))
    # merge a too-small tail into its predecessor
    if len(chunks) > 1 and count(chunks[-1][0]) < lo // 3:
        (a, sa), (b, sb) = chunks[-2], chunks[-1]
        chunks[-2:] = [(a + sa + b, sb)]
    return chunks


class SessionState:
    """Per-session segmentation memo. Built fresh for every conversation (never global)."""

    def __init__(self, session_id: str, tokenizer, cfg, canonicalize_kinds=("tool_result", "attachment")):
        self.session_id = session_id
        self.tok = tokenizer
        self.cfg = cfg
        self.canon_kinds = canonicalize_kinds
        self._msg_memo: Dict[int, Tuple[str, List[Tuple[str, str]], int, str]] = {}
        self.canonicalizer_saved: Dict[int, int] = {}

    def _message_chunks(self, i: int, m: Dict[str, Any]):
        content = m.get("content") or ""
        key = (i, sha12(content))
        hit = self._msg_memo.get(i)
        if hit and hit[0] == key[1]:
            return hit[1], hit[2], hit[3]
        before = self.tok.count(content)
        canon_kind = ""
        if m.get("kind") in self.canon_kinds:
            content, canon_kind = canonicalize(content)
        self.canonicalizer_saved[i] = before - self.tok.count(content)
        chunks = chunk_text(content, self.tok.count, self.cfg.chunk_min_tokens, self.cfg.chunk_max_tokens) \
            if m.get("kind") not in ("system", "query") else [(content, "")]
        self._msg_memo[i] = (key[1], chunks, before, canon_kind)
        return chunks, before, canon_kind

    def segment(self, messages: List[Dict[str, Any]]) -> List[Segment]:
        """Segment the whole message list. Pure function of `messages` (memoized per message)."""
        segs: List[Segment] = []
        first_by_hash: Dict[str, Segment] = {}
        counts: Dict[str, int] = {}
        sh_list: List[Tuple[Segment, set]] = []
        for i, m in enumerate(messages):
            chunks, before, _ = self._message_chunks(i, m)
            tags = {t: True for t in (m.get("tags") or [])}
            if m.get("role") == "system":
                tags["static"] = True
            for p, (text, sep) in enumerate(chunks):
                sid = sha12(text)
                s = Segment(id=sid, key=f"{i}.{p}", index=len(segs), msg_index=i, part=p, role=m["role"],
                            kind=m.get("kind") or ("system" if m["role"] == "system" else "chat"),
                            content=text, tokens=self.tok.count(text), sep=sep, tags=dict(tags),
                            turn_index=int(m.get("turn", 0)), mid=m.get("mid", f"m{i:02d}"),
                            canon_tokens_before=before if p == 0 else 0)
                if s.kind == "gold_answer":
                    s.kind = "chat"
                if sid in first_by_hash and s.kind not in ("query",):
                    s.dup_of, s.dup_type = first_by_hash[sid].key, "exact"
                    counts[sid] += 1
                    s.repeat_count = counts[sid]
                else:
                    first_by_hash.setdefault(sid, s)
                    counts.setdefault(sid, 0)
                    if s.tokens >= self.cfg.min_segment_tokens and s.kind not in ("query",):
                        sh = shingles(text)
                        best, best_s = 0.0, None
                        for o, osh in sh_list:
                            j = jaccard(sh, osh)
                            if j > best:
                                best, best_s = j, o
                        s.near_dup_score = best
                        if best >= self.cfg.neardup_jaccard and best_s is not None:
                            s.dup_of, s.dup_type = best_s.key, "near"
                        sh_list.append((s, sh))
                segs.append(s)
        return segs

    def canonicalizer_tokens_saved(self, n_messages: int) -> int:
        return sum(v for i, v in self.canonicalizer_saved.items() if i < n_messages)


def reassemble(segs_by_msg: Dict[int, List[Tuple[Segment, str]]], messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for i, m in enumerate(messages):
        parts = segs_by_msg.get(i)
        if parts is None:
            continue
        text = "".join(t + (s.sep if k < len(parts) - 1 else "") for k, (s, t) in enumerate(parts))
        out.append({"role": m["role"], "content": text})
    return out

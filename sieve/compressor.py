"""LLMLingua-2 wrapper (CPU only) with a deterministic memo, digit-split repair, and a loud
query-aware extractive fallback if LLMLingua cannot be loaded."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sieve.verifier import extract_atoms, recall

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _atom_regex(atom: str) -> re.Pattern:
    parts = []
    for ch in atom:
        if ch.isalnum():
            parts.append(re.escape(ch))
        else:
            parts.append(r"\s*" + re.escape(ch) + r"?\s*")
    return re.compile("".join(parts))


class Compressor:
    def __init__(self, cfg, tokenizer, memo_path: Optional[Path] = None, force_fallback: bool = False,
                 use_memo: bool = True):
        self.cfg = cfg
        self.tok = tokenizer
        self.use_memo = use_memo
        self.memo: Dict[str, Dict[str, Any]] = {}
        self.memo_path = Path(memo_path) if memo_path else None
        self.memo_hits = 0
        self.calls = 0
        self.total_ms = 0.0
        self.fallback = True
        self.load_error: Optional[str] = None
        self._pc = None
        self._lock = threading.Lock()
        self.last_word_labels: Optional[List[Tuple[str, int]]] = None
        if not force_fallback:
            try:
                import logging
                import warnings
                warnings.filterwarnings("ignore")
                logging.getLogger("transformers").setLevel(logging.ERROR)
                os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
                from llmlingua import PromptCompressor
                import llmlingua
                self._pc = PromptCompressor(model_name=cfg.compressor_model, use_llmlingua2=True, device_map="cpu")
                self.fallback = False
                self.version = f"llmlingua2-{getattr(llmlingua, '__version__', '?')}|{cfg.compressor_model}"
            except Exception as e:
                self.load_error = f"{type(e).__name__}: {e}"
        if self.fallback:
            self.version = "extractive-fallback-v1"
            print("\n" + "!" * 78 + "\n!! WARNING: LLMLingua-2 unavailable -> using extractive FALLBACK compressor."
                  f"\n!! Every record will carry compressor_fallback=true. Reason: {self.load_error}\n" + "!" * 78,
                  file=sys.stderr)
        if self.memo_path and self.memo_path.exists() and use_memo:
            for line in self.memo_path.read_text().splitlines():
                try:
                    r = json.loads(line)
                    self.memo[r["k"]] = r
                except Exception:
                    pass

    def key(self, text: str, ratio: float) -> str:
        h = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"{h}|{ratio:.2f}|{self.version}"

    def _repair(self, original: str, out: str) -> str:
        """LLMLingua re-joins sub-tokens with spaces ("4.1" -> "4. 1"). Restore atoms that are
        present in spaced form; never invents text that is not in the original."""
        for kind, atoms in extract_atoms(original).items():
            if kind not in ("number", "version", "id", "date"):
                continue
            for a in sorted(atoms, key=len, reverse=True):
                if a in out or not re.search(r"\W", a):
                    continue
                out = _atom_regex(a).sub(a, out, count=1)
        return out

    def _llmlingua(self, text: str, ratio: float) -> Tuple[str, Optional[List[Tuple[str, int]]]]:
        r = self._pc.compress_prompt(text, rate=ratio, force_tokens=["\n", ".", "?", "!"],
                                     chunk_end_tokens=[".", "\n"], drop_consecutive=True,
                                     force_reserve_digit=True, return_word_label=True)
        labels = None
        lab = r.get("fn_labeled_original_prompt")
        if lab:
            labels = []
            for item in lab.split("\t\t|\t\t"):
                w, _, k = item.rpartition(" ")
                if w:
                    labels.append((w, int(k) if k.isdigit() else 1))
        return r["compressed_prompt"], labels

    def _fallback(self, text: str, ratio: float, query: str = "") -> str:
        """Query-aware extractive sentence selection (not first-N), original order kept."""
        sents = [s for s in SENT_SPLIT.split(text) if s.strip()]
        if len(sents) <= 1:
            return text
        q = set(re.findall(r"\w+", query.lower()))
        budget = ratio * self.tok.count(text)
        scored = []
        for i, s in enumerate(sents):
            w = set(re.findall(r"\w+", s.lower()))
            digits = sum(c.isdigit() for c in s)
            scored.append((len(w & q) + 0.2 * digits + 0.1 * sum(x[:1].isupper() for x in s.split()), -i, i))
        keep, used = set(), 0
        for _, _, i in sorted(scored, reverse=True):
            n = self.tok.count(sents[i])
            if used + n > budget and keep:
                continue
            keep.add(i)
            used += n
        return " ".join(sents[i] for i in sorted(keep))

    def compress(self, text: str, ratio: float, query: str = "") -> Dict[str, Any]:
        """Same (content, ratio, version) -> identical string, always."""
        if ratio >= 1.0 or not text.strip():
            return {"text": text, "tokens": self.tok.count(text), "memo_hit": False, "ms": 0.0}
        k = self.key(text, ratio)
        if self.use_memo and k in self.memo:
            self.memo_hits += 1
            r = self.memo[k]
            return {"text": r["text"], "tokens": r["tokens"], "memo_hit": True, "ms": 0.0}
        t0 = time.perf_counter()
        labels = None
        with self._lock:
            if self.fallback:
                out = self._fallback(text, ratio, query="")  # query-independent so the memo key holds
            else:
                try:
                    out, labels = self._llmlingua(text, ratio)
                except Exception as e:  # pragma: no cover
                    print(f"!! LLMLingua call failed ({e}); passthrough", file=sys.stderr)
                    out = text
            out = self._repair(text, out)
        ms = (time.perf_counter() - t0) * 1000
        self.calls += 1
        self.total_ms += ms
        self.last_word_labels = labels
        rec = {"k": k, "text": out, "tokens": self.tok.count(out)}
        if self.use_memo:
            self.memo[k] = rec
            if self.memo_path:
                self.memo_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.memo_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
        return {"text": out, "tokens": rec["tokens"], "memo_hit": False, "ms": ms, "labels": labels}

    def compress_verified(self, text: str, ratio: float, kind: str, use_verifier: bool = True,
                          extra_atoms=()) -> Dict[str, Any]:
        """Compress; if numeric/ID atoms are lost, raise the ratio one bucket and retry; finally
        passthrough. Returns text + verifier record."""
        buckets = sorted(self.cfg.ratio_buckets)
        ratio = self.cfg.snap_ratio(ratio, kind)
        atoms = extract_atoms(text, extra_atoms) if use_verifier else {}
        retries, ms, memo_hits = 0, 0.0, 0
        lost_first: List[str] = []
        while True:
            r = self.compress(text, ratio)
            ms += r["ms"]
            memo_hits += int(r["memo_hit"])
            if not use_verifier or ratio >= 1.0:
                ver = recall(atoms, r["text"], text) if use_verifier else None
                return {"text": r["text"], "tokens": r["tokens"], "ratio": ratio, "ms": ms, "memo_hits": memo_hits,
                        "verifier": ({"recall": ver["overall_recall"], "numeric_recall": ver["numeric_recall"],
                                      "context_recall": ver["context_recall"],
                                      "retries": retries, "reinflated": retries > 0, "lost": lost_first}
                                     if ver else None)}
            ver = recall(atoms, r["text"], text)
            ok = ver["numeric_recall"] >= self.cfg.verifier_min_recall and \
                ver["overall_recall"] >= self.cfg.verifier_overall_min_recall and \
                ver["context_recall"] >= self.cfg.verifier_min_context_recall
            if ok:
                return {"text": r["text"], "tokens": r["tokens"], "ratio": ratio, "ms": ms, "memo_hits": memo_hits,
                        "verifier": {"recall": ver["overall_recall"], "numeric_recall": ver["numeric_recall"],
                                     "context_recall": ver["context_recall"],
                                     "retries": retries, "reinflated": retries > 0, "lost": lost_first}}
            if not lost_first:
                lost_first = ver["lost"][:12]
            higher = [b for b in buckets if b > ratio]
            retries += 1
            if not higher or retries > self.cfg.verifier_max_retries:
                ratio = 1.0
            else:
                ratio = higher[0]

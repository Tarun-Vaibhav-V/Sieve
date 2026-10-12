"""Deterministic fake Ollama client so the whole pipeline runs without a GPU.

Everything it reports is SIMULATED (timings derive from token counts; prefix reuse is
a token-level common-prefix with the previous request). Records carry mock=true.
The "model" answers by retrieving the context sentence with the highest content-word
overlap with the question; it calls expand_context when a stub's keys match better
than any visible sentence.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

from sieve.ollama_client import ChatResult, OllamaClient, normalize_seg_id

STOP = set("""a an the of to in on for and or is are was were be been by with at from as that this
these those it its what which who whom whose when where how why does do did can could should would
will shall may might must please answer question using only conversation information mentioned
current currently our we us you your i me my about into than then there their they them yes no
not one word short reply concise""".split())
STUB_RE = re.compile(r"\[ref:([0-9a-f]{12}) \|[^\]]*?keys: ([^|\]]*)\|[^\]]*\]")
WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.\-:/]*")
SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

SIM_PREFILL_S_PER_TOK = 0.0012
SIM_DECODE_TPS = 30.0
SIM_LOAD_S = 2.5


def _terms(text: str) -> List[str]:
    return [w.lower().strip(".:-/") for w in WORD_RE.findall(text) if w.lower().strip(".:-/") not in STOP]


class MockClient(OllamaClient):
    is_mock = True

    def __init__(self, cfg, tokenizer=None):
        self.cfg = cfg
        self.tok = tokenizer
        self._prev: List[str] = []
        self._loaded = False
        self.calls = 0

    def version(self) -> str:
        return "mock"

    def list_models(self) -> List[str]:
        return [self.cfg.model]

    def show(self, model=None) -> Dict[str, Any]:
        return {"capabilities": ["completion", "tools"], "details": {"quantization_level": "MOCK"}}

    def ps(self) -> Dict[str, Any]:
        if not self._loaded:
            return {"size": 0, "size_vram": 0, "gpu_pct": 0.0}
        return {"size": 6 * 2**30, "size_vram": 6 * 2**30, "gpu_pct": 1.0}

    def unload(self) -> None:
        self._loaded = False
        self._prev = []

    @staticmethod
    def _render(messages, tools) -> List[str]:
        parts = []
        if tools:
            parts.append(json.dumps(tools, sort_keys=True))
        for m in messages:
            parts.append(f"<{m['role']}>")
            parts.append(m.get("content") or "")
            for tc in m.get("tool_calls") or []:
                parts.append(json.dumps(tc, sort_keys=True))
        return " ".join(parts).split()

    def _answer(self, messages, tools) -> Dict[str, Any]:
        q_idx = max(i for i, m in enumerate(messages) if m["role"] == "user")
        last = (messages[q_idx].get("content") or "").split("\n\n(Stubs with keys")[0]  # Sieve's tail hint is not context
        question = last.split("\n\n[Prefetched context")[0]
        qt = set(_terms(question))
        ctx_msgs = [m for i, m in enumerate(messages) if i != q_idx] + \
                   ([{"role": "user", "content": last[len(question):]}])
        best, best_score = None, 0.0
        stubs = []
        for m in ctx_msgs:
            text = m.get("content") or ""
            for sid, keys in STUB_RE.findall(text):
                stubs.append((sid, keys))
            text = STUB_RE.sub(" ", text)
            for s in SENT_RE.split(text):
                st = set(_terms(s))
                if not st:
                    continue
                score = len(qt & st) / (len(qt) ** 0.5 + 1e-9)
                if score >= best_score and score > 0:  # >= : later sentences win ties
                    best, best_score = s.strip(), score
        expanded = {m.get("content", "")[:40] for m in messages if m["role"] == "tool"}
        if tools and stubs:
            for sid, keys in stubs:
                kt = set(_terms(keys))
                ks = len(qt & kt) / (len(qt) ** 0.5 + 1e-9)
                already = any(sid in (m.get("content") or "")[:80] for m in messages if m["role"] == "tool")
                if ks > best_score and ks > 0.3 and not already and not expanded & {sid}:
                    return {"tool_calls": [{"function": {"name": "expand_context",
                                                         "arguments": {"seg_id": sid}}}]}
        if best is None or best_score < 0.55:
            return {"content": "Not mentioned in the conversation."}
        return {"content": best[:300]}

    def chat(self, messages, tools=None, options=None, num_ctx=None, keep_alive=None) -> ChatResult:
        self.calls += 1
        toks = self._render(messages, tools)
        common = 0
        for a, b in zip(self._prev, toks):
            if a != b:
                break
            common += 1
        self._prev = toks
        n_prompt = int(len(toks) * 1.3)
        processed = max(1, n_prompt - int(common * 1.3))
        ctx = num_ctx or self.cfg.num_ctx
        if n_prompt > ctx:  # simulate Ollama truncation: whole prompt reprocessed
            processed = ctx
        out = self._answer(messages, tools) if (options or {}).get("num_predict", 99) > 4 else {"content": "OK"}
        res = ChatResult()
        res.content = out.get("content", "")
        res.tool_calls = out.get("tool_calls", [])
        res.eval_count = max(1, (self.tok.count(res.content) if self.tok else len(res.content) // 4) or 12)
        res.load_s = 0.0 if self._loaded else SIM_LOAD_S
        self._loaded = True
        res.prompt_eval_count = processed
        res.prompt_eval_s = processed * SIM_PREFILL_S_PER_TOK
        res.eval_s = res.eval_count / SIM_DECODE_TPS
        res.decode_tps = SIM_DECODE_TPS
        res.ttft_s = res.load_s + res.prompt_eval_s + 1 / SIM_DECODE_TPS
        res.total_s = res.load_s + res.prompt_eval_s + res.eval_s
        res.total_duration_s = res.total_s
        res.vram_peak_mb = 6144.0
        res.vram_source = "mock"
        res.done_reason = "stop"
        return res

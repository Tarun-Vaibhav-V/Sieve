"""Experimental conditions C0-C6 (+ ablations). All share model, options, questions, seeds.
Sieve's plan is computed first so truncate / uniform_lingua match its final token budget."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from sieve.classifier import LaneClassifier, model_path
from sieve.compressor import Compressor
from sieve.features import Similarity
from sieve.orchestrator import SievePipeline
from sieve.segmenter import SessionState
from sieve.tokenizer import get_tokenizer

CONDITIONS = ["baseline", "truncate", "uniform_lingua", "sieve_rules", "sieve_full", "sieve_nomemo"]
REFERENCE = "reference_bigctx"
ABLATIONS = ["sieve_full+noverifier", "sieve_full+noprefetch", "sieve_full+noledger", "sieve_full+clf_heuristic",
             "sieve_full+core6"]
DESCRIPTIONS = {
    "baseline": "C0 raw prompt, append-only history",
    "truncate": "C1 drop oldest messages to C4's token budget",
    "uniform_lingua": "C2 LLMLingua-2 at one uniform rate matched to C4's budget",
    "sieve_rules": "C3 hard rules + heuristic labeller (no ML, no verifier)",
    "sieve_full": "C4 rules + classifier + ledger/epochs + verifier + tools/prefetch",
    "sieve_nomemo": "C5 C4 re-routed and re-compressed every turn, no ledger, no memo",
    "reference_bigctx": "C6 raw prompt with num_ctx=8192 (accuracy upper bound)",
}


class Runtime:
    """Shared heavy objects (tokenizer, embedder, compressor, classifiers)."""

    def __init__(self, cfg, force_approx_tokens: bool = False, force_tfidf: bool = False,
                 force_fallback_compressor: bool = False):
        self.cfg = cfg
        cal = Path(cfg.runs_dir) / "env.json"
        self.tok = get_tokenizer(cfg.tokenizer_model, force_approx_tokens, cal)
        self.sim = Similarity(cfg.embed_model, force_tfidf)
        memo = Path(cfg.runs_dir) / ".cache" / "compress_memo.jsonl"
        self.comp = Compressor(cfg, self.tok, memo, force_fallback_compressor)
        self._comp_nomemo = None
        self._clf: Dict[str, LaneClassifier] = {}
        self.force_fallback_compressor = force_fallback_compressor

    @property
    def comp_nomemo(self) -> Compressor:
        if self._comp_nomemo is None:
            c = Compressor.__new__(Compressor)
            c.__dict__.update(self.comp.__dict__)
            c.memo, c.use_memo, c.memo_path, c.memo_hits, c.calls, c.total_ms = {}, False, None, 0, 0, 0.0
            self._comp_nomemo = c
        return self._comp_nomemo

    def classifier(self, labels: str = "oracle", features: str = "all") -> LaneClassifier:
        k = f"{labels}_{features}"
        if k not in self._clf:
            p = model_path(self.cfg, labels, features)
            if not p.exists():
                from sieve.cli import train_models
                train_models(self.cfg, self, eval_report=False)
            self._clf[k] = LaneClassifier.load(p, self.cfg)
        return self._clf[k]

    def pipeline(self, name: str) -> SievePipeline:
        base, _, flag = name.partition("+")
        cfg = self.cfg
        if base == "sieve_rules":
            return SievePipeline(cfg, self.tok, self.sim, self.comp, router="heuristic", use_verifier=False, name=name)
        if base == "sieve_nomemo":
            return SievePipeline(cfg, self.tok, self.sim, self.comp_nomemo, router="classifier",
                                 classifier=self.classifier(cfg.classifier_labels), use_ledger=False, name=name)
        if base != "sieve_full":
            raise KeyError(name)
        kw = dict(router="classifier", classifier=self.classifier(cfg.classifier_labels))
        if flag == "noverifier":
            kw["use_verifier"] = False
        elif flag == "noprefetch":
            kw["use_prefetch"] = False
            kw["tool_mode"] = "native"
        elif flag == "noledger":
            kw["use_ledger"] = False
        elif flag == "clf_heuristic":
            kw["classifier"] = self.classifier("heuristic")
        elif flag == "core6":
            kw["classifier"] = self.classifier(cfg.classifier_labels, "core6")
        elif flag:
            raise KeyError(name)
        return SievePipeline(cfg, self.tok, self.sim, self.comp, name=name, **kw)


def raw_messages(history: List[Dict[str, Any]], question: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{"role": m["role"], "content": m["content"]} for m in history] + \
           [{"role": "user", "content": question["content"]}]


class Condition:
    uses_tools = False
    num_ctx: Optional[int] = None

    def __init__(self, name: str, rt: Runtime, budgets: Optional[Dict[str, List[int]]] = None):
        self.name, self.rt, self.cfg, self.budgets = name, rt, rt.cfg, budgets or {}

    def start_case(self, case: Dict[str, Any]):
        self.case = case

    def build(self, ep_idx: int, history, question) -> Dict[str, Any]:
        raise NotImplementedError

    def _plain(self, msgs, t_overhead_ms=0.0, **extra) -> Dict[str, Any]:
        n = self.rt.tok.count_messages(msgs)
        out = dict(messages=msgs, tools=None, prompt_tokens=n,
                   timers=dict(segment=0.0, classify=0.0, compress=t_overhead_ms, verify=0.0, assemble=0.0),
                   lanes={"compress": 0, "cache": 0, "externalize": 0}, epoch_id=0, rebuild_tokens=n,
                   memo_hits=0, reinflations=0, hard_truncation=False, prefetched=[],
                   canonicalizer_tokens_saved=0, compressor_fallback=False)
        out.update(extra)
        return out


class Baseline(Condition):
    def build(self, ep_idx, history, question):
        return self._plain(raw_messages(history, question))


class Reference(Baseline):
    num_ctx = 8192


class Truncate(Condition):
    def build(self, ep_idx, history, question):
        budget = self.budgets[self.case["case_id"]][ep_idx]
        msgs = raw_messages(history, question)
        sys_msgs = [m for m in msgs[:1] if m["role"] == "system"]
        rest = msgs[len(sys_msgs):-1]
        q = msgs[-1]
        dropped = 0
        while rest and self.rt.tok.count_messages(sys_msgs + rest + [q]) > budget:
            rest.pop(0)
            dropped += 1
        return self._plain(sys_msgs + rest + [q], dropped_messages=dropped, matched_budget=budget)


class UniformLingua(Condition):
    """One uniform LLMLingua-2 rate on every non-protected segment, matched to C4's budget."""

    def start_case(self, case):
        super().start_case(case)
        self.state = SessionState(case["case_id"], self.rt.tok, self.cfg, canonicalize_kinds=())

    def build(self, ep_idx, history, question):
        budget = self.budgets[self.case["case_id"]][ep_idx]
        msgs = [dict(m) for m in history] + [dict(question, kind="query")]
        segs = self.state.segment(msgs)
        protected = lambda s: s.role == "system" or s.kind == "query" or s.tags.get("pinned") or s.tags.get("never_compress")
        raw = raw_messages(history, question)
        raw_n = self.rt.tok.count_messages(raw)
        unprot = sum(s.tokens for s in segs if not protected(s))
        t0 = time.perf_counter()
        rate = 1.0 if raw_n <= budget or unprot == 0 else max(0.1, min(1.0, 1 - (raw_n - budget) / unprot))
        rate = round(rate * 20) / 20
        for _ in range(6):
            texts = {}
            for s in segs:
                texts[s.key] = s.content if protected(s) or rate >= 1.0 or s.tokens < 8 else \
                    self.rt.comp.compress(s.content, rate)["text"]
            by_msg: Dict[int, List] = {}
            for s in segs:
                by_msg.setdefault(s.msg_index, []).append((s, texts[s.key]))
            out = []
            for i, m in enumerate(msgs):
                parts = by_msg.get(i, [])
                out.append({"role": m["role"], "content": "".join(t + (s.sep if k < len(parts) - 1 else "")
                                                                  for k, (s, t) in enumerate(parts))})
            n = self.rt.tok.count_messages(out)
            if n <= budget * 1.02 or rate <= 0.1:
                break
            rate = max(0.1, round((rate - 0.05) * 20) / 20)
        ms = (time.perf_counter() - t0) * 1000
        return self._plain(out, ms, uniform_rate=rate, matched_budget=budget,
                           compressor_fallback=self.rt.comp.fallback)


class SieveCondition(Condition):
    uses_tools = True

    def __init__(self, name, rt, budgets=None):
        super().__init__(name, rt, budgets)
        self.pipeline = rt.pipeline(name)

    def start_case(self, case):
        super().start_case(case)
        self.session = self.pipeline.new_session(f"{self.name}|{case['case_id']}")

    def build(self, ep_idx, history, question):
        return self.session.build(history, question)


def make_condition(name: str, rt: Runtime, budgets=None) -> Condition:
    if name == "baseline":
        return Baseline(name, rt, budgets)
    if name == REFERENCE:
        return Reference(name, rt, budgets)
    if name == "truncate":
        return Truncate(name, rt, budgets)
    if name == "uniform_lingua":
        return UniformLingua(name, rt, budgets)
    return SieveCondition(name, rt, budgets)


def plan_budgets(cases, rt: Runtime) -> Dict[str, List[int]]:
    """Run C4's planner (no model calls) to get its final prompt tokens per evaluation point."""
    pipe = rt.pipeline("sieve_full")
    out = {}
    for case in cases:
        sess = pipe.new_session(f"plan|{case['case_id']}")
        toks = []
        for ep in case["eval_points"]:
            qi = ep["question_index"]
            plan = sess.build(case["messages"][:qi], case["messages"][qi])
            toks.append(plan["prompt_tokens"])
        out[case["case_id"]] = toks
    return out

"""Sieve pipeline: segment -> rules/classifier -> lanes -> ledger/epochs -> layout -> tools/prefetch."""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from sieve.budget import allocate, context_report
from sieve.features import compute_features
from sieve.ledger import Emitted, Ledger, first_divergence
from sieve.rules import Decision, RuleContext, apply_rules, heuristic_label
from sieve.segmenter import Segment, SessionState
from sieve.store import Store
from sieve.verifier import extract_atoms, flat

SIEVE_LINE = ("Note: some earlier context was replaced by stubs of the form [ref:ID | kind | size | summary | keys | ...]. "
              "If a stub may contain the answer, call the expand_context tool with that ID to read the full text "
              "before answering; never guess a stub's contents.")
PINNED_HEADER = "Pinned reference documents:"
MAX_HINT_REFS = 6
TAIL_HINT = ("\n\n(Stubs with keys: {refs}. If one of these may hold the answer, call expand_context with its id "
             "before answering.)")
EXPAND_TOOL = {"type": "function", "function": {
    "name": "expand_context",
    "description": "Return the full original text of an earlier context segment that was replaced by a [ref:ID | ...] stub.",
    "parameters": {"type": "object", "properties": {
        "seg_id": {"type": "string", "description": "The 12-character id that follows 'ref:' in the stub."}},
        "required": ["seg_id"]}}}


class SievePipeline:
    def __init__(self, cfg, tok, sim, compressor, router: str = "classifier", classifier=None,
                 use_ledger: bool = True, use_verifier: bool = True, use_prefetch: bool = True,
                 tool_mode: Optional[str] = None, name: str = "sieve_full"):
        self.cfg, self.tok, self.sim, self.comp = cfg, tok, sim, compressor
        self.router, self.clf = router, classifier
        self.use_ledger, self.use_verifier, self.use_prefetch = use_ledger, use_verifier, use_prefetch
        self.tool_mode = tool_mode or cfg.tool_mode
        self.name = name
        if router == "classifier" and classifier is None:
            raise ValueError("classifier router needs a trained classifier (run `python -m sieve train`)")

    def new_session(self, session_id: str, store_path=None) -> "SieveSession":
        return SieveSession(self, session_id, store_path)


class SieveSession:
    def __init__(self, p: SievePipeline, session_id: str, store_path=None):
        self.p = p
        self.cfg = p.cfg
        self.session_id = session_id
        self.state = SessionState(session_id, p.tok, p.cfg)
        self.store = Store(p.cfg, p.tok, session_id, store_path)
        self.ledger = Ledger()
        self.tools_on = False
        self.turns = 0

    # ---------------- routing ----------------
    def decide(self, seg: Segment, emitted_lanes: Dict[str, str]) -> Decision:
        stub_tokens = self.p.tok.count(self.store.stub_for(seg)) if seg.tokens >= self.cfg.min_segment_tokens else 10**6
        rr = apply_rules(seg, self.cfg, emitted_lanes, atoms_of=extract_atoms, ctx=RuleContext(stub_tokens))
        if rr["decision"] is not None:
            d = rr["decision"]
            d.reasons = rr["fired"]
            return d
        if self.p.router == "heuristic":
            d = heuristic_label(seg, self.cfg)
        else:
            d = self.p.clf.predict(seg)
        d.forbid_externalize = rr["forbid_externalize"]
        d.reasons = rr["fired"] + d.reasons
        if d.lane == "externalize" and d.forbid_externalize:
            d.lane, d.ratio = "compress", self.cfg.snap_ratio(max(0.7, d.ratio if d.ratio < 1 else 0.7), seg.kind)
            d.reasons.append("externalize_forbidden->compress")
        return d

    def can_externalize(self, seg: Segment, d: Decision) -> bool:
        if d.forbid_externalize or seg.kind in ("query", "system") or seg.tags.get("pinned"):
            return False
        if seg.tokens < self.cfg.min_segment_tokens:
            return False
        return self.p.tok.count(self.store.stub_for(seg)) <= 0.5 * seg.tokens

    def emit(self, seg: Segment, d: Decision, segs_by_key: Dict[str, Segment], timers: Dict[str, float]) -> Emitted:
        e = Emitted(segment_id=seg.id, key=seg.key, lane=d.lane, ratio=d.ratio, text=seg.content,
                    tokens=seg.tokens, source=d.source, compressor_fallback=False, backref=d.backref,
                    pinned=bool(seg.tags.get("pinned")))
        if d.backref:
            orig = segs_by_key.get(d.backref)
            oid = orig.id if orig else d.backref
            e.text = f"[repeat: same content as ref:{oid} earlier in this conversation]"
        elif d.lane == "externalize":
            e.text = self.store.put(seg, seg.turn_index)
            e.ratio = 1.0
        elif d.lane == "compress" and d.ratio < 1.0:
            t0 = time.perf_counter()
            r = self.p.comp.compress_verified(seg.content, d.ratio, seg.kind, use_verifier=self.p.use_verifier)
            dt = (time.perf_counter() - t0) * 1000
            timers["compress"] += r["ms"]
            timers["verify"] += max(0.0, dt - r["ms"])
            timers["memo_hits"] += r["memo_hits"]
            e.text, e.ratio, e.verifier = r["text"], r["ratio"], r["verifier"]
            e.compressor_fallback = self.p.comp.fallback
            if r["verifier"] and r["verifier"]["reinflated"]:
                timers["reinflations"] += 1
        e.tokens = self.p.tok.count(e.text)
        return e

    # ---------------- layout ----------------
    def render(self, messages: List[Dict[str, Any]], segs: List[Segment], emitted: Dict[str, Emitted],
               tail_text: str = "") -> tuple:
        """Chronological layout. A message whose chunks are all externalized gets ONE merged stub.
        Returns (messages, groups) with groups = {stub_id: [member segment ids]}."""
        by_msg: Dict[int, List[tuple]] = {}
        pinned: List[str] = []
        for s in segs:
            e = emitted[s.key]
            if e.dropped:
                continue
            text = e.text
            if e.pinned and s.role != "system":
                pinned.append(text)
                text = f"[pinned document ref:{s.id} is shown in the system prompt]"
            by_msg.setdefault(s.msg_index, []).append((s, text))
        groups: Dict[str, List[str]] = {}
        out = []
        for i, m in enumerate(messages):
            parts = by_msg.get(i)
            if not parts:
                continue
            if len(parts) > 1 and all(emitted[s.key].lane == "externalize" and not emitted[s.key].backref
                                      for s, _ in parts):
                whole = "".join(s.content + (s.sep if k < len(parts) - 1 else "") for k, (s, _) in enumerate(parts))
                g = self.store.put_group(whole, parts[0][0].kind, parts[0][0].turn_index)
                groups[g.id] = [s.id for s, _ in parts]
                content = self.store.stub_for(g)
            else:
                content = "".join(t + (s.sep if k < len(parts) - 1 else "") for k, (s, t) in enumerate(parts))
                for s, _ in parts:
                    if emitted[s.key].lane == "externalize" and not emitted[s.key].backref:
                        groups[s.id] = [s.id]
            if m["role"] == "system":
                content = content + "\n\n" + SIEVE_LINE
                if pinned:
                    content += "\n\n" + PINNED_HEADER + "\n\n" + "\n\n".join(pinned)
            out.append({"role": m["role"], "content": content})
        if out and out[0]["role"] != "system":
            sys_c = SIEVE_LINE + (("\n\n" + PINNED_HEADER + "\n\n" + "\n\n".join(pinned)) if pinned else "")
            out.insert(0, {"role": "system", "content": sys_c})
        if tail_text:
            out[-1] = {"role": out[-1]["role"], "content": out[-1]["content"] + tail_text}
        return out, groups

    def tools(self) -> Optional[List[Dict[str, Any]]]:
        if self.p.tool_mode == "prefetch" or not self.tools_on:
            return None
        return [EXPAND_TOOL]

    def hint_allowance(self) -> int:
        if self.p.tool_mode == "prefetch":
            return 0
        one = "ref:0123456789ab (keys: Abcdefgh Ijklmn, 12,345, XY-12345, 2 March 2029, Qrstuv)"
        return self.p.tok.count(TAIL_HINT.format(refs=", ".join([one] * MAX_HINT_REFS)))

    # ---------------- main entry ----------------
    def build(self, history: List[Dict[str, Any]], question: Dict[str, Any], force_epoch: bool = False,
              gold_atoms: Optional[List[str]] = None) -> Dict[str, Any]:
        cfg = self.cfg
        timers = dict(segment=0.0, classify=0.0, compress=0.0, verify=0.0, assemble=0.0, memo_hits=0,
                      reinflations=0)
        t0 = time.perf_counter()
        msgs = [dict(m) for m in history] + [dict(question, kind="query")]
        segs = self.state.segment(msgs)
        current_turn = int(question.get("turn", 0))
        compute_features(segs, question.get("content", ""), current_turn, self.p.sim, self.store.expand_counts)
        segs_by_key = {s.key: s for s in segs}
        for s in segs:  # near-dup safety: every atom of the new version must already be in the original
            if s.dup_type == "near":
                o = segs_by_key.get(s.dup_of)
                s.features["_near_atoms_subset"] = float(o is not None and flat(extract_atoms(s.content)) <=
                                                         flat(extract_atoms(o.content)))
        timers["segment"] = (time.perf_counter() - t0) * 1000

        decisions: Dict[str, Decision] = {}
        emitted: Dict[str, Emitted] = {}
        replayed: List[str] = []
        new_keys: List[str] = []
        tool_t = self.p.tok.tools_tokens([EXPAND_TOOL]) if self.p.tool_mode != "prefetch" else 0
        hint_t = self.hint_allowance()
        trigger = cfg.epoch_trigger_fraction * cfg.num_ctx - cfg.reserve_output_tokens

        def route_and_emit(seg_list: List[Segment], lanes: Dict[str, str]):
            for s in seg_list:
                tc = time.perf_counter()
                d = self.decide(s, lanes)
                timers["classify"] += (time.perf_counter() - tc) * 1000
                decisions[s.key] = d
                emitted[s.key] = self.emit(s, d, segs_by_key, timers)
                lanes[s.key] = d.lane

        def exact_tokens(tail: str = "") -> int:
            m, _ = self.render(msgs, segs, emitted, tail)
            on = self.tools_on or any(e.lane == "externalize" and not e.dropped for e in emitted.values())
            return self.p.tok.count_messages(m, [EXPAND_TOOL] if (on and self.p.tool_mode != "prefetch") else None)

        epoch_reason = None
        if self.p.use_ledger and self.ledger.entries and not force_epoch:
            lanes: Dict[str, str] = {}
            for s in segs:
                if s.key in self.ledger.by_key:
                    le = self.ledger.by_key[s.key]
                    emitted[s.key] = Emitted(**{k: v for k, v in asdict(le).items() if k != "epoch_id"})
                    decisions[s.key] = Decision(s.id, s.key, "ledger_replay", le.lane, le.ratio, backref=le.backref)
                    lanes[s.key] = le.lane
                    replayed.append(s.key)
            new = [s for s in segs if s.key not in self.ledger.by_key]
            new_keys = [s.key for s in new]
            route_and_emit(new, lanes)
            if exact_tokens() + hint_t > trigger:
                epoch_reason = "budget"
        else:
            epoch_reason = "forced" if force_epoch else ("session_start" if self.ledger.epoch_id < 0 else "no_ledger")

        alloc_log = None
        if epoch_reason:
            decisions.clear()
            emitted.clear()
            replayed = []
            new_keys = [s.key for s in segs]
            route_and_emit(segs, {})
            items = [{"seg": s, "dec": decisions[s.key], "em": emitted[s.key]} for s in segs]
            target = int(min(cfg.budget_limit, cfg.epoch_hysteresis * cfg.epoch_trigger_tokens))

            def emit_fn(seg, d):
                return self.emit(seg, d, segs_by_key, timers)

            def run_alloc(tgt: int, limit: int):
                cur = exact_tokens()
                body = sum(0 if it["em"].dropped else it["em"].tokens for it in items)
                on = self.tools_on or any(it["em"].lane == "externalize" for it in items)
                overhead = cur - body + (0 if on else tool_t) + hint_t
                log = allocate(items, overhead, tgt, limit, cfg, emit_fn, self.can_externalize)
                for it in items:
                    decisions[it["seg"].key], emitted[it["seg"].key] = it["dec"], it["em"]
                return log

            if exact_tokens() + hint_t > trigger:
                alloc_log = run_alloc(target, cfg.budget_limit)
            # exact re-check (merged stubs / template overhead are only known after rendering)
            for _ in range(3):
                if exact_tokens() + hint_t <= cfg.budget_limit:
                    break
                more = run_alloc(cfg.budget_limit - (exact_tokens() + hint_t - cfg.budget_limit) - 1, cfg.budget_limit - 1)
                if alloc_log:
                    for k in ("lowered", "externalized", "dropped"):
                        alloc_log[k] += more[k]
                    alloc_log["hard_truncation"] = alloc_log["hard_truncation"] or more["hard_truncation"]
                    alloc_log["end_tokens"] = more["end_tokens"]
                else:
                    alloc_log = more

        self.tools_on = self.tools_on or any(e.lane == "externalize" and not e.dropped for e in emitted.values())

        # ---------- prefetch + tail hint (inside the final user message only) ----------
        ta = time.perf_counter()
        self.store.begin_request()
        prefetched: List[Dict[str, Any]] = []
        prefetch_text = ""
        base_msgs, groups = self.render(msgs, segs, emitted, "")
        tools = self.tools()
        base_tokens = self.p.tok.count_messages(base_msgs, tools)
        sim_of = {s.id: s.features.get("query_similarity", 0.0) for s in segs}
        gsim = {g: max(sim_of.get(m, 0.0) for m in mem) for g, mem in groups.items()}
        if self.p.use_prefetch and self.p.tool_mode in ("prefetch", "hybrid"):
            thr = cfg.prefetch_sim_threshold if self.p.tool_mode == "hybrid" else min(cfg.prefetch_sim_threshold, 0.5)
            room = cfg.budget_limit - base_tokens - hint_t
            for g in sorted(groups, key=lambda g: (-gsim[g], g)):
                if gsim[g] < thr:
                    break
                n = self.store.data[g]["tokens"]
                if n + 20 > room:
                    continue
                txt = self.store.expand(g, self.session_id, current_turn)
                if txt.startswith("[expand_context error"):
                    break
                prefetch_text += f"\n\n[Prefetched context for ref:{g}]\n" + txt.split("\n", 1)[1]
                room -= n + 20
                prefetched.append({"seg": g, "tokens": n, "sim": round(gsim[g], 4), "members": groups[g]})
        tail_text = prefetch_text
        if tools:  # without thinking the model rarely expands unprompted; a tail hint is cache-safe
            done = {p["seg"] for p in prefetched}
            keys = {g: self.store.data[g]["keys"] for g in groups}
            refs = [g for g in sorted(groups, key=lambda g: (not keys[g], -gsim[g], g)) if g not in done][:MAX_HINT_REFS]
            if refs:
                tail_text += TAIL_HINT.format(refs=", ".join(
                    f"ref:{r} (keys: {', '.join(keys[r]) if keys[r] else '-'})" for r in refs))
        final_msgs = self.render(msgs, segs, emitted, tail_text)[0] if tail_text else base_msgs
        prompt_tokens = self.p.tok.count_messages(final_msgs, tools)
        div = first_divergence(self.ledger.last_messages, base_msgs)
        tools_changed = bool(self.ledger.last_messages is not None and self.ledger.last_tools_on != bool(tools))
        if tools_changed:
            div = 0  # tools render inside the system block: the whole prefix changes
        rebuild_tokens = self.p.tok.count_messages(final_msgs[div:], tools if div == 0 else None) \
            if self.ledger.last_messages is not None else prompt_tokens
        timers["assemble"] = (time.perf_counter() - ta) * 1000

        # ---------- ledger bookkeeping ----------
        if epoch_reason or not self.p.use_ledger:
            self.ledger.reset([emitted[s.key] for s in segs])
        else:
            for k in new_keys:
                self.ledger.append(emitted[k])
        self.ledger.last_messages = base_msgs
        self.ledger.last_tools_on = bool(tools)
        self.ledger.history.append({"turn": current_turn, "replayed": len(replayed), "new": len(new_keys),
                                    "epoch": bool(epoch_reason), "epoch_reason": epoch_reason,
                                    "epoch_id": self.ledger.epoch_id, "rebuild_tokens": rebuild_tokens,
                                    "first_changed_msg": div, "prompt_tokens": prompt_tokens})
        self.turns += 1
        lanes = {"compress": 0, "cache": 0, "externalize": 0}
        for e in emitted.values():
            if not e.dropped:
                lanes[e.lane] = lanes.get(e.lane, 0) + 1
        return {
            "messages": final_msgs, "tools": tools, "prompt_tokens": prompt_tokens,
            "prompt_tokens_no_tail": base_tokens,
            "segments": segs, "decisions": decisions, "emitted": emitted, "groups": groups,
            "epoch_id": self.ledger.epoch_id, "epoch_reason": epoch_reason, "rebuild_tokens": rebuild_tokens,
            "first_changed_msg": div, "replayed": replayed, "new": new_keys, "alloc": alloc_log,
            "hard_truncation": bool(alloc_log and alloc_log["hard_truncation"]),
            "prefetched": prefetched, "lanes": lanes, "timers": timers,
            "canonicalizer_tokens_saved": self.state.canonicalizer_tokens_saved(len(msgs)),
            "compressor_fallback": any(e.compressor_fallback for e in emitted.values()),
            "memo_hits": int(timers["memo_hits"]), "reinflations": int(timers["reinflations"]),
            **context_report(prompt_tokens, self.cfg.num_ctx),
        }

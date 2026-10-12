import json

import pytest

from sieve.workloads.generator import generate_case, generate_suite
from sieve.workloads.scoring import score_answer


def test_segmenter_stable(cfg, rt):
    from sieve.segmenter import SessionState
    case = generate_case("C", 0, 7, "long")
    a = SessionState("s1", rt.tok, cfg).segment(case["messages"])
    b = SessionState("s2", rt.tok, cfg).segment(case["messages"])
    assert [(s.key, s.id, s.tokens) for s in a] == [(s.key, s.id, s.tokens) for s in b]
    assert all(s.tokens <= cfg.chunk_max_tokens * 1.3 for s in a if s.kind == "doc")
    # chunking is lossless: re-joining chunks restores the message
    msg = case["messages"][1]["content"]
    parts = [s for s in a if s.msg_index == 1]
    assert "".join(s.content + (s.sep if i < len(parts) - 1 else "") for i, s in enumerate(parts)) == msg


def test_repeat_counters_reset_per_session(cfg, rt):
    from sieve.segmenter import SessionState
    case = generate_case("D", 0, 7, "mid")
    msgs = case["messages"]
    s1 = SessionState("a", rt.tok, cfg).segment(msgs)
    assert any(s.repeat_count >= 1 for s in s1)
    first_doc_only = msgs[:3]
    s2 = SessionState("b", rt.tok, cfg).segment(first_doc_only)
    assert all(s.repeat_count == 0 for s in s2)


def test_rules_override_classifier(cfg, rt, trained):
    from sieve.conditions import Runtime
    pipe = rt.pipeline("sieve_full")
    case = generate_case("J", 3, 7, "mid")  # tagged never_compress / never_externalize / pinned
    sess = pipe.new_session("t")
    ep = case["eval_points"][0]
    plan = sess.build(case["messages"][:ep["question_index"]], case["messages"][ep["question_index"]])
    for s in plan["segments"]:
        d = plan["decisions"][s.key]
        if s.role == "system":
            assert d.source == "rule:system_cache" and d.lane == "cache"
        if s.kind == "query":
            assert d.ratio == 1.0 and d.source == "rule:current_query"
        if s.tags.get("never_compress"):
            assert d.lane == "cache" and plan["emitted"][s.key].text == s.content
        if s.tags.get("never_externalize"):
            assert d.lane != "externalize"
        if s.tags.get("pinned"):
            assert d.lane == "cache"


def test_stub_and_memo_deterministic(cfg, rt):
    from sieve.segmenter import SessionState
    from sieve.store import Store
    case = generate_case("B", 1, 7, "long")
    seg = [s for s in SessionState("x", rt.tok, cfg).segment(case["messages"]) if s.tokens > 150][0]
    a = Store(cfg, rt.tok, "x").stub_for(seg)
    b = Store(cfg, rt.tok, "y").stub_for(seg)
    assert a == b and a.startswith(f"[ref:{seg.id} |") and "expand_context" in a
    c1 = rt.comp.compress(seg.content, 0.5)
    c2 = rt.comp.compress(seg.content, 0.5)
    assert c1["text"] == c2["text"] and c2["memo_hit"]


@pytest.mark.skipif(bool(__import__("os").environ.get("SIEVE_SKIP_HEAVY")), reason="downloads the LLMLingua-2 model")
def test_llmlingua_memo_deterministic(cfg, rt):
    from sieve.compressor import Compressor
    comp = Compressor(cfg, rt.tok)
    if comp.fallback:
        pytest.skip("LLMLingua-2 not available")
    text = generate_case("C", 1, 7, "mid")["messages"][1]["content"][:1500]
    a = comp.compress(text, 0.5)["text"]
    comp.memo.clear()
    b = comp.compress(text, 0.5)["text"]
    assert a == b


def test_ledger_replays_byte_identically(cfg, rt, trained):
    pipe = rt.pipeline("sieve_full")
    case = generate_case("K", 0, 7, "long")
    sess = pipe.new_session("led")
    prev = None
    for ep in case["eval_points"]:
        qi = ep["question_index"]
        plan = sess.build(case["messages"][:qi], case["messages"][qi])
        if prev is not None and plan["epoch_reason"] is None:
            for k, e in prev.items():
                assert plan["emitted"][k].text == e.text
            assert plan["replayed"]
        prev = dict(plan["emitted"])


def test_epoch_trigger_threshold(cfg, rt, trained):
    pipe = rt.pipeline("sieve_full")
    sess = pipe.new_session("ep")
    sysm = {"role": "system", "content": "You answer questions.", "kind": "system", "turn": 0}
    from sieve.workloads.generator import Gen
    g = Gen("epoch")
    hist = [sysm]
    reasons = []
    for t in range(1, 20):
        hist = hist + [{"role": "user", "content": g.paragraph(6), "kind": "chat", "turn": t},
                       {"role": "assistant", "content": g.filler(700), "kind": "chat", "turn": t}]
        plan = sess.build(hist, {"role": "user", "content": "What was discussed?", "turn": t + 1})
        projected_ok = plan["prompt_tokens"] <= cfg.budget_limit or plan["hard_truncation"]
        assert projected_ok
        reasons.append(plan["epoch_reason"])
    assert reasons[0] == "session_start"
    assert "budget" in reasons  # history grew past epoch_trigger_fraction * num_ctx at some turn
    assert reasons.count(None) >= 1  # and some turns were plain replays


def test_budget_never_exceeds_limit(cfg, rt, trained):
    pipe = rt.pipeline("sieve_full")
    for cid in [("B", 0), ("E", 2), ("K", 1), ("C", 4)]:
        case = generate_case(cid[0], cid[1], 7, "long")
        sess = pipe.new_session(f"b{cid}")
        for ep in case["eval_points"]:
            plan = sess.build(case["messages"][:ep["question_index"]], case["messages"][ep["question_index"]])
            no_tail = rt.tok.count_messages(sess.ledger.last_messages, plan["tools"])
            assert no_tail <= cfg.budget_limit or plan["hard_truncation"], (cid, no_tail)


def test_verifier_reinflates_when_atom_dropped(cfg, rt):
    from sieve.compressor import Compressor
    comp = Compressor(cfg, rt.tok, force_fallback=True, use_memo=False)
    text = ("The weather was fine and everyone enjoyed lunch. " * 6 +
            "Unit KX-48213 runs firmware v4.12.3 at 5,585 rpm. " + "Nothing else happened that day. " * 6)
    orig = comp.compress
    comp.compress = lambda t, r, query="": ({"text": t.replace("5,585", ""), "tokens": rt.tok.count(t), "memo_hit": False, "ms": 0}
                                            if r < 1 else orig(t, r))
    out = comp.compress_verified(text, 0.3, "doc", use_verifier=True)
    assert out["verifier"]["reinflated"] and out["ratio"] == 1.0 and "5,585" in out["text"]
    assert "5,585" in out["verifier"]["lost"]


def test_store_unknown_id_and_budget(cfg, rt):
    from sieve.segmenter import SessionState
    from sieve.store import Store
    st = Store(cfg, rt.tok, "s")
    assert st.expand("deadbeef0000").startswith("[expand_context error")
    case = generate_case("E", 0, 7, "long")
    segs = [s for s in SessionState("x", rt.tok, cfg).segment(case["messages"]) if s.tokens > 200]
    for s in segs:
        st.put(s, 1)
    st.begin_request()
    used = 0
    for s in segs:
        out = st.expand(s.id)
        assert out
    assert st.request_expanded_tokens <= cfg.expansion_budget_tokens
    assert st.expand(segs[0].id).startswith("[expand_context error") or st.request_expanded_tokens <= cfg.expansion_budget_tokens
    other = Store(cfg, rt.tok, "other-session")
    assert other.expand(segs[0].id).startswith("[expand_context error")


def test_tool_loop_appends_at_tail(cfg, rt):
    from sieve.mock_client import MockClient
    from sieve.orchestrator import EXPAND_TOOL
    from sieve.store import Store
    cl = MockClient(cfg, rt.tok)

    class S:
        id, kind, tokens = "abcdef123456", "doc", 80
        content = "Archive: the peak throughput measured for Quillmar was 48,213 requests per second."
    st = Store(cfg, rt.tok, "s")
    st.put(S, 1)
    stub = '[ref:abcdef123456 | doc | 80t | "Archive" | keys: Quillmar, peak throughput | use expand_context("abcdef123456") to read]'
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": stub},
            {"role": "assistant", "content": "ok"}, {"role": "user", "content": "What was the peak throughput for Quillmar?"}]
    out = cl.run_with_tools(msgs, [EXPAND_TOOL], lambda sid: st.expand(sid), 2)
    assert out["tool_rounds"] == 1
    tail = out["messages"][len(msgs):]
    assert out["messages"][:len(msgs)] == msgs
    assert tail[0]["role"] == "assistant" and tail[0]["tool_calls"] and tail[1]["role"] == "tool"
    assert "48,213" in out["answer"]


def test_generator_deterministic_and_gold_only_in_needed():
    a = generate_suite("smoke", 7)
    b = generate_suite("smoke", 7)
    assert json.dumps(a) == json.dumps(b)
    assert json.dumps(generate_suite("smoke", 8)) != json.dumps(a)
    for case in generate_suite("standard", 7):
        msgs = {m["mid"]: m for m in case["messages"]}
        for ep in case["eval_points"]:
            for f in ep["facts"]:
                assert any(f in msgs[n]["content"] for n in ep["needed_segment_ids"]), (case["case_id"], f)
            if ep["answer_kind"] in ("refusal", "yesno"):
                continue
            for m in case["messages"][:ep["question_index"]]:
                if m["mid"] in ep["needed_segment_ids"] or m["kind"] == "gold_answer":
                    continue
                for gold in ep["gold"]:
                    if len(gold) >= 4:
                        assert gold not in m["content"], (case["case_id"], gold, m["mid"])


def test_scoring():
    ep = dict(answer_kind="number", numeric=4812, gold=["4,812"], gold_all=[], forbid=[])
    assert score_answer("It was 4,812 requests per second.", ep)
    assert not score_answer("It was 4,813.", ep)
    ep = dict(answer_kind="text", gold=["Doran Psatik", "Psatik"], gold_all=[], forbid=[])
    assert score_answer("The owner is Doran Psatik.", ep) and not score_answer("The owner is Doran.", ep)
    ep = dict(answer_kind="text", gold=["22 April 2028"], gold_all=["22", "April", "2028"], forbid=["3 May 2027"])
    assert score_answer("It moved to April 22, 2028.", ep)
    assert not score_answer("It is 3 May 2027.", ep)
    ep = dict(answer_kind="yesno", gold=["no"], gold_all=[], forbid=[])
    assert score_answer("No, it was excluded.", ep) and not score_answer("Yes.", ep)
    ep = dict(answer_kind="refusal", gold=["not mentioned"], gold_all=[], forbid=[])
    assert score_answer("Not mentioned.", ep) and not score_answer("It was 42 units.", ep)


def test_bootstrap_ci():
    from sieve.metrics import bootstrap_ci
    m, lo, hi = bootstrap_ci([1.0] * 50 + [0.0] * 50)
    assert abs(m - 0.5) < 1e-9 and lo < 0.5 < hi and hi - lo < 0.3
    m, lo, hi = bootstrap_ci([2.0] * 20)
    assert m == lo == hi == 2.0


def test_canonicalizer():
    from sieve.canonicalizer import canonicalize
    js = '```json\n' + json.dumps({"a": [{"x": 1, "n": None}, {"x": 1, "n": None}, {"x": 2, "t": []}], "z": ""}) + '\n```'
    out, kinds = canonicalize(js)
    assert "null" not in out and '"_repeat":2' in out and kinds == "json"
    log = "```\nINFO [a] ok\nINFO [a] ok\nINFO [a] ok\nERROR [b] bad\n```"
    out, _ = canonicalize(log)
    assert "×3" in out and "ERROR [b] bad" in out

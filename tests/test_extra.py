"""Tests for mechanisms added beyond the spec's list: cost-sensitive routing, near-duplicate
safety, the context verifier, timing-derived prefix reuse, the fallback guard, CLI/report/paper
generation, and the mock client."""
import json

import numpy as np
import pytest

from sieve.workloads.generator import Gen, generate_case


def test_epoch_fires_exactly_at_threshold(cfg, rt, trained):
    """A replay turn (no epoch) must sit at or below the trigger; the first turn above it is an epoch."""
    pipe = rt.pipeline("sieve_full")
    sess = pipe.new_session("thr")
    g = Gen("thr")
    hist = [{"role": "system", "content": "You answer questions.", "kind": "system", "turn": 0}]
    trigger = cfg.epoch_trigger_fraction * cfg.num_ctx - cfg.reserve_output_tokens
    saw_epoch = False
    for t in range(1, 20):
        hist = hist + [{"role": "user", "content": g.paragraph(6), "kind": "chat", "turn": t},
                       {"role": "assistant", "content": g.filler(700), "kind": "chat", "turn": t}]
        plan = sess.build(hist, {"role": "user", "content": "What was discussed?", "turn": t + 1})
        if plan["epoch_reason"] is None:
            assert plan["prompt_tokens_no_tail"] + sess.hint_allowance() <= trigger
        if plan["epoch_reason"] == "budget":
            saw_epoch = True
    assert saw_epoch


def test_expected_cost_prefers_safe_lane(cfg, trained):
    from sieve.classifier import LaneClassifier, model_path
    clf = LaneClassifier.load(model_path(cfg, "oracle"), cfg)
    # LANES = [cache, compress, externalize]; externalize is more likely but hiding a needed segment costs 5
    lane, ratio, cost, src = clf.decide_from(np.array([0.0, 0.4, 0.6]), 0.7, "doc")
    assert lane == "compress" and src == "classifier"
    lane, *_ = clf.decide_from(np.array([0.0, 0.1, 0.9]), 0.7, "doc")
    assert lane == "externalize"
    lane, ratio, _, src = clf.decide_from(np.array([0.34, 0.33, 0.33]), 0.5, "doc")
    assert (lane, ratio, src) == ("compress", 0.9, "fallback")  # below the confidence threshold


def test_near_duplicate_with_changed_value_is_not_backreferenced(cfg, rt, trained):
    g = Gen("nd")
    body = g.filler(260)
    a = body + " The rate limit for Quorvex is 4,812 requests per minute."
    b = body + " The rate limit for Quorvex is 9,377 requests per minute."
    hist = [{"role": "system", "content": "Answer.", "kind": "system", "turn": 0},
            {"role": "user", "content": a, "kind": "doc", "turn": 1},
            {"role": "assistant", "content": "Noted.", "kind": "chat", "turn": 1},
            {"role": "user", "content": b, "kind": "doc", "turn": 2},
            {"role": "assistant", "content": "Noted.", "kind": "chat", "turn": 2}]
    plan = rt.pipeline("sieve_full").new_session("nd").build(hist, {"role": "user", "content": "What is the rate limit?", "turn": 3})
    near = [s for s in plan["segments"] if s.dup_type == "near"]
    assert near, "fixture should produce a near-duplicate"
    for s in near:
        assert plan["decisions"][s.key].backref is None
    assert "9,377" in "\n".join(m["content"] for m in plan["messages"])


def test_context_recall_catches_dropped_relation():
    from sieve.verifier import extract_atoms, recall
    o = "Change board improves rollout. The on-call owner for the Xetouglour service is Miadouth Ryltum. Review shared."
    bad = "change board improves rollout. - owner Xetouglour Miadouth Ryltum. review shared."
    good = "change board improves rollout. on-call owner Xetouglour service Miadouth Ryltum. review shared."
    assert recall(extract_atoms(o), bad, o)["context_recall"] < 0.5
    assert recall(extract_atoms(o), good, o)["context_recall"] >= 0.5


def test_digit_repair_restores_split_numbers(cfg, rt):
    from sieve.compressor import Compressor
    c = Compressor(cfg, rt.tok, force_fallback=True, use_memo=False)
    assert c._repair("Unit runs at 5,585 rpm, firmware 4.1.", "Unit 5 585 rpm firmware 4. 1.") == "Unit 5,585 rpm firmware 4.1."


def test_timing_derived_prefix_reuse(cfg, rt):
    from sieve.bench import make_record
    from sieve.conditions import Baseline
    from sieve.ollama_client import ChatResult
    case = generate_case("A", 0, 7, "short")
    ep = case["eval_points"][0]
    qi = ep["question_index"]
    cond = Baseline("baseline", rt)
    raw = [{"role": m["role"], "content": m["content"]} for m in case["messages"][:qi + 1]]
    plan = cond.build(0, case["messages"][:qi], case["messages"][qi])
    n = plan["prompt_tokens"]
    fit = {"a": 0.002, "b": 0.0}
    r = ChatResult(content="x", prompt_eval_count=n, prompt_eval_s=0.25 * fit["a"] * n, eval_count=5, eval_s=0.2,
                   decode_tps=25.0, ttft_s=0.3, total_s=0.5)
    out = {"answer": "x", "rounds": [r], "tool_rounds": 0, "expansions": [], "ttft_s": 0.3, "total_s": 0.5,
           "prompt_eval_count": n, "first_prompt_eval_count": n, "prompt_eval_s": r.prompt_eval_s, "eval_count": 5,
           "load_s": 0.0, "decode_tps": 25.0, "vram_peak_mb": None, "vram_source": "x", "thinking_leak": False,
           "messages": plan["messages"]}
    env = {"cold_prefill_fit": fit, "prefix_reuse_method": "timing", "decode_tps": 28.0}
    rec = make_record("t", cond, case, ep, 0, False, plan, raw, out, rt, {}, env)
    assert rec["prefix_reuse_pct_count"] == 0.0           # the server's count shows no reuse...
    assert abs(rec["prefix_reuse_pct"] - 0.75) < 1e-3     # ...but prefill took 25% of the cold prediction
    assert rec["tokens_processed_est"] == round(n * 0.25)
    assert rec["perf_degraded"] is False
    out["decode_tps"] = 10.0
    assert make_record("t", cond, case, ep, 0, False, plan, raw, out, rt, {}, env)["perf_degraded"] is True


def test_bench_refuses_degraded_components(cfg, tmp_path, monkeypatch):
    from sieve import cli
    (tmp_path / "env.json").write_text(json.dumps({"ollama_version": "test", "model": cfg.model}))
    with pytest.raises(SystemExit) as e:
        cli.main(["--approx-tokens", "--tfidf", "--fallback-compressor", "--set", f"runs_dir={tmp_path}",
                  "bench", "--tier", "smoke", "--conditions", "baseline"])
    assert e.value.code == 2


def test_mock_client_is_deterministic(cfg, rt):
    from sieve.mock_client import MockClient
    case = generate_case("C", 0, 7, "mid")
    ep = case["eval_points"][0]
    msgs = [{"role": m["role"], "content": m["content"]} for m in case["messages"][:ep["question_index"] + 1]]
    a, b = MockClient(cfg, rt.tok).chat(msgs), MockClient(cfg, rt.tok).chat(msgs)
    assert (a.content, a.prompt_eval_count, a.total_s) == (b.content, b.prompt_eval_count, b.total_s)
    c = MockClient(cfg, rt.tok)
    c.chat(msgs)
    assert c.chat(msgs).prompt_eval_count < a.prompt_eval_count  # simulated prefix reuse on a repeat


def test_config_overrides():
    from sieve.config import load_config
    c = load_config(overrides={"num_ctx": "8192", "think": "false", "ratio_buckets": "[0.5, 1.0]"})
    assert c.num_ctx == 8192 and c.think is False and c.ratio_buckets == [0.5, 1.0]
    assert c.snap_ratio(0.2, "doc") == 0.5
    with pytest.raises(SystemExit):
        load_config(overrides={"no_such_key": "1"})


def test_cli_gen_report_html_and_paper_source(cfg, rt, trained, tmp_path):
    """End-to-end CLI on a mock run: gen -> bench -> report -> html; paper .tex is generated from the same logs."""
    from sieve import cli
    from sieve.paper import build_tex
    from sieve.report_html import load_all
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "models").mkdir()
    for f in (cfg.runs_dir and __import__("pathlib").Path(cfg.runs_dir) / "models").glob("*.joblib"):
        (runs / "models" / f.name).write_bytes(f.read_bytes())
    base = ["--mock", "--tfidf", "--fallback-compressor", "--set", f"runs_dir={runs}"]
    cli.main(base + ["gen", "--tier", "smoke", "--seed", "7"])
    cli.main(base + ["bench", "--tier", "smoke", "--conditions", "baseline,sieve_full,sieve_nomemo", "--repeats", "1",
                     "--run-id", "m"])
    run = runs / "m"
    assert (run / "claims.md").read_text().startswith("# claims.md")
    assert "MOCK RUN" in (run / "claims.md").read_text()
    cli.main(base + ["html", "--run", "m", "--out", str(tmp_path / "site")])
    html = (tmp_path / "site" / "results.html").read_text()
    assert "MOCK RUN" in html and "<table" in html
    D = load_all(runs, "m")
    router = {"per_lane": {"externalize": {"precision": 1, "recall": 1}}, "macro_f1": 1, "ratio_mae": 0,
              "dangerous_errors": 0, "n_needed": 1}
    D["clf"] = {"routers": {"clf_oracle": router, "heuristic_labeller": router}, "n_rows": 1, "n_groups": 1,
                "inference_ms_per_segment": 0.01}
    tex = build_tex(D, smoke=True)
    acc = D["agg"]["by_condition"]["sieve_full"]["accuracy"]
    assert f"{acc:.3f}" in tex  # numbers in the paper are the logged ones
    assert "\\begin{document}" in tex and "\\end{document}" in tex


def test_experiments_run_in_mock_mode(cfg, rt, trained, tmp_path):
    """selfcheck, cachetest, every lane demo and the break-even sweep run end to end on the mock client."""
    from sieve.experiments import breakeven, cachetest, lane_demo_case, run_case, selfcheck
    from sieve.mock_client import MockClient
    cl = MockClient(cfg, rt.tok)
    env = selfcheck(cfg, cl, rt)
    assert env["mock"] and env["tool_call_supported"] and env["think_disabled_ok"]
    ct = cachetest(cfg, cl, rt)
    labels = [r["label"][:4] for r in ct["cachetest"]]
    assert labels == ["(a) ", "(b) ", "(c1)", "(c2)"] and ct["cold_prefill_fit"]["a"] > 0
    for lane in ("compress", "cache", "externalize"):
        res = run_case(cfg, cl, rt, lane_demo_case(lane), ["baseline", "sieve_full"], tmp_path / "demo", show=True)
        assert len(res["baseline"]) == len(res["sieve_full"]) >= 1
    ext = run_case(cfg, cl, rt, lane_demo_case("externalize"), ["sieve_full"], show=False)["sieve_full"]
    assert any(r["lanes"]["externalize"] for r in ext)
    be = breakeven(cfg, cl, rt, n_cases=2, ps=(0.0, 1.0))
    assert len(be["rows"]) == 4 and {s["p"] for s in be["summary"]} == {0.0, 1.0}

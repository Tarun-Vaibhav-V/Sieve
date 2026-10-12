"""selfcheck, cachetest, demos, and the externalize break-even sweep."""
from __future__ import annotations

import json
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sieve.bench import call_model, make_record, save_artifacts
from sieve.conditions import Runtime, make_condition, raw_messages
from sieve.orchestrator import EXPAND_TOOL, SIEVE_LINE, TAIL_HINT
from sieve.store import Store
from sieve.viz import ledger_view, prompt_diff, result_panel, segment_table, sparkline, waterfall
from sieve.workloads.generator import Conv, Gen, fact_name, fact_number, facts_of

console = Console()
OLLAMA_ENV = ["OLLAMA_FLASH_ATTENTION", "OLLAMA_KV_CACHE_TYPE", "OLLAMA_NUM_PARALLEL", "OLLAMA_MAX_LOADED_MODELS",
              "OLLAMA_CONTEXT_LENGTH", "OLLAMA_KEEP_ALIVE"]


def _ok(t: Table, name: str, ok: Optional[bool], detail: str):
    t.add_row(name, {True: "[green]PASS", False: "[red]FAIL", None: "[yellow]NOTE"}[ok], detail)


def _server_env() -> Dict[str, Optional[str]]:
    """Best effort: read the env of the running `ollama serve` (macOS `ps eww`); else this shell's env."""
    import os
    out = {k: os.environ.get(k) for k in OLLAMA_ENV}
    try:
        txt = subprocess.run(["ps", "eww", "-o", "command=", "-p",
                              subprocess.run(["pgrep", "-f", "ollama serve"], capture_output=True, text=True).stdout.split()[0]],
                             capture_output=True, text=True, timeout=3).stdout
        for k in OLLAMA_ENV:
            for tok in txt.split():
                if tok.startswith(k + "="):
                    out[k] = tok.split("=", 1)[1]
    except Exception:
        pass
    return out


def selfcheck(cfg, client, rt: Runtime) -> Dict[str, Any]:
    t = Table(title="Sieve selfcheck", header_style="bold")
    for c in ("check", "status", "detail"):
        t.add_column(c)
    env: Dict[str, Any] = {"model": cfg.model, "num_ctx": cfg.num_ctx, "mock": client.is_mock,
                           "time": time.strftime("%Y-%m-%d %H:%M:%S"), "platform": platform.platform(),
                           "machine": platform.machine(), "python": platform.python_version()}
    try:
        env["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        env["mem_gb"] = round(int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 2**30, 1)
    except Exception:
        pass
    try:
        env["ollama_version"] = client.version()
        _ok(t, "ollama reachable", True, f"{cfg.ollama_host} version {env['ollama_version']}")
    except Exception as e:
        _ok(t, "ollama reachable", False, str(e))
        console.print(t)
        return env
    env["server_env"] = _server_env()
    models = client.list_models()
    present = cfg.model in models or any(m.split(":")[0] == cfg.model for m in models)
    _ok(t, "model present", present, cfg.model if present else f"{cfg.model} not in {models}")
    try:
        sh = client.show()
        env["capabilities"] = sh.get("capabilities")
        env["quantization"] = (sh.get("details") or {}).get("quantization_level")
        env["parameter_size"] = (sh.get("details") or {}).get("parameter_size")
    except Exception:
        pass
    w = client.warmup()
    env["load_s"] = w.load_s
    ps = client.ps()
    env["ps"] = ps
    _ok(t, "GPU/CPU split (/api/ps)", None,
        f"size {ps.get('size', 0) / 2**30:.2f} GiB, size_vram {ps.get('size_vram', 0) / 2**30:.2f} GiB -> "
        f"{100 * ps.get('gpu_pct', 0):.0f}% on GPU")
    # thinking disabled?
    r = client.chat([{"role": "user", "content": "Is 391 a prime number? Answer yes or no."}])
    leak = r.thinking_leak
    env["think_check"] = {"answer": r.content, "eval_count": r.eval_count, "thinking_chars": len(r.thinking)}
    _ok(t, "think:false effective", not leak, f"answer={r.content!r} eval_count={r.eval_count} thinking={len(r.thinking)} chars")
    env["think_disabled_ok"] = not leak
    # tool round trip
    store = Store(cfg, rt.tok, "selfcheck")

    class _S:  # minimal segment for the store
        id, kind, tokens, content = "a1b2c3d4e5f6", "doc", 60, ("Archive note: the peak throughput measured for "
                                                               "project Quillmar was 48,213 requests per second.")
    store.put(_S, 1)
    # hand-written stub without the answer in `keys:` so the model must expand to answer
    stub = (f'[ref:{_S.id} | doc | 60t | "Archive note about project Quillmar" | keys: Quillmar | '
            f'use expand_context("{_S.id}") to read]')
    msgs = [{"role": "system", "content": "Answer briefly.\n\n" + SIEVE_LINE},
            {"role": "user", "content": "Archived context: " + stub},
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": "What was the peak throughput measured for project Quillmar?"
                                        + TAIL_HINT.format(refs=f"ref:{_S.id} (keys: Quillmar)")}]
    out = client.run_with_tools(msgs, [EXPAND_TOOL], lambda sid: store.expand(sid), cfg.max_tool_rounds)
    tool_ok = out["tool_rounds"] > 0 and any(x["seg"] == _S.id for x in out["expansions"])
    env["tool_call_supported"] = tool_ok
    env["tool_check"] = {"tool_rounds": out["tool_rounds"], "answer": out["answer"], "expansions": out["expansions"]}
    _ok(t, "tool-call round trip", tool_ok, f"rounds={out['tool_rounds']} answer={out['answer'][:60]!r}"
        + ("" if tool_ok else " -> tool_mode will default to prefetch"))
    # token calibration (cold requests)
    sys_m = {"role": "system", "content": "You are a calibration helper. Reply with one word."}
    u1 = {"role": "user", "content": "The quick brown fox jumps over the lazy dog near the riverbank today."}
    a1 = {"role": "assistant", "content": "Noted."}
    u2 = {"role": "user", "content": "Please confirm you received the previous sentence."}
    client.flush_prefix()
    r2 = client.chat([sys_m, u1], options={"num_predict": 1})
    client.flush_prefix()
    r4 = client.chat([sys_m, u1, a1, u2], options={"num_predict": 1})
    c2 = sum(rt.tok.count(m["content"]) for m in (sys_m, u1))
    c4 = c2 + rt.tok.count(a1["content"]) + rt.tok.count(u2["content"])
    per_msg = round((r4.prompt_eval_count - r2.prompt_eval_count - (c4 - c2)) / 2)
    per_req = r2.prompt_eval_count - c2 - 2 * per_msg
    env["token_calibration"] = {"tokenizer": cfg.tokenizer_model, "per_message": per_msg, "per_request": per_req,
                                "pec2": r2.prompt_eval_count, "pec4": r4.prompt_eval_count, "approx": rt.tok.approx}
    if not client.is_mock and not rt.tok.approx:
        rt.tok.per_message, rt.tok.per_request = per_msg, per_req
    _ok(t, "tokenizer", not rt.tok.approx, f"{cfg.tokenizer_model} ({'APPROX chars/3.6' if rt.tok.approx else 'exact'}); "
        f"template overhead per_message={per_msg} per_request={per_req}")
    # prefill / decode speed (cold)
    g = Gen("calib")
    long_text = g.filler(1800)
    client.flush_prefix()
    rl = client.chat([{"role": "user", "content": long_text + "\n\nSummarize the above in two sentences."}],
                     options={"num_predict": 48})
    env["prefill_tps"] = rl.prompt_eval_count / rl.prompt_eval_s if rl.prompt_eval_s else None
    env["prefill_ms_per_token"] = 1000 * rl.prompt_eval_s / rl.prompt_eval_count if rl.prompt_eval_count else None
    env["decode_tps"] = rl.decode_tps
    env["calib_long_tokens_tokenizer"] = rt.tok.count_messages([{"role": "user", "content": long_text + "\n\nSummarize the above in two sentences."}])
    env["calib_long_prompt_eval_count"] = rl.prompt_eval_count
    _ok(t, "speed (cold)", None, f"prefill {env['prefill_tps'] or 0:.0f} tok/s, decode {rl.decode_tps:.1f} tok/s; "
        f"tokenizer {env['calib_long_tokens_tokenizer']} vs Ollama {rl.prompt_eval_count} tokens")
    s = rt.comp.compress(("The quarterly review of the Kalvoro platform showed latency improved by 23 percent, and "
                          "the team led by Doran Psatik will ship version 4.1 on 12 March 2027. ") * 2, 0.5)
    _ok(t, "LLMLingua-2", not rt.comp.fallback, f"{rt.comp.version}: {s['text'][:70]!r}" if not rt.comp.fallback
        else f"FALLBACK ({rt.comp.load_error})")
    env["compressor"] = rt.comp.version
    env["compressor_fallback"] = rt.comp.fallback
    _ok(t, "query similarity", rt.sim.method == "minilm", f"{rt.sim.method}" + (f" ({rt.sim.load_error})" if rt.sim.load_error else ""))
    env["similarity"] = rt.sim.method
    env["vram_source"] = w.vram_source
    console.print(t)
    if not client.is_mock:
        console.print("[dim]Recommended server env: OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 "
                      "OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1. Observed: "
                      + ", ".join(f"{k}={v}" for k, v in env['server_env'].items() if v) + "[/]")
    return env


def fit_cold_prefill(client, rt: Runtime, sizes=(250, 500, 1000, 1800, 2600, 3400)) -> Dict[str, Any]:
    """Cold prefill time vs prompt length, from novel prompts (a nonce defeats the server's RAM
    prompt cache). Fit t(n) = a*n + b*n^2 (least squares, b >= 0)."""
    import numpy as np
    nonce = "%08x" % random.SystemRandom().getrandbits(32)
    pts = []
    for n in sizes:
        g = Gen(f"cold-{nonce}-{n}")
        txt = f"[{nonce}-{n}] " + g.filler(n)
        client.flush_prefix()
        r = client.chat([{"role": "user", "content": txt + "\nReply OK."}], options={"num_predict": 1})
        pts.append((rt.tok.count_messages([{"role": "user", "content": txt + "\nReply OK."}]), r.prompt_eval_s,
                    r.prompt_eval_count))
    X = np.array([[p[0], p[0] ** 2] for p in pts], float)
    y = np.array([p[1] for p in pts], float)
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    if b < 0:
        a, b = float((X[:, 0] @ y) / (X[:, 0] @ X[:, 0])), 0.0
    return {"a": float(a), "b": float(b), "points": pts, "nonce": nonce}


def predicted_cold_s(n: int, fit: Optional[Dict[str, Any]]) -> Optional[float]:
    if not fit:
        return None
    return fit["a"] * n + fit["b"] * n * n


def cachetest(cfg, client, rt: Runtime) -> Dict[str, Any]:
    """Prefix reuse is judged from prefill TIME: on Ollama's llama-server backend prompt_eval_count
    reports the full prompt even when the KV prefix is reused (observed on this machine)."""
    fit = fit_cold_prefill(client, rt)
    nonce = "%08x" % random.SystemRandom().getrandbits(32)
    g = Gen(f"cachetest-{nonce}")
    P = f"[session {nonce}] " + g.filler(1800)
    sysm = {"role": "system", "content": "Answer in one short sentence."}
    q1 = "In one sentence, what is the general mood of the notes above?"
    q2 = "In one sentence, name one process mentioned in the notes above."

    def pair(p1: str, p2: str, label: str) -> Dict[str, Any]:
        client.flush_prefix()
        m1 = [sysm, {"role": "user", "content": p1}, {"role": "assistant", "content": "I have read it."},
              {"role": "user", "content": q1}]
        m2 = [sysm, {"role": "user", "content": p2}, {"role": "assistant", "content": "I have read it."},
              {"role": "user", "content": q2}]
        a = client.chat(m1)
        b = client.chat(m2)
        n2 = rt.tok.count_messages(m2)
        cold2 = predicted_cold_s(n2, fit)
        return {"label": label, "first_prompt_eval_count": a.prompt_eval_count, "first_prompt_eval_s": a.prompt_eval_s,
                "first_predicted_cold_s": predicted_cold_s(rt.tok.count_messages(m1), fit),
                "second_prompt_eval_count": b.prompt_eval_count, "second_prompt_eval_s": b.prompt_eval_s,
                "second_total_tokens": n2, "second_predicted_cold_s": cold2,
                "reuse_pct_count": max(0.0, 1 - b.prompt_eval_count / max(1, n2)),
                "reuse_pct_timing": max(0.0, min(1.0, 1 - b.prompt_eval_s / cold2)) if cold2 else None,
                "ttft1": a.ttft_s, "ttft2": b.ttft_s}

    res = [pair(P, P, "(a) same prefix P, new question")]
    words = P.split(" ")
    k = len(words) // 10
    res.append(pair("[b] " + P, "[b] " + " ".join(words[:k] + ["zebra"] + words[k + 1:]),
                    "(b) one word changed at ~10% depth"))
    Pc = "[c] " + P
    c5 = rt.comp.compress(Pc, 0.5)["text"]
    c7 = rt.comp.compress(Pc, 0.7)["text"]
    res.append(pair(c5, c7, "(c1) re-compressed each turn (0.5 then 0.7)"))
    Pd = "[d] " + P
    d5 = rt.comp.compress(Pd, 0.5)["text"]
    res.append(pair(d5, rt.comp.compress(Pd, 0.5)["text"], "(c2) memoized compressed form (identical)"))
    by_count = res[0]["second_prompt_eval_count"] < 0.5 * res[0]["first_prompt_eval_count"]
    by_time = (res[0]["reuse_pct_timing"] or 0) > 0.5
    t = Table(title="cachetest: KV prefix reuse (timing-derived)", header_style="bold")
    for c in ("experiment", "2nd tokens", "Ollama count", "2nd prefill s", "cold pred s", "reuse % (time)",
              "reuse % (count)", "TTFT1", "TTFT2"):
        t.add_column(c)
    for r in res:
        t.add_row(r["label"], str(r["second_total_tokens"]), str(r["second_prompt_eval_count"]),
                  f"{r['second_prompt_eval_s']:.2f}", f"{r['second_predicted_cold_s'] or 0:.2f}",
                  f"{100 * (r['reuse_pct_timing'] or 0):.0f}", f"{100 * r['reuse_pct_count']:.0f}",
                  f"{r['ttft1']:.2f}", f"{r['ttft2']:.2f}")
    console.print(t)
    console.print(f"cold prefill fit: t(n) = {fit['a'] * 1000:.3f} ms*n + {fit['b'] * 1e6:.4f} us*n^2")
    if by_time:
        console.print("[green bold]Prefix reuse OBSERVED[/] (prefill time collapses on a shared prefix)."
                      + ("" if by_count else " Note: Ollama's prompt_eval_count does NOT drop on reuse here, so "
                                             "Sieve reports timing-derived reuse."))
    else:
        console.print(Panel("[bold red]Prefix reuse NOT observed on this model/server.[/] Prefix-reuse metrics will be "
                            "reported as 'not observed'; everything else keeps running.", style="red"))
    return {"cachetest": res, "prefix_reuse_supported": by_time, "prefix_reuse_count_drops": by_count,
            "prefix_reuse_method": "timing" if not by_count else "count", "cold_prefill_fit": fit,
            "memo_restores_reuse": (res[3]["reuse_pct_timing"] or 0) > (res[2]["reuse_pct_timing"] or 0) + 0.3}


# ---------------- demos ----------------

def lane_demo_case(lane: str) -> Dict[str, Any]:
    g = Gen(f"demo-{lane}")
    if lane == "compress":
        c = Conv("demo:compress", "C", g, "mid", 2400)
        c.system()
        f1, f2 = fact_number(g), fact_name(g)
        d = c.user("Please read this project report carefully:\n\n" + g.embed(g.filler(2000), facts_of(f1) + facts_of(f2)),
                   kind="doc")
        c.assistant("I have read the report.")
        c.ask(f1, [d])
    elif lane == "cache":
        c = Conv("demo:cache", "D", g, "mid", 2600)
        f1, f2, f3 = fact_number(g), fact_name(g), fact_number(g)
        sys_mid = c.system("Operations handbook:\n\n" + g.embed(g.filler(900), facts_of(f3)))
        doc = g.embed(g.filler(700), facts_of(f1) + facts_of(f2))
        d1 = c.user("Reference document:\n\n" + doc, kind="doc")
        c.assistant("Got it.")
        c.ask(f1, [d1])
        d2 = c.user("Reference document:\n\n" + doc, kind="doc")
        c.assistant("Got it.")
        c.ask(f2, [d1, d2])
        c.ask(f3, [sys_mid])
    else:
        c = Conv("demo:externalize", "B", g, "long", 5200)
        c.system()
        old = fact_number(g)
        o = c.user("Notes from the kickoff meeting:\n\n" + g.embed(g.filler(1100), facts_of(old)), kind="doc")
        c.assistant("Thanks, I have the kickoff notes. " + g.paragraph(3))
        for _ in range(8):
            c.chat_exchange(380)
        rec = fact_name(g)
        r = c.user("Quick status update: " + g.paragraph(2) + " " + facts_of(rec)[0])
        c.assistant("Noted.")
        c.ask(rec, [r])            # should NOT need expansion
        c.ask(old, [o])            # needs the stubbed kickoff notes -> expansion expected
    out = c.out()
    out["seed"] = 0
    return out


def run_case(cfg, client, rt: Runtime, case: Dict[str, Any], conditions: List[str], run_dir: Optional[Path] = None,
             show: bool = True, env: Optional[Dict] = None) -> Dict[str, List[Dict[str, Any]]]:
    from sieve.conditions import plan_budgets
    budgets = plan_budgets([case], rt)
    results: Dict[str, List[Dict[str, Any]]] = {}
    conds = {}
    for cname in conditions:  # one pass per condition, cold at the first evaluation point
        cond = make_condition(cname, rt, budgets)
        cond.start_case(case)
        conds[cname] = cond
        results[cname] = []
        client.unload()  # clears the server's RAM prompt cache so the first point is truly cold
        client.warmup(cond.num_ctx)
        for ep in case["eval_points"]:
            qi = ep["question_index"]
            hist, q = case["messages"][:qi], case["messages"][qi]
            cold = ep["eval_point"] == 0
            if cold:
                client.flush_prefix(cond.num_ctx)
            plan = cond.build(ep["eval_point"], hist, q)
            store = getattr(getattr(cond, "session", None), "store", None)
            out = call_model(client, cond, plan, store, int(q.get("turn", 0)))
            raw = raw_messages(hist, q)
            rec = make_record("demo", cond, case, ep, 0, cold, plan, raw, out, rt, client.ps(), env or {})
            rec["_plan"], rec["_raw"], rec["_out"] = plan, raw, out
            results[cname].append(rec)
            if run_dir:
                save_artifacts(run_dir, rec, raw, plan, out)
    if show:
        for k, ep in enumerate(case["eval_points"]):
            console.rule(f"[bold]{case['case_id']} — evaluation point {k} ({ep['question_type']})")
            for cname in conditions:
                r = results[cname][k]
                if r["_plan"].get("segments"):
                    console.print(segment_table(r["_plan"], f"{cname}: segment decisions"))
                    console.print(Panel(waterfall(r["_plan"]), title=f"{cname}: token waterfall (before ░ / after █)"))
                    console.print(Panel(prompt_diff(r["_raw"], r["_plan"]["messages"]),
                                        title=f"{cname}: prompt diff (red = removed, orange = stub/back-ref, cyan = added)"))
                    ver = [(s.key, r["_plan"]["emitted"][s.key].verifier) for s in r["_plan"]["segments"]
                           if r["_plan"]["emitted"][s.key].verifier]
                    if ver:
                        vt = Table(title=f"{cname}: verifier", header_style="bold")
                        for col in ("seg", "final ratio", "recall", "numeric recall", "retries", "reinflated", "lost on 1st try"):
                            vt.add_column(col)
                        for key, v in ver:
                            vt.add_row(key, f"{r['_plan']['emitted'][key].ratio:.1f}", f"{v['recall']:.2f}",
                                       f"{v['numeric_recall']:.2f}", str(v["retries"]), str(v["reinflated"]),
                                       ", ".join(v["lost"][:6]))
                        console.print(vt)
                    if r["expansions"] or r["prefetch_count"]:
                        console.print(f"[dark_orange]{cname}: expansions {r['expansions']} prefetched "
                                      f"{r['_plan']['prefetched']}[/]")
                    else:
                        console.print(f"[dim]{cname}: no expansion[/]")
            console.print(result_panel(ep["question"], ep["gold"], [results[c][k] for c in conditions]))
        for cname in conditions:
            sess = getattr(conds[cname], "session", None)
            if sess:
                console.print(ledger_view(sess.ledger.history))
            console.print(f"{cname:>16} per-turn TTFT {sparkline([r['ttft_s'] for r in results[cname]])} "
                          + " ".join(f"{r['ttft_s']:.2f}s" for r in results[cname]))
    return results


def breakeven(cfg, client, rt: Runtime, n_cases: int = 8, ps=(0.0, 0.25, 0.5, 0.75, 1.0)) -> Dict[str, Any]:
    """Sweep the probability that a question needs the externalized document; measure net time
    (Sieve overhead + model time) of sieve_full vs baseline, all cold."""
    rows = []
    for p in ps:
        rng = random.Random(int(p * 100))
        need = set(rng.sample(range(n_cases), round(p * n_cases)))
        for i in range(n_cases):
            g = Gen(f"breakeven-{i}")
            c = Conv(f"BE:{i}", "B", g, "long", 5000)
            c.system()
            old = fact_number(g)
            o = c.user("Archive notes:\n\n" + g.embed(g.filler(1500), facts_of(old)), kind="doc")
            c.assistant("Archived. " + g.paragraph(2))
            for _ in range(7):
                c.chat_exchange(300)
            rec = fact_name(g)
            r = c.user("Status: " + facts_of(rec)[0] + " " + g.paragraph(1))
            c.assistant("Noted.")
            if i in need:
                c.ask(old, [o])
            else:
                c.ask(rec, [r])
            case = c.out()
            case["seed"] = 0
            res = run_case(cfg, client, rt, case, ["baseline", "sieve_full"], show=False)
            b, s = res["baseline"][0], res["sieve_full"][0]
            ov = sum(s["sieve_overhead_ms"].values()) / 1000
            rows.append({"p": p, "case": i, "needs_doc": i in need, "baseline_s": b["total_s"], "sieve_s": s["total_s"] + ov,
                         "sieve_overhead_s": ov, "baseline_correct": b["correct"], "sieve_correct": s["correct"],
                         "tool_rounds": s["tool_rounds"], "prefetch": s["prefetch_count"],
                         "baseline_tokens": b["tokens_final_total"], "sieve_tokens": s["tokens_final_total"]})
    summary = []
    for p in ps:
        rs = [r for r in rows if r["p"] == p]
        summary.append({"p": p, "baseline_s": sum(r["baseline_s"] for r in rs) / len(rs),
                        "sieve_s": sum(r["sieve_s"] for r in rs) / len(rs),
                        "baseline_acc": sum(r["baseline_correct"] for r in rs) / len(rs),
                        "sieve_acc": sum(r["sieve_correct"] for r in rs) / len(rs),
                        "expand_rate": sum(r["tool_rounds"] > 0 for r in rs) / len(rs)})
    be = None
    for a, b in zip(summary, summary[1:]):
        da, db = a["sieve_s"] - a["baseline_s"], b["sieve_s"] - b["baseline_s"]
        if da <= 0 < db or da >= 0 > db:
            be = a["p"] + (b["p"] - a["p"]) * (-da) / (db - da)
            break
    t = Table(title="Externalize break-even (cold, net time = overhead + model)", header_style="bold")
    for c in ("P(question needs stubbed doc)", "baseline s", "sieve s", "Δ s", "baseline acc", "sieve acc", "expand rate"):
        t.add_column(c)
    for s in summary:
        t.add_row(f"{s['p']:.2f}", f"{s['baseline_s']:.2f}", f"{s['sieve_s']:.2f}", f"{s['sieve_s'] - s['baseline_s']:+.2f}",
                  f"{s['baseline_acc']:.2f}", f"{s['sieve_acc']:.2f}", f"{s['expand_rate']:.2f}")
    console.print(t)
    console.print(f"Empirical break-even p: {be:.2f}" if be is not None else
                  "No crossing in the swept range (Sieve " + ("always faster" if all(s['sieve_s'] < s['baseline_s'] for s in summary)
                                                              else "never faster") + ").")
    return {"rows": rows, "summary": summary, "break_even_p": be}

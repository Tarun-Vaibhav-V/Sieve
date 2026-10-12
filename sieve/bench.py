"""Benchmark protocol: one full pass per condition (never interleaved), cold/warm marking,
resumable JSONL logging, per-call artifacts."""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from sieve.conditions import Condition, Runtime, make_condition, plan_budgets, raw_messages
from sieve.metrics import JsonlLogger, completed_conversations, load_records
from sieve.verifier import atom_recall_in
from sieve.workloads.scoring import score_answer

console = Console()


def seg_needed(seg, ep) -> bool:
    return any(f and f in seg.content for f in ep.get("facts") or [])


def trace_of(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for s in plan.get("segments") or []:
        d = plan["decisions"].get(s.key)
        e = plan["emitted"].get(s.key)
        out.append({"key": s.key, "id": s.id, "role": s.role, "kind": s.kind, "turn": s.turn_index,
                    "tokens_before": s.tokens, "tokens_after": 0 if e.dropped else e.tokens,
                    "features": {k: v for k, v in s.features.items() if not k.startswith("_")},
                    "dup_of": s.dup_of, "dup_type": s.dup_type,
                    "decision": {"source": d.source, "lane": d.lane, "ratio": d.ratio, "probs": d.probs,
                                 "confidence": d.confidence, "expected_cost": d.expected_cost,
                                 "reasons": d.reasons, "backref": d.backref},
                    "emitted": {"lane": e.lane, "ratio": e.ratio, "verifier": e.verifier, "dropped": e.dropped,
                                "text": e.text if len(e.text) < 600 else e.text[:600] + " ..."}})
    return out


def text_of(msgs) -> str:
    return "\n".join((m.get("content") or "") for m in msgs)


def call_model(client, cond: Condition, plan: Dict[str, Any], store=None, turn=None) -> Dict[str, Any]:
    cfg = cond.cfg
    if plan.get("tools") and store is not None:
        out = client.run_with_tools(plan["messages"], plan["tools"],
                                    lambda sid: store.expand(sid, None, turn), cfg.max_tool_rounds, num_ctx=cond.num_ctx)
    else:
        r = client.chat(plan["messages"], num_ctx=cond.num_ctx)
        out = {"answer": r.content, "rounds": [r], "tool_rounds": 0, "expansions": [], "ttft_s": r.ttft_s,
               "total_s": r.total_s, "prompt_eval_count": r.prompt_eval_count,
               "first_prompt_eval_count": r.prompt_eval_count, "prompt_eval_s": r.prompt_eval_s,
               "eval_count": r.eval_count, "load_s": r.load_s, "decode_tps": r.decode_tps,
               "vram_peak_mb": r.vram_peak_mb, "vram_source": r.vram_source, "thinking_leak": r.thinking_leak,
               "unanswered_tool_call": False, "messages": plan["messages"]}
    return out


def make_record(run_id, cond, case, ep, rep, cold, plan, raw_msgs, out, rt, client_ps, env) -> Dict[str, Any]:
    tok = rt.tok
    raw_n = tok.count_messages(raw_msgs)
    final_n = plan["prompt_tokens"]
    pec_first = out["first_prompt_eval_count"]
    first_pe_s = out["rounds"][0].prompt_eval_s if out.get("rounds") else out["prompt_eval_s"]
    fit = env.get("cold_prefill_fit")
    reuse_count = max(0.0, min(1.0, 1 - pec_first / final_n)) if final_n else 0.0
    cold_pred = (fit["a"] * final_n + fit["b"] * final_n ** 2) if fit else None
    reuse_time = max(0.0, min(1.0, 1 - first_pe_s / cold_pred)) if cold_pred else None
    method = env.get("prefix_reuse_method", "count")
    reuse = reuse_time if (method == "timing" and reuse_time is not None) else reuse_count
    segs = plan.get("segments") or []
    needed_ext = set()
    for s in segs:
        e = plan["emitted"].get(s.key)
        if e and e.lane == "externalize" and not e.dropped and seg_needed(s, ep):
            needed_ext.add(s.id)
    groups = plan.get("groups") or {}
    seg_by_id = {s.id: s for s in segs}

    def members(ref):
        return [seg_by_id[m] for m in groups.get(ref, [ref]) if m in seg_by_id]

    covered = set()
    for p in plan.get("prefetched") or []:
        covered |= {m.id for m in members(p["seg"])}
    expansions = []
    for x in out.get("expansions") or []:
        ms = members(x["seg"])
        expansions.append({"seg": x["seg"], "tokens": sum(m.tokens for m in ms),
                           "needed": any(seg_needed(m, ep) for m in ms), "valid": bool(ms)})
        covered |= {m.id for m in ms}
    missed = len(needed_ext - covered)
    wasted = sum(1 for x in expansions if not x["needed"]) + \
        sum(1 for p in plan.get("prefetched") or [] if not any(seg_needed(m, ep) for m in members(p["seg"])))
    raw_text = text_of(raw_msgs)
    atoms = [a for a in ep["critical_atoms"] if a in raw_text]
    final_text = text_of(plan["messages"])
    tool_text = final_text + "\n" + "\n".join(m.get("content") or "" for m in out.get("messages", [])[len(plan["messages"]):])
    correct = score_answer(out["answer"], ep)
    t = plan.get("timers") or {}
    rec = {
        "run_id": run_id, "condition": cond.name, "family": case["family"], "case_id": case["case_id"],
        "eval_point": ep["eval_point"], "repeat": rep, "cold": cold, "question_type": ep["question_type"],
        "length_bucket": case["length_bucket"], "num_ctx": cond.num_ctx or cond.cfg.num_ctx,
        "tokens_raw_total": raw_n, "tokens_final_total": final_n, "tokens_saved": raw_n - final_n,
        "compression_ratio": round(final_n / raw_n, 4) if raw_n else 1.0,
        "canonicalizer_tokens_saved": plan.get("canonicalizer_tokens_saved", 0),
        "context_utilization": round(final_n / (cond.num_ctx or cond.cfg.num_ctx), 4),
        "output_headroom": (cond.num_ctx or cond.cfg.num_ctx) - final_n,
        "baseline_overflow": raw_n > cond.cfg.num_ctx, "hard_truncation": plan.get("hard_truncation", False),
        "prompt_eval_count": out["prompt_eval_count"], "first_prompt_eval_count": pec_first,
        "prefix_reuse_pct": round(reuse, 4), "prefix_reuse_method": method,
        "prefix_reuse_pct_count": round(reuse_count, 4),
        "prefix_reuse_pct_timing": None if reuse_time is None else round(reuse_time, 4),
        "first_prompt_eval_s": first_pe_s, "predicted_cold_prefill_s": cold_pred,
        "tokens_processed_est": int(round(final_n * (1 - reuse))) if method == "timing" else pec_first,
        "prompt_eval_s": out["prompt_eval_s"], "load_s": out["load_s"],
        "ttft_s": out["ttft_s"], "total_s": out["total_s"], "decode_tps": out["decode_tps"],
        "eval_count": out["eval_count"],
        "sieve_overhead_ms": {k: round(float(t.get(k, 0.0)), 2) for k in ("segment", "classify", "compress", "verify", "assemble")},
        "gpu_pct": client_ps.get("gpu_pct", 0.0), "vram_peak_mb": out.get("vram_peak_mb"),
        "vram_source": out.get("vram_source"), "lanes": plan.get("lanes"),
        "epoch_id": plan.get("epoch_id", 0), "epoch_reason": plan.get("epoch_reason"),
        "rebuild_tokens": plan.get("rebuild_tokens", final_n),
        "memo_hits": plan.get("memo_hits", 0), "reinflations": plan.get("reinflations", 0),
        "tool_rounds": out["tool_rounds"], "expansions": expansions, "missed_expansions": missed,
        "wasted_expansions": wasted, "prefetch_count": len(plan.get("prefetched") or []),
        "needed_externalized": len(needed_ext), "unanswered_tool_call": out.get("unanswered_tool_call", False),
        "answer": out["answer"], "correct": bool(correct),
        "atom_recall": round(atom_recall_in(final_text, atoms), 4),
        "atom_recall_with_tools": round(atom_recall_in(tool_text, atoms), 4),
        "thinking_leak": out["thinking_leak"], "approx_tokens": rt.tok.approx,
        # health: decode far below the selfcheck speed means the machine was swapping (latency invalid)
        "perf_degraded": bool(env.get("decode_tps") and out["decode_tps"] and out["eval_count"] >= 4
                              and out["decode_tps"] < 0.5 * env["decode_tps"]),
        "compressor_fallback": plan.get("compressor_fallback", False), "mock": bool(env.get("mock")),
        "uniform_rate": plan.get("uniform_rate"), "dropped_messages": plan.get("dropped_messages"),
        "rounds": [dict(r.metrics(), tool_calls=r.tool_calls) for r in out["rounds"]],
    }
    return rec


def save_artifacts(run_dir: Path, rec, raw_msgs, plan, out):
    d = run_dir / rec["case_id"].replace(":", "_") / rec["condition"] / f"ep{rec['eval_point']}_r{rec['repeat']}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "raw_prompt.json").write_text(json.dumps(raw_msgs, indent=1))
    (d / "final_prompt.json").write_text(json.dumps({"messages": plan["messages"], "tools": plan.get("tools")}, indent=1))
    (d / "response.txt").write_text(out["answer"] + "\n")
    tr = {"metrics": {k: v for k, v in rec.items() if k != "answer"}, "segments": trace_of(plan),
          "alloc": plan.get("alloc"), "tool_messages": out.get("messages", [])[len(plan["messages"]):],
          "prefetched": plan.get("prefetched")}
    (d / "trace.json").write_text(json.dumps(tr, indent=1, default=str))
    from sieve.viz import prompt_diff_text
    (d / "diff.txt").write_text(prompt_diff_text(raw_msgs, plan["messages"]))


def estimate(cases, conditions, repeats, env) -> Dict[str, float]:
    n_eval = sum(len(c["eval_points"]) for c in cases)
    calls = n_eval * len(conditions) * repeats
    flushes = len(cases) * len(conditions) * repeats
    dt = env.get("decode_tps") or 25.0
    fit = env.get("cold_prefill_fit") or {"a": 1 / (env.get("prefill_tps") or 400.0), "b": 0.0}
    n = min(3200.0, 0.8 * sum(c["est_tokens"] for c in cases) / max(1, len(cases)))
    cold_s = fit["a"] * n + fit["b"] * n * n
    frac_cold = flushes / max(1, calls)
    # cold points pay full prefill; warm points mostly reuse the prefix (observed ~85%)
    per_call = frac_cold * cold_s + (1 - frac_cold) * 0.3 * cold_s + 25 / dt + 0.6
    sec = calls * per_call + flushes * 0.6 + len(conditions) * repeats * 12
    return {"llm_calls": calls + flushes, "eval_calls": calls, "est_seconds": sec}


def run_bench(cfg, client, rt: Runtime, cases, conditions: List[str], repeats: int, run_dir: Path,
              run_id: str, env: Dict[str, Any], resume: bool = True, seed: int = 7) -> Path:
    log_path = run_dir / "results.jsonl"
    logger = JsonlLogger(log_path)
    plans_path = run_dir / "plans.json"
    if plans_path.exists():
        budgets = json.loads(plans_path.read_text())
    else:
        console.print("[bold]Planning[/] C4 budgets (no model calls) ...")
        t0 = time.perf_counter()
        budgets = plan_budgets(cases, rt)
        plans_path.write_text(json.dumps(budgets))
        console.print(f"  planned {len(cases)} cases in {time.perf_counter() - t0:.1f}s")
    done = completed_conversations(load_records(log_path), cases) if resume else set()
    est = estimate(cases, conditions, repeats, env)
    console.print(f"[bold]Estimate:[/] {est['llm_calls']} LLM calls (+ tool rounds), "
                  f"~{est['est_seconds'] / 60:.0f} min wall-clock{' (simulated)' if client.is_mock else ''}. "
                  f"{len(done)} conversation-passes already complete (resume).")
    total = sum(len(c["eval_points"]) for c in cases) * len(conditions) * repeats
    with Progress(TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                  console=console) as prog:
        task = prog.add_task("bench", total=total)
        for rep in range(repeats):
            order = list(conditions)
            random.Random(seed * 1000 + rep).shuffle(order)
            for cname in order:
                todo = [c for c in cases if (c["case_id"], cname, rep) not in done]
                prog.update(task, advance=sum(len(c["eval_points"]) for c in cases if c not in todo))
                if not todo:
                    continue
                cond = make_condition(cname, rt, budgets)
                prog.update(task, description=f"r{rep} {cname:<24}")
                client.unload()
                client.warmup(cond.num_ctx)  # excluded from stats
                for case in todo:
                    cond.start_case(case)
                    for ep in case["eval_points"]:
                        qi = ep["question_index"]
                        history, question = case["messages"][:qi], case["messages"][qi]
                        cold = ep["eval_point"] == 0
                        if cold:
                            client.flush_prefix(cond.num_ctx)
                        plan = cond.build(ep["eval_point"], history, question)
                        store = getattr(getattr(cond, "session", None), "store", None)
                        out = call_model(client, cond, plan, store, int(question.get("turn", 0)))
                        try:
                            ps = client.ps()
                        except Exception:
                            ps = {}
                        raw = raw_messages(history, question)
                        rec = make_record(run_id, cond, case, ep, rep, cold, plan, raw, out, rt, ps, env)
                        logger.write(rec)
                        save_artifacts(run_dir, rec, raw, plan, out)
                        prog.advance(task)
    return log_path

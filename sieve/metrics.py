"""JSONL logging, aggregation, paired bootstrap CIs, and data-derived claims.md."""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

SCHEMA_KEYS = ["run_id", "condition", "family", "case_id", "eval_point", "repeat", "cold", "tokens_raw_total",
               "tokens_final_total", "tokens_saved", "compression_ratio", "canonicalizer_tokens_saved",
               "context_utilization", "output_headroom", "baseline_overflow", "hard_truncation",
               "prompt_eval_count", "prefix_reuse_pct", "prompt_eval_s", "load_s", "ttft_s", "total_s",
               "decode_tps", "eval_count", "sieve_overhead_ms", "gpu_pct", "vram_peak_mb", "lanes", "epoch_id",
               "rebuild_tokens", "memo_hits", "reinflations", "tool_rounds", "expansions", "missed_expansions",
               "wasted_expansions", "prefetch_count", "answer", "correct", "atom_recall", "thinking_leak",
               "approx_tokens", "compressor_fallback"]


def rkey(r: Dict[str, Any]) -> Tuple:
    return (r["case_id"], r["condition"], r["repeat"], r["eval_point"])


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, rec: Dict[str, Any]):
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def load_records(path: Path) -> List[Dict[str, Any]]:
    """Last write wins per (case, condition, repeat, eval_point)."""
    path = Path(path)
    if not path.exists():
        return []
    out: Dict[Tuple, Dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        out[rkey(r)] = r
    return list(out.values())


def completed_conversations(records: List[Dict[str, Any]], cases: List[Dict[str, Any]]) -> set:
    """(case_id, condition, repeat) whose every evaluation point is logged -> skipped on resume."""
    n_eval = {c["case_id"]: len(c["eval_points"]) for c in cases}
    seen = defaultdict(set)
    for r in records:
        seen[(r["case_id"], r["condition"], r["repeat"])].add(r["eval_point"])
    return {k for k, v in seen.items() if len(v) >= n_eval.get(k[0], 10**9)}


def validate(rec: Dict[str, Any]) -> List[str]:
    return [k for k in SCHEMA_KEYS if k not in rec]


# ---------------- statistics ----------------

def bootstrap_ci(x: Iterable[float], n: int = 2000, seed: int = 0, stat=np.mean, alpha: float = 0.05) -> Tuple[float, float, float]:
    x = np.asarray(list(x), dtype=float)
    if len(x) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    boots = np.array([stat(x[i]) for i in idx])
    return float(stat(x)), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def _q(xs, q):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.quantile(xs, q)) if xs else None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def summarize(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not recs:
        return {}
    ov = [sum((r.get("sieve_overhead_ms") or {}).get(k, 0) for k in ("segment", "classify", "compress", "verify", "assemble"))
          for r in recs]
    exp_rate = _mean([1.0 if r.get("tool_rounds", 0) > 0 else 0.0 for r in recs])
    out = {
        "n": len(recs),
        "accuracy": _mean([float(r["correct"]) for r in recs]),
        "token_reduction_pct": _mean([100 * (1 - r["tokens_final_total"] / r["tokens_raw_total"]) for r in recs if r["tokens_raw_total"]]),
        "tokens_final_mean": _mean([r["tokens_final_total"] for r in recs]),
        "tokens_raw_mean": _mean([r["tokens_raw_total"] for r in recs]),
        "ttft_median": _q([r["ttft_s"] for r in recs], .5), "ttft_mean": _mean([r["ttft_s"] for r in recs]),
        "ttft_p95": _q([r["ttft_s"] for r in recs], .95),
        "total_median": _q([r["total_s"] for r in recs], .5), "total_mean": _mean([r["total_s"] for r in recs]),
        "total_p95": _q([r["total_s"] for r in recs], .95),
        "prompt_eval_count_mean": _mean([r["prompt_eval_count"] for r in recs]),
        "tokens_processed_mean": _mean([r.get("tokens_processed_est", r["prompt_eval_count"]) for r in recs]),
        "prefill_s_median": _q([r.get("first_prompt_eval_s") for r in recs], .5),
        "prefix_reuse_pct_mean": _mean([r["prefix_reuse_pct"] for r in recs]),
        "atom_recall_mean": _mean([r["atom_recall"] for r in recs]),
        "context_utilization_mean": _mean([r["context_utilization"] for r in recs]),
        "output_headroom_mean": _mean([r["output_headroom"] for r in recs]),
        "overflow_rate": _mean([float(r["tokens_final_total"] > r.get("num_ctx", 4096)) for r in recs]),
        "expand_rate": exp_rate,
        "missed_expansions": sum(r.get("missed_expansions", 0) for r in recs),
        "wasted_expansions": sum(r.get("wasted_expansions", 0) for r in recs),
        "prefetch_count": sum(r.get("prefetch_count", 0) for r in recs),
        "sieve_overhead_ms_mean": _mean(ov), "sieve_overhead_ms_p95": _q(ov, .95),
        "net_latency_mean": _mean([r["total_s"] + o / 1000 for r, o in zip(recs, ov)]),
        "vram_peak_mb_max": max([r.get("vram_peak_mb") or 0 for r in recs]) or None,
        "decode_tps_median": _q([r["decode_tps"] for r in recs if r.get("decode_tps")], .5),
        "hard_truncations": sum(bool(r.get("hard_truncation")) for r in recs),
        "reinflations": sum(r.get("reinflations", 0) for r in recs),
        "thinking_leaks": sum(bool(r.get("thinking_leak")) for r in recs),
    }
    return out


def cumulative_prefill(recs: List[Dict[str, Any]]) -> Dict[str, float]:
    """Mean over conversations of summed tokens actually processed (timing-derived estimate when
    Ollama's prompt_eval_count does not drop on reuse)."""
    by = defaultdict(lambda: defaultdict(int))
    for r in recs:
        by[r["condition"]][(r["case_id"], r["repeat"])] += r.get("tokens_processed_est", r["prompt_eval_count"])
    return {c: float(np.mean(list(v.values()))) for c, v in by.items()}


def paired(recs: List[Dict[str, Any]], cond: str, ref: str = "baseline", subset=None) -> Dict[str, Any]:
    idx = {rkey(r)[0:1] + rkey(r)[2:]: r for r in recs if r["condition"] == ref}
    pairs = []
    for r in recs:
        if r["condition"] != cond:
            continue
        b = idx.get(rkey(r)[0:1] + rkey(r)[2:])
        if b is None or (subset and not subset(b)):
            continue
        pairs.append((r, b))
    if not pairs:
        return {"n": 0}
    acc = [float(a["correct"]) - float(b["correct"]) for a, b in pairs]
    tok = [100 * (1 - a["tokens_final_total"] / b["tokens_final_total"]) for a, b in pairs if b["tokens_final_total"]]
    ttft = [a["ttft_s"] - b["ttft_s"] for a, b in pairs if a["ttft_s"] is not None and b["ttft_s"] is not None]
    pe = [a.get("tokens_processed_est", a["prompt_eval_count"]) - b.get("tokens_processed_est", b["prompt_eval_count"])
          for a, b in pairs]
    pf = [a["first_prompt_eval_s"] - b["first_prompt_eval_s"] for a, b in pairs if "first_prompt_eval_s" in a]
    tot = [a["total_s"] - b["total_s"] for a, b in pairs]
    return {"n": len(pairs), "acc_delta": bootstrap_ci(acc), "token_reduction_pct": bootstrap_ci(tok),
            "ttft_delta_s": bootstrap_ci(ttft, stat=np.median), "prompt_eval_delta": bootstrap_ci(pe),
            "total_delta_s": bootstrap_ci(tot, stat=np.median), "prefill_delta_s": bootstrap_ci(pf, stat=np.median)}


def aggregate(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    conds = sorted({r["condition"] for r in recs})
    fams = sorted({r["family"] for r in recs})
    out = {"by_condition": {}, "by_condition_cold": {}, "by_condition_warm": {}, "by_family": {}, "paired": {},
           "paired_cold": {}, "paired_overflow": {}, "cumulative_prefill": cumulative_prefill(recs),
           "by_qtype": {}, "by_bucket": {}}
    for c in conds:
        rc = [r for r in recs if r["condition"] == c]
        out["by_condition"][c] = summarize(rc)
        out["by_condition_cold"][c] = summarize([r for r in rc if r["cold"]])
        out["by_condition_warm"][c] = summarize([r for r in rc if not r["cold"]])
        out["by_family"][c] = {f: summarize([r for r in rc if r["family"] == f]) for f in fams}
        out["by_qtype"][c] = {q: summarize([r for r in rc if r.get("question_type") == q])
                              for q in sorted({r.get("question_type") for r in rc})}
        out["by_bucket"][c] = {b: summarize([r for r in rc if r.get("length_bucket") == b])
                               for b in ("short", "mid", "long")}
        if c != "baseline":
            out["paired"][c] = paired(recs, c)
            out["paired_cold"][c] = paired(recs, c, subset=lambda b: b["cold"])
            out["paired_overflow"][c] = paired(recs, c, subset=lambda b: b["baseline_overflow"])
    return out


def _fmt_ci(t, unit="", pct=False, digits=1):
    m, lo, hi = t
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return "n/a"
    k = 100 if pct else 1
    return f"{m * k:+.{digits}f}{unit} (95% CI {lo * k:+.{digits}f} to {hi * k:+.{digits}f})"


def write_claims(agg: Dict[str, Any], env: Dict[str, Any], path: Path, mock: bool) -> str:
    L = ["# claims.md — statements derived only from this run's logs", ""]
    if mock:
        L += ["> **MOCK RUN.** All model behaviour and timings are simulated by `mock_client.py`. "
              "Nothing below is evidence about a real model.", ""]
    L += [f"- Model: `{env.get('model')}`; num_ctx {env.get('num_ctx')}; Ollama {env.get('ollama_version')}; "
          f"workloads are synthetic (invented entities).", ""]
    reuse = env.get("prefix_reuse_supported")
    L.append(f"- KV prefix reuse on this model/server: **{'observed' if reuse else 'NOT observed'}** "
             f"(cachetest). " + ("" if reuse else "Prefix-reuse metrics are reported as 'not observed'."))
    L.append("")
    bc = agg["by_condition"]
    supported, unsupported = [], []
    for c, p in agg["paired"].items():
        if not p.get("n"):
            continue
        acc, tok = p["acc_delta"], p["token_reduction_pct"]
        L.append(f"## {c} vs baseline (paired, n={p['n']})")
        L.append(f"- Accuracy delta: {_fmt_ci(acc, ' pp', pct=True)}")
        L.append(f"- Prompt-token reduction: {_fmt_ci(tok, '%')}")
        L.append(f"- Median TTFT delta: {_fmt_ci(p['ttft_delta_s'], ' s', digits=2)}")
        L.append(f"- Mean prompt tokens processed delta: {_fmt_ci(p['prompt_eval_delta'], ' tok', digits=0)}")
        pc = agg["paired_cold"].get(c, {})
        if pc.get("n"):
            L.append(f"- Cold points only (n={pc['n']}): median TTFT delta {_fmt_ci(pc['ttft_delta_s'], ' s', digits=2)}")
        po = agg["paired_overflow"].get(c, {})
        if po.get("n"):
            L.append(f"- Where baseline overflowed num_ctx (n={po['n']}): accuracy delta {_fmt_ci(po['acc_delta'], ' pp', pct=True)}")
        L.append("")
        if tok[1] > 0:
            supported.append(f"{c} reduces prompt tokens vs baseline (CI lower bound {tok[1]:.1f}%).")
        else:
            unsupported.append(f"{c} reduces prompt tokens (CI includes 0).")
        if acc[1] >= -0.05:
            supported.append(f"{c} accuracy is within -5 pp of baseline at the CI lower bound ({acc[1]*100:+.1f} pp).")
        else:
            unsupported.append(f"{c} preserves accuracy (CI lower bound {acc[1]*100:+.1f} pp is below -5 pp).")
        if p["ttft_delta_s"][2] < 0:
            supported.append(f"{c} has lower median TTFT than baseline (CI upper bound {p['ttft_delta_s'][2]:+.2f} s).")
        else:
            unsupported.append(f"{c} is faster than baseline in TTFT (CI upper bound {p['ttft_delta_s'][2]:+.2f} s >= 0).")
    if "sieve_full" in bc and "sieve_nomemo" in bc:
        cp = agg["cumulative_prefill"]
        L.append("## Cache-vs-compress (ledger) evidence")
        L.append(f"- Cumulative prompt tokens processed per conversation: sieve_full {cp.get('sieve_full', 0):.0f} vs "
                 f"sieve_nomemo {cp.get('sieve_nomemo', 0):.0f} vs baseline {cp.get('baseline', 0):.0f}.")
        L.append(f"- Mean prefix reuse: sieve_full {100 * (bc['sieve_full'].get('prefix_reuse_pct_mean') or 0):.1f}% vs "
                 f"sieve_nomemo {100 * (bc['sieve_nomemo'].get('prefix_reuse_pct_mean') or 0):.1f}%.")
        L.append("")
        if cp.get("sieve_full", 0) < cp.get("sieve_nomemo", 0):
            supported.append("The ledger reduces cumulative prefill tokens relative to re-routing every turn (C4 < C5).")
        else:
            unsupported.append("The ledger reduces cumulative prefill relative to C5.")
    L.append("## Supported by this run")
    L += [f"- {s}" for s in supported] or ["- (none)"]
    L.append("")
    L.append("## Not supported by this run")
    L += [f"- {s}" for s in unsupported] or ["- (none)"]
    L += ["- Any claim about real-world (non-synthetic) conversations.",
          "- Any claim about models other than the one listed above.",
          "- Classifier training accuracy on heuristic labels (circular; never cited)."]
    text = "\n".join(L) + "\n"
    Path(path).write_text(text)
    return text

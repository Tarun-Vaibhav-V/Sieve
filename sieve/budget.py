"""Token budget allocator: tighten lowest-priority segments first (lower ratio, then
externalize); never touch protected segments; hard truncation only as a last resort."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple


def context_report(prompt_tokens: int, num_ctx: int) -> Dict[str, Any]:
    return {"context_utilization": round(prompt_tokens / num_ctx, 4), "output_headroom": num_ctx - prompt_tokens}


def allocate(items: List[Dict[str, Any]], overhead: int, target: int, limit: int, cfg,
             emit: Callable[[Any, Any], Any], can_externalize: Callable[[Any, Any], bool]) -> Dict[str, Any]:
    """items: [{"seg", "dec", "em"}] in prompt order (mutated in place).
    emit(seg, new_decision) -> Emitted. Returns allocation log."""
    total = lambda: overhead + sum(0 if it["em"].dropped else it["em"].tokens for it in items)
    log = {"start_tokens": total(), "target": target, "limit": limit, "lowered": 0, "externalized": 0,
           "dropped": 0, "hard_truncation": False}
    if total() <= target:
        log["end_tokens"] = total()
        return log
    cands = [it for it in items if not it["dec"].protected and it["seg"].kind != "query"]
    cands.sort(key=lambda it: (round(it["seg"].features.get("query_similarity", 0), 3), it["seg"].turn_index,
                               it["seg"].index))
    # pass 1: lower ratios to the per-kind minimum
    for it in cands:
        if total() <= target:
            break
        d = it["dec"]
        floor = cfg.snap_ratio(cfg.min_ratio_by_kind.get(it["seg"].kind, 0.3), it["seg"].kind)
        if d.lane in ("compress", "cache") and it["em"].ratio > floor and not d.backref:
            nd = type(d)(**{**d.__dict__, "lane": "compress", "ratio": floor, "source": "budget",
                            "reasons": d.reasons + [f"budget_lower->{floor}"]})
            it["dec"], it["em"] = nd, emit(it["seg"], nd)
            log["lowered"] += 1
    # pass 2: externalize
    for it in cands:
        if total() <= target:
            break
        d = it["dec"]
        if d.lane != "externalize" and can_externalize(it["seg"], d):
            nd = type(d)(**{**d.__dict__, "lane": "externalize", "ratio": 1.0, "source": "budget",
                            "reasons": d.reasons + ["budget_externalize"]})
            it["dec"], it["em"] = nd, emit(it["seg"], nd)
            log["externalized"] += 1
    # pass 3: drop oldest unprotected (hard truncation) only if over the hard limit
    if total() > limit:
        for it in sorted(cands, key=lambda it: (it["seg"].turn_index, it["seg"].index)):
            if total() <= limit:
                break
            it["em"].dropped = True
            it["em"].text = ""
            log["dropped"] += 1
        log["hard_truncation"] = log["dropped"] > 0
    log["end_tokens"] = total()
    return log

"""Hard-rule layer, checked before the classifier. Rules always win."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sieve.segmenter import Segment

LANES = ["cache", "compress", "externalize"]


@dataclass
class Decision:
    segment_id: str
    key: str
    source: str                  # rule:<name> | classifier | heuristic | fallback | budget
    lane: str                    # compress | cache | externalize
    ratio: float = 1.0
    probs: Dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    expected_cost: Optional[float] = None
    reasons: List[str] = field(default_factory=list)
    backref: Optional[str] = None          # key of the segment this one refers back to
    forbid_externalize: bool = False
    protected: bool = False                # budget allocator may not touch it

    @property
    def passthrough(self) -> bool:
        return self.lane == "compress" and self.ratio >= 1.0 and not self.backref


@dataclass
class RuleContext:
    stub_tokens_estimate: int = 70


def _match(seg: Segment, match: Dict[str, Any]) -> bool:
    if "kind" in match and seg.kind != match["kind"]:
        return False
    if "role" in match and seg.role != match["role"]:
        return False
    if "tag" in match and not seg.tags.get(match["tag"]):
        return False
    if "contains" in match and match["contains"] not in seg.content:
        return False
    return True


def apply_rules(seg: Segment, cfg, emitted_keys: Dict[str, str], atoms_of=None,
                ctx: RuleContext = RuleContext()) -> Dict[str, Any]:
    """Returns {"decision": Decision|None, "forbid_externalize": bool, "fired": [names]}.

    emitted_keys: key -> lane of segments already emitted earlier in this prompt/session
    (used for back-references). atoms_of: callable(text)->set, for the near-dup safety check.
    """
    fired: List[str] = []
    forbid = False

    def D(name, lane, ratio=1.0, **kw):
        fired.append(name)
        return {"decision": Decision(seg.id, seg.key, f"rule:{name}", lane, ratio, reasons=[name], **kw),
                "forbid_externalize": forbid, "fired": fired}

    # (1) current user turn: untouched
    if seg.kind == "query":
        return D("current_query", "compress", 1.0, protected=True)
    # (2) system prompt: cache
    if seg.role == "system":
        return D("system_cache", "cache", 1.0, protected=True)
    # (3) tags
    if seg.tags.get("never_externalize"):
        forbid = True
        fired.append("never_externalize")
    if seg.tags.get("never_compress"):
        return D("never_compress", "cache", 1.0, protected=True, forbid_externalize=forbid)
    if seg.tags.get("pinned"):
        return D("pinned", "cache", 1.0, protected=True, forbid_externalize=forbid)
    # (3b) user-extensible rules from config.yaml
    for r in cfg.extra_rules or []:
        if _match(seg, r.get("match", {})):
            lane = r.get("lane", "compress")
            if lane == "externalize" and forbid:
                continue
            return D(r.get("name", "extra_rule"), lane, float(r.get("ratio", 1.0)), forbid_externalize=forbid)
    # (4) repeat of something already in the prompt -> tiny back-reference
    if seg.dup_of and seg.dup_of in emitted_keys:
        if seg.dup_type == "exact":
            return D("exact_repeat_backref", "cache", 1.0, backref=seg.dup_of, protected=True)
        # near-dup: only safe if every fact-like atom of this version already appears in the original
        if seg.features.get("_near_atoms_subset", 0.0) >= 1.0:
            return D("near_dup_backref", "cache", 1.0, backref=seg.dup_of, protected=True)
        fired.append("near_dup_atoms_differ")
    # (5) too short to be worth it
    if seg.tokens < cfg.min_segment_tokens:
        return D("short_passthrough", "compress", 1.0, forbid_externalize=forbid)
    # (6) stub would be > 50% of the original
    if ctx.stub_tokens_estimate > 0.5 * seg.tokens:
        forbid = True
        fired.append("stub_too_large")
    return {"decision": None, "forbid_externalize": forbid, "fired": fired}


def heuristic_label(seg: Segment, cfg) -> Decision:
    """The original hand-written labeller (used by C3 and for heuristic training labels)."""
    f = seg.features
    q, rec, toks = f.get("query_similarity", 0), f.get("recency_norm", 0), seg.tokens
    if f.get("is_system") or seg.tags.get("static") or seg.tags.get("pinned") or f.get("repeat_count", 0) > 0:
        lane, ratio = "cache", 1.0
    elif rec >= 0.5 and q < 0.35 and toks >= 80:
        lane, ratio = "externalize", 1.0
    elif q >= 0.5:
        lane, ratio = "compress", 0.9
    elif q >= 0.35:
        lane, ratio = "compress", 0.7
    else:
        lane, ratio = "compress", 0.5
    if lane == "compress":
        ratio = cfg.snap_ratio(ratio, seg.kind)
    return Decision(seg.id, seg.key, "heuristic", lane, ratio, reasons=[f"q={q:.2f}", f"rec={rec:.2f}"])

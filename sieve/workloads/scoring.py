"""Answer scoring: normalized containment, numeric tolerance, yes/no, refusal patterns."""
from __future__ import annotations

import re
from typing import Any, Dict, List

REFUSAL_RE = re.compile(
    r"not (?:been )?(?:mentioned|specified|provided|stated|available|given|included|listed|present|found|"
    r"in the conversation|recorded|known)|no (?:information|mention|data|record)|unknown|"
    r"(?:cannot|can't|unable to) (?:be )?(?:determine|find|say|tell|answer)|(?:don't|do not) (?:know|have)|"
    r"does(?:n't| not) (?:say|mention|specify|state|include)|isn't (?:mentioned|specified|stated)", re.I)
NUM_RE = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


def normalize(s: str) -> str:
    s = s.lower().replace("’", "'")
    s = re.sub(r"[*_`\"]", "", s)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)  # 1,284 -> 1284
    s = re.sub(r"[^\w.\- ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_numbers(s: str) -> List[float]:
    out = []
    for m in NUM_RE.findall(s):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def is_refusal(answer: str) -> bool:
    return bool(REFUSAL_RE.search(answer))


def score_answer(answer: str, ep: Dict[str, Any]) -> bool:
    a = answer or ""
    na = normalize(a)
    kind = ep["answer_kind"]
    if not na:
        return False
    if kind == "refusal":
        return is_refusal(a) and not parse_numbers(a)
    if kind == "yesno":
        m = re.search(r"\b(yes|no)\b", na)
        return bool(m) and m.group(1) == ep["gold"][0]
    for f in ep.get("forbid") or []:
        if normalize(f) in na and not any(normalize(g) in na for g in ep["gold"]):
            return False
    if kind == "number":
        target = float(ep["numeric"])
        tol = 1e-6 if target == int(target) else 1e-3 * abs(target)  # integers must match exactly
        if any(abs(x - target) <= tol for x in parse_numbers(a)):
            return True
        return False
    if ep.get("gold_all"):
        if all(normalize(t) in na.split() or normalize(t) in na for t in ep["gold_all"]):
            return True
    padded = f" {na} "
    for gold in ep["gold"]:
        ng = normalize(gold)
        if ng and (f" {ng} " in padded or f" {ng}." in padded or (len(ng) > 6 and ng in na)):
            return True
    return False

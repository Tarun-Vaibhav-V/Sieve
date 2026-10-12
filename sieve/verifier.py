"""Rule-based faithfulness check (no LLM call): Critical Atom Recall."""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Set, Tuple

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
PATTERNS: List[Tuple[str, str]] = [
    ("date", rf"\b\d{{1,2}} (?:{MONTHS}) \d{{4}}\b|\b(?:{MONTHS}) \d{{1,2}},? \d{{4}}\b|\b\d{{4}}-\d\d-\d\d\b"),
    ("id", r"\b[A-Z]{2,}-\d{3,}\b|\bE\d{4}\b|\b[0-9a-f]{8,}\b"),
    ("version", r"\bv?\d+\.\d+(?:\.\d+)+\b|\bv\d+\.\d+\b"),
    ("number", r"\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|\$?\d+(?:\.\d+)?%?"),
    ("quoted", r"\"[^\"\n]{2,60}\""),
    ("entity", r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b"),
    ("name", r"(?<=[a-z,;:] )[A-Z][a-z]{3,}\b"),
]
NUMERIC_KINDS = {"date", "id", "version", "number"}
COMMON = set("""The This That These Those In On For And Or It We Our Several Most From None If As Going
Quick Thanks Noted Update Correction Project Unit Record Legal Keep Pinned Archive Switching One Let Here
Please Reference Tool Spec Notes Design Decision Team Operations After Answer Yes No Can Could What Any
How Who Which Got Understood Calling Build""".split())
COMPILED = [(k, re.compile(p)) for k, p in PATTERNS]


def extract_atoms(text: str, extra: Iterable[str] = ()) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {k: set() for k, _ in PATTERNS}
    claimed: List[Tuple[int, int]] = []
    for kind, rx in COMPILED:
        for m in rx.finditer(text):
            a, b = m.span()
            if any(a < y and b > x for x, y in claimed):
                continue
            s = m.group(0)
            if kind == "number" and len(s.strip("$%")) < 2:
                continue  # single digits are too noisy to be critical
            if kind in ("entity", "name"):
                words = [w for w in s.split() if w not in COMMON]
                if not words:
                    continue
                s = " ".join(words) if kind == "entity" else s
            out[kind].add(s)
            claimed.append((a, b))
    for e in extra:
        if e and e in text:
            out.setdefault("critical", set()).add(e)
    return out


def flat(atoms: Dict[str, Set[str]]) -> Set[str]:
    return set().union(*atoms.values()) if atoms else set()


CTX_STOP = set("""the a an of to in on for and or is are was were be been by with at from as that this these those it its
our we us you was has have had will would should could can may might must not no than then there their they them which who
what when where how""".split())
WORD = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")


def context_recall(original: str, compressed: str, atoms: Iterable[str], window: int = 12) -> Dict[str, object]:
    """Relation check: for each atom, the content words of its original sentence must mostly survive
    near the atom in the compressed text (LLMLingua can keep 'Draenyl Kroumnaesk' but drop
    'product lead', which keeps atom recall at 1.0 while destroying the fact)."""
    worst, bad = 1.0, []
    comp_words = compressed.split()
    atoms = list(atoms)
    atom_words = {w.lower() for x in atoms for w in WORD.findall(x)}
    for a in atoms:
        i = original.find(a)
        j = compressed.find(a)
        if i < 0 or j < 0:
            continue
        st = max(original.rfind(". ", 0, i), original.rfind("\n", 0, i)) + 1
        en_c = [x for x in (original.find(". ", i), original.find("\n", i)) if x >= 0]
        sent = original[st:min(en_c) if en_c else len(original)]
        # relation words only: entity/number words are atoms and are checked on their own
        cw = {w.lower() for w in WORD.findall(sent) if w.lower() not in CTX_STOP and w.lower() not in atom_words}
        if not cw:
            continue
        k = len(compressed[:j].split())
        near = {w.lower() for w in WORD.findall(" ".join(comp_words[max(0, k - window): k + window + len(a.split())]))}
        frac = len(cw & near) / len(cw)
        if frac < worst:
            worst = frac
        if frac < 0.5:
            bad.append(a)
    return {"context_recall": worst, "context_lost": bad}


def recall(atoms: Dict[str, Set[str]], compressed: str, original: str = "") -> Dict[str, object]:
    num = set().union(*[v for k, v in atoms.items() if k in NUMERIC_KINDS or k == "critical"]) if atoms else set()
    allx = flat(atoms)
    lost = sorted(a for a in allx if a not in compressed)
    lost_num = [a for a in lost if a in num]
    ctx = context_recall(original, compressed, sorted(allx - set(lost))) if original else \
        {"context_recall": 1.0, "context_lost": []}
    return {
        "numeric_recall": 1.0 if not num else 1 - len(lost_num) / len(num),
        "overall_recall": 1.0 if not allx else 1 - len(lost) / len(allx),
        "lost": lost + [f"(context) {a}" for a in ctx["context_lost"]], "n_atoms": len(allx), "n_numeric": len(num),
        "context_recall": ctx["context_recall"],
    }


def atom_recall_in(text: str, atoms: Iterable[str]) -> float:
    atoms = [a for a in atoms if a]
    if not atoms:
        return 1.0
    return sum(a in text for a in atoms) / len(atoms)

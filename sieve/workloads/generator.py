"""Programmatic, seed-deterministic conversation generator with ground truth.

Every case is a scripted message list plus 1-4 evaluation points. Entities are invented
(codenames, people, figures) so the model cannot answer from prior knowledge. Filler comes
from a template grammar with no digits and no invented entities, so gold facts live only in
the messages listed in `needed_segment_ids` (workload-level segments = messages, ids "mNN").
"""
from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional, Tuple

from sieve.workloads import vocab as V

FAMILIES = "ABCDEFGHIJK"
TIERS = {
    "smoke":    dict(A=1, B=2, C=2, D=1, E=2, F=1, G=1, H=1, I=0, J=0, K=1),
    "standard": dict(A=6, B=10, C=10, D=6, E=8, F=8, G=6, H=6, I=4, J=4, K=8),
    "full":     dict(A=12, B=20, C=20, D=12, E=24, F=16, G=16, H=12, I=10, J=14, K=20),
}
FAMILY_DESC = {
    "A": "Long system prompt, short chat (CACHE)",
    "B": "Long chat, old irrelevant history (EXTERNALIZE)",
    "C": "Verbose document Q&A (COMPRESS)",
    "D": "Same document re-pasted / paraphrased (CACHE + dedup)",
    "E": "Needle in haystack (router + verifier)",
    "F": "Topic switches, late question about early detail (expansion)",
    "G": "Number/ID/entity dense text (verifier stress)",
    "H": "Bloated tool results: JSON, logs, CSV (canonicalizer + compress)",
    "I": "Code context Q&A (compress limits)",
    "J": "Adversarial edge cases (robustness)",
    "K": "Mixed realistic sessions, 10-20 turns (all lanes + epochs)",
}
BUCKET_PATTERN = ["long", "mid", "short", "mid", "long", "mid", "short", "mid", "long", "short"]
CHARS_PER_TOKEN = 5.8  # template prose, calibrated once against the Qwen3 tokenizer (DECISIONS D7)
DENSE_CHARS_PER_TOKEN = {"json": 3.0, "csv": 2.6, "log": 4.4, "code": 3.4}


def est_tokens(text: str, cpt: float = CHARS_PER_TOKEN) -> int:
    """Generation-time size estimate (tokenizer-independent so generation is environment-stable)."""
    return int(len(text) / cpt) + 1


SYSTEM_BASE = ("You are a precise assistant for the {org} team. Answer the user's question in one short "
               "sentence using only information in this conversation. If the answer is not present in "
               "the conversation, reply exactly: Not mentioned.")


class Gen:
    def __init__(self, seed: str):
        self.r = random.Random(seed)
        self.used: set = set()

    # ---------------- entities ----------------
    def codename(self, syl: Optional[int] = None) -> str:
        on = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "qu", "th", "kr",
              "dr", "st", "br", "gl", "sh", "x"]
        vw = ["a", "e", "i", "o", "u", "ae", "ia", "ou", "y"]
        co = ["", "n", "r", "l", "k", "x", "th", "sk", "rn", "m", "v"]
        for _ in range(1000):
            n = syl or self.r.choice([2, 2, 3])
            w = "".join(self.r.choice(on) + self.r.choice(vw) + (self.r.choice(co) if i == n - 1 or self.r.random() < .3 else "")
                        for i in range(n))
            w = w.capitalize()
            if w not in self.used and 5 <= len(w) <= 11:
                self.used.add(w)
                return w
        raise RuntimeError("codename space exhausted")

    def person(self) -> str:
        return f"{self.codename(2)} {self.codename()}"

    def office(self) -> str:
        return f"{self.codename()} {self.r.choice(V.OFFICE_SUFFIX)}"

    def vendor(self) -> str:
        return f"{self.codename()} {self.r.choice(V.VENDOR_SUFFIX)}"

    def number(self, lo: int, hi: int) -> int:
        while True:
            n = self.r.randint(lo, hi)
            if n not in self.used and n % 10 != 0:
                self.used.add(n)
                return n

    def date(self) -> Tuple[str, List[str]]:
        d, m, y = self.r.randint(2, 28), self.r.choice(V.MONTHS), self.r.randint(2027, 2031)
        return f"{d} {m} {y}", [str(d), m, str(y)]

    def version(self) -> str:
        return f"v{self.r.randint(2, 9)}.{self.r.randint(10, 39)}.{self.r.randint(1, 9)}"

    def ident(self) -> str:
        a = "".join(self.r.choice("ABCDEFGHJKLMNPRSTVWXZ") for _ in range(2))
        return f"{a}-{self.r.randint(10000, 99999)}"

    def hexid(self) -> str:
        return "".join(self.r.choice("0123456789abcdef") for _ in range(8))

    # ---------------- filler ----------------
    def _slots(self) -> Dict[str, str]:
        c = self.r.choice
        return dict(adj=c(V.ADJ), adj2=c(V.ADJ), noun=c(V.NOUN), noun2=c(V.NOUN), np=c(V.NOUN_PHRASE),
                    verb=c(V.VERB), verbb=c(V.VERB_BASE), state=c(V.STATE))

    def sentence(self) -> str:
        return self.r.choice(V.TEMPLATES).format(**self._slots())

    def paragraph(self, n: Optional[int] = None) -> str:
        return " ".join(self.sentence() for _ in range(n or self.r.randint(4, 8)))

    def filler(self, tokens: int, para_tokens: Tuple[int, int] = (90, 180)) -> str:
        """Filler prose of approximately `tokens` tokens, split into paragraphs."""
        paras, total = [], 0
        while total < tokens:
            want = min(self.r.randint(*para_tokens), max(20, tokens - total))
            p = []
            while est_tokens(" ".join(p)) < want:
                p.append(self.sentence())
            text = " ".join(p)
            paras.append(text)
            total += est_tokens(text)
        return "\n\n".join(paras)

    def user_ask(self) -> str:
        return self.r.choice(V.USER_ASKS).format(**self._slots())

    def code_filler(self, n_funcs: int) -> str:
        out = []
        for _ in range(n_funcs):
            v, n = self.r.choice(V.CODE_VERB), self.r.choice(V.CODE_NOUN)
            m = self.r.choice(V.CODE_NOUN)
            out.append(
                f"def {v}_{n}_{m}s(items, strict=False):\n"
                f"    \"\"\"{v.capitalize()} {n} {m}s before the {self.r.choice(V.NOUN)} step.\"\"\"\n"
                f"    result = []\n"
                f"    for item in items:\n"
                f"        if strict and not item:\n"
                f"            continue\n"
                f"        result.append(item)\n"
                f"    return result\n")
        return "\n\n".join(out)

    def embed(self, filler_text: str, facts: List[str], positions: Optional[List[float]] = None) -> str:
        """Insert fact sentences into filler paragraphs at relative positions (0..1)."""
        paras = filler_text.split("\n\n")
        positions = positions or [self.r.random() for _ in facts]
        slots = sorted(zip(positions, range(len(facts))))
        for pos, i in reversed(slots):
            k = min(len(paras) - 1, int(pos * len(paras)))
            sents = paras[k].split(". ")
            j = self.r.randint(0, max(0, len(sents) - 1))
            sents.insert(j, facts[i].rstrip("."))
            paras[k] = ". ".join(sents)
            if not paras[k].endswith("."):
                paras[k] += "."
        return "\n\n".join(paras)


# ---------------- facts / questions ----------------

def fact_number(g: Gen, subject: Optional[str] = None) -> Dict[str, Any]:
    metric, unit, lo, hi, money = g.r.choice(V.METRICS)
    c = subject or g.codename()
    n = g.number(lo, hi)
    val = f"${n:,}" if money else f"{n:,} {unit}"
    return dict(qtype="lookup_number", subject=c, metric=metric,
                sentence=f"The {metric} for {c} was measured at {val}.",
                question=f"What was the {metric} measured for {c}?",
                gold=[f"{n:,}", str(n)], answer_kind="number", numeric=n, atoms=[c, f"{n:,}"],
                gold_answer=f"The {metric} for {c} was {val}.")


def fact_name(g: Gen, subject: Optional[str] = None) -> Dict[str, Any]:
    role = g.r.choice(V.ROLES)
    c = subject or g.codename()
    p = g.person()
    return dict(qtype="lookup_name", subject=c,
                sentence=f"The {role} for the {c} service is {p}.",
                question=f"Who is the {role} for the {c} service?",
                gold=[p, p.split()[1]], answer_kind="text", atoms=[c, p],
                gold_answer=f"The {role} for {c} is {p}.")


def fact_multihop(g: Gen) -> Dict[str, Any]:
    c, p, o = g.codename(), g.person(), g.office()
    return dict(qtype="multi_hop", subject=c,
                sentences=[f"Project {c} is led by {p}.", f"{p} works out of the {o} office."],
                question=f"Which office does the lead of Project {c} work out of?",
                gold=[o, o.split()[0]], answer_kind="text", atoms=[c, p, o],
                gold_answer=f"The lead of {c}, {p}, works out of the {o} office.")


def fact_decision(g: Gen) -> Dict[str, Any]:
    comp = g.r.choice(V.COMPONENTS)
    v1, v2 = g.vendor(), g.vendor()
    c = g.codename()
    return dict(qtype="decision_recall", subject=c,
                sentence=f"After the review we decided to use {v1} as the vendor for the {c} {comp} tier, rejecting {v2}.",
                question=f"Which vendor did we decide to use for the {c} {comp} tier?",
                gold=[v1, v1.split()[0]], answer_kind="text", atoms=[c, v1], forbid=[v2.split()[0]],
                gold_answer=f"We chose {v1} for the {c} {comp} tier.")


def fact_update(g: Gen) -> Dict[str, Any]:
    c = g.codename()
    if g.r.random() < 0.5:
        d1, _ = g.date()
        d2, parts = g.date()
        return dict(qtype="value_update", subject=c,
                    sentences=[f"The launch date for {c} is set to {d1}.",
                               f"Update: the launch date for {c} has moved to {d2}."],
                    question=f"What is the latest launch date for {c}?",
                    gold=[d2], gold_all=parts, answer_kind="text", atoms=[c, d1, d2], forbid=[d1],
                    gold_answer=f"The latest launch date for {c} is {d2}.")
    n1, n2 = g.number(50, 9000), g.number(50, 9000)
    return dict(qtype="value_update", subject=c,
                sentences=[f"The rate limit for {c} is {n1:,} requests per minute.",
                           f"Correction: the rate limit for {c} is now {n2:,} requests per minute."],
                question=f"What is the current rate limit for {c}, in requests per minute?",
                gold=[f"{n2:,}", str(n2)], answer_kind="number", numeric=n2, atoms=[c, f"{n1:,}", f"{n2:,}"],
                gold_answer=f"The current rate limit for {c} is {n2:,} requests per minute.")


def fact_negation(g: Gen) -> Dict[str, Any]:
    c = g.codename()
    items = [g.codename() for _ in range(4)]
    excluded = items[3]
    ask_pos = g.r.random() < 0.4
    target = items[g.r.randint(0, 2)] if ask_pos else excluded
    return dict(qtype="negation", subject=c,
                sentence=(f"The approved component list for release {c} includes {items[0]}, {items[1]} and "
                          f"{items[2]}; {excluded} was explicitly excluded."),
                question=f"Answer yes or no: is {target} included in the approved component list for release {c}?",
                gold=["yes" if ask_pos else "no"], answer_kind="yesno", atoms=[c] + items,
                gold_answer=("Yes" if ask_pos else "No") + f", {target} is " + ("" if ask_pos else "not ") + "included.")


def fact_unanswerable(g: Gen, about: Dict[str, Any]) -> Dict[str, Any]:
    """Ask for a metric of an existing subject that is never stated anywhere."""
    used = about.get("metric")
    metric, unit, *_ = g.r.choice([m for m in V.METRICS if m[0] != used])
    c = about["subject"]
    return dict(qtype="unanswerable", subject=c, question=f"What was the {metric} measured for {c}?",
                gold=["not mentioned"], answer_kind="refusal", atoms=[c],
                gold_answer="Not mentioned.", sentences=[])


def facts_of(f: Dict[str, Any]) -> List[str]:
    return f.get("sentences") or ([f["sentence"]] if f.get("sentence") else [])


# ---------------- conversation builder ----------------

class Conv:
    def __init__(self, case_id: str, family: str, g: Gen, bucket: str, target: int):
        self.case_id, self.family, self.g, self.bucket, self.target = case_id, family, g, bucket, target
        self.messages: List[Dict[str, Any]] = []
        self.evals: List[Dict[str, Any]] = []
        self.turn = 0
        self.org = g.codename()

    def _add(self, role, content, kind, tags=()) -> str:
        mid = f"m{len(self.messages):02d}"
        self.messages.append(dict(mid=mid, role=role, kind=kind, content=content, tags=sorted(tags),
                                  turn=self.turn))
        return mid

    def system(self, extra: str = "", tags=("static",)) -> str:
        text = SYSTEM_BASE.format(org=self.org) + (("\n\n" + extra) if extra else "")
        return self._add("system", text, "system", tags)

    def user(self, content, kind="chat", tags=()) -> str:
        self.turn += 1
        return self._add("user", content, kind, tags)

    def assistant(self, content, kind="chat") -> str:
        return self._add("assistant", content, kind)

    def chat_exchange(self, tokens: int):
        """Filler user question + verbose assistant answer of about `tokens` tokens."""
        u = self.g.user_ask()
        self.user(u + (" " + self.g.paragraph(2) if self.g.r.random() < .5 else ""))
        self.assistant(self.g.filler(max(40, tokens - est_tokens(u)), (80, 160)))

    def ask(self, f: Dict[str, Any], needed: List[str]):
        self.user(f["question"], kind="query")
        qi = len(self.messages) - 1
        self.evals.append(dict(
            eval_point=len(self.evals), question_index=qi, question=f["question"], gold=f["gold"],
            gold_all=f.get("gold_all", []), forbid=f.get("forbid", []), answer_kind=f["answer_kind"],
            numeric=f.get("numeric"), question_type=f["qtype"], critical_atoms=f["atoms"],
            facts=facts_of(f), needed_segment_ids=sorted(set(needed))))
        self.assistant(f["gold_answer"], kind="gold_answer")

    def tokens(self) -> int:
        return sum(est_tokens(m["content"]) + 5 for m in self.messages)

    def out(self) -> Dict[str, Any]:
        return dict(case_id=self.case_id, family=self.family, family_desc=FAMILY_DESC[self.family],
                    length_bucket=self.bucket, target_tokens=self.target,
                    est_tokens=self.tokens(), messages=self.messages, eval_points=self.evals)


def pick_fact(g: Gen, kinds=("number", "name", "multihop", "decision", "update", "negation")):
    k = g.r.choice(kinds)
    return {"number": fact_number, "name": fact_name, "multihop": fact_multihop, "decision": fact_decision,
            "update": fact_update, "negation": fact_negation}[k](g)


# ---------------- families ----------------

def fam_A(c: Conv, n: int):
    """Long system prompt (handbook with facts), short chat; questions about the handbook."""
    g = c.g
    facts = [fact_number(g), fact_name(g), fact_decision(g), fact_number(g), fact_negation(g)]
    body = g.embed(g.filler(int(c.target * 0.8)), [s for f in facts for s in facts_of(f)])
    sys_mid = c.system(f"Operations handbook for {c.org}:\n\n" + body, tags=("static",))
    asked = [facts[0], facts[1 + n % 3], facts[4]] if c.bucket != "short" else [facts[0], facts[1]]
    for i, f in enumerate(asked):
        c.chat_exchange(int(c.target * 0.03) + 40)
        c.ask(f, [sys_mid])
    if n % 2 == 0:
        c.ask(fact_unanswerable(g, facts[0]), [])


def fam_B(c: Conv, n: int):
    """Long chat: an early message with an old fact, many irrelevant turns, recent facts."""
    g = c.g
    c.system()
    old = fact_number(g) if n % 2 else fact_name(g)
    old_mid = c.user("Notes from the kickoff meeting:\n\n" + g.embed(g.filler(int(c.target * 0.18)), facts_of(old)),
                     kind="doc")
    c.assistant("Thanks, I have the kickoff notes. " + g.paragraph(3))
    n_turns = 4 + n % 4
    for _ in range(n_turns):
        c.chat_exchange(int(c.target * 0.55 / n_turns))
    recent = fact_name(g) if n % 2 else fact_number(g)
    rec_mid = c.user("Quick status update: " + g.paragraph(2) + " " + facts_of(recent)[0] + " " + g.paragraph(2))
    c.assistant("Noted. " + g.paragraph(2))
    c.ask(recent, [rec_mid])
    c.chat_exchange(int(c.target * 0.05) + 40)
    if n % 3 != 2:  # most cases also ask about the old detail (expansion needed if externalized)
        c.ask(old, [old_mid])
    else:
        c.ask(fact_unanswerable(g, recent), [])


def fam_C(c: Conv, n: int):
    """A verbose document pasted once; lookup / name / multi-hop questions."""
    g = c.g
    c.system()
    f1, f2, f3 = fact_number(g), fact_name(g), fact_multihop(g)
    facts = facts_of(f1) + facts_of(f2) + facts_of(f3)
    doc = g.embed(g.filler(int(c.target * 0.85)), facts, [g.r.random() for _ in facts])
    d = c.user("Please read this project report carefully:\n\n" + doc, kind="doc")
    c.assistant("I have read the report.")
    order = [f1, f2, f3] if n % 2 == 0 else [f3, f1, f2]
    for f in order[: 2 if c.bucket == "short" else 3]:
        c.ask(f, [d])


def fam_D(c: Conv, n: int):
    """Same document re-pasted exactly, then paraphrased (near-duplicate)."""
    g = c.g
    c.system()
    f1, f2, f3 = fact_number(g), fact_name(g), fact_decision(g)
    facts = facts_of(f1) + facts_of(f2) + facts_of(f3)
    doc = g.embed(g.filler(int(c.target * 0.33)), facts)
    d1 = c.user("Reference document:\n\n" + doc, kind="doc")
    c.assistant("Got it.")
    c.ask(f1, [d1])
    c.chat_exchange(60)
    d2 = c.user("Reference document:\n\n" + doc, kind="doc")  # exact repeat
    c.assistant("Got it.")
    c.ask(f2, [d1, d2])
    # paraphrase: swap a few filler words per paragraph, facts unchanged
    paras = doc.split("\n\n")
    para2 = []
    for p in paras:
        w = p.split(" ")
        for _ in range(max(1, len(w) // 60)):
            k = g.r.randrange(len(w))
            if w[k].lower() in V.ADJ:
                w[k] = g.r.choice(V.ADJ)
        para2.append(" ".join(w))
    d3 = c.user("Reference document (lightly edited):\n\n" + "\n\n".join(para2), kind="doc")
    c.assistant("Got it.")
    c.ask(f3, [d1, d2, d3])


def fam_E(c: Conv, n: int):
    """Needle in a haystack at 10/50/90% depth."""
    g = c.g
    c.system()
    depth = [0.1, 0.5, 0.9][n % 3]
    f = fact_number(g) if n % 2 == 0 else fact_name(g)
    hay = g.filler(int(c.target * 0.9))
    doc = g.embed(hay, facts_of(f), [depth])
    d = c.user("Here is a long set of meeting notes:\n\n" + doc, kind="doc")
    c.assistant("I have the notes.")
    c.ask(f, [d])
    if n % 4 == 1:
        c.ask(fact_unanswerable(g, f), [])


def fam_F(c: Conv, n: int):
    """Several topic blocks; a late question about an early detail."""
    g = c.g
    c.system()
    early = pick_fact(g, ("number", "name", "decision"))
    e_mid = c.user(f"Let's start with topic {g.codename()}. " + g.paragraph(3) + " " +
                   " ".join(facts_of(early)) + " " + g.paragraph(3))
    c.assistant(g.filler(int(c.target * 0.08)))
    topics = 2 + n % 3
    late = None
    for t in range(topics):
        c.user(f"Switching topics to {g.codename()}. " + g.paragraph(2))
        c.assistant(g.filler(int(c.target * 0.6 / topics)))
        if t == topics - 1:
            late = pick_fact(g, ("number", "name"))
            l_mid = c.user("One more detail on this topic: " + " ".join(facts_of(late)) + " " + g.paragraph(2))
            c.assistant("Understood. " + g.paragraph(2))
    c.ask(late, [l_mid])
    c.ask(early, [e_mid])


def fam_G(c: Conv, n: int):
    """Dense spec sheet: many units with numbers, IDs, versions; questions on specific fields."""
    g = c.g
    c.system()
    units = []
    lines = []
    n_units = max(4, int(c.target * 0.85 / 75))
    for _ in range(n_units):
        u = dict(id=g.ident(), fw=g.version(), rpm=g.number(800, 9800), duty=round(g.r.uniform(10, 95), 1),
                 owner=g.person(), batch=g.hexid())
        units.append(u)
        lines.append(f"Unit {u['id']} runs firmware {u['fw']} at {u['rpm']:,} rpm with a {u['duty']} percent duty "
                     f"cycle; it was commissioned by {u['owner']} from batch {u['batch']}. " + g.sentence())
    paras = ["\n".join(lines[i:i + 4]) for i in range(0, len(lines), 4)]
    d = c.user("Spec sheet for the turbine fleet:\n\n" + "\n\n".join(paras), kind="doc")
    c.assistant("I have the spec sheet.")
    picks = g.r.sample(units, 3)
    qs = [
        dict(qtype="lookup_number", question=f"What rpm does unit {picks[0]['id']} run at?",
             gold=[f"{picks[0]['rpm']:,}", str(picks[0]['rpm'])], answer_kind="number", numeric=picks[0]['rpm'],
             atoms=[picks[0]['id'], f"{picks[0]['rpm']:,}"], gold_answer=f"{picks[0]['rpm']:,} rpm.",
             sentences=[ln for ln in lines if picks[0]['id'] in ln]),
        dict(qtype="lookup_name", question=f"Which firmware version does unit {picks[1]['id']} run?",
             gold=[picks[1]['fw'], picks[1]['fw'][1:]], answer_kind="text", atoms=[picks[1]['id'], picks[1]['fw']],
             gold_answer=f"Firmware {picks[1]['fw']}.", sentences=[ln for ln in lines if picks[1]['id'] in ln]),
        dict(qtype="lookup_name", question=f"Who commissioned unit {picks[2]['id']}?",
             gold=[picks[2]['owner'], picks[2]['owner'].split()[1]], answer_kind="text",
             atoms=[picks[2]['id'], picks[2]['owner']], gold_answer=f"{picks[2]['owner']}.",
             sentences=[ln for ln in lines if picks[2]['id'] in ln]),
    ]
    for q in qs[: 2 if c.bucket == "short" else 3]:
        c.ask(q, [d])


def fam_H(c: Conv, n: int):
    """Bloated tool results: JSON (nulls, repeated records), logs (repeated lines), CSV (dup rows)."""
    g = c.g
    c.system()
    kind = ["json", "log", "csv"][n % 3]
    budget = int(c.target * 0.85)
    if kind == "json":
        regions = [g.codename() for _ in range(4)]
        target_id = f"ORD-{g.number(10000, 99999)}"
        target_region = g.codename()
        recs = []
        while est_tokens(json.dumps(recs, indent=2), DENSE_CHARS_PER_TOKEN["json"]) < budget:
            recs.append({"order_id": f"ORD-{g.number(10000, 99999)}", "region": g.r.choice(regions),
                         "status": g.r.choice(["shipped", "pending", "held"]), "notes": None, "tags": [],
                         "gift_wrap": None, "coupon": ""})
            if g.r.random() < 0.35:
                recs.append(dict(recs[-1]))  # identical repeated record
        recs.insert(len(recs) // 2 + g.r.randint(-3, 3), {"order_id": target_id, "region": target_region,
                    "status": "shipped", "notes": None, "tags": [], "gift_wrap": None, "coupon": ""})
        body = "```json\n" + json.dumps({"orders": recs, "next_page": None}, indent=2) + "\n```"
        q = dict(qtype="lookup_name", question=f"Which region is order {target_id} assigned to?",
                 gold=[target_region], answer_kind="text", atoms=[target_id, target_region],
                 gold_answer=f"Order {target_id} is assigned to {target_region}.",
                 sentences=[f'"order_id": "{target_id}"'])
        tool = "list_orders"
    elif kind == "log":
        svcs = [g.codename().lower() for _ in range(5)]
        code = f"E{g.number(1000, 9999)}"
        culprit = g.r.choice(svcs)
        lines = []
        while est_tokens("\n".join(lines), DENSE_CHARS_PER_TOKEN["log"]) < budget:
            s = g.r.choice(svcs)
            line = f"INFO [{s}] heartbeat ok; queue depth nominal"
            lines.extend([line] * g.r.randint(1, 6))
            if g.r.random() < 0.3:
                lines.append(f"WARN [{s}] retrying upstream call")
        err = f"ERROR [{culprit}] failed with code {code}: upstream contract violation"
        lines.insert(int(len(lines) * g.r.uniform(0.2, 0.8)), err)
        body = "```\n" + "\n".join(lines) + "\n```"
        q = dict(qtype="lookup_name", question=f"Which service logged error code {code}?",
                 gold=[culprit], answer_kind="text", atoms=[code, culprit],
                 gold_answer=f"The {culprit} service logged {code}.", sentences=[err])
        tool = "tail_logs"
    else:
        skus = []
        rows = ["sku,warehouse,quantity,status"]
        whs = [g.codename() for _ in range(3)]
        while est_tokens("\n".join(rows), DENSE_CHARS_PER_TOKEN["csv"]) < budget:
            sku = f"SKU-{g.number(10000, 99999)}"
            skus.append(sku)
            row = f"{sku},{g.r.choice(whs)},{g.number(1, 9999)},active   "
            rows.append(row)
            if g.r.random() < 0.3:
                rows.append(row)  # duplicate row
        pick = g.r.choice(rows[1:]).strip()
        sku, wh, qty, _ = pick.split(",")
        body = "```csv\n" + "\n".join(rows) + "\n```"
        q = dict(qtype="lookup_number", question=f"What quantity is listed for {sku}?",
                 gold=[qty], answer_kind="number", numeric=int(qty), atoms=[sku, qty],
                 gold_answer=f"{sku} has quantity {qty}.", sentences=[pick])
        tool = "export_inventory"
    c.user(f"I ran a tool for the {c.org} account.")
    c.assistant(f"Calling `{tool}` now.")
    t = c.user(f"Tool output from `{tool}`:\n{body}", kind="tool_result")
    c.assistant("I have the tool output.")
    c.ask(q, [t])


def fam_I(c: Conv, n: int):
    """Code context Q&A."""
    g = c.g
    c.system()
    fn = f"fetch_{g.codename().lower()}_ledger"
    retry = g.number(3, 97)
    err = f"{g.codename()}TimeoutError"
    raiser = f"poll_{g.codename().lower()}_queue"
    target_fn = (f"def {fn}(client, retry_limit={retry}, backoff=0.5):\n"
                 f"    \"\"\"Fetch the ledger with retries.\"\"\"\n"
                 f"    for attempt in range(retry_limit):\n"
                 f"        result = client.get('/ledger')\n"
                 f"        if result.ok:\n"
                 f"            return result\n"
                 f"    return None\n")
    raise_fn = (f"def {raiser}(queue):\n"
                f"    \"\"\"Poll the queue once.\"\"\"\n"
                f"    if queue.stalled():\n"
                f"        raise {err}('queue stalled')\n"
                f"    return queue.pop()\n")
    k = max(3, int(c.target * 0.85 / 62))
    blocks = g.code_filler(k).split("\n\n")
    blocks.insert(g.r.randint(0, len(blocks)), target_fn)
    blocks.insert(g.r.randint(0, len(blocks)), raise_fn)
    d = c.user("Here is the module source:\n\n```python\n" + "\n\n".join(blocks) + "\n```", kind="doc")
    c.assistant("I have the module.")
    c.ask(dict(qtype="lookup_number", question=f"What is the default value of retry_limit in {fn}?",
               gold=[str(retry)], answer_kind="number", numeric=retry, atoms=[fn, str(retry)],
               gold_answer=f"The default retry_limit is {retry}.", sentences=[f"def {fn}(client, retry_limit={retry}"]),
          [d])
    c.ask(dict(qtype="lookup_name", question=f"Which function raises {err}?", gold=[raiser], answer_kind="text",
               atoms=[err, raiser], gold_answer=f"{raiser} raises {err}.", sentences=[f"raise {err}("]), [d])


def fam_J(c: Conv, n: int):
    """Adversarial edge cases, cycling through sub-types."""
    g = c.g
    sub = n % 7
    c.family_sub = sub
    if sub == 0:  # too short to benefit
        c.system()
        f = fact_number(g)
        m = c.user(f["sentence"] + " " + g.sentence())
        c.assistant("Noted.")
        c.ask(f, [m])
    elif sub == 1:  # all history relevant: should not be compressed away
        c.system()
        fs = [fact_number(g), fact_name(g), fact_decision(g), fact_number(g)]
        mids = []
        for f in fs:
            mids.append(c.user("Record this: " + facts_of(f)[0] + " " + g.paragraph(2)))
            c.assistant("Recorded.")
        for f, m in list(zip(fs, mids))[:3]:
            c.ask(f, [m])
    elif sub == 2:  # paraphrased repeats with identical values
        c.system()
        f = fact_number(g)
        a = c.user(g.paragraph(3) + " " + f["sentence"] + " " + g.paragraph(3), kind="doc")
        c.assistant("Thanks.")
        b = c.user("To repeat in other words: " + g.paragraph(2) + " " + f["sentence"] + " " + g.paragraph(2), kind="doc")
        c.assistant("Thanks.")
        c.ask(f, [a, b])
    elif sub == 3:  # tagged segments
        c.system()
        f1, f2, f3 = fact_number(g), fact_name(g), fact_decision(g)
        a = c.user("Legal text (do not alter):\n\n" + g.embed(g.filler(int(c.target * .25)), facts_of(f1)),
                   kind="doc", tags=("never_compress",))
        c.assistant("Understood.")
        b = c.user("Keep this visible:\n\n" + g.embed(g.filler(int(c.target * .25)), facts_of(f2)),
                   kind="doc", tags=("never_externalize",))
        c.assistant("Understood.")
        p = c.user("Pinned reference:\n\n" + g.embed(g.filler(int(c.target * .25)), facts_of(f3)),
                   kind="doc", tags=("pinned",))
        c.assistant("Understood.")
        c.ask(f1, [a])
        c.ask(f2, [b])
        c.ask(f3, [p])
    elif sub == 4:  # contradicting values across turns: latest wins
        c.system()
        f = fact_update(g)
        a = c.user(g.paragraph(2) + " " + f["sentences"][0])
        c.assistant("Noted.")
        for _ in range(2):
            c.chat_exchange(int(c.target * .15))
        b = c.user(f["sentences"][1] + " " + g.paragraph(2))
        c.assistant("Noted.")
        c.ask(f, [a, b])
    elif sub == 5:  # needed segment externalized on purpose
        c.system()
        f = fact_number(g)
        d = c.user("Archive excerpt:\n\n" + g.embed(g.filler(int(c.target * .5)), facts_of(f)), kind="doc",
                   tags=("force_externalize",))
        c.assistant("Archived.")
        c.chat_exchange(int(c.target * .2))
        c.ask(f, [d])
    else:  # unanswerable with a tempting near-miss value present
        c.system()
        f = fact_number(g)
        m = c.user(g.filler(int(c.target * .5)) + " " + f["sentence"])
        c.assistant("Noted.")
        c.ask(fact_unanswerable(g, f), [])
        c.ask(f, [m])


def fam_K(c: Conv, n: int):
    """Mixed realistic session, 10-20 turns, several evaluation points."""
    g = c.g
    pol = fact_name(g)
    sys_mid = c.system("Team policies:\n\n" + g.embed(g.filler(int(c.target * .1)), facts_of(pol)))
    f_doc = fact_number(g)
    f_upd = fact_update(g)
    f_dec = fact_decision(g)
    d = c.user("Design doc draft:\n\n" + g.embed(g.filler(int(c.target * .22)), facts_of(f_doc) + [f_upd["sentences"][0]]),
               kind="doc")
    c.assistant("I've read the design doc. " + g.paragraph(2))
    c.ask(f_doc, [d])
    n_chat = 3 + n % 4
    for _ in range(n_chat):
        c.chat_exchange(int(c.target * .3 / n_chat))
    t = c.user("Tool output from `build_report`:\n```\n" + "\n".join(
        [f"INFO [build] step {w} completed" for w in g.r.sample(V.NOUN, 8)] * 3) + "\n```", kind="tool_result")
    c.assistant("Build looks fine.")
    dm = c.user("Decision log: " + f_dec["sentence"] + " " + g.paragraph(2))
    c.assistant("Logged. " + g.paragraph(1))
    c.ask(f_dec, [dm])
    for _ in range(2):
        c.chat_exchange(int(c.target * .08))
    um = c.user(f_upd["sentences"][1] + " " + g.paragraph(1))
    c.assistant("Thanks for the update.")
    c.ask(f_upd, [d, um])
    c.chat_exchange(60)
    c.ask(pol, [sys_mid])


BUILDERS = dict(A=fam_A, B=fam_B, C=fam_C, D=fam_D, E=fam_E, F=fam_F, G=fam_G, H=fam_H, I=fam_I, J=fam_J, K=fam_K)


def target_for(bucket: str, g: Gen) -> int:
    lo, hi = {"long": (4400, 7000), "mid": (1600, 3400), "short": (600, 1350)}[bucket]
    return g.r.randint(lo, hi)


def generate_case(family: str, n: int, seed: int, bucket: str) -> Dict[str, Any]:
    g = Gen(f"sieve-{seed}-{family}-{n}")
    if family == "J" and n % 7 == 0:
        bucket = "short"
    c = Conv(f"{family}:{n}", family, g, bucket, target_for(bucket, g))
    BUILDERS[family](c, n)
    out = c.out()
    out["seed"] = seed
    out["subtype"] = getattr(c, "family_sub", None)
    return out


def generate_suite(tier: str, seed: int) -> List[Dict[str, Any]]:
    counts = TIERS[tier]
    cases, k = [], 0
    for fam in FAMILIES:
        for n in range(counts[fam]):
            bucket = BUCKET_PATTERN[(k + seed) % len(BUCKET_PATTERN)]
            cases.append(generate_case(fam, n, seed, bucket))
            k += 1
    return cases

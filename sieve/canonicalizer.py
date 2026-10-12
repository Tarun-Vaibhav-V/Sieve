"""Deterministic attachment / tool-result normalizers (text only)."""
from __future__ import annotations

import csv
import io
import json
import re
from typing import Any, Tuple

FENCE_RE = re.compile(r"```([a-zA-Z]*)\n(.*?)\n```", re.S)


def sniff(body: str, hint: str = "") -> str:
    h = hint.lower()
    s = body.strip()
    if h in ("json",) or (s[:1] in "[{" and s[-1:] in "]}"):
        try:
            json.loads(s)
            return "json"
        except Exception:
            pass
    if h in ("csv", "tsv"):
        return h
    if h in ("python", "py", "js", "ts", "go", "java", "c", "cpp", "rust", "code"):
        return "code"
    if re.search(r"<\s*(html|div|p|span|body|table)\b", s, re.I):
        return "html"
    lines = s.splitlines()
    if len(lines) >= 3 and sum(bool(re.match(r"^\s*(INFO|WARN|ERROR|DEBUG|TRACE|\d{4}-\d\d-\d\d)", l)) for l in lines) > len(lines) / 2:
        return "log"
    if len(lines) >= 3 and all(l.count(",") == lines[0].count(",") and l.count(",") > 0 for l in lines[:20]):
        return "csv"
    if h in ("md", "markdown") or re.search(r"^#{1,6} |\*\*|\[[^\]]+\]\([^)]+\)", s, re.M):
        return "markdown"
    return "text"


def _prune(o: Any) -> Any:
    if isinstance(o, dict):
        out = {k: _prune(v) for k, v in o.items()}
        return {k: v for k, v in out.items() if v not in (None, "", [], {})}
    if isinstance(o, list):
        items = [_prune(v) for v in o]
        res, i = [], 0
        while i < len(items):  # collapse identical consecutive records with a count
            j = i
            while j + 1 < len(items) and items[j + 1] == items[i]:
                j += 1
            if j > i and isinstance(items[i], dict):
                res.append(dict(items[i], **{"_repeat": j - i + 1}))
            else:
                res.extend(items[i:j + 1])
            i = j + 1
        return res
    return o


def _dump(o: Any, top: bool = True) -> str:
    """Minified JSON, but one record per line inside lists so it chunks on line boundaries."""
    c = lambda x: json.dumps(x, separators=(",", ":"), ensure_ascii=False)
    if isinstance(o, list) and o and all(isinstance(x, dict) for x in o):
        return "[\n" + ",\n".join(c(x) for x in o) + "\n]"
    if isinstance(o, dict) and top:
        return "{" + ",".join(f"{c(k)}:{_dump(v, False)}" for k, v in o.items()) + "}"
    return c(o)


def canon_json(s: str) -> str:
    return _dump(_prune(json.loads(s)))


def canon_log(s: str) -> str:
    out, prev, n = [], None, 0
    for line in s.splitlines() + [None]:
        if line is not None:
            line = line.rstrip()
        norm = re.sub(r"\d", "#", line) if line is not None else None
        pnorm = re.sub(r"\d", "#", prev) if prev is not None else None
        if prev is not None and line == prev:
            n += 1
            continue
        if prev is not None:
            out.append(prev + (f" ×{n}" if n > 1 else ""))
        prev, n = line, 1
        _ = (norm, pnorm)  # near-identical (digit-only) lines are kept: digits may be facts
    return "\n".join(out)


def canon_csv(s: str, delim: str = ",") -> str:
    rows = [r for r in csv.reader(io.StringIO(s), delimiter=delim)]
    if not rows:
        return s
    header, body = rows[0], rows[1:]
    counts, order = {}, []
    for r in body:
        t = tuple(c.strip() for c in r)
        if t not in counts:
            order.append(t)
            counts[t] = 0
        counts[t] += 1
    has_dup = any(v > 1 for v in counts.values())
    out = io.StringIO()
    w = csv.writer(out, delimiter=delim, lineterminator="\n")
    w.writerow([h.strip() for h in header] + (["count"] if has_dup else []))
    for t in order:
        w.writerow(list(t) + ([counts[t]] if has_dup else []))
    return out.getvalue().rstrip("\n")


def canon_code(s: str) -> str:
    lines = [l.rstrip() for l in s.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines))


def canon_markup(s: str) -> str:
    s = re.sub(r"<script.*?</script>|<style.*?</style>", "", s, flags=re.S | re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\*\*|__|^#{1,6} ", "", s, flags=re.M)
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", s)).strip()


NORMALIZERS = {"json": canon_json, "log": canon_log, "csv": canon_csv, "tsv": lambda s: canon_csv(s, "\t"),
               "code": canon_code, "html": canon_markup}


def canonicalize(text: str) -> Tuple[str, str]:
    """Normalize every fenced block (or the whole text if unfenced). Returns (text, kinds)."""
    kinds = []

    def fix(body: str, hint: str) -> str:
        k = sniff(body, hint)
        fn = NORMALIZERS.get(k)
        kinds.append(k)
        if not fn:
            return body
        try:
            return fn(body)
        except Exception:
            return body

    if FENCE_RE.search(text):
        new = FENCE_RE.sub(lambda m: f"```{m.group(1)}\n{fix(m.group(2), m.group(1))}\n```", text)
    else:
        new = fix(text, "")
    return new, ",".join(kinds)


def ocr_image(path: str) -> str:  # interface stub, disabled by default
    raise NotImplementedError("OCR is out of scope for Sieve (text-only attachments).")


def transcribe_audio(path: str) -> str:  # interface stub, disabled by default
    raise NotImplementedError("ASR is out of scope for Sieve (text-only attachments).")

"""Static HTML report: explanation.html + results.html (neo-brutalist black / white / yellow).

Every number on the pages is read from a run's logs (results.jsonl, summary.json, env.json),
the classifier evaluation, the cachetest and the break-even sweep. Nothing is typed in by hand.
"""
from __future__ import annotations

import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sieve.metrics import aggregate, bootstrap_ci, load_records

E = html.escape
COND_ORDER = ["baseline", "truncate", "uniform_lingua", "sieve_rules", "sieve_full", "sieve_nomemo",
              "reference_bigctx", "sieve_full+noverifier", "sieve_full+noprefetch", "sieve_full+noledger",
              "sieve_full+clf_heuristic", "sieve_full+core6"]
COND_LABEL = {"baseline": "C0 baseline", "truncate": "C1 truncate", "uniform_lingua": "C2 uniform LLMLingua",
              "sieve_rules": "C3 Sieve rules", "sieve_full": "C4 Sieve full", "sieve_nomemo": "C5 Sieve no-ledger",
              "reference_bigctx": "C6 8k-context reference", "sieve_full+noverifier": "C4 − verifier",
              "sieve_full+noprefetch": "C4 − prefetch (tool only)", "sieve_full+noledger": "C4 − ledger",
              "sieve_full+clf_heuristic": "C4 heuristic-trained clf", "sieve_full+core6": "C4 core-6 features"}
MAIN = ["baseline", "truncate", "uniform_lingua", "sieve_rules", "sieve_full", "sieve_nomemo", "reference_bigctx"]
ABL = ["sieve_full", "sieve_full+noverifier", "sieve_full+noprefetch", "sieve_full+noledger",
       "sieve_full+clf_heuristic", "sieve_full+core6"]
FAM_DESC = {
    "A": "Long system prompt, short chat", "B": "Long chat, old irrelevant history", "C": "Verbose document Q&A",
    "D": "Same document re-pasted", "E": "Needle in a haystack", "F": "Topic switches, late recall",
    "G": "Number / ID dense text", "H": "Bloated tool output", "I": "Code Q&A", "J": "Adversarial edge cases",
    "K": "Mixed 10–20 turn sessions"}

CSS = """
/* Layout: one 1120px column of hard-edged slabs; ink rules and offset shadows do the separating. */
:root{
  --paper:#fbfaf4; --card:#ffffff; --ink:#0b0b0b; --sub:#4a4a44; --line:#0b0b0b;
  --yellow:#ffd60a; --yellow-ink:#0b0b0b; --hl:#fff1a6; --good:#1f7a3a; --bad:#b3261e; --grid:#d9d6c8;
  --display:"Archivo Black","Arial Black",Impact,sans-serif;
  --body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  --shadow:6px 6px 0 var(--ink); --b:3px solid var(--line);
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --paper:#111110; --card:#1b1b19; --ink:#f4f2e8; --sub:#b9b6a8; --line:#f4f2e8;
  --yellow:#ffd60a; --yellow-ink:#0b0b0b; --hl:#3a3410; --good:#5fd07e; --bad:#ff7a6e; --grid:#3a3a35; color-scheme:dark}}
:root[data-theme="dark"]{
  --paper:#111110; --card:#1b1b19; --ink:#f4f2e8; --sub:#b9b6a8; --line:#f4f2e8;
  --yellow:#ffd60a; --yellow-ink:#0b0b0b; --hl:#3a3410; --good:#5fd07e; --bad:#ff7a6e; --grid:#3a3a35; color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);font:16px/1.6 var(--body);margin:0;padding-inline:16px;padding-block:0 64px}
.wrap{max-width:1120px;margin:0 auto;display:flex;flex-direction:column;gap:40px}
a{color:var(--ink);text-decoration-thickness:2px;text-underline-offset:3px}
a:focus-visible,button:focus-visible,select:focus-visible,input:focus-visible{outline:3px solid var(--yellow);outline-offset:2px}
nav.top{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;border-bottom:var(--b);padding-block:14px;position:sticky;top:env(safe-area-inset-top,0px);background:var(--paper);z-index:5}
nav.top .brand{font-family:var(--display);font-size:22px;letter-spacing:.02em}
nav.top .links{display:flex;flex-wrap:wrap;gap:8px}
nav.top .links a{font:600 13px var(--mono);text-transform:uppercase;letter-spacing:.06em;border:2px solid var(--line);padding:4px 10px;text-decoration:none;background:var(--card)}
nav.top .links a.on{background:var(--yellow);color:var(--yellow-ink)}
header.hero{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(0,1fr);gap:28px;align-items:end;padding-top:24px}
@media (max-width:820px){header.hero{grid-template-columns:1fr}}
h1{font-family:var(--display);font-size:clamp(40px,7vw,84px);line-height:.92;margin:0;letter-spacing:-.01em;text-wrap:balance}
h1 mark{background:var(--yellow);color:var(--yellow-ink);padding:0 .12em;box-decoration-break:clone}
h2{font-family:var(--display);font-size:clamp(26px,3.4vw,38px);line-height:1.05;margin:0;text-wrap:balance}
h3{font:700 19px/1.3 var(--body);margin:0}
.eyebrow{font:600 12px var(--mono);letter-spacing:.12em;text-transform:uppercase;color:var(--sub)}
.lede{font-size:19px;max-width:62ch;margin:0}
.meta{border:var(--b);background:var(--card);box-shadow:var(--shadow);padding:16px 18px;font:13px/1.7 var(--mono)}
.meta div{display:flex;justify-content:space-between;gap:12px;border-bottom:1px dashed var(--grid)}
.meta div:last-child{border:0}
.meta b{font-weight:600}
section{display:flex;flex-direction:column;gap:18px}
section > .head{display:flex;flex-direction:column;gap:6px;border-top:var(--b);padding-top:14px}
.prose{max-width:68ch;display:flex;flex-direction:column;gap:12px}
.prose p{margin:0}
.slab{border:var(--b);background:var(--card);box-shadow:var(--shadow);padding:18px;min-width:0}
.slab.yellow{background:var(--yellow);color:var(--yellow-ink)}
.grid2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}
.grid3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:22px}
.grid4{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:18px}
@media (max-width:900px){.grid3,.grid4{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:620px){.grid2,.grid3,.grid4{grid-template-columns:1fr}}
.tile{border:var(--b);background:var(--card);box-shadow:var(--shadow);padding:14px 16px;display:flex;flex-direction:column;gap:4px;min-width:0}
.tile .v{font-family:var(--display);font-size:clamp(28px,4vw,40px);line-height:1}
.tile .k{font:600 12px var(--mono);letter-spacing:.08em;text-transform:uppercase}
.tile .s{font:12px/1.5 var(--mono);color:var(--sub)}
.tile.hot{background:var(--yellow);color:var(--yellow-ink)}
.tile.hot .s{color:var(--yellow-ink)}
.scroll{overflow-x:auto;border:var(--b);background:var(--card);box-shadow:var(--shadow)}
table{border-collapse:collapse;width:100%;font:13px/1.45 var(--mono);font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap;vertical-align:top}
th:first-child,td:first-child{text-align:left}
thead th{background:var(--ink);color:var(--paper);font-weight:600;text-transform:uppercase;font-size:11px;letter-spacing:.06em;position:sticky;top:0}
tr.focus td{background:var(--hl)}
td.l,th.l{text-align:left}
td.wrap{white-space:normal;min-width:240px;max-width:420px}
.pos{color:var(--good);font-weight:600}.neg{color:var(--bad);font-weight:600}
.ci{color:var(--sub)}
.pill{display:inline-block;border:2px solid var(--line);padding:0 6px;font:600 11px/1.6 var(--mono);text-transform:uppercase;letter-spacing:.05em}
.pill.y{background:var(--yellow);color:var(--yellow-ink)}
.pill.ok{background:var(--good);color:var(--paper);border-color:var(--good)}
.pill.no{background:var(--bad);color:var(--paper);border-color:var(--bad)}
.lane{display:inline-block;width:10px;height:10px;border:2px solid var(--line);margin-right:6px;vertical-align:-1px}
.lane.compress{background:var(--ink)}.lane.cache{background:var(--card)}.lane.externalize{background:var(--yellow)}
figure{margin:0;display:flex;flex-direction:column;gap:8px;min-width:0}
figcaption{font:12px/1.5 var(--mono);color:var(--sub)}
svg{display:block;max-width:100%;height:auto}
svg text{font-family:var(--mono);font-size:12px;fill:var(--ink)}
section{scroll-margin-top:96px}
svg .ax{stroke:var(--ink);stroke-width:2}
svg .gr{stroke:var(--grid);stroke-width:1}
svg .bar{fill:var(--ink)}
svg .bar.y{fill:var(--yellow);stroke:var(--ink);stroke-width:2}
svg .bar.w{fill:var(--card);stroke:var(--ink);stroke-width:2}
svg .whisk{stroke:var(--sub);stroke-width:2}
svg .ln{fill:none;stroke:var(--ink);stroke-width:3}
svg .ln.y{stroke:var(--yellow)}
svg .ln.d{stroke-dasharray:6 4}
svg .dot{fill:var(--yellow);stroke:var(--ink);stroke-width:2}
svg .dot.k{fill:var(--ink)}
svg .muted{fill:var(--sub)}
svg .lbl{font-weight:600}
pre.code{margin:0;font:12.5px/1.55 var(--mono);white-space:pre-wrap;word-break:break-word;background:var(--card);border:var(--b);box-shadow:var(--shadow);padding:14px;max-height:360px;overflow:auto}
pre.code .rm{background:var(--bad);color:var(--paper);text-decoration:line-through}
pre.code .st{background:var(--yellow);color:var(--yellow-ink)}
ul.tight{margin:0;padding-left:20px;display:flex;flex-direction:column;gap:6px}
.claims li{max-width:90ch}
.controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center}
.controls label{font:600 12px var(--mono);text-transform:uppercase;letter-spacing:.06em;display:flex;gap:6px;align-items:center}
select,input[type=search],button{font:13px var(--mono);border:2px solid var(--line);background:var(--card);color:var(--ink);padding:6px 8px;border-radius:0}
button{cursor:pointer;background:var(--yellow);color:var(--yellow-ink);font-weight:600;box-shadow:3px 3px 0 var(--ink)}
button:active{transform:translate(2px,2px);box-shadow:1px 1px 0 var(--ink)}
.flow{display:flex;flex-wrap:wrap;gap:10px;align-items:stretch}
.flow .step{border:var(--b);background:var(--card);padding:10px 12px;font:600 13px var(--mono);flex:1 1 140px;min-width:0;box-shadow:4px 4px 0 var(--ink)}
.flow .step small{display:block;font-weight:400;color:var(--sub);margin-top:4px;font-size:11.5px;line-height:1.45}
.flow .arrow{align-self:center;font:700 20px var(--mono)}
.note{border-left:6px solid var(--yellow);padding:8px 12px;background:var(--card);font-size:14px;max-width:80ch}
footer{border-top:var(--b);padding-top:14px;font:12px/1.6 var(--mono);color:var(--sub)}
@media (prefers-reduced-motion:no-preference){.tile,.slab{transition:transform .12s, box-shadow .12s}.tile:hover{transform:translate(-2px,-2px);box-shadow:8px 8px 0 var(--ink)}}
"""
FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo+Black&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600;700&display=swap">')


# ---------------- formatting helpers ----------------

def f(x, d=1, pct=False, signed=False, unit=""):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    v = x * 100 if pct else x
    s = f"{v:+.{d}f}" if signed else f"{v:.{d}f}"
    return s + unit


def ci_txt(t, d=1, pct=False, unit=""):
    if not t or t[0] is None or (isinstance(t[0], float) and math.isnan(t[0])):
        return "–"
    k = 100 if pct else 1
    cls = "pos" if t[1] * k > 0 else ("neg" if t[2] * k < 0 else "")
    return (f'<span class="{cls}">{t[0] * k:+.{d}f}{unit}</span> '
            f'<span class="ci">[{t[1] * k:+.{d}f}, {t[2] * k:+.{d}f}]</span>')


FULL_DOC = True  # local files get a complete document; set False for an Artifact publish (skeleton is added there)


def page(title: str, active: str, body: str, run_label: str) -> str:
    nav = (f'<nav class="top"><div class="brand">SIEVE</div><div class="links">'
           f'<a href="explanation.html" class="{"on" if active == "exp" else ""}">How it works</a>'
           f'<a href="results.html" class="{"on" if active == "res" else ""}">Results</a>'
           f'<a href="results.html#records">Records</a></div></nav>')
    inner = (f'<title>{E(title)}</title><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
             f'{FONTS}<style>{CSS}</style>', f'<div class="wrap">{nav}{body}'
             f'<footer>Generated from {E(run_label)} by <code>python -m sieve html</code>. Every number is computed '
             f'from the logged runs; workloads are synthetic.</footer></div>')
    if FULL_DOC:
        return f'<!doctype html><html lang="en"><head><meta charset="utf-8">{inner[0]}</head><body>{inner[1]}</body></html>'
    return inner[0] + inner[1]


# ---------------- SVG charts (one linear scale per chart) ----------------

def svg_hbars(rows: List[Dict[str, Any]], xmin: float, xmax: float, fmt, width=720, unit="", ticks=5,
              title="") -> str:
    """rows: {label, v, lo?, hi?, cls?}. Horizontal bars from xmin (or 0) with optional CI whiskers."""
    lw = int(min(260, 20 + 7.6 * max([len(r["label"]) for r in rows] + [4])))
    rp, rh, top = 70, 30, 18
    h = top + rh * len(rows) + 30
    pw = width - lw - rp
    X = lambda v: lw + (v - xmin) / (xmax - xmin) * pw
    out = [f'<svg viewBox="0 0 {width} {h}" role="img" aria-label="{E(title)}">']
    for i in range(ticks + 1):
        v = xmin + (xmax - xmin) * i / ticks
        out.append(f'<line class="gr" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{top - 6}" y2="{h - 26}"/>'
                   f'<text x="{X(v):.1f}" y="{h - 10}" text-anchor="middle" class="muted">{fmt(v)}{unit}</text>')
    zero = X(max(xmin, min(0, xmax)))
    for i, r in enumerate(rows):
        y = top + i * rh
        x0, x1 = sorted((zero, X(r["v"])))
        out.append(f'<text x="{lw - 10}" y="{y + 16}" text-anchor="end" class="lbl">{E(r["label"])}</text>')
        out.append(f'<rect class="bar {r.get("cls", "")}" x="{x0:.1f}" y="{y + 4}" width="{max(1.5, x1 - x0):.1f}" height="{rh - 12}"/>')
        if r.get("lo") is not None:
            a, b = X(r["lo"]), X(r["hi"])
            yy = y + rh / 2 - 2
            out.append(f'<line class="whisk" x1="{a:.1f}" x2="{b:.1f}" y1="{yy}" y2="{yy}"/>'
                       f'<line class="whisk" x1="{a:.1f}" x2="{a:.1f}" y1="{yy - 5}" y2="{yy + 5}"/>'
                       f'<line class="whisk" x1="{b:.1f}" x2="{b:.1f}" y1="{yy - 5}" y2="{yy + 5}"/>')
        tx = max(x1, X(r["hi"]) if r.get("hi") is not None else x1) + 6
        out.append(f'<text x="{tx:.1f}" y="{y + 16}">{fmt(r["v"])}{unit}</text>')
    out.append(f'<line class="ax" x1="{zero:.1f}" x2="{zero:.1f}" y1="{top - 6}" y2="{h - 26}"/></svg>')
    return "".join(out)


def svg_scatter(points: List[Dict[str, Any]], xr, yr, xlab, ylab, width=720, height=380, title="") -> str:
    """points: {label, x, y, xlo, xhi, ylo, yhi, hot}. Crosshair CIs."""
    l, r, t, b = 64, 24, 20, 48
    pw, ph = width - l - r, height - t - b
    X = lambda v: l + (v - xr[0]) / (xr[1] - xr[0]) * pw
    Y = lambda v: t + (1 - (v - yr[0]) / (yr[1] - yr[0])) * ph
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{E(title)}">']
    for i in range(6):
        xv = xr[0] + (xr[1] - xr[0]) * i / 5
        yv = yr[0] + (yr[1] - yr[0]) * i / 5
        out.append(f'<line class="gr" x1="{X(xv):.1f}" x2="{X(xv):.1f}" y1="{t}" y2="{t + ph}"/>'
                   f'<text x="{X(xv):.1f}" y="{t + ph + 16}" text-anchor="middle" class="muted">{xv:.0f}</text>'
                   f'<line class="gr" x1="{l}" x2="{l + pw}" y1="{Y(yv):.1f}" y2="{Y(yv):.1f}"/>'
                   f'<text x="{l - 8}" y="{Y(yv) + 4:.1f}" text-anchor="end" class="muted">{yv:+.0f}</text>')
    if yr[0] < 0 < yr[1]:
        out.append(f'<line class="ax" x1="{l}" x2="{l + pw}" y1="{Y(0):.1f}" y2="{Y(0):.1f}"/>')
    out.append(f'<line class="ax" x1="{l}" x2="{l}" y1="{t}" y2="{t + ph}"/><line class="ax" x1="{l}" x2="{l + pw}" y1="{t + ph}" y2="{t + ph}"/>')
    out.append(f'<text x="{l + pw / 2}" y="{height - 8}" text-anchor="middle" class="lbl">{E(xlab)}</text>'
               f'<text transform="translate(14,{t + ph / 2}) rotate(-90)" text-anchor="middle" class="lbl">{E(ylab)}</text>')
    placed: List[Tuple[float, float]] = []
    for p in points:
        cx, cy = X(p["x"]), Y(p["y"])
        out.append(f'<line class="whisk" x1="{X(p["xlo"]):.1f}" x2="{X(p["xhi"]):.1f}" y1="{cy:.1f}" y2="{cy:.1f}"/>'
                   f'<line class="whisk" x1="{cx:.1f}" x2="{cx:.1f}" y1="{Y(p["ylo"]):.1f}" y2="{Y(p["yhi"]):.1f}"/>'
                   f'<circle class="dot {"" if p.get("hot") else "k"}" cx="{cx:.1f}" cy="{cy:.1f}" r="{8 if p.get("hot") else 6}"/>')
        ly = cy - 12
        while any(abs(ly - py) < 13 and abs(cx - px) < 120 for px, py in placed):
            ly += 14
        placed.append((cx, ly))
        anchor = "end" if cx > l + pw * 0.75 else "start"
        dx = -10 if anchor == "end" else 10
        out.append(f'<text x="{cx + dx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" class="lbl">{E(p["label"])}</text>')
    out.append("</svg>")
    return "".join(out)


def svg_lines(series: List[Dict[str, Any]], xs: Sequence[float], yr, xlab, ylab, width=720, height=320,
              xfmt=lambda v: f"{v:.2f}", yfmt=lambda v: f"{v:.0f}", title="") -> str:
    """series: {label, ys, cls}."""
    l, r, t, b = 64, 140, 20, 48
    pw, ph = width - l - r, height - t - b
    X = lambda v: l + (v - xs[0]) / (xs[-1] - xs[0] or 1) * pw
    Y = lambda v: t + (1 - (v - yr[0]) / (yr[1] - yr[0])) * ph
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{E(title)}">']
    for i in range(6):
        yv = yr[0] + (yr[1] - yr[0]) * i / 5
        out.append(f'<line class="gr" x1="{l}" x2="{l + pw}" y1="{Y(yv):.1f}" y2="{Y(yv):.1f}"/>'
                   f'<text x="{l - 8}" y="{Y(yv) + 4:.1f}" text-anchor="end" class="muted">{yfmt(yv)}</text>')
    for xv in xs:
        out.append(f'<text x="{X(xv):.1f}" y="{t + ph + 16}" text-anchor="middle" class="muted">{xfmt(xv)}</text>')
    out.append(f'<line class="ax" x1="{l}" x2="{l}" y1="{t}" y2="{t + ph}"/><line class="ax" x1="{l}" x2="{l + pw}" y1="{t + ph}" y2="{t + ph}"/>')
    for s in series:
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, s["ys"]) if y is not None)
        out.append(f'<polyline class="ln {s.get("cls", "")}" points="{pts}"/>')
        for x, y in zip(xs, s["ys"]):
            if y is not None:
                out.append(f'<circle class="dot {"" if "y" in s.get("cls", "") else "k"}" cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4.5"/>')
        last = [(x, y) for x, y in zip(xs, s["ys"]) if y is not None][-1]
        out.append(f'<text x="{X(last[0]) + 10:.1f}" y="{Y(last[1]) + 4:.1f}" class="lbl">{E(s["label"])}</text>')
    out.append(f'<text x="{l + pw / 2}" y="{height - 8}" text-anchor="middle" class="lbl">{E(xlab)}</text>'
               f'<text transform="translate(14,{t + ph / 2}) rotate(-90)" text-anchor="middle" class="lbl">{E(ylab)}</text></svg>')
    return "".join(out)


def svg_stack(rows: List[Tuple[str, Dict[str, float]]], width=720, title="") -> str:
    """100% stacked lane mix per condition."""
    lw, rp, rh, top = 190, 20, 30, 10
    h = top + rh * len(rows) + 34
    pw = width - lw - rp
    out = [f'<svg viewBox="0 0 {width} {h}" role="img" aria-label="{E(title)}">']
    cls = {"compress": "bar", "cache": "bar w", "externalize": "bar y"}
    for i, (lab, mix) in enumerate(rows):
        y = top + i * rh
        tot = sum(mix.values()) or 1
        x = lw
        out.append(f'<text x="{lw - 10}" y="{y + 16}" text-anchor="end" class="lbl">{E(lab)}</text>')
        for k in ("compress", "cache", "externalize"):
            w = pw * mix.get(k, 0) / tot
            if w > 0:
                out.append(f'<rect class="{cls[k]}" x="{x:.1f}" y="{y + 4}" width="{w:.1f}" height="{rh - 12}"/>')
                if w > 34:
                    out.append(f'<text x="{x + w / 2:.1f}" y="{y + 16}" text-anchor="middle" '
                               f'style="fill:{"var(--paper)" if k == "compress" else "var(--ink)"}">{100 * mix.get(k, 0) / tot:.0f}%</text>')
                x += w
    ly = top + rh * len(rows) + 18
    for j, (k, name) in enumerate((("compress", "compress"), ("cache", "cache"), ("externalize", "externalize"))):
        out.append(f'<rect class="{cls[k]}" x="{lw + j * 130}" y="{ly - 10}" width="14" height="12"/>'
                   f'<text x="{lw + j * 130 + 20}" y="{ly}">{name}</text>')
    out.append("</svg>")
    return "".join(out)


# ---------------- data assembly ----------------

def load_all(runs_dir: Path, run: str) -> Dict[str, Any]:
    rd = runs_dir / run
    recs = load_records(rd / "results.jsonl")
    env_run = json.loads((rd / "env.json").read_text()) if (rd / "env.json").exists() else {}
    env = json.loads((runs_dir / "env.json").read_text()) if (runs_dir / "env.json").exists() else {}
    cases = [json.loads(l) for l in (rd / "cases.jsonl").read_text().splitlines() if l.strip()] if (rd / "cases.jsonl").exists() else []
    clf = runs_dir / "models" / "classifier_eval.json"
    be = runs_dir / "breakeven.json"
    return {"recs": recs, "agg": aggregate(recs), "env_run": env_run, "env": env, "cases": cases,
            "clf": json.loads(clf.read_text()) if clf.exists() else None,
            "be": json.loads(be.read_text()) if be.exists() else None,
            "claims": (rd / "claims.md").read_text() if (rd / "claims.md").exists() else "", "run": run,
            "demo": runs_dir / "demo"}


def acc_ci(recs, cond):
    xs = [float(r["correct"]) for r in recs if r["condition"] == cond]
    return bootstrap_ci(xs) if xs else (None, None, None)


def conds_present(recs, order=COND_ORDER):
    have = {r["condition"] for r in recs}
    return [c for c in order if c in have]


# ---------------- results page ----------------

def results_page(D: Dict[str, Any]) -> str:
    recs, agg, env, er = D["recs"], D["agg"], D["env"], D["env_run"]
    bc, pc = agg["by_condition"], agg["paired"]
    cond_all = conds_present(recs)
    main = [c for c in MAIN if c in cond_all]
    mock = bool(er.get("mock"))
    n_cases = len({r["case_id"] for r in recs})
    n_eval = len({(r["case_id"], r["eval_point"]) for r in recs})
    S = bc.get("sieve_full", {})
    B = bc.get("baseline", {})
    P = pc.get("sieve_full", {})
    cold = agg["paired_cold"].get("sieve_full", {})
    over = agg["paired_overflow"].get("sieve_full", {})
    parts = []
    # hero
    parts.append(f'''<header class="hero"><div style="display:flex;flex-direction:column;gap:16px">
<div class="eyebrow">Benchmark results · run {E(D["run"])}{" · MOCK" if mock else ""}</div>
<h1>Fewer tokens, <mark>same answers?</mark></h1>
<p class="lede">Seven prompt-handling conditions ran on the same {n_eval} questions from {n_cases} synthetic conversations, on one local model.
Each condition ran in its own pass. Differences are paired against the raw-prompt baseline, with 95% bootstrap intervals.</p></div>
<div class="meta">
<div><span>model</span><b>{E(str(env.get("model", er.get("model"))))} · {E(str(env.get("quantization", "")))}</b></div>
<div><span>hardware</span><b>{E(str(env.get("cpu", "")))} · {E(str(env.get("mem_gb", "")))} GB</b></div>
<div><span>GPU share</span><b>{f((env.get("ps") or {}).get("gpu_pct", 0), 0, pct=True)}% ({f((env.get("ps") or {}).get("size_vram", 0) / 2**30, 2)} GiB)</b></div>
<div><span>Ollama</span><b>{E(str(env.get("ollama_version", "")))}</b></div>
<div><span>num_ctx / num_predict</span><b>{er.get("config", {}).get("num_ctx", 4096)} / {er.get("config", {}).get("num_predict", 64)}</b></div>
<div><span>think:false verified</span><b>{"yes" if env.get("think_disabled_ok") else "no"}</b></div>
<div><span>model calls logged</span><b>{len(recs)}</b></div>
<div><span>prefix reuse method</span><b>{E(str(env.get("prefix_reuse_method", "count")))}</b></div>
</div></header>''')
    if mock:
        parts.append('<div class="slab yellow"><b>MOCK RUN.</b> Model behaviour and timings are simulated. Nothing here is evidence about a real model.</div>')
    # headline tiles
    tiles = [
        ("hot", "Prompt tokens, C4 vs C0", f(-P.get("token_reduction_pct", [None])[0] if P.get("n") else None, 1, unit="%") if P.get("n") else "–",
         f'95% CI {f(-P["token_reduction_pct"][2], 1)} to {f(-P["token_reduction_pct"][1], 1)}%' if P.get("n") else ""),
        ("", "Accuracy, C4 − C0", f(P["acc_delta"][0], 1, pct=True, signed=True, unit=" pp") if P.get("n") else "–",
         f'95% CI {f(P["acc_delta"][1], 1, pct=True, signed=True)} to {f(P["acc_delta"][2], 1, pct=True, signed=True)} pp · n={P.get("n", 0)}' if P.get("n") else ""),
        ("", "Cold TTFT, C4 − C0 (median)", f(cold["ttft_delta_s"][0], 2, signed=True, unit=" s") if cold.get("n") else "–",
         f'95% CI {f(cold["ttft_delta_s"][1], 2, signed=True)} to {f(cold["ttft_delta_s"][2], 2, signed=True)} s · first question of each chat' if cold.get("n") else ""),
        ("", "Output headroom", f'{S.get("output_headroom_mean", 0):.0f}' if S else "–",
         f'mean free tokens in num_ctx (C0: {B.get("output_headroom_mean", 0):.0f})' if B else ""),
    ]
    parts.append('<div class="grid4">' + "".join(
        f'<div class="tile {c}"><span class="k">{E(k)}</span><span class="v">{v}</span><span class="s">{s}</span></div>'
        for c, k, v, s in tiles) + "</div>")
    # main table
    rows = []
    for c in main:
        s, sc, sw = bc[c], agg["by_condition_cold"].get(c, {}), agg["by_condition_warm"].get(c, {})
        a = acc_ci(recs, c)
        rows.append(f'<tr class="{"focus" if c == "sieve_full" else ""}"><td>{E(COND_LABEL.get(c, c))}</td><td>{s["n"]}</td>'
                    f'<td>{f(a[0], 3)} <span class="ci">[{f(a[1], 2)}, {f(a[2], 2)}]</span></td>'
                    f'<td>{f(s["tokens_final_mean"], 0)}</td><td>{f(s["token_reduction_pct"], 1)}</td><td>{f(s["output_headroom_mean"], 0)}</td>'
                    f'<td>{f(sc.get("ttft_median"), 2)}</td><td>{f(sw.get("ttft_median"), 2)}</td><td>{f(s.get("ttft_p95"), 2)}</td>'
                    f'<td>{f(s.get("prefix_reuse_pct_mean"), 0, pct=True)}%</td><td>{f(s.get("atom_recall_mean"), 3)}</td>'
                    f'<td>{f(s.get("expand_rate"), 2)}</td><td>{f(s.get("sieve_overhead_ms_mean"), 0)}</td>'
                    f'<td>{f(s.get("decode_tps_median"), 1)}</td></tr>')
    parts.append(f'''<section id="overview"><div class="head"><span class="eyebrow">All evaluation points</span><h2>Every condition at a glance</h2></div>
<div class="prose"><p>C4 is the full Sieve pipeline. C1 and C2 are matched per question to the prompt size C4 produced, so they compare
strategies at the same token budget. C6 is the raw prompt with twice the context window, as an upper bound for accuracy.
TTFT is time to the first answer token, including any tool round-trips.</p></div>
<div class="scroll"><table><thead><tr><th>condition</th><th>n</th><th>accuracy [95% CI]</th><th>prompt tok</th><th>tok saved %</th><th>headroom</th>
<th>TTFT cold</th><th>TTFT warm</th><th>TTFT p95</th><th>prefix reuse</th><th>atom recall</th><th>expand rate</th><th>Sieve ms</th><th>decode tok/s</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>
<p class="note">Decode speed (tok/s) does not depend on the prompt, so it should stay roughly flat across conditions; it does.
Prefix reuse is estimated from prefill time because this Ollama build reports the full prompt count even on a cache hit (see How it works).</p></section>''')
    # accuracy vs tokens scatter + paired table
    pts = []
    for c in main:
        if c == "baseline" or not pc.get(c, {}).get("n"):
            continue
        p = pc[c]
        pts.append({"label": COND_LABEL.get(c, c).split(" ", 1)[0],
                    "x": p["token_reduction_pct"][0], "xlo": p["token_reduction_pct"][1], "xhi": p["token_reduction_pct"][2],
                    "y": 100 * p["acc_delta"][0], "ylo": 100 * p["acc_delta"][1], "yhi": 100 * p["acc_delta"][2],
                    "hot": c == "sieve_full"})
    if pts:
        xmax = max(10, math.ceil(max(p["xhi"] for p in pts) / 10) * 10)
        xmin = min(0, math.floor(min(p["xlo"] for p in pts) / 10) * 10)
        ylo = math.floor(min(p["ylo"] for p in pts) / 10) * 10
        yhi = math.ceil(max(p["yhi"] for p in pts) / 10) * 10
        if ylo == yhi:
            yhi += 10
        scat = svg_scatter(pts, (xmin, xmax), (min(ylo, -10), max(yhi, 10)), "prompt tokens saved vs C0 (%)",
                           "accuracy change vs C0 (pp)", width=1000, height=420, title="accuracy vs token savings")
    else:
        scat = ""
    prow = []
    for c in cond_all:
        if c == "baseline" or not pc.get(c, {}).get("n"):
            continue
        p, po, pcd = pc[c], agg["paired_overflow"].get(c, {}), agg["paired_cold"].get(c, {})
        prow.append(f'<tr class="{"focus" if c == "sieve_full" else ""}"><td>{E(COND_LABEL.get(c, c))}</td><td>{p["n"]}</td>'
                    f'<td>{ci_txt(p["acc_delta"], 1, pct=True)}</td><td>{ci_txt(p["token_reduction_pct"], 1)}</td>'
                    f'<td>{ci_txt(p["ttft_delta_s"], 2)}</td><td>{ci_txt(pcd.get("ttft_delta_s"), 2) if pcd.get("n") else "–"}</td>'
                    f'<td>{ci_txt(p.get("prefill_delta_s"), 2)}</td>'
                    f'<td>{ci_txt(po["acc_delta"], 1, pct=True) if po.get("n") else "–"} <span class="ci">n={po.get("n", 0)}</span></td></tr>')
    parts.append(f'''<section id="paired"><div class="head"><span class="eyebrow">Paired against C0, same question</span><h2>Savings against accuracy</h2></div>
<div class="prose"><p>Up and to the right is better: the condition sends fewer tokens and gets at least as many answers right.
A vertical interval that crosses zero means this run cannot tell the accuracy apart from the baseline.
The last table column covers only questions whose raw prompt did not fit in {er.get("config", {}).get("num_ctx", 4096)} tokens, where Ollama truncated it.</p></div>
<figure class="slab">{scat}<figcaption>Each point is one condition ({E(" · ".join(COND_LABEL[c] for c in main if c != "baseline"))}).
Whiskers are 95% bootstrap intervals on paired per-question differences. The yellow point is C4.</figcaption></figure>
<div class="scroll"><table><thead><tr><th>condition</th><th>pairs</th><th>Δ accuracy pp [CI]</th><th>tokens saved % [CI]</th>
<th>Δ TTFT median s [CI]</th><th>Δ TTFT cold s [CI]</th><th>Δ prefill s [CI]</th><th>Δ acc where C0 overflowed</th></tr></thead>
<tbody>{"".join(prow)}</tbody></table></div></section>''')
    # latency: cold vs warm bars
    rows_c = [{"label": COND_LABEL.get(c, c), "v": agg["by_condition_cold"][c].get("ttft_median") or 0,
               "cls": "y" if c == "sieve_full" else ""} for c in main if agg["by_condition_cold"].get(c)]
    rows_w = [{"label": COND_LABEL.get(c, c), "v": agg["by_condition_warm"][c].get("ttft_median") or 0,
               "cls": "y" if c == "sieve_full" else ""} for c in main if agg["by_condition_warm"].get(c)]
    tmax = max([r["v"] for r in rows_c + rows_w] + [0.5]) * 1.15
    cp = agg["cumulative_prefill"]
    rows_cp = [{"label": COND_LABEL.get(c, c), "v": cp[c], "cls": "y" if c == "sieve_full" else ""} for c in main if c in cp]
    cmax = max([r["v"] for r in rows_cp] + [1]) * 1.15
    parts.append(f'''<section id="latency"><div class="head"><span class="eyebrow">Latency</span><h2>Cold questions and warm questions</h2></div>
<div class="prose"><p>A <b>cold</b> question is the first one in a conversation: the model has never seen this prompt, so every token is prefilled.
<b>Warm</b> questions come later, and most of their prompt is a prefix the server already holds in its KV cache. The raw prompt is
strongest in the warm case. Sieve's main case is cold prefill and headroom.</p></div>
<div class="grid2"><figure class="slab">{svg_hbars(rows_c, 0, tmax, lambda v: f"{v:.1f}", unit=" s", title="median TTFT cold")}
<figcaption>Median TTFT, cold questions (seconds).</figcaption></figure>
<figure class="slab">{svg_hbars(rows_w, 0, tmax, lambda v: f"{v:.1f}", unit=" s", title="median TTFT warm")}
<figcaption>Median TTFT, warm questions (same scale).</figcaption></figure></div>
<figure class="slab">{svg_hbars(rows_cp, 0, cmax, lambda v: f"{v:.0f}", title="cumulative prefill")}
<figcaption>Tokens actually prefilled per conversation, summed over its questions (timing-derived estimate).
C5 re-routes every turn, so its compressed prefix keeps changing and is prefilled again.</figcaption></figure></section>''')
    # cache-vs-compress
    ct = env.get("cachetest") or []
    ct_rows = "".join(f'<tr><td class="l">{E(r["label"])}</td><td>{r["second_total_tokens"]}</td><td>{r["second_prompt_eval_count"]}</td>'
                      f'<td>{f(r["second_prompt_eval_s"], 2)}</td><td>{f(r.get("second_predicted_cold_s"), 2)}</td>'
                      f'<td><b>{f(r.get("reuse_pct_timing"), 0, pct=True)}%</b></td><td>{f(r["ttft2"], 2)}</td></tr>' for r in ct)
    ct_bars = svg_hbars([{"label": r["label"], "v": 100 * (r.get("reuse_pct_timing") or 0),
                          "cls": "y" if "memo" in r["label"] or "same prefix" in r["label"] else ""} for r in ct],
                        0, 100, lambda v: f"{v:.0f}", unit="%", width=1000, title="cachetest reuse")
    s4, s5 = bc.get("sieve_full", {}), bc.get("sieve_nomemo", {})
    parts.append(f'''<section id="cache"><div class="head"><span class="eyebrow">The cache-vs-compress conflict</span><h2>Compression breaks the KV cache unless the bytes stay put</h2></div>
<figure class="slab">{ct_bars}<figcaption>cachetest: share of the second request's prefill that the server skipped.</figcaption></figure>
<div class="prose"><p>The ledger stores the exact string emitted for each segment and replays it on later turns. In the benchmark, C4 averaged
<b>{f(s4.get("prefix_reuse_pct_mean"), 0, pct=True)}%</b> prefix reuse with <b>{f(s4.get("sieve_overhead_ms_mean"), 0)} ms</b> of Sieve CPU time per call.
C5, which re-routes and re-compresses every turn, averaged <b>{f(s5.get("prefix_reuse_pct_mean"), 0, pct=True)}%</b> reuse and
<b>{f(s5.get("sieve_overhead_ms_mean"), 0)} ms</b> of CPU per call.</p>
<p>Cold prefill fit used for the timing estimate: t(n) = {f((env.get("cold_prefill_fit") or {}).get("a", 0) * 1000, 3)} ms·n + {f((env.get("cold_prefill_fit") or {}).get("b", 0) * 1e6, 4)} µs·n².</p></div>
<div class="scroll"><table><thead><tr><th>cachetest experiment</th><th>tokens</th><th>Ollama count</th><th>prefill s</th><th>cold prediction s</th><th>reuse</th><th>TTFT s</th></tr></thead><tbody>{ct_rows}</tbody></table></div></section>''')
    # families heat table
    fams = sorted({r["family"] for r in recs})
    fam_rows = []
    for fa in fams:
        cells = []
        base = agg["by_family"].get("baseline", {}).get(fa, {}).get("accuracy")
        for c in main:
            v = agg["by_family"].get(c, {}).get(fa, {}).get("accuracy")
            cls = ""
            if base is not None and v is not None and c != "baseline":
                cls = "pos" if v > base + 1e-9 else ("neg" if v < base - 1e-9 else "")
            cells.append(f'<td class="{cls}">{f(v, 2)}</td>')
        tr = agg["by_family"].get("sieve_full", {}).get(fa, {}).get("token_reduction_pct")
        nn = agg["by_family"].get("baseline", {}).get(fa, {}).get("n", 0)
        fam_rows.append(f'<tr><td>{fa} · {E(FAM_DESC.get(fa, ""))}</td><td>{nn}</td>{"".join(cells)}<td>{f(tr, 1)}%</td></tr>')
    qtypes = sorted({r.get("question_type") for r in recs})
    q_rows = []
    for q in qtypes:
        cells = []
        base = agg["by_qtype"].get("baseline", {}).get(q, {}).get("accuracy")
        for c in main:
            v = agg["by_qtype"].get(c, {}).get(q, {}).get("accuracy")
            cls = "pos" if (base is not None and v is not None and c != "baseline" and v > base + 1e-9) else (
                "neg" if (base is not None and v is not None and c != "baseline" and v < base - 1e-9) else "")
            cells.append(f'<td class="{cls}">{f(v, 2)}</td>')
        q_rows.append(f'<tr><td>{E(q)}</td><td>{agg["by_qtype"].get("baseline", {}).get(q, {}).get("n", 0)}</td>{"".join(cells)}</tr>')
    b_rows = []
    for bk in ("short", "mid", "long"):
        cells = "".join(f'<td>{f(agg["by_bucket"].get(c, {}).get(bk, {}).get("accuracy"), 2)}</td>' for c in main)
        tr = agg["by_bucket"].get("sieve_full", {}).get(bk, {}).get("token_reduction_pct")
        b_rows.append(f'<tr><td>{bk}</td><td>{agg["by_bucket"].get("baseline", {}).get(bk, {}).get("n", 0)}</td>{cells}<td>{f(tr, 1)}%</td></tr>')
    head = "".join(f"<th>{E(COND_LABEL.get(c, c).split(' ')[0])}</th>" for c in main)
    parts.append(f'''<section id="families"><div class="head"><span class="eyebrow">Breakdown</span><h2>Where each condition wins and loses</h2></div>
<div class="prose"><p>Accuracy by workload family, question type and prompt length. Green means above C0 for the same questions; red means below.
Groups are small (n = questions per row), so treat single rows as anecdotes.</p></div>
<div class="scroll"><table><thead><tr><th>family</th><th>n</th>{head}<th>C4 tok saved</th></tr></thead><tbody>{"".join(fam_rows)}</tbody></table></div>
<div class="grid2"><div class="scroll"><table><thead><tr><th>question type</th><th>n</th>{head}</tr></thead><tbody>{"".join(q_rows)}</tbody></table></div>
<div class="scroll"><table><thead><tr><th>raw length</th><th>n</th>{head}<th>C4 saved</th></tr></thead><tbody>{"".join(b_rows)}</tbody></table></div></div></section>''')
    # lanes + expansions
    lanes = defaultdict(lambda: defaultdict(float))
    for r in recs:
        for k, v in (r.get("lanes") or {}).items():
            lanes[r["condition"]][k] += v
    lane_rows = [(COND_LABEL.get(c, c), lanes[c]) for c in cond_all if c.startswith("sieve") and sum(lanes[c].values())]
    ex_rows = []
    for c in cond_all:
        if not c.startswith("sieve"):
            continue
        rc = [r for r in recs if r["condition"] == c]
        ne = sum(r.get("needed_externalized", 0) for r in rc)
        ex_rows.append(f'<tr class="{"focus" if c == "sieve_full" else ""}"><td>{E(COND_LABEL.get(c, c))}</td>'
                       f'<td>{sum(r["tool_rounds"] > 0 for r in rc)}</td><td>{sum(len(r["expansions"]) for r in rc)}</td>'
                       f'<td>{sum(r["prefetch_count"] for r in rc)}</td><td>{ne}</td><td>{sum(r["missed_expansions"] for r in rc)}</td>'
                       f'<td>{sum(r["wasted_expansions"] for r in rc)}</td><td>{sum(bool(r.get("unanswered_tool_call")) for r in rc)}</td>'
                       f'<td>{sum(r.get("reinflations", 0) for r in rc)}</td><td>{sum(bool(r.get("hard_truncation")) for r in rc)}</td>'
                       f'<td>{sum(r.get("epoch_id", 0) for r in rc if r["eval_point"] == max(x["eval_point"] for x in rc if x["case_id"] == r["case_id"]))}</td></tr>')
    parts.append(f'''<section id="lanes"><div class="head"><span class="eyebrow">Routing behaviour</span><h2>What the router did</h2></div>
<figure class="slab">{svg_stack(lane_rows, title="lane mix")}<figcaption>Share of emitted segments per lane, summed over every call.</figcaption></figure>
<div class="scroll"><table><thead><tr><th>condition</th><th>calls w/ tool</th><th>expansions</th><th>prefetched</th><th>needed seg. stubbed</th>
<th>missed</th><th>wasted</th><th>tool call, no answer</th><th>verifier re-inflations</th><th>hard truncations</th><th>budget epochs</th></tr></thead>
<tbody>{"".join(ex_rows)}</tbody></table></div>
<p class="note"><b>Missed</b>: a segment that held the answer was stubbed and neither expanded nor prefetched. <b>Wasted</b>: an expansion or prefetch of a segment
the question did not need. <b>Budget epochs</b>: full re-routes forced by the budget, summed over conversations.</p></section>''')
    # ablations
    ab = [c for c in ABL if c in cond_all]
    if len(ab) > 1:
        arows = []
        ref = bc["sieve_full"]
        for c in ab:
            s = bc[c]
            a = acc_ci(recs, c)
            arows.append(f'<tr class="{"focus" if c == "sieve_full" else ""}"><td>{E(COND_LABEL.get(c, c))}</td><td>{s["n"]}</td>'
                         f'<td>{f(a[0], 3)} <span class="ci">[{f(a[1], 2)}, {f(a[2], 2)}]</span></td>'
                         f'<td>{f(s["tokens_final_mean"], 0)}</td><td>{f(s["token_reduction_pct"], 1)}</td>'
                         f'<td>{f(agg["by_condition_cold"].get(c, {}).get("ttft_median"), 2)}</td><td>{f(agg["by_condition_warm"].get(c, {}).get("ttft_median"), 2)}</td>'
                         f'<td>{f(s.get("prefix_reuse_pct_mean"), 0, pct=True)}%</td><td>{f(agg["cumulative_prefill"].get(c), 0)}</td>'
                         f'<td>{f(s.get("expand_rate"), 2)}</td><td>{s.get("reinflations", 0)}</td></tr>')
        parts.append(f'''<section id="ablations"><div class="head"><span class="eyebrow">Ablations of C4</span><h2>Take one part away at a time</h2></div>
<div class="scroll"><table><thead><tr><th>variant</th><th>n</th><th>accuracy [CI]</th><th>prompt tok</th><th>saved %</th><th>TTFT cold</th><th>TTFT warm</th>
<th>reuse</th><th>prefill tok / conv</th><th>expand rate</th><th>re-inflations</th></tr></thead><tbody>{"".join(arows)}</tbody></table></div></section>''')
    # classifier
    clf = D["clf"]
    if clf:
        crow = []
        names = {"rules_only": "Hard rules only", "heuristic_labeller": "Heuristic labeller (C3)", "clf_heuristic": "Classifier, heuristic labels",
                 "clf_oracle": "Classifier, oracle labels (C4)", "clf_oracle_core6": "Classifier, oracle, core-6 features"}
        for k in ("rules_only", "heuristic_labeller", "clf_heuristic", "clf_oracle", "clf_oracle_core6"):
            r = clf["routers"].get(k)
            if not r:
                continue
            e = r["per_lane"]["externalize"]
            crow.append(f'<tr class="{"focus" if k == "clf_oracle" else ""}"><td>{names[k]}</td><td>{r["macro_f1"]:.3f}</td>'
                        f'<td>{r["per_lane"]["cache"]["f1"]:.3f}</td><td>{r["per_lane"]["compress"]["f1"]:.3f}</td>'
                        f'<td>{e["precision"]:.2f} / {e["recall"]:.2f} / {e["f1"]:.2f}</td><td>{r["ratio_mae"] if r["ratio_mae"] is not None else "–"}</td>'
                        f'<td><b>{r["dangerous_errors"]}</b> / {r["n_needed"]}</td><td>{r["externalize_rate"]:.3f}</td></tr>')
        cm = clf["routers"]["clf_oracle"]["confusion"]
        labs = ["cache", "compress", "externalize"]
        cm_html = "".join(f'<tr><td>{labs[i]}</td>' + "".join(f'<td>{"<b>" if i == j else ""}{v}{"</b>" if i == j else ""}</td>' for j, v in enumerate(row)) + "</tr>"
                          for i, row in enumerate(cm))
        imp = clf["feature_importances"][:10]
        imp_svg = svg_hbars([{"label": n, "v": v, "cls": "y" if i == 0 else ""} for i, (n, v) in enumerate(imp)], 0,
                            max(v for _, v in imp) * 1.15, lambda v: f"{v:.2f}", width=560, title="feature importance")
        cal = "".join(f'<tr><td>{b["bin"]}</td><td>{b["n"]}</td><td>{b["accuracy"]:.3f}</td><td>{b["mean_conf"]:.3f}</td></tr>' for b in clf["calibration"])
        parts.append(f'''<section id="classifier"><div class="head"><span class="eyebrow">Held-out classifier evaluation</span><h2>The router, scored on conversations it never saw</h2></div>
<div class="prose"><p>{clf["n_rows"]} segment decisions from {clf["n_groups"]} training conversations (seeds disjoint from the benchmark), split
<b>by conversation</b> with GroupKFold. Ground truth is the oracle label built from workload gold. A <b>dangerous error</b> sends a segment
that a current or future question needs to EXTERNALIZE. Inference: {clf["inference_ms_per_segment"]:.3f} ms per segment on CPU.</p>
<p class="note">The oracle label is partly a fixed function of the features (segment age, size, repeats), so a high F1 mostly shows that the rule is
learnable. The evidence that matters is downstream: the accuracy and token columns above for C3, C4 and the classifier ablations.</p></div>
<div class="scroll"><table><thead><tr><th>router</th><th>macro-F1</th><th>cache F1</th><th>compress F1</th><th>externalize P / R / F1</th><th>ratio MAE</th><th>dangerous errors / needed</th><th>externalize rate</th></tr></thead><tbody>{"".join(crow)}</tbody></table></div>
<div class="grid3" style="align-items:start"><div class="scroll"><table><thead><tr><th>oracle \\ predicted</th><th>cache</th><th>compress</th><th>externalize</th></tr></thead><tbody>{cm_html}</tbody></table></div>
<figure class="slab" style="grid-column:span 1">{imp_svg}<figcaption>Gradient-boosting feature importance (oracle labels).</figcaption></figure>
<div class="scroll"><table><thead><tr><th>confidence</th><th>n</th><th>accuracy</th><th>mean conf</th></tr></thead><tbody>{cal}</tbody></table></div></div></section>''')
    # break-even
    be = D["be"]
    if be:
        xs = [s["p"] for s in be["summary"]]
        yb = [s["baseline_s"] for s in be["summary"]]
        ys = [s["sieve_s"] for s in be["summary"]]
        ymax = max(yb + ys) * 1.2
        chart = svg_lines([{"label": "C0 baseline", "ys": yb, "cls": "d"}, {"label": "C4 Sieve", "ys": ys, "cls": "y"}], xs, (0, ymax),
                          "P(question needs the stubbed document)", "mean net time per question (s)", yfmt=lambda v: f"{v:.1f}",
                          title="break-even")
        brow = "".join(f'<tr><td>{s["p"]:.2f}</td><td>{s["baseline_s"]:.2f}</td><td>{s["sieve_s"]:.2f}</td><td>{s["sieve_s"] - s["baseline_s"]:+.2f}</td>'
                       f'<td>{s["baseline_acc"]:.2f}</td><td>{s["sieve_acc"]:.2f}</td><td>{s["expand_rate"]:.2f}</td></tr>' for s in be["summary"])
        bep = be.get("break_even_p")
        verdict = (f"The curves cross at p ≈ <b>{bep:.2f}</b>." if bep is not None else
                   "The curves do not cross in the swept range.")
        parts.append(f'''<section id="breakeven"><div class="head"><span class="eyebrow">Externalize break-even</span><h2>When does hiding a document stop paying?</h2></div>
<div class="grid2" style="align-items:start"><figure class="slab">{chart}<figcaption>{len(be["rows"]) // max(1, len(xs))} conversations per point, all cold. Net time = Sieve CPU + model time, including tool round-trips.</figcaption></figure>
<div class="prose"><p>Each conversation holds a large archived document that Sieve externalizes. As the share of questions that actually need it grows,
Sieve pays for more expansion round-trips. {verdict}</p>
<div class="scroll"><table><thead><tr><th>p</th><th>C0 s</th><th>C4 s</th><th>Δ s</th><th>C0 acc</th><th>C4 acc</th><th>expand rate</th></tr></thead><tbody>{brow}</tbody></table></div></div></div></section>''')
    # losses (auto-derived)
    loss = []
    for fa in fams:
        b = agg["by_family"].get("baseline", {}).get(fa, {}).get("accuracy")
        s = agg["by_family"].get("sieve_full", {}).get(fa, {}).get("accuracy")
        if b is not None and s is not None and s < b - 1e-9:
            loss.append(f"Family {fa} ({FAM_DESC.get(fa, '')}): C4 accuracy {s:.2f} vs C0 {b:.2f}.")
    sw, bw = agg["by_condition_warm"].get("sieve_full", {}), agg["by_condition_warm"].get("baseline", {})
    if sw.get("ttft_median") and bw.get("ttft_median") and sw["ttft_median"] > bw["ttft_median"]:
        loss.append(f"Warm questions: C4 median TTFT {sw['ttft_median']:.2f} s vs C0 {bw['ttft_median']:.2f} s (the raw prompt's prefix is already cached).")
    shb = agg["by_bucket"].get("sieve_full", {}).get("short", {})
    if shb:
        loss.append(f"Short prompts (<1.5k tokens): C4 saves only {f(shb.get('token_reduction_pct'), 1)}% of tokens, while still paying {f(shb.get('sieve_overhead_ms_mean'), 0)} ms of CPU per call.")
    for r in [r for r in recs if r["condition"] == "sieve_full" and r.get("missed_expansions")][:1]:
        loss.append(f"{sum(x.get('missed_expansions', 0) for x in recs if x['condition'] == 'sieve_full')} needed segments were stubbed and never expanded.")
    claims = D["claims"]
    sup = claims.split("## Supported by this run")[-1].split("## Not supported by this run")[0] if claims else ""
    nsup = claims.split("## Not supported by this run")[-1] if claims else ""
    li = lambda t: "".join(f"<li>{E(x.strip()[2:])}</li>" for x in t.strip().splitlines() if x.strip().startswith("- "))
    parts.append(f'''<section id="claims"><div class="head"><span class="eyebrow">claims.md, generated from the logs</span><h2>What this run supports</h2></div>
<div class="grid2" style="align-items:start"><div class="slab"><h3>Supported</h3><ul class="tight claims">{li(sup)}</ul></div>
<div class="slab"><h3>Not supported by this run</h3><ul class="tight claims">{li(nsup)}</ul></div></div>
<div class="slab yellow"><h3>Where Sieve loses</h3><ul class="tight">{"".join(f"<li>{E(x)}</li>" for x in loss) or "<li>No family where C4 is below C0 in this run.</li>"}</ul></div></section>''')
    # records explorer
    keep = ["condition", "family", "case_id", "eval_point", "cold", "question_type", "tokens_raw_total", "tokens_final_total",
            "correct", "ttft_s", "total_s", "prefix_reuse_pct", "tool_rounds", "answer", "baseline_overflow", "atom_recall"]
    data = [{k: r.get(k) for k in keep} for r in sorted(recs, key=lambda r: (COND_ORDER.index(r["condition"]) if r["condition"] in COND_ORDER else 99, r["case_id"], r["eval_point"]))]
    for d in data:
        d["ttft_s"] = round(d["ttft_s"] or 0, 3)
        d["total_s"] = round(d["total_s"] or 0, 3)
        d["answer"] = (d["answer"] or "")[:160]
    cases = {c["case_id"]: c for c in D["cases"]}
    qs = {f'{c["case_id"]}|{ep["eval_point"]}': [ep["question"], ep["gold"][0]] for c in cases.values() for ep in c["eval_points"]}
    copts = "".join(f'<option value="{E(c)}">{E(COND_LABEL.get(c, c))}</option>' for c in cond_all)
    fopts = "".join(f'<option value="{fa}">{fa} · {E(FAM_DESC.get(fa, ""))}</option>' for fa in fams)
    parts.append(f'''<section id="records"><div class="head"><span class="eyebrow">Publishable records</span><h2>Every logged model call</h2></div>
<div class="prose"><p>{len(recs)} records, one per model call, taken from <code>runs/{E(D["run"])}/results.jsonl</code>. For each call the run directory also holds
the raw prompt, the final Sieve prompt, the response, the per-segment decision trace and a prompt diff, at <code>&lt;case&gt;/&lt;condition&gt;/ep&lt;k&gt;_r0/</code>.</p></div>
<div class="controls"><label for="fc">condition <select id="fc"><option value="">all</option>{copts}</select></label>
<label for="ff">family <select id="ff"><option value="">all</option>{fopts}</select></label>
<label for="fk">result <select id="fk"><option value="">all</option><option value="1">correct</option><option value="0">wrong</option></select></label>
<label for="fq">search <input id="fq" type="search" placeholder="case, answer…"></label>
<button id="copy" type="button">Copy filtered as CSV</button><span id="cnt" class="eyebrow"></span></div>
<div class="scroll" style="max-height:620px"><table id="rt"><thead><tr><th>condition</th><th>case</th><th>q#</th><th>cold</th><th class="l">question / gold</th>
<th>raw tok</th><th>sent tok</th><th>TTFT s</th><th>reuse</th><th>tools</th><th class="l">answer</th><th>ok</th></tr></thead><tbody></tbody></table></div>
<div class="controls"><button id="more" type="button">Show 150 more</button></div>
<script>
const R={json.dumps(data, separators=(",", ":"))};
const Q={json.dumps(qs, separators=(",", ":"))};
const L={json.dumps({c: COND_LABEL.get(c, c) for c in cond_all})};
let lim=150;
const $=id=>document.getElementById(id);
const esc=s=>String(s==null?"":s).replace(/[&<>"]/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}}[c]));
function rows(){{const c=$("fc").value,f=$("ff").value,k=$("fk").value,q=$("fq").value.toLowerCase();
 return R.filter(r=>(!c||r.condition===c)&&(!f||r.family===f)&&(!k||(r.correct?"1":"0")===k)&&(!q||(r.case_id+" "+r.answer+" "+((Q[r.case_id+"|"+r.eval_point]||[""])[0])).toLowerCase().includes(q)));}}
function draw(){{const rs=rows();$("cnt").textContent=rs.length+" matching";
 $("rt").tBodies[0].innerHTML=rs.slice(0,lim).map(r=>{{const qq=Q[r.case_id+"|"+r.eval_point]||["",""];
 return `<tr><td>${{esc(L[r.condition]||r.condition)}}</td><td>${{esc(r.case_id)}}</td><td>${{r.eval_point}}</td><td>${{r.cold?"cold":"warm"}}</td>
 <td class="l wrap">${{esc(qq[0])}}<br><span class="ci">gold: ${{esc(qq[1])}}</span></td><td>${{r.tokens_raw_total}}${{r.baseline_overflow?" ⚑":""}}</td><td>${{r.tokens_final_total}}</td>
 <td>${{r.ttft_s.toFixed(2)}}</td><td>${{Math.round(100*r.prefix_reuse_pct)}}%</td><td>${{r.tool_rounds}}</td><td class="l wrap">${{esc(r.answer)}}</td>
 <td><span class="pill ${{r.correct?"ok":"no"}}">${{r.correct?"yes":"no"}}</span></td></tr>`}}).join("");
 $("more").hidden=rs.length<=lim;}}
["fc","ff","fk"].forEach(i=>$(i).addEventListener("change",()=>{{lim=150;draw();}}));
$("fq").addEventListener("input",()=>{{lim=150;draw();}});
$("more").addEventListener("click",()=>{{lim+=150;draw();}});
$("copy").addEventListener("click",async()=>{{const rs=rows();const cols=["condition","case_id","eval_point","cold","question_type","tokens_raw_total","tokens_final_total","ttft_s","total_s","prefix_reuse_pct","tool_rounds","correct","answer"];
 const csv=[cols.join(",")].concat(rs.map(r=>cols.map(c=>JSON.stringify(r[c]==null?"":r[c])).join(","))).join("\\n");
 try{{await navigator.clipboard.writeText(csv);$("copy").textContent="Copied "+rs.length+" rows";}}catch(e){{$("copy").textContent="Copy blocked: use results.jsonl";}}
 setTimeout(()=>{{$("copy").textContent="Copy filtered as CSV";}},2000);}});
draw();
</script></section>''')
    return page("Sieve Benchmark Results", "res", "".join(parts), f"run {D['run']}")


# ---------------- explanation page ----------------

def _demo_trace(demo_dir: Path, name: str, cond: str = "sieve_full", ep: int = 0) -> Optional[Dict[str, Any]]:
    p = demo_dir / f"demo_{name}" / cond / f"ep{ep}_r0" / "trace.json"
    return json.loads(p.read_text()) if p.exists() else None


def _diff_html(orig: str, comp: str, limit: int = 900) -> str:
    import difflib
    a, b = orig.split(), comp.split()
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            out.append(E(" ".join(a[i1:i2])))
        if op in ("delete", "replace"):
            out.append(f'<span class="rm">{E(" ".join(a[i1:i2]))}</span>')
        if op in ("insert", "replace"):
            out.append(f'<span class="st">{E(" ".join(b[j1:j2]))}</span>')
    s = " ".join(out)
    return s


def explanation_page(D: Dict[str, Any]) -> str:
    env, er, recs, cases = D["env"], D["env_run"], D["recs"], D["cases"]
    cfg = er.get("config", {})
    demo = D["demo"]
    parts = []
    parts.append(f'''<header class="hero"><div style="display:flex;flex-direction:column;gap:16px">
<div class="eyebrow">How Sieve works · methods</div>
<h1>Route every segment of a prompt into <mark>one of three lanes</mark>.</h1>
<p class="lede">A chat model has one fixed context window, shared by what you send and what it writes back. Real conversations waste that window
in three ways: verbose text, static text resent every turn, and bulky material kept in full when it is rarely needed.
Sieve sits between the chat and the model, splits each request into segments, and gives each segment the cheapest treatment that keeps the answer intact.</p></div>
<div class="slab yellow"><h3>Claim under test</h3><p style="margin:8px 0 0">A per-segment routing policy across compress / cache / externalize reduces prompt tokens and
increases context headroom with accuracy preserved, while a ledger/epoch mechanism controls the KV-prefix invalidation that compression causes.</p></div></header>''')
    parts.append('''<section id="lanes"><div class="head"><span class="eyebrow">The three lanes</span><h2>One lane per segment</h2></div><div class="grid3">
<div class="slab"><span class="pill">COMPRESS</span><h3 style="margin-top:10px">Shorten what must be sent</h3><p>LLMLingua-2 drops low-information tokens on the CPU.
The keep-ratio is chosen per segment from 0.3, 0.5, 0.7, 0.9 or 1.0. A rule-based verifier checks that numbers, IDs, dates, names, and the words around them, survived;
if not, the segment is re-compressed at a gentler ratio.</p></div>
<div class="slab"><span class="pill">CACHE</span><h3 style="margin-top:10px">Keep static text byte-identical</h3><p>System prompts, pinned documents and exact repeats are never re-compressed.
They sit at the front of the prompt so the server's KV cache can skip them. A document pasted twice becomes a one-line back-reference.</p></div>
<div class="slab"><span class="pill y">EXTERNALIZE</span><h3 style="margin-top:10px">Move bulky, old text out</h3><p>The segment is replaced by a short deterministic stub with a summary and up to five key entities.
The full text sits in a store. The model can read it back through an <code>expand_context</code> tool call, or Sieve prefetches it when it clearly matches the question.</p></div></div></section>''')
    parts.append('''<section id="pipeline"><div class="head"><span class="eyebrow">Per request</span><h2>The pipeline</h2></div>
<div class="flow"><div class="step">Canonicalize<small>JSON minified, nulls dropped, repeated log lines and CSV rows collapsed with ×N</small></div><div class="arrow">→</div>
<div class="step">Segment<small>150–400-token chunks on paragraph boundaries; sha256 ids; exact and near-duplicate detection per session</small></div><div class="arrow">→</div>
<div class="step">Rules<small>current question untouched; system → cache; tags; repeats → back-reference; tiny → passthrough</small></div><div class="arrow">→</div>
<div class="step">Classifier<small>gradient boosting on 16 features; minimum expected cost; low confidence → compress 0.9</small></div><div class="arrow">→</div>
<div class="step">Ledger + budget<small>replay earlier strings verbatim; re-route only in an epoch; tighten the lowest-priority segments first</small></div><div class="arrow">→</div>
<div class="step">Layout + tools<small>system → pinned → history in order → question; prefetch and stub hint go in the final message only</small></div></div>
<p class="note">Rules always beat the classifier. Everything Sieve adds (stubs, the tool definition, the system note, the tail hint) is counted in the token totals.</p></section>''')
    # real examples from demo traces
    ex = []
    t = _demo_trace(demo, "compress")
    if t:
        segs = [s for s in t["segments"] if s["emitted"]["lane"] == "compress" and s["emitted"]["ratio"] < 1 and s["kind"] == "doc"]
        if segs:
            s = max(segs, key=lambda s: s["tokens_before"] - s["tokens_after"])
            ex.append(f'''<div class="slab"><span class="pill">COMPRESS · real trace</span><h3 style="margin-top:10px">Segment {E(s["key"])}: {s["tokens_before"]} → {s["tokens_after"]} tokens at ratio {s["emitted"]["ratio"]}</h3>
<p>Router: <code>{E(s["decision"]["source"])}</code>, confidence {s["decision"]["confidence"]}. Verifier: recall {f((s["emitted"]["verifier"] or {}).get("recall"), 2)},
context recall {f((s["emitted"]["verifier"] or {}).get("context_recall"), 2)}, retries {(s["emitted"]["verifier"] or {}).get("retries")}.</p>
<pre class="code">{E(s["emitted"]["text"])}</pre></div>''')
    t = _demo_trace(demo, "cache", ep=1)
    if t:
        br = [s for s in t["segments"] if s["decision"].get("backref")]
        if br:
            s = br[0]
            ex.append(f'''<div class="slab"><span class="pill">CACHE · real trace</span><h3 style="margin-top:10px">Re-pasted document → back-reference</h3>
<p>Segment {E(s["key"])} ({s["tokens_before"]} tokens) is byte-identical to one already in the prompt. Rule <code>{E(s["decision"]["source"])}</code> emits:</p>
<pre class="code"><span class="st">{E(s["emitted"]["text"])}</span></pre><p>{s["tokens_after"]} tokens instead of {s["tokens_before"]}.</p></div>''')
    t = _demo_trace(demo, "externalize", ep=1)
    if t:
        st = [s for s in t["segments"] if s["emitted"]["lane"] == "externalize"]
        tm = t.get("tool_messages") or []
        m = t["metrics"]
        rp = demo / "demo_externalize" / "sieve_full" / "ep1_r0" / "response.txt"
        m = dict(m, answer=rp.read_text().strip() if rp.exists() else "")
        if st:
            s = st[0]
            ex.append(f'''<div class="slab"><span class="pill y">EXTERNALIZE · real trace</span><h3 style="margin-top:10px">Stub, then expansion on demand</h3>
<p>Segment {E(s["key"])} ({s["tokens_before"]} tokens) became this stub:</p><pre class="code"><span class="st">{E(s["emitted"]["text"])}</span></pre>
<p>For the question that needed it, the model made <b>{m.get("tool_rounds", 0)}</b> tool round(s): {E(", ".join(x["seg"] for x in m.get("expansions", [])) or "none")}.
Its answer: “{E(m.get("answer", "") if "answer" in m else "")}”.</p></div>''')
    if ex:
        parts.append(f'<section id="examples"><div class="head"><span class="eyebrow">From the demo runs on the real model</span><h2>What the lanes look like in practice</h2></div><div class="grid3" style="align-items:start">{"".join(ex)}</div></section>')
    # ledger
    ct = env.get("cachetest") or []
    ct_bars = svg_hbars([{"label": r["label"].split(")")[0] + ")", "v": 100 * (r.get("reuse_pct_timing") or 0),
                          "cls": "y" if ("memo" in r["label"] or "same prefix" in r["label"]) else ""} for r in ct], 0, 100,
                        lambda v: f"{v:.0f}", unit="%", width=520, title="cachetest") if ct else ""
    parts.append(f'''<section id="ledger"><div class="head"><span class="eyebrow">The KV-cache problem</span><h2>Compression and caching fight. The ledger referees.</h2></div>
<div class="grid2" style="align-items:start"><div class="prose">
<p>A local server like Ollama keeps the attention state (KV cache) of the last prompt. If the next prompt starts with the same tokens, it skips recomputing them.
Compression breaks this: if turn 5 compresses an old message differently from turn 4, every token after that point has to be prefilled again.</p>
<p>Sieve's <b>ledger</b> records the exact string emitted for every segment. On later turns it replays those strings byte-for-byte and routes only the new segments.
Compression is memoized on <code>(content hash, ratio, compressor version)</code>, so the same input always yields the same string.</p>
<p>A full re-route, an <b>epoch</b>, happens only when the projected prompt plus the {cfg.get("reserve_output_tokens", 256)}-token output reserve would pass
{int(100 * cfg.get("epoch_trigger_fraction", 0.8))}% of the {cfg.get("num_ctx", 4096)}-token window. The allocator then aims at {int(100 * cfg.get("epoch_hysteresis", 0.85))}% of that line, so the next turns do not re-trigger it.</p></div>
<figure class="slab">{ct_bars}<figcaption>Measured on this machine (cachetest). Share of the second request's prefill the server skipped:
(a) same prefix, (b) one word changed at 10% depth, (c1) re-compressed at a new ratio, (c2) the memoized compressed string.</figcaption></figure></div>
<p class="note"><b>Measurement detail.</b> On Ollama {E(str(env.get("ollama_version", "")))} (llama-server backend), <code>prompt_eval_count</code> reports the whole prompt even on a cache hit.
Sieve therefore estimates reuse from prefill <i>time</i>: it fits cold prefill time against prompt length from novel prompts, then takes reuse = 1 − observed ÷ predicted cold time.
The cachetest above validates the method: 95% reuse when nothing changed, about 8% when the change sits 10% of the way in.</p></section>''')
    # classifier
    parts.append(f'''<section id="router"><div class="head"><span class="eyebrow">The router</span><h2>Rules first, then a cost-aware classifier</h2></div>
<div class="grid2" style="align-items:start"><div class="prose">
<p>Each segment not claimed by a rule gets 16 features: the original six (is_system, is_last_turn, token count, recency in turns, repeat count, similarity to the current
question by MiniLM) plus kind one-hots, digit density, near-duplicate score and prior expansion count.</p>
<p>A gradient-boosting model gives lane probabilities. Sieve picks the lane with the lowest <b>expected cost</b>, not the most likely one.
Hiding a segment that is needed costs {cfg.get("cost_matrix", {}).get("needed_externalized", 5)}; other mistakes cost 1. A random forest chooses the
compression ratio, snapped to a bucket and floored per kind (documents 0.3, chat 0.4, instructions 0.9).</p>
<p>Training data comes from {len(cfg.get("train_seeds", []))} generator seeds that are disjoint from the benchmark seed. Labels come from the workload's ground truth (oracle) or from the
hand-written heuristic. Splits are by conversation, never by segment.</p></div>
<div class="scroll"><table><thead><tr><th>true \\ predicted</th><th>cache</th><th>compress</th><th>externalize</th></tr></thead><tbody>
<tr><td>cache</td><td>0</td><td>1</td><td>1</td></tr><tr><td>compress (needed)</td><td>1</td><td>0</td><td><b>{cfg.get("cost_matrix", {}).get("needed_externalized", 5)}</b></td></tr>
<tr><td>externalize</td><td>1</td><td>{cfg.get("cost_matrix", {}).get("needless_compress_of_externalizable", 1)}</td><td>0</td></tr></tbody></table></div></div></section>''')
    # experimental design
    fam_n = defaultdict(lambda: [0, 0])
    for c in cases:
        fam_n[c["family"]][0] += 1
        fam_n[c["family"]][1] += len(c["eval_points"])
    frows = "".join(f'<tr><td>{fa}</td><td class="l">{E(FAM_DESC.get(fa, ""))}</td><td>{n[0]}</td><td>{n[1]}</td></tr>' for fa, n in sorted(fam_n.items()))
    crow = "".join(f'<tr><td>{E(COND_LABEL[c])}</td><td class="l wrap">{E(d)}</td></tr>' for c, d in [
        ("baseline", "Raw prompt, append-only history."),
        ("truncate", "Drop the oldest whole messages until the prompt fits C4's token count for the same question."),
        ("uniform_lingua", "One LLMLingua-2 rate on every unprotected segment, matched to C4's token count."),
        ("sieve_rules", "Hard rules plus the hand-written heuristic labeller; no ML, no verifier."),
        ("sieve_full", "Rules + classifier + ledger/epochs + verifier + tool loop and prefetch."),
        ("sieve_nomemo", "C4 re-routed and re-compressed every turn, with no ledger and no memo."),
        ("reference_bigctx", "Raw prompt with an 8,192-token window: an upper bound for accuracy, run in its own pass.")])
    qt = defaultdict(int)
    for c in cases:
        for ep in c["eval_points"]:
            qt[ep["question_type"]] += 1
    qrow = ", ".join(f"{E(k)} ({v})" for k, v in sorted(qt.items()))
    parts.append(f'''<section id="design"><div class="head"><span class="eyebrow">Experimental design</span><h2>How the benchmark is built</h2></div>
<div class="grid2" style="align-items:start"><div class="prose">
<p><b>Workloads.</b> {len(cases)} conversations generated from seed {E(str(er.get("seed", 7)))}. Every entity is invented (codenames, people, figures), so the model cannot answer from memory.
Filler text comes from a template grammar that contains no digits and no invented names, so a gold fact can only be found in the messages that ground truth says hold it.</p>
<p><b>Evaluation points.</b> Each conversation has 1–4 questions. After a question is answered, the scripted gold answer is appended, so a wrong answer never leaks into later turns.
Question types: {qrow}.</p>
<p><b>Protocol.</b> One full pass per condition, never interleaved. The model is unloaded and warmed up before each pass. The first question of every conversation is <i>cold</i>
(a novel prompt after a fresh load); later ones are <i>warm</i>. Temperature 0, seed 0, <code>think: false</code>, num_ctx {cfg.get("num_ctx", 4096)}, num_predict {cfg.get("num_predict", 64)}.</p>
<p><b>Scoring.</b> Normalized containment for names, exact integer match for numbers, all of day, month and year for dates, first yes/no word for yes/no questions, and a refusal
pattern for unanswerable ones. Differences against C0 are paired by question, with 95% bootstrap intervals (2,000 resamples).</p></div>
<div style="display:flex;flex-direction:column;gap:18px;min-width:0"><div class="scroll"><table><thead><tr><th>family</th><th class="l">stresses</th><th>chats</th><th>questions</th></tr></thead><tbody>{frows}</tbody></table></div>
<div class="scroll"><table><thead><tr><th>condition</th><th class="l">what it does</th></tr></thead><tbody>{crow}</tbody></table></div></div></div></section>''')
    parts.append(f'''<section id="env"><div class="head"><span class="eyebrow">Setup and deviations</span><h2>What ran, and where it differs from the plan</h2></div>
<div class="grid2" style="align-items:start"><div class="meta">
<div><span>machine</span><b>{E(str(env.get("cpu", "")))}, {E(str(env.get("mem_gb", "")))} GB unified</b></div>
<div><span>model</span><b>{E(str(env.get("model", "")))} ({E(str(env.get("parameter_size", "")))}, {E(str(env.get("quantization", "")))})</b></div>
<div><span>GPU share (/api/ps)</span><b>{f((env.get("ps") or {}).get("gpu_pct", 0), 0, pct=True)}%</b></div>
<div><span>Ollama</span><b>{E(str(env.get("ollama_version", "")))}</b></div>
<div><span>compressor</span><b>{E(str(env.get("compressor", "")).split("|")[-1].split("/")[-1])}</b></div>
<div><span>similarity</span><b>{E(str(env.get("similarity", "")))}</b></div>
<div><span>cold prefill</span><b>{f(env.get("prefill_tps"), 0)} tok/s</b></div>
<div><span>decode</span><b>{f(env.get("decode_tps"), 1)} tok/s</b></div>
<div><span>tokenizer vs Ollama</span><b>{env.get("calib_long_tokens_tokenizer")} vs {env.get("calib_long_prompt_eval_count")} tokens</b></div>
<div><span>tool round-trip</span><b>{"works" if env.get("tool_call_supported") else "failed"}</b></div></div>
<div class="prose"><ul class="tight">
<li>The plan targeted Qwen3.5-9B on a 6 GB RTX 4050 with partial CPU offload. This run used qwen3:14b on an Apple M5 Pro, fully on the GPU.</li>
<li>Prefix reuse is timing-derived, because this Ollama build does not lower <code>prompt_eval_count</code> on a cache hit.</li>
<li>With thinking off, the model rarely calls tools unprompted. Sieve adds one line to the final message listing up to six stubs and their keys. It sits in the tail, so the cached prefix is untouched.</li>
<li>The verifier also checks the words around each fact. LLMLingua was seen keeping a name while dropping “product lead”.</li>
<li>All of these, and about thirty smaller choices, are listed in <code>DECISIONS.md</code>.</li></ul></div></div></section>''')
    parts.append('''<section id="limits"><div class="head"><span class="eyebrow">Read the results with these in mind</span><h2>Limitations</h2></div>
<div class="slab yellow"><ul class="tight">
<li>Synthetic, small-scale benchmark on one model and one machine. The results describe this suite, not real conversations.</li>
<li>Stub summaries are extractive; attachments are text only (OCR and audio are disabled stubs).</li>
<li>Oracle labels are partly a function of the features, so classifier F1 is not evidence on its own.</li>
<li>Warm-cache baselines are strong: a raw prompt whose prefix is already cached is cheap to extend.</li>
<li>The partial-offload regime of the original 6 GB target was not exercised.</li></ul></div>
<div class="prose"><p>Reproduce with <code>python -m sieve selfcheck</code>, <code>cachetest</code>, <code>train --eval</code>, <code>bench --tier standard --conditions all --with-reference --ablations</code>,
<code>breakeven</code> and <code>html</code>. The no-GPU path is <code>python -m sieve --mock bench --tier smoke</code>.</p></div></section>''')
    return page("How Sieve Works", "exp", "".join(parts), f"run {D['run']}")


def write_html(runs_dir: Path, run: str, out_dir: Path, full_doc: bool = True) -> List[Path]:
    global FULL_DOC
    FULL_DOC = full_doc
    D = load_all(Path(runs_dir), run)
    out_dir.mkdir(parents=True, exist_ok=True)
    a = out_dir / "results.html"
    b = out_dir / "explanation.html"
    a.write_text(results_page(D))
    b.write_text(explanation_page(D))
    return [b, a]

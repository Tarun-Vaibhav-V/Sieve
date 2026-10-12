"""Rich terminal rendering: segment tables, waterfalls, prompt diffs, ledger views, sparklines,
plus optional matplotlib PNGs for `report`."""
from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()
LANE_STYLE = {"compress": "blue", "cache": "green", "externalize": "dark_orange", "passthrough": "grey58"}
FEAT_SHORT = [("is_system", "sys"), ("is_last_turn", "last"), ("token_count_norm", "tok"), ("recency_norm", "rec"),
              ("repeat_count", "rep"), ("query_similarity", "qsim"), ("digit_density", "dig"),
              ("near_dup_score", "ndup"), ("prior_expand_count", "pexp")]
SPARK = "▁▂▃▄▅▆▇█"


def lane_label(d) -> str:
    if d.lane == "compress" and d.ratio >= 1.0 and not d.backref:
        return "passthrough"
    return d.lane


def segment_table(plan: Dict[str, Any], title: str = "Segment decisions") -> Table:
    t = Table(title=title, show_lines=False, header_style="bold", expand=False)
    for col in ["seg", "role", "kind", "tok→", "→tok"] + [s for _, s in FEAT_SHORT] + \
               ["p(ca/co/ex)", "lane", "ratio", "source", "conf"]:
        t.add_column(col, no_wrap=True)
    for s in plan["segments"]:
        d, e = plan["decisions"][s.key], plan["emitted"][s.key]
        lab = "dropped" if e.dropped else lane_label(d)
        style = LANE_STYLE.get(lab, "red")
        p = d.probs or {}
        probs = "/".join(f"{p.get(k, 0):.2f}" for k in ("cache", "compress", "externalize")) if p else "-"
        src = d.source.replace("rule:", "R:")
        if d.backref:
            src += "↩"
        t.add_row(s.key, s.role[:4], s.kind[:6], str(s.tokens), str(0 if e.dropped else e.tokens),
                  *[f"{s.features.get(k, 0):.2f}" if isinstance(s.features.get(k, 0), float) else str(s.features.get(k))
                    for k, _ in FEAT_SHORT],
                  probs, lab, f"{e.ratio:.1f}", src[:22], f"{d.confidence:.2f}", style=style)
    return t


def waterfall(plan: Dict[str, Any], width: int = 46) -> Text:
    segs = plan["segments"]
    mx = max([s.tokens for s in segs] + [1])
    out = Text()
    for s in segs:
        e = plan["emitted"][s.key]
        after = 0 if e.dropped else e.tokens
        lab = lane_label(plan["decisions"][s.key])
        b = int(round(width * s.tokens / mx))
        a = int(round(width * after / mx))
        out.append(f"{s.key:>6} ", style="bold")
        out.append("█" * a, style=LANE_STYLE.get(lab, "red"))
        out.append("░" * max(0, b - a), style="grey35")
        out.append(f" {s.tokens}→{after}\n")
    return out


def prompt_diff_text(raw_msgs, final_msgs) -> str:
    a = "\n".join(f"<{m['role']}> {m.get('content') or ''}" for m in raw_msgs).split()
    b = "\n".join(f"<{m['role']}> {m.get('content') or ''}" for m in final_msgs).split()
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            out.append(" ".join(a[i1:i2]))
        if op in ("delete", "replace"):
            out.append("[-" + " ".join(a[i1:i2]) + "-]")
        if op in ("insert", "replace"):
            out.append("{+" + " ".join(b[j1:j2]) + "+}")
    return " ".join(out)


def prompt_diff(raw_msgs, final_msgs, max_chars: int = 6000) -> Text:
    """Removed words red/strikethrough, stubs orange, other inserted text cyan."""
    a = "\n".join(f"<{m['role']}> {m.get('content') or ''}" for m in raw_msgs).split(" ")
    b = "\n".join(f"<{m['role']}> {m.get('content') or ''}" for m in final_msgs).split(" ")
    out = Text()
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            out.append(" ".join(a[i1:i2]) + " ")
        if op in ("delete", "replace"):
            out.append(" ".join(a[i1:i2]) + " ", style="red strike")
        if op in ("insert", "replace"):
            ins = " ".join(b[j1:j2])
            out.append(ins + " ", style="bold dark_orange" if ("[ref:" in ins or "[repeat:" in ins or "expand_context" in ins) else "cyan")
        if len(out) > max_chars:
            out.append("\n… (diff truncated; full diff in diff.txt)", style="italic")
            break
    return out


def sparkline(xs: List[Optional[float]]) -> str:
    v = [x for x in xs if x is not None]
    if not v:
        return ""
    lo, hi = min(v), max(v)
    return "".join(SPARK[int((x - lo) / (hi - lo + 1e-12) * (len(SPARK) - 1))] if x is not None else " " for x in xs)


def result_panel(case_q: str, gold, rows: List[Dict[str, Any]]) -> Table:
    t = Table(title=f"Q: {case_q}\ngold: {gold}", header_style="bold")
    cols = ["condition", "answer", "correct", "tokens", "Δtok", "TTFT s", "total s", "processed", "reuse %",
            "VRAM MB", "tools"]
    for c in cols:
        t.add_column(c, overflow="fold", max_width=48 if c == "answer" else None)
    base = rows[0]
    for r in rows:
        t.add_row(r["condition"], r["answer"][:200], "✔" if r["correct"] else "✘", str(r["tokens_final_total"]),
                  f"{r['tokens_final_total'] - base['tokens_final_total']:+d}",
                  f"{r['ttft_s']:.2f}" if r["ttft_s"] is not None else "-", f"{r['total_s']:.2f}",
                  str(r["prompt_eval_count"]), f"{100 * r['prefix_reuse_pct']:.0f}",
                  f"{(r.get('vram_peak_mb') or 0):.0f}", str(r["tool_rounds"]),
                  style="green" if r["correct"] else "red")
    return t


def ledger_view(history: List[Dict[str, Any]]) -> Table:
    t = Table(title="Ledger (per evaluation turn)", header_style="bold")
    for c in ["turn", "replayed", "new", "epoch", "reason", "epoch_id", "first changed msg", "rebuild tokens"]:
        t.add_column(c)
    for h in history:
        t.add_row(str(h["turn"]), str(h["replayed"]), str(h["new"]), "●" if h["epoch"] else "", str(h["epoch_reason"] or ""),
                  str(h["epoch_id"]), str(h["first_changed_msg"]), str(h["rebuild_tokens"]),
                  style="yellow" if h["epoch"] else None)
    return t


def condition_table(agg: Dict[str, Any], title: str = "By condition") -> Table:
    t = Table(title=title, header_style="bold")
    cols = [("n", "n", "{:.0f}"), ("acc", "accuracy", "{:.3f}"), ("tok red %", "token_reduction_pct", "{:.1f}"),
            ("prompt tok", "tokens_final_mean", "{:.0f}"), ("TTFT med", "ttft_median", "{:.2f}"),
            ("TTFT p95", "ttft_p95", "{:.2f}"), ("total med", "total_median", "{:.2f}"),
            ("processed", "prompt_eval_count_mean", "{:.0f}"), ("reuse %", "prefix_reuse_pct_mean", "{:.1%}"),
            ("atom rec", "atom_recall_mean", "{:.3f}"), ("expand rate", "expand_rate", "{:.2f}"),
            ("overhead ms", "sieve_overhead_ms_mean", "{:.0f}"), ("headroom", "output_headroom_mean", "{:.0f}")]
    t.add_column("condition")
    for c, _, _ in cols:
        t.add_column(c, justify="right")
    for cond, s in agg.items():
        t.add_row(cond, *[(f.format(s[k]) if s.get(k) is not None else "-") for _, k, f in cols])
    return t


def confusion_table(cm, labels, title) -> Table:
    t = Table(title=title, header_style="bold")
    t.add_column("true \\ pred")
    for l in labels:
        t.add_column(l, justify="right")
    for l, row in zip(labels, cm):
        t.add_row(l, *[str(v) for v in row])
    return t


# ---------------- optional PNGs ----------------

def write_pngs(recs: List[Dict[str, Any]], agg: Dict[str, Any], out_dir: Path, clf_eval: Optional[Dict] = None) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    conds = list(agg["by_condition"].keys())

    def bar(key, fname, ylabel):
        fig, ax = plt.subplots(figsize=(8, 3.5))
        vals = [agg["by_condition"][c].get(key) or 0 for c in conds]
        ax.bar(conds, vals, color="#222")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=130)
        plt.close(fig)
        files.append(fname)

    bar("accuracy", "accuracy_by_condition.png", "accuracy")
    bar("tokens_final_mean", "tokens_by_condition.png", "mean prompt tokens")
    lanes = {c: {"compress": 0, "cache": 0, "externalize": 0} for c in conds}
    for r in recs:
        for k, v in (r.get("lanes") or {}).items():
            lanes[r["condition"]][k] = lanes[r["condition"]].get(k, 0) + v
    fig, ax = plt.subplots(figsize=(8, 3.5))
    bottom = [0] * len(conds)
    for lane, col in (("compress", "#3b6fd4"), ("cache", "#2e9c48"), ("externalize", "#e8890c")):
        v = [lanes[c][lane] for c in conds]
        ax.bar(conds, v, bottom=bottom, label=lane, color=col)
        bottom = [a + b for a, b in zip(bottom, v)]
    ax.legend()
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / "lane_distribution.png", dpi=130)
    plt.close(fig)
    files.append("lane_distribution.png")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for c in conds:
        by_ep = {}
        for r in recs:
            if r["condition"] == c and r["ttft_s"] is not None:
                by_ep.setdefault(r["eval_point"], []).append(r["ttft_s"])
        xs = sorted(by_ep)
        ax.plot(xs, [sorted(by_ep[x])[len(by_ep[x]) // 2] for x in xs], marker="o", label=c)
    ax.set_xlabel("evaluation point (turn order)")
    ax.set_ylabel("median TTFT (s)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "ttft_per_turn.png", dpi=130)
    plt.close(fig)
    files.append("ttft_per_turn.png")
    if clf_eval:
        cm = clf_eval["routers"]["clf_oracle"]["confusion"]
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.imshow(cm, cmap="Greys")
        labs = ["cache", "compress", "externalize"]
        ax.set_xticks(range(3), labs, rotation=30)
        ax.set_yticks(range(3), labs)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i][j], ha="center", va="center", color="#e8b30c")
        ax.set_xlabel("predicted")
        ax.set_ylabel("oracle")
        fig.tight_layout()
        fig.savefig(out_dir / "confusion_matrix.png", dpi=130)
        plt.close(fig)
        files.append("confusion_matrix.png")
    return files

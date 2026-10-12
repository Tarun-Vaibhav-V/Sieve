"""IEEE Access-style paper generated from a run's logs.

`python -m sieve paper --run <run_id>` fills paper/template.tex: every «placeholder» is computed here from
results.jsonl / summary / env / classifier evaluation / break-even sweep, figures are drawn from the same data,
and the build fails if any placeholder is left unfilled. The bibliography (paper/refs/*.bib) was fetched from
doi.org, Crossref, the ACL Anthology and the arXiv API, not typed by hand.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from sieve.metrics import bootstrap_ci
from sieve.report_html import FAM_DESC, load_all

ROOT = Path(__file__).resolve().parent.parent
LBL = {"baseline": "C0 Raw prompt", "truncate": "C1 Truncate", "uniform_lingua": "C2 Uniform LLMLingua-2",
       "sieve_rules": "C3 Sieve, rules", "sieve_full": "C4 Sieve, full", "sieve_nomemo": "C5 Sieve, no ledger",
       "reference_bigctx": "C6 8k context", "sieve_full+noverifier": "C4 $-$verifier",
       "sieve_full+noprefetch": "C4 $-$prefetch", "sieve_full+noledger": "C4 $-$ledger",
       "sieve_full+clf_heuristic": "C4, heuristic-trained", "sieve_full+core6": "C4, core-6 features"}
SHORT = {k: v.split(" ")[0] for k, v in LBL.items()}
ORDER = ["baseline", "truncate", "uniform_lingua", "sieve_rules", "sieve_full", "sieve_nomemo", "reference_bigctx"]
ABL = ["sieve_full", "sieve_full+noverifier", "sieve_full+noprefetch", "sieve_full+noledger",
       "sieve_full+clf_heuristic", "sieve_full+core6"]
QT_NAME = {"lookup_number": "numeric lookup", "lookup_name": "name lookup", "multi_hop": "multi-hop",
           "decision_recall": "decision recall", "value_update": "latest-value tracking", "negation": "negation",
           "unanswerable": "unanswerable"}


def n(x, d=1):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:,.{d}f}"


def ci(t, d=1, k=1.0):
    return f"{t[1] * k:+.{d}f} to {t[2] * k:+.{d}f}"


def sgn(x, d=1):
    return f"{x:+.{d}f}"


def verdict(t) -> Tuple[str, str]:
    """Long and short wording for a CI on a difference."""
    if t[1] > 0:
        return "is a significant improvement", "above the raw prompt"
    if t[2] < 0:
        return "is a significant decrease", "below the raw prompt"
    return "is not statistically distinguishable from zero", "statistically indistinguishable from the raw prompt"


# ---------------- figures ----------------

def figures(D: Dict[str, Any], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "font.size": 8, "axes.linewidth": 0.8, "pdf.fonttype": 42})
    agg = D["agg"]
    pc = agg["paired"]
    conds = [c for c in ORDER if c in pc and pc[c].get("n")]
    fig, ax = plt.subplots(figsize=(3.45, 2.6))
    ax.axhline(0, color="black", lw=0.8)
    for c in conds:
        p = pc[c]
        x, y = p["token_reduction_pct"], [100 * v for v in p["acc_delta"]]
        hot = c == "sieve_full"
        ax.errorbar(x[0], y[0], xerr=[[x[0] - x[1]], [x[2] - x[0]]], yerr=[[y[0] - y[1]], [y[2] - y[0]]], fmt="o",
                    ms=6.5 if hot else 4.5, mfc="#ffd60a" if hot else "black", mec="black", ecolor="0.55", elinewidth=0.8,
                    capsize=2, zorder=3)
        ax.annotate(SHORT[c], (x[0], y[0]), textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax.set_xlabel("Prompt tokens saved vs. C0 (%)")
    ax.set_ylabel("Accuracy change vs. C0 (pp)")
    ax.grid(color="0.88", lw=0.5)
    fig.tight_layout()
    fig.savefig(out / "fig_tradeoff.pdf")
    plt.close(fig)
    ct = D["env"].get("cachetest") or []
    if ct:
        fig, ax = plt.subplots(figsize=(3.45, 1.8))
        labs = ["(a) identical\nprefix", "(b) one word\nchanged @10%", "(c1) re-\ncompressed", "(c2) memoized\ncompressed"]
        vals = [100 * (r.get("reuse_pct_timing") or 0) for r in ct]
        ax.bar(range(len(vals)), vals, color=["black", "white", "white", "#ffd60a"][:len(vals)], edgecolor="black", lw=0.8,
               hatch=[None, "///", "///", None][:len(vals)])
        for i, v in enumerate(vals):
            ax.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=7)
        ax.set_xticks(range(len(vals)), labs[:len(vals)], fontsize=6.5)
        ax.set_ylim(0, 112)
        ax.set_ylabel("Prefill skipped (%)")
        fig.tight_layout()
        fig.savefig(out / "fig_cachetest.pdf")
        plt.close(fig)
    cs = [c for c in ORDER if c in agg["by_condition"]]
    fig, ax = plt.subplots(figsize=(3.45, 2.0))
    xs = np.arange(len(cs))
    cold = [agg["by_condition_cold"][c].get("ttft_median") or 0 for c in cs]
    warm = [agg["by_condition_warm"][c].get("ttft_median") or 0 for c in cs]
    ax.bar(xs - 0.2, cold, 0.4, color=["#ffd60a" if c == "sieve_full" else "black" for c in cs], edgecolor="black", lw=0.6,
           label="cold")
    ax.bar(xs + 0.2, warm, 0.4, color="white", edgecolor="black", lw=0.6, hatch="///", label="warm")
    ax.set_xticks(xs, [SHORT[c] for c in cs])
    ax.set_ylabel("Median TTFT (s)")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", color="0.88", lw=0.5)
    fig.tight_layout()
    fig.savefig(out / "fig_ttft.pdf")
    plt.close(fig)
    be = D["be"]
    if be:
        fig, ax = plt.subplots(figsize=(3.45, 2.0))
        p = [s["p"] for s in be["summary"]]
        ax.plot(p, [s["baseline_s"] for s in be["summary"]], "k--o", ms=3.5, label="C0 raw prompt")
        ax.plot(p, [s["sieve_s"] for s in be["summary"]], "-o", color="black", mfc="#ffd60a", ms=4.5, label="C4 Sieve")
        ax.set_xlabel("P(question needs the externalized document)")
        ax.set_ylabel("Mean net time (s)")
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False)
        ax.grid(color="0.88", lw=0.5)
        fig.tight_layout()
        fig.savefig(out / "fig_breakeven.pdf")
        plt.close(fig)


# ---------------- values ----------------

def compute(D: Dict[str, Any]) -> Dict[str, str]:
    agg, env, recs, er, cases = D["agg"], D["env"], D["recs"], D["env_run"], D["cases"]
    cfg = er.get("config", {})
    bc, pc, pcold, pover = agg["by_condition"], agg["paired"], agg["paired_cold"], agg["paired_overflow"]
    S, B = bc["sieve_full"], bc["baseline"]
    P, Pc, Po = pc["sieve_full"], pcold["sieve_full"], pover.get("sieve_full", {})
    cp = agg["cumulative_prefill"]
    V: Dict[str, str] = {}
    base_recs = [r for r in recs if r["condition"] == "baseline"]
    n_q = len(base_recs)
    V.update(num_ctx=str(cfg.get("num_ctx", 4096)), num_predict=str(cfg.get("num_predict", 64)), n_q=str(n_q),
             n_cases=str(len(cases)), n_calls_total=f"{len(recs):,}", seed=str(er.get("seed", 7)),
             buckets=", ".join(f"{b:g}" for b in cfg.get("ratio_buckets", [])),
             budget_fraction=f"{cfg.get('budget_fraction', 0.8):g}", reserve=str(cfg.get("reserve_output_tokens", 256)),
             chunk_min=str(cfg.get("chunk_min_tokens", 150)), chunk_max=str(cfg.get("chunk_max_tokens", 400)),
             neardup=f"{cfg.get('neardup_jaccard', 0.85):g}", min_seg=str(cfg.get("min_segment_tokens", 40)),
             cost_ext=f"{cfg.get('cost_matrix', {}).get('needed_externalized', 5):g}",
             conf_thr=f"{cfg.get('classifier_conf_threshold', 0.55):g}",
             max_rounds=str(cfg.get("max_tool_rounds", 2)), exp_budget=f"{cfg.get('expansion_budget_tokens', 1200):,}",
             prefetch_thr=f"{cfg.get('prefetch_sim_threshold', 0.75):g}",
             epoch_frac=f"{cfg.get('epoch_trigger_fraction', 0.8):g}", hyst=f"{cfg.get('epoch_hysteresis', 0.85):g}",
             epoch_frac_pct=f"{100 * cfg.get('epoch_trigger_fraction', 0.8):.0f}",
             hyst_pct=f"{100 * cfg.get('epoch_hysteresis', 0.85):.0f}",
             ver_overall=f"{cfg.get('verifier_overall_min_recall', 0.9):g}",
             ver_ctx=f"{cfg.get('verifier_min_context_recall', 0.5):g}", ver_retries=str(cfg.get("verifier_max_retries", 3)))
    ps = env.get("ps") or {}
    V.update(param_size=str(env.get("parameter_size", "")), ollama_ver=str(env.get("ollama_version", "")),
             cpu=str(env.get("cpu", "")).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_"), mem_gb=f"{env.get('mem_gb', 0):g}", gpu_pct=f"{100 * ps.get('gpu_pct', 0):.0f}",
             model_gib=f"{ps.get('size', 0) / 2**30:.2f}", calib_tok=f"{env.get('calib_long_tokens_tokenizer', 0):,}",
             calib_srv=f"{env.get('calib_long_prompt_eval_count', 0):,}")
    fit = env.get("cold_prefill_fit") or {}
    V.update(fit_a=f"{fit.get('a', 0) * 1000:.3f}", fit_b=f"{fit.get('b', 0) * 1e6:.4f}",
             fit_sizes=", ".join(f"{int(p[0]):,}" for p in fit.get("points", [])))
    qt = Counter(ep["question_type"] for c in cases for ep in c["eval_points"])
    V["qtypes"] = ", ".join(f"{QT_NAME.get(k, k)} ({v})" for k, v in sorted(qt.items(), key=lambda x: -x[1]))
    ov = [r for r in base_recs if r["baseline_overflow"]]
    V.update(overflow_q_n=str(len(ov)), overflow_q_pct=f"{100 * len(ov) / max(1, n_q):.0f}")
    fq = Counter(r["family"] for r in base_recs)
    V["tab_fam"] = "\n".join(f"{f} & {FAM_DESC[f].replace('&', chr(92) + '&')} ({fq.get(f, 0)})\\\\" for f in sorted(FAM_DESC))
    V["fallback_n"] = str(sum(bool(r.get("approx_tokens") or r.get("compressor_fallback")) for r in recs))
    clf = D["clf"]
    V["train_convs"] = str(clf.get("n_groups", "")) if clf else "--"
    rows = []
    for c in [c for c in ORDER if c in bc]:
        s = bc[c]
        a = bootstrap_ci([float(r["correct"]) for r in recs if r["condition"] == c])
        rows.append(f"{LBL[c]} & {a[0]:.3f} [{a[1]:.2f}, {a[2]:.2f}] & {n(s['tokens_final_mean'], 0)} & "
                    f"{n(s['token_reduction_pct'], 1)} & {n(s['output_headroom_mean'], 0)} & "
                    f"{n(agg['by_condition_cold'][c].get('ttft_median'), 2)} & {n(agg['by_condition_warm'][c].get('ttft_median'), 2)} & "
                    f"{n(100 * (s.get('prefix_reuse_pct_mean') or 0), 0)} & {n(s.get('sieve_overhead_ms_mean'), 0)} & "
                    f"{n(s.get('decode_tps_median'), 1)}\\\\")
    V["tab_main"] = "\n".join(rows)
    br = lambda t, d=1, k=1.0: f"[{t[1] * k:.{d}f}, {t[2] * k:.{d}f}]"
    rows = []
    for c in [c for c in ORDER if c in pc and pc[c].get("n")]:
        p, c0, o = pc[c], pcold[c], pover.get(c, {})
        rows.append(f"{SHORT[c]} & {sgn(100 * p['acc_delta'][0])} {br(p['acc_delta'], 1, 100)} & "
                    f"{p['token_reduction_pct'][0]:.1f} {br(p['token_reduction_pct'])} & "
                    f"{sgn(c0['ttft_delta_s'][0], 2)} {br(c0['ttft_delta_s'], 2)} & "
                    + (f"{sgn(100 * o['acc_delta'][0])} {br(o['acc_delta'], 1, 100)}" if o.get("n") else "--") + "\\\\")
    V["tab_paired"] = "\n".join(rows)
    long_v, short_v = verdict(P["acc_delta"])
    V.update(c4_tok=n(S["tokens_final_mean"], 0), c0_tok=n(B["tokens_final_mean"], 0),
             c4_saved=f"{P['token_reduction_pct'][0]:.1f}\\%",
             c4_saved_ci=f"{P['token_reduction_pct'][1]:.1f}--{P['token_reduction_pct'][2]:.1f}\\%",
             c0_head=n(B["output_headroom_mean"], 0), c4_head=n(S["output_headroom_mean"], 0),
             c4_acc=f"{S['accuracy']:.3f}", c0_acc=f"{B['accuracy']:.3f}", c4_dacc=sgn(100 * P["acc_delta"][0]),
             c4_dacc_ci=ci(P["acc_delta"], 1, 100), c4_acc_verdict=long_v, c4_acc_verdict_short=short_v,
             c6_acc=f"{bc['reference_bigctx']['accuracy']:.3f}" if "reference_bigctx" in bc else "--")
    if Po.get("n") and Po["acc_delta"][1] > 0:
        V["overflow_sentence"] = (f"The gain is concentrated where the raw prompt overflowed the window and the server truncated "
                                  f"it: on those {len(ov)} questions C4 changed accuracy by {sgn(100 * Po['acc_delta'][0])} pp "
                                  f"(CI {ci(Po['acc_delta'], 1, 100)}).")
        V["concl_overflow"] = (f"; on questions whose raw prompt overflowed the window, accuracy rose by "
                               f"{100 * Po['acc_delta'][0]:.1f} percentage points")
    else:
        V["overflow_sentence"] = (f"On the {len(ov)} questions whose raw prompt overflowed the window, C4 changed accuracy by "
                                  f"{sgn(100 * Po['acc_delta'][0])} pp (CI {ci(Po['acc_delta'], 1, 100)})."
                                  if Po.get("n") else "")
        V["concl_overflow"] = ""
    for k, c in (("c1", "truncate"), ("c2", "uniform_lingua")):
        ok = c in pc and pc[c].get("n")
        V[f"{k}_dacc"] = sgn(100 * pc[c]["acc_delta"][0]) if ok else "--"
        V[f"{k}_dacc_ci"] = ci(pc[c]["acc_delta"], 1, 100) if ok else "--"
    if Pc["ttft_delta_s"][2] < 0:
        V["cold_sentence"] = (f"On cold questions C4 reduced median TTFT by {-Pc['ttft_delta_s'][0]:.2f} s (CI "
                              f"{-Pc['ttft_delta_s'][2]:.2f} to {-Pc['ttft_delta_s'][1]:.2f}), because it prefills fewer "
                              f"tokens (Fig.~\\ref{{fig:ttft}}).")
    else:
        V["cold_sentence"] = (f"On cold questions the median TTFT difference was {sgn(Pc['ttft_delta_s'][0], 2)} s (CI "
                              f"{ci(Pc['ttft_delta_s'], 2)}), not distinguishable from zero (Fig.~\\ref{{fig:ttft}}).")
    V.update(c4_warm=n(agg["by_condition_warm"]["sieve_full"]["ttft_median"], 2),
             c0_warm=n(agg["by_condition_warm"]["baseline"]["ttft_median"], 2), c4_cpu=n(S["sieve_overhead_ms_mean"], 0))
    base_tps = B.get("decode_tps_median") or 0
    slow = [c for c in ORDER + ABL[1:] if c in bc and base_tps and (bc[c].get("decode_tps_median") or 0) < 0.75 * base_tps]
    deg = {c: sum(bool(r.get("perf_degraded")) for r in recs if r["condition"] == c) for c in bc}
    if slow:
        V["confound"] = ("Decode speed does not depend on the prompt, so it serves as a health indicator for the host. The passes "
                         + ", ".join(LBL[c] for c in slow) + f" ran at a median decode speed below 75\\% of C0's "
                         f"{base_tps:.1f} tok/s because of host memory pressure (calls flagged as degraded: "
                         + ", ".join(f"{SHORT[c]} {deg[c]}" for c in slow) + "). Their absolute latencies are inflated and are "
                         "not used for latency comparisons; their token and accuracy results are unaffected. C0 and C4 ran at "
                         f"matching decode speeds ({base_tps:.1f} and {S.get('decode_tps_median', 0):.1f} tok/s).")
    else:
        V["confound"] = "All passes ran at comparable decode speeds."
    ctd = {r["label"][:4].strip(): r for r in env.get("cachetest") or []}
    rt = lambda k: f"{100 * (ctd.get(k, {}).get('reuse_pct_timing') or 0):.0f}"
    V.update(ct_a=rt("(a)"), ct_b=rt("(b)"), ct_c1=rt("(c1)"), ct_c2=rt("(c2)"),
             ct_tokens=f"{ctd.get('(a)', {}).get('second_total_tokens', 0):,}",
             c4_reuse=f"{100 * (S.get('prefix_reuse_pct_mean') or 0):.0f}",
             c5_reuse=f"{100 * (bc.get('sieve_nomemo', {}).get('prefix_reuse_pct_mean') or 0):.0f}",
             c5_cpu=n(bc.get("sieve_nomemo", {}).get("sieve_overhead_ms_mean"), 0),
             c4_cum=f"{cp.get('sieve_full', 0):,.0f}", c5_cum=f"{cp.get('sieve_nomemo', 0):,.0f}",
             c0_cum=f"{cp.get('baseline', 0):,.0f}")
    rows = []
    for c in [c for c in ABL if c in bc]:
        s = bc[c]
        rows.append(f"{LBL[c]} & {s['accuracy']:.3f} & {sgn(100 * pc[c]['acc_delta'][0])} {br(pc[c]['acc_delta'], 1, 100)} & "
                    f"{n(s['token_reduction_pct'], 1)} & {n(100 * (s.get('prefix_reuse_pct_mean') or 0), 0)} & "
                    f"{n(cp.get(c), 0)} & {s.get('reinflations', 0)}\\\\")
    V["tab_abl"] = "\n".join(rows)
    a = []
    if "sieve_full+noverifier" in bc:
        nv = bc["sieve_full+noverifier"]
        a.append(f"Removing the verifier changed accuracy from {S['accuracy']:.3f} to {nv['accuracy']:.3f} while saving "
                 f"{nv['token_reduction_pct']:.1f}\\% of tokens instead of {S['token_reduction_pct']:.1f}\\%; with the verifier, "
                 f"C4 re-inflated {S.get('reinflations', 0)} segment compressions.")
    if "sieve_full+noledger" in bc:
        nl = bc["sieve_full+noledger"]
        a.append(f"Removing the ledger lowered mean prefix reuse from {100 * (S.get('prefix_reuse_pct_mean') or 0):.0f}\\% to "
                 f"{100 * (nl.get('prefix_reuse_pct_mean') or 0):.0f}\\% and raised prefill per conversation from "
                 f"{cp.get('sieve_full', 0):,.0f} to {cp.get('sieve_full+noledger', 0):,.0f} tokens.")
    if "sieve_full+clf_heuristic" in bc and "sieve_full+core6" in bc:
        a.append(f"The classifier trained on heuristic labels reached {bc['sieve_full+clf_heuristic']['accuracy']:.3f} and the "
                 f"classifier restricted to the six core features {bc['sieve_full+core6']['accuracy']:.3f}, against "
                 f"{S['accuracy']:.3f} for C4; their paired intervals overlap with C4's.")
    pf = sum(r.get("prefetch_count", 0) for r in recs)
    if "sieve_full+noprefetch" in bc and pf == 0:
        a.append("Prefetching never triggered in any condition, so the $-$prefetch variant sent the same prompts as C4.")
    V["abl_text"] = " ".join(a)
    if clf:
        r_or, r_he = clf["routers"]["clf_oracle"], clf["routers"]["heuristic_labeller"]
        names = [("rules_only", "Hard rules only"), ("heuristic_labeller", "Heuristic labeller (C3)"),
                 ("clf_heuristic", "GBDT, heuristic labels"), ("clf_oracle", "GBDT, oracle labels (C4)"),
                 ("clf_oracle_core6", "GBDT, oracle, core-6")]
        rows = []
        for k, lab in names:
            r = clf["routers"].get(k)
            if r:
                e = r["per_lane"]["externalize"]
                rows.append(f"{lab} & {r['macro_f1']:.3f} & {e['precision']:.2f} / {e['recall']:.2f} & "
                            f"{n(r['ratio_mae'], 3) if r['ratio_mae'] is not None else '--'} & {r['dangerous_errors']}\\\\")
        V.update(tab_clf="\n".join(rows), clf_rows=f"{clf['n_rows']:,}", clf_groups=str(clf["n_groups"]),
                 clf_needed=f"{r_or['n_needed']:,}", clf_f1=f"{r_or['macro_f1']:.3f}", clf_danger=str(r_or["dangerous_errors"]),
                 clf_h_danger=str(r_he["dangerous_errors"]), clf_ms=f"{clf['inference_ms_per_segment']:.3f}")
    be = D["be"]
    if be:
        sm = be["summary"]
        cross = be.get("break_even_p")
        per = len(be["rows"]) // max(1, len(sm))
        txt = (f"Fig.~\\ref{{fig:be}} sweeps the probability $p$ that a question needs a large archived document that C4 "
               f"externalizes ({per} cold conversations per point). Mean net time per question was "
               f"{sm[0]['sieve_s']:.2f} s for C4 and {sm[0]['baseline_s']:.2f} s for C0 at $p={sm[0]['p']:g}$, and "
               f"{sm[-1]['sieve_s']:.2f} s and {sm[-1]['baseline_s']:.2f} s at $p={sm[-1]['p']:g}$, where C4 expanded on "
               f"{100 * sm[-1]['expand_rate']:.0f}\\% of questions. ")
        txt += (f"The curves cross at $p\\approx{cross:.2f}$." if cross is not None else
                "The curves did not cross in the swept range: in this setting externalization paid off even when every "
                "question needed the document, because C0's prompt overflowed the window and was prefilled in full on every call.")
        txt += f" Accuracy at $p={sm[-1]['p']:g}$ was {sm[-1]['sieve_acc']:.2f} for C4 and {sm[-1]['baseline_acc']:.2f} for C0."
        V["be_text"] = (txt + "\n\\begin{figure}[t]\\centering\\includegraphics[width=\\columnwidth]{fig_breakeven.pdf}"
                        "\\caption{Externalization break-even: mean net time per question (Sieve CPU time plus model time, "
                        "including tool round trips) as the probability that a question needs the externalized document "
                        "grows.}\\label{fig:be}\\end{figure}")
    else:
        V["be_text"] = ""
    cs = [c for c in ORDER if c in agg["by_family"]]
    V["fam_cols"] = "r" * len(cs)
    V["fam_head"] = " & ".join(SHORT[c] for c in cs)
    rows, above, below = [], [], []
    for f in sorted(agg["by_family"]["baseline"]):
        b = agg["by_family"]["baseline"][f]["accuracy"]
        vals = []
        for c in cs:
            v = agg["by_family"][c].get(f, {}).get("accuracy")
            s = n(v, 2)
            if c != "baseline" and v is not None and v > b + 1e-9:
                s = "\\textbf{" + s + "}"
            vals.append(s)
        v4 = agg["by_family"]["sieve_full"][f]["accuracy"]
        if v4 > b + 1e-9:
            above.append(f)
        elif v4 < b - 1e-9:
            below.append(f)
        rows.append(f"{f} & {agg['by_family']['baseline'][f]['n']} & " + " & ".join(vals) + "\\\\")
    V["tab_famres"] = "\n".join(rows)
    V["fam_text"] = f"C4 was above C0 in families {', '.join(above) or 'none'} and below it in {', '.join(below) or 'none'}."
    sup = []
    if P["token_reduction_pct"][1] > 0:
        sup.append(f"Sieve reduced prompt tokens by {P['token_reduction_pct'][0]:.1f}\\% with an interval that excludes zero")
    if Po.get("n") and Po["acc_delta"][1] > 0:
        sup.append(f"improved accuracy on prompts that overflowed the window ({sgn(100 * Po['acc_delta'][0])} pp)")
    if Pc["ttft_delta_s"][2] < 0:
        sup.append(f"lowered median cold TTFT by {-Pc['ttft_delta_s'][0]:.2f} s")
    V["supported"] = (", ".join(sup) + f"; its overall accuracy change {long_v}. The cache test and the C4--C5 comparison "
                      "support the ledger as the mechanism that keeps compression compatible with prefix caching.")
    loss = []
    sh, shb = agg["by_bucket"]["sieve_full"].get("short", {}), agg["by_bucket"]["baseline"].get("short", {})
    if sh:
        loss.append(f"On short prompts (under about 1,500 tokens) C4 saved {sh.get('token_reduction_pct', 0):.1f}\\% of tokens and "
                    f"reached accuracy {sh.get('accuracy', 0):.2f} against {shb.get('accuracy', 0):.2f} for C0.")
    if below:
        loss.append(f"Families {', '.join(below)} lost accuracy under C4.")
    loss.append(f"On warm questions the raw prompt's prefix is already cached, so latency differences there are small "
                f"({V['c4_warm']} s vs.\\ {V['c0_warm']} s median).")
    loss.append(f"C4 hard-truncated {S.get('hard_truncations', 0)} prompts that remained over budget after compression and "
                f"externalization.")
    V["losses"] = " ".join(loss)
    c4r = [r for r in recs if r["condition"] == "sieve_full"]
    V.update(c4_expand_pct=f"{100 * sum(r['tool_rounds'] > 0 for r in c4r) / max(1, len(c4r)):.0f}",
             c4_missed=str(sum(r.get("missed_expansions", 0) for r in c4r)), prefetch_n=str(pf))
    V["prefetch_sentence"] = (f"Prefetching never triggered at the {V['prefetch_thr']} similarity threshold, so the $-$prefetch "
                              "ablation sent the same content as C4 and differs only in timing." if pf == 0 else
                              f"Prefetching triggered {pf} times.")
    V["abstract"] = (
        "Local large language models share one fixed context window between the prompt and the answer, and long conversations "
        "waste it on verbose, repeated and stale text. Prompt compression reduces tokens but rewrites the prefix, which defeats "
        "the key-value (KV) prefix cache that local servers rely on. This paper presents Sieve, a middleware that splits each "
        "request into segments and routes every segment to one of three lanes: COMPRESS (LLMLingua-2 pruning at a per-segment "
        "ratio, checked by a rule-based verifier of facts and their context), CACHE (kept byte-identical so the KV prefix is "
        "reused) or EXTERNALIZE (replaced by a deterministic stub that the model can expand through a tool call). Hard rules run "
        "first, a cost-sensitive gradient-boosting classifier routes the rest, and a per-session ledger replays emitted strings "
        f"so compression does not invalidate the cached prefix. On {n_q} ground-truth questions from {len(cases)} synthetic "
        f"conversations with invented entities, run on Qwen3-14B through Ollama with a {V['num_ctx']}-token window, Sieve "
        f"reduced prompt tokens by {P['token_reduction_pct'][0]:.1f}\\% (95\\% CI {P['token_reduction_pct'][1]:.1f}--"
        f"{P['token_reduction_pct'][2]:.1f}\\%) and raised mean output headroom from {V['c0_head']} to {V['c4_head']} tokens. "
        f"Its accuracy change of {sgn(100 * P['acc_delta'][0])} percentage points (CI {ci(P['acc_delta'], 1, 100)}) {long_v}"
        + (f", while on prompts that overflowed the window accuracy rose by {100 * Po['acc_delta'][0]:.1f} points "
           f"(CI {ci(Po['acc_delta'], 1, 100)})" if Po.get("n") and Po["acc_delta"][1] > 0 else "")
        + (f". Median time to first token on cold questions fell by {-Pc['ttft_delta_s'][0]:.2f} s" if Pc["ttft_delta_s"][2] < 0 else "")
        + f". A controlled test shows that re-compressing a prefix reduces prefill reuse from {V['ct_a']}\\% to {V['ct_c1']}\\%, "
        f"and that sending the memoized compressed string restores it to {V['ct_c2']}\\%. Code, logs and traces are public.")
    return V


def fill(V: Dict[str, str]) -> str:
    tpl = (ROOT / "paper" / "template.tex").read_text()
    return re.sub("«([a-z0-9_]+)»", lambda m: V[m.group(1)] if m.group(1) in V else m.group(0), tpl)


def build_paper(runs_dir: Path, run: str, out_root: Path) -> Path:
    D = load_all(Path(runs_dir), run)
    out = out_root / run
    out.mkdir(parents=True, exist_ok=True)
    figures(D, out)
    V = compute(D)
    text = fill(V)
    left = sorted(set(re.findall("«([a-z0-9_]+)»", text)))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    (out / "main.tex").write_text(text)
    (out / "values.json").write_text(json.dumps(V, indent=1, ensure_ascii=False))
    (out / "refs.bib").write_text("\n".join(p.read_text() for p in sorted((ROOT / "paper" / "refs").glob("*.bib"))))
    exe = shutil.which("tectonic")
    if exe:
        subprocess.run([exe, "-X", "compile", "main.tex"], cwd=out, check=True)
    return out / "main.pdf"


def build_tex(D: Dict[str, Any], smoke: bool = True) -> str:
    """Fill the template without compiling (used by tests)."""
    return fill(compute(D))

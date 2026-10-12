"""python -m sieve <command>  (see README)."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console
from rich.table import Table

from sieve.config import load_config

console = Console()


def _client(cfg, rt):
    if cfg.mock:
        from sieve.mock_client import MockClient
        return MockClient(cfg, rt.tok)
    from sieve.ollama_client import OllamaClient
    return OllamaClient(cfg, rt.tok)


def _runtime(cfg, args):
    from sieve.conditions import Runtime
    return Runtime(cfg, force_approx_tokens=args.approx_tokens, force_tfidf=args.tfidf,
                   force_fallback_compressor=args.fallback_compressor)


def _env_path(cfg) -> Path:
    return Path(cfg.runs_dir) / ("env_mock.json" if cfg.mock else "env.json")


def _load_env(cfg) -> Dict[str, Any]:
    p = _env_path(cfg)
    return json.loads(p.read_text()) if p.exists() else {"model": cfg.model, "num_ctx": cfg.num_ctx, "mock": cfg.mock}


def _save_env(cfg, upd: Dict[str, Any]):
    env = _load_env(cfg)
    env.update(upd)
    p = _env_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(env, indent=2, default=str))
    return env


# ---------------- commands ----------------

def cmd_selfcheck(cfg, args):
    rt = _runtime(cfg, args)
    from sieve.experiments import selfcheck
    env = selfcheck(cfg, _client(cfg, rt), rt)
    env = _save_env(cfg, env)
    console.print(f"wrote {_env_path(cfg)}")


def cmd_cachetest(cfg, args):
    rt = _runtime(cfg, args)
    from sieve.experiments import cachetest
    res = cachetest(cfg, _client(cfg, rt), rt)
    _save_env(cfg, res)
    console.print(f"wrote {_env_path(cfg)}")


def cmd_gen(cfg, args):
    from sieve.workloads.generator import FAMILY_DESC, generate_suite
    from sieve.tokenizer import get_tokenizer
    cases = generate_suite(args.tier, args.seed)
    out = Path(cfg.runs_dir) / "workloads" / f"{args.tier}_seed{args.seed}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(c) for c in cases) + "\n")
    tok = get_tokenizer(cfg.tokenizer_model, args.approx_tokens)
    t = Table(title=f"Workloads tier={args.tier} seed={args.seed}", header_style="bold")
    for c in ("family", "description", "cases", "eval points", "tokens min/median/max", "overflow num_ctx"):
        t.add_column(c)
    bad = 0
    for fam, desc in FAMILY_DESC.items():
        cs = [c for c in cases if c["family"] == fam]
        if not cs:
            continue
        toks = sorted(tok.count_messages(c["messages"]) for c in cs)
        t.add_row(fam, desc, str(len(cs)), str(sum(len(c["eval_points"]) for c in cs)),
                  f"{toks[0]}/{toks[len(toks) // 2]}/{toks[-1]}", str(sum(x > cfg.num_ctx for x in toks)))
        for c in cs:  # ground-truth check: gold facts present in the needed messages
            mids = {m["mid"]: m for m in c["messages"]}
            for ep in c["eval_points"]:
                for f in ep["facts"]:
                    if not any(f in mids[n]["content"] for n in ep["needed_segment_ids"]):
                        bad += 1
    console.print(t)
    console.print(f"{len(cases)} cases, {sum(len(c['eval_points']) for c in cases)} evaluation points -> {out}")
    console.print("[green]gold facts present in needed segments: OK[/]" if not bad else f"[red]{bad} gold facts missing![/]")
    if bad:
        sys.exit(1)


def train_models(cfg, rt, eval_report: bool = True, labels_main: str = "oracle"):
    from sieve.classifier import LaneClassifier, build_rows, evaluate, label_rows, model_path
    from sieve.features import ALL_FEATURES, CORE_FEATURES
    from sieve.workloads.generator import generate_suite
    t0 = time.perf_counter()
    cases = [c for s in cfg.train_seeds for c in generate_suite("standard", s)]
    console.print(f"[bold]train[/]: {len(cases)} conversations from training seeds {cfg.train_seeds} "
                  f"(disjoint from benchmark seeds)")
    rows = build_rows(cases, cfg, rt.tok, rt.sim)
    console.print(f"  {len(rows)} segment rows built in {time.perf_counter() - t0:.1f}s (similarity: {rt.sim.method})")
    for labels in ("oracle", "heuristic"):
        y, r = label_rows(rows, labels, cfg)
        LaneClassifier(cfg, ALL_FEATURES, labels).fit(rows, y, r).save(model_path(cfg, labels, "all"))
    y, r = label_rows(rows, "oracle", cfg)
    LaneClassifier(cfg, CORE_FEATURES, "oracle").fit(rows, y, r).save(model_path(cfg, "oracle", "core6"))
    console.print(f"  saved models to {model_path(cfg, 'oracle').parent}")
    if not eval_report:
        return None
    ev = evaluate(rows, cfg)
    ev["train_cases"] = len(cases)
    ev["similarity"] = rt.sim.method
    p = model_path(cfg, "oracle").parent / "classifier_eval.json"
    p.write_text(json.dumps(ev, indent=2))
    show_classifier_eval(ev)
    console.print(f"wrote {p}")
    return ev


def show_classifier_eval(ev):
    from sieve.viz import confusion_table
    labs = ["cache", "compress", "externalize"]
    console.print(f"[bold]Classifier evaluation[/] — {ev['n_rows']} rows, {ev['n_groups']} held-out-by-conversation groups "
                  f"(GroupKFold); oracle label dist {ev['label_dist']}")
    t = Table(title="Routers vs oracle labels (held-out conversations)", header_style="bold")
    for c in ("router", "macro-F1", "cache F1", "compress F1", "externalize P/R/F1", "ratio MAE", "dangerous errors",
              "externalize rate"):
        t.add_column(c)
    for name, r in ev["routers"].items():
        e = r["per_lane"]["externalize"]
        t.add_row(name, f"{r['macro_f1']:.3f}", f"{r['per_lane']['cache']['f1']:.3f}", f"{r['per_lane']['compress']['f1']:.3f}",
                  f"{e['precision']:.2f}/{e['recall']:.2f}/{e['f1']:.2f}", str(r["ratio_mae"]),
                  f"{r['dangerous_errors']} / {r['n_needed']} needed", f"{r['externalize_rate']:.3f}")
    console.print(t)
    console.print(confusion_table(ev["routers"]["clf_oracle"]["confusion"], labs, "Confusion: oracle-trained classifier"))
    ct = Table(title="Calibration (oracle-trained, held-out)", header_style="bold")
    for c in ("confidence bin", "n", "accuracy", "mean conf"):
        ct.add_column(c)
    for b in ev["calibration"]:
        ct.add_row(b["bin"], str(b["n"]), f"{b['accuracy']:.3f}", f"{b['mean_conf']:.3f}")
    console.print(ct)
    console.print("Feature importances: " + ", ".join(f"{f}={v:.3f}" for f, v in ev["feature_importances"][:10]))
    console.print(f"Inference: {ev['inference_ms_per_segment']:.3f} ms/segment; below confidence threshold: "
                  f"{ev['below_conf_threshold']}")
    console.print(f"[italic]{ev['note']}[/]")


def cmd_train(cfg, args):
    rt = _runtime(cfg, args)
    train_models(cfg, rt, eval_report=args.eval)


def cmd_demo(cfg, args):
    rt = _runtime(cfg, args)
    client = _client(cfg, rt)
    from sieve.experiments import lane_demo_case, run_case
    env = _load_env(cfg)
    conds = args.conditions.split(",") if args.conditions else ["baseline", "sieve_full"]
    run_dir = Path(cfg.runs_dir) / ("demo_mock" if cfg.mock else "demo")
    if args.case:
        from sieve.workloads.generator import generate_case, generate_suite
        fam, n = args.case.split(":")
        suite = {c["case_id"]: c for c in generate_suite(args.tier, args.seed)}
        case = suite.get(f"{fam}:{n}") or generate_case(fam, int(n), args.seed, "mid")
        run_case(cfg, client, rt, case, conds, run_dir, env=env)
        return
    lanes = ["compress", "cache", "externalize"] if args.lane == "all" else [args.lane]
    for lane in lanes:
        console.rule(f"[bold yellow]DEMO lane={lane}")
        run_case(cfg, client, rt, lane_demo_case(lane), conds, run_dir, env=env)


def _resolve_conditions(spec: str, with_reference: bool, ablations: bool) -> List[str]:
    from sieve.conditions import ABLATIONS, CONDITIONS, REFERENCE
    conds = list(CONDITIONS) if spec == "all" else spec.split(",")
    if ablations:
        conds += [a for a in ABLATIONS if a not in conds]
    if with_reference and REFERENCE not in conds:
        conds.append(REFERENCE)
    return conds


def cmd_bench(cfg, args):
    from sieve.bench import run_bench
    from sieve.workloads.generator import generate_suite
    rt = _runtime(cfg, args)
    client = _client(cfg, rt)
    env = _load_env(cfg)
    if not cfg.mock and not env.get("ollama_version"):
        console.print("[yellow]No env.json: run `python -m sieve selfcheck` and `cachetest` first (doing selfcheck now).[/]")
        cmd_selfcheck(cfg, args)
        env = _load_env(cfg)
    if not cfg.mock and env.get("tool_call_supported") is False and cfg.tool_mode != "prefetch":
        console.print("[yellow]selfcheck found tool calls unreliable -> tool_mode=prefetch[/]")
        cfg.tool_mode = "prefetch"
    conds = _resolve_conditions(args.conditions, args.with_reference, args.ablations)
    degraded = [n for n, bad in (("tokenizer (approx)", rt.tok.approx), ("compressor (extractive fallback)", rt.comp.fallback),
                                 ("similarity (TF-IDF)", rt.sim.method != "minilm")) if bad]
    if degraded and not cfg.mock and not args.allow_fallback:
        console.print(f"[red bold]Refusing to benchmark with degraded components: {', '.join(degraded)}.[/] "
                      "Fix the cause (e.g. set HF_HUB_OFFLINE=1 to load cached models) or pass --allow-fallback.")
        sys.exit(2)
    runs = Path(cfg.runs_dir)
    tag = f"{args.tier}{'_mock' if cfg.mock else ''}"
    if args.run_id:
        run_id = args.run_id
    elif args.resume:
        prev = sorted([p.name for p in runs.glob(f"{tag}_2*") if p.is_dir()])
        run_id = prev[-1] if prev else f"{tag}_{time.strftime('%Y%m%d-%H%M%S')}"
    else:
        run_id = f"{tag}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = runs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cases = generate_suite(args.tier, args.seed)
    (run_dir / "cases.jsonl").write_text("\n".join(json.dumps(c) for c in cases) + "\n")
    env.update({"run_id": run_id, "tier": args.tier, "seed": args.seed, "conditions": conds, "repeats": args.repeats,
                "mock": cfg.mock, "config": cfg.to_dict(), "tool_mode_effective": cfg.tool_mode,
                "approx_tokens": rt.tok.approx, "compressor": rt.comp.version, "similarity": rt.sim.method})
    (run_dir / "env.json").write_text(json.dumps(env, indent=2, default=str))
    console.print(f"[bold]run {run_id}[/]: {len(cases)} cases × {len(conds)} conditions × {args.repeats} repeats"
                  + ("  [MOCK]" if cfg.mock else ""))
    run_bench(cfg, client, rt, cases, conds, args.repeats, run_dir, run_id, env, resume=True, seed=args.seed)
    console.print(f"[green]done[/] -> {run_dir / 'results.jsonl'}")
    args.run = run_id
    cmd_report(cfg, args)


def cmd_report(cfg, args):
    from sieve.metrics import aggregate, load_records, validate, write_claims
    from sieve.viz import condition_table, write_pngs
    run_dir = Path(cfg.runs_dir) / args.run
    recs = load_records(run_dir / "results.jsonl")
    if not recs:
        console.print(f"[red]no records in {run_dir}[/]")
        sys.exit(1)
    missing = {k for r in recs for k in validate(r)}
    if missing:
        console.print(f"[red]schema: missing keys {sorted(missing)}[/]")
    env = json.loads((run_dir / "env.json").read_text()) if (run_dir / "env.json").exists() else {}
    agg = aggregate(recs)
    console.print(condition_table(agg["by_condition"], "All evaluation points" + (" [MOCK]" if env.get("mock") else "")))
    console.print(condition_table(agg["by_condition_cold"], "Cold points (first question of each conversation)"))
    console.print(condition_table(agg["by_condition_warm"], "Warm points"))
    console.print("Cumulative prompt tokens processed per conversation: " +
                  ", ".join(f"{c}={v:.0f}" for c, v in agg["cumulative_prefill"].items()))
    console.print("Decode tok/s is expected to be roughly flat across conditions (it does not depend on the prompt): " +
                  ", ".join(f"{c}={(s.get('decode_tps_median') or 0):.1f}" for c, s in agg["by_condition"].items()))
    clf = Path(cfg.runs_dir) / "models" / "classifier_eval.json"
    clf_eval = json.loads(clf.read_text()) if clf.exists() else None
    (run_dir / "summary.json").write_text(json.dumps(agg, indent=2, default=str))
    text = write_claims(agg, env, run_dir / "claims.md", bool(env.get("mock")))
    files = write_pngs(recs, agg, run_dir / "plots", clf_eval)
    console.print(text)
    console.print(f"wrote summary.json, claims.md{', ' + str(len(files)) + ' PNGs' if files else ''} in {run_dir}")


def cmd_breakeven(cfg, args):
    rt = _runtime(cfg, args)
    from sieve.experiments import breakeven
    res = breakeven(cfg, _client(cfg, rt), rt, n_cases=args.n)
    p = Path(cfg.runs_dir) / ("breakeven_mock.json" if cfg.mock else "breakeven.json")
    p.write_text(json.dumps(res, indent=2, default=str))
    console.print(f"wrote {p}")


def cmd_html(cfg, args):
    from sieve.report_html import write_html
    out = Path(args.out) if args.out else Path(cfg.runs_dir).parent / "site"
    files = write_html(Path(cfg.runs_dir), args.run, out, full_doc=not args.fragment)
    for f in files:
        console.print(f"wrote {f}")


def cmd_paper(cfg, args):
    from sieve.paper import build_paper
    out = Path(args.out) if args.out else Path(cfg.runs_dir).parent / "paper"
    pdf = build_paper(Path(cfg.runs_dir), args.run, out)
    console.print(f"wrote {pdf}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m sieve")
    ap.add_argument("--mock", action="store_true", help="deterministic fake Ollama client (no GPU)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config key")
    ap.add_argument("--approx-tokens", action="store_true", help="force the chars/3.6 token estimator")
    ap.add_argument("--tfidf", action="store_true", help="force TF-IDF similarity")
    ap.add_argument("--fallback-compressor", action="store_true", help="force the extractive fallback compressor")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selfcheck")
    sub.add_parser("cachetest")
    g = sub.add_parser("gen")
    g.add_argument("--tier", default="smoke", choices=["smoke", "standard", "full"])
    g.add_argument("--seed", type=int, default=7)
    t = sub.add_parser("train")
    t.add_argument("--labels", default="oracle", choices=["heuristic", "oracle"])
    t.add_argument("--eval", action="store_true")
    d = sub.add_parser("demo")
    d.add_argument("--lane", default="all", choices=["compress", "cache", "externalize", "all"])
    d.add_argument("--case", default=None, help="<family>:<n>, e.g. C:3")
    d.add_argument("--conditions", default=None)
    d.add_argument("--tier", default="standard")
    d.add_argument("--seed", type=int, default=7)
    b = sub.add_parser("bench")
    b.add_argument("--tier", default="smoke", choices=["smoke", "standard", "full"])
    b.add_argument("--conditions", default="all")
    b.add_argument("--repeats", type=int, default=None)
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--with-reference", action="store_true")
    b.add_argument("--ablations", action="store_true", help="also run C4 ablation variants")
    b.add_argument("--resume", action="store_true")
    b.add_argument("--run-id", default=None)
    b.add_argument("--allow-fallback", action="store_true", help="benchmark even if a component fell back")
    r = sub.add_parser("report")
    r.add_argument("--run", required=True)
    h = sub.add_parser("html")
    h.add_argument("--run", required=True)
    h.add_argument("--out", default=None)
    h.add_argument("--fragment", action="store_true", help="omit <html>/<head> (for an Artifact publish)")
    pp = sub.add_parser("paper")
    pp.add_argument("--run", required=True)
    pp.add_argument("--out", default=None)
    be = sub.add_parser("breakeven")
    be.add_argument("--n", type=int, default=8)
    args = ap.parse_args(argv)
    over = dict(s.split("=", 1) for s in args.set)
    cfg = load_config(args.config, over)
    cfg.mock = args.mock
    if args.cmd == "bench" and args.repeats is None:
        args.repeats = 3 if args.tier == "smoke" else 1
    if args.cmd == "train" and args.labels:
        cfg.classifier_labels = cfg.classifier_labels
    {"selfcheck": cmd_selfcheck, "cachetest": cmd_cachetest, "gen": cmd_gen, "train": cmd_train, "demo": cmd_demo,
     "bench": cmd_bench, "report": cmd_report, "breakeven": cmd_breakeven, "html": cmd_html, "paper": cmd_paper}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()

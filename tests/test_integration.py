import json

from sieve.metrics import load_records, validate


def test_mock_bench_smoke_end_to_end_and_resume(cfg, rt, trained, tmp_path):
    from sieve.bench import run_bench
    from sieve.conditions import CONDITIONS
    from sieve.mock_client import MockClient
    from sieve.workloads.generator import generate_suite
    client = MockClient(cfg, rt.tok)
    cases = generate_suite("smoke", 7)
    env = {"mock": True, "model": cfg.model, "num_ctx": cfg.num_ctx}
    path = run_bench(cfg, client, rt, cases, CONDITIONS, 1, tmp_path, "t", env)
    recs = load_records(path)
    n_eval = sum(len(c["eval_points"]) for c in cases)
    assert len(recs) == n_eval * len(CONDITIONS)
    assert all(not validate(r) for r in recs)
    assert any(r["baseline_overflow"] for r in recs if r["condition"] == "baseline")
    assert any(r["lanes"]["externalize"] for r in recs if r["condition"] == "sieve_full")
    assert all(r["mock"] for r in recs)
    # per-call artifacts exist
    r = recs[0]
    d = tmp_path / r["case_id"].replace(":", "_") / r["condition"] / f"ep{r['eval_point']}_r0"
    assert (d / "trace.json").exists() and (d / "diff.txt").exists() and (d / "final_prompt.json").exists()
    # resume: nothing is re-run
    calls = client.calls
    run_bench(cfg, client, rt, cases, CONDITIONS, 1, tmp_path, "t", env)
    assert client.calls - calls <= 2 * len(CONDITIONS) + 2   # only unload/warmup, no evaluation calls
    assert len(load_records(path)) == len(recs)

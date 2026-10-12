# claims.md — statements derived only from this run's logs

- Model: `qwen3:14b`; num_ctx 4096; Ollama 0.32.15; workloads are synthetic (invented entities).

- KV prefix reuse on this model/server: **observed** (cachetest). 

## sieve_full vs baseline (paired, n=29)
- Accuracy delta: +10.3 pp (95% CI -3.4 to +24.1)
- Prompt-token reduction: +35.5% (95% CI +27.4 to +42.7)
- Median TTFT delta: -0.41 s (95% CI -1.54 to -0.12)
- Mean prompt tokens processed delta: -296 tok (95% CI -691 to +120)
- Cold points only (n=12): median TTFT delta -1.67 s (95% CI -4.18 to -0.63)
- Where baseline overflowed num_ctx (n=9): accuracy delta +33.3 pp (95% CI -11.1 to +77.8)

## sieve_nomemo vs baseline (paired, n=29)
- Accuracy delta: +10.3 pp (95% CI -3.4 to +24.1)
- Prompt-token reduction: +36.7% (95% CI +28.6 to +44.3)
- Median TTFT delta: +0.12 s (95% CI -0.28 to +1.36)
- Mean prompt tokens processed delta: +99 tok (95% CI -350 to +532)
- Cold points only (n=12): median TTFT delta -0.50 s (95% CI -3.09 to +0.58)
- Where baseline overflowed num_ctx (n=9): accuracy delta +33.3 pp (95% CI -11.1 to +77.8)

## sieve_rules vs baseline (paired, n=29)
- Accuracy delta: +0.0 pp (95% CI -13.8 to +13.8)
- Prompt-token reduction: +36.8% (95% CI +28.6 to +44.0)
- Median TTFT delta: -0.04 s (95% CI -0.66 to +0.03)
- Mean prompt tokens processed delta: -278 tok (95% CI -612 to +92)
- Cold points only (n=12): median TTFT delta -0.86 s (95% CI -3.11 to +0.03)
- Where baseline overflowed num_ctx (n=9): accuracy delta +11.1 pp (95% CI -22.2 to +44.4)

## truncate vs baseline (paired, n=29)
- Accuracy delta: -31.0 pp (95% CI -48.3 to -13.8)
- Prompt-token reduction: +57.2% (95% CI +43.4 to +69.9)
- Median TTFT delta: -0.62 s (95% CI -1.50 to -0.14)
- Mean prompt tokens processed delta: -683 tok (95% CI -1028 to -355)
- Cold points only (n=12): median TTFT delta -2.60 s (95% CI -5.37 to -1.17)
- Where baseline overflowed num_ctx (n=9): accuracy delta +0.0 pp (95% CI +0.0 to +0.0)

## uniform_lingua vs baseline (paired, n=29)
- Accuracy delta: +0.0 pp (95% CI -13.8 to +13.8)
- Prompt-token reduction: +37.4% (95% CI +29.3 to +44.6)
- Median TTFT delta: -0.03 s (95% CI -0.73 to +0.01)
- Mean prompt tokens processed delta: -170 tok (95% CI -552 to +225)
- Cold points only (n=12): median TTFT delta -0.81 s (95% CI -3.49 to +0.01)
- Where baseline overflowed num_ctx (n=9): accuracy delta +22.2 pp (95% CI +0.0 to +55.6)

## Cache-vs-compress (ledger) evidence
- Cumulative prompt tokens processed per conversation: sieve_full 2182 vs sieve_nomemo 3134 vs baseline 2896.
- Mean prefix reuse: sieve_full 44.1% vs sieve_nomemo 17.0%.

## Supported by this run
- sieve_full reduces prompt tokens vs baseline (CI lower bound 27.4%).
- sieve_full accuracy is within -5 pp of baseline at the CI lower bound (-3.4 pp).
- sieve_full has lower median TTFT than baseline (CI upper bound -0.12 s).
- sieve_nomemo reduces prompt tokens vs baseline (CI lower bound 28.6%).
- sieve_nomemo accuracy is within -5 pp of baseline at the CI lower bound (-3.4 pp).
- sieve_rules reduces prompt tokens vs baseline (CI lower bound 28.6%).
- truncate reduces prompt tokens vs baseline (CI lower bound 43.4%).
- truncate has lower median TTFT than baseline (CI upper bound -0.14 s).
- uniform_lingua reduces prompt tokens vs baseline (CI lower bound 29.3%).
- The ledger reduces cumulative prefill tokens relative to re-routing every turn (C4 < C5).

## Not supported by this run
- sieve_nomemo is faster than baseline in TTFT (CI upper bound +1.36 s >= 0).
- sieve_rules preserves accuracy (CI lower bound -13.8 pp is below -5 pp).
- sieve_rules is faster than baseline in TTFT (CI upper bound +0.03 s >= 0).
- truncate preserves accuracy (CI lower bound -48.3 pp is below -5 pp).
- uniform_lingua preserves accuracy (CI lower bound -13.8 pp is below -5 pp).
- uniform_lingua is faster than baseline in TTFT (CI upper bound +0.01 s >= 0).
- Any claim about real-world (non-synthetic) conversations.
- Any claim about models other than the one listed above.
- Classifier training accuracy on heuristic labels (circular; never cited).

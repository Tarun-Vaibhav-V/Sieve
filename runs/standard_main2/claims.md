# claims.md — statements derived only from this run's logs

- Model: `qwen3:14b`; num_ctx 4096; Ollama 0.32.15; workloads are synthetic (invented entities).

- KV prefix reuse on this model/server: **observed** (cachetest). 

## reference_bigctx vs baseline (paired, n=180)
- Accuracy delta: +13.3 pp (95% CI +8.3 to +18.3)
- Prompt-token reduction: +0.0% (95% CI +0.0 to +0.0)
- Median TTFT delta: +0.46 s (95% CI +0.30 to +0.86)
- Mean prompt tokens processed delta: +307 tok (95% CI +137 to +498)
- Cold points only (n=76): median TTFT delta +2.14 s (95% CI +1.61 to +2.80)
- Where baseline overflowed num_ctx (n=47): accuracy delta +51.1 pp (95% CI +36.2 to +66.0)

## sieve_full vs baseline (paired, n=180)
- Accuracy delta: +5.0 pp (95% CI -0.6 to +11.1)
- Prompt-token reduction: +31.0% (95% CI +27.8 to +34.0)
- Median TTFT delta: -0.25 s (95% CI -0.49 to -0.10)
- Mean prompt tokens processed delta: -356 tok (95% CI -504 to -210)
- Cold points only (n=76): median TTFT delta -1.12 s (95% CI -1.49 to -0.66)
- Where baseline overflowed num_ctx (n=47): accuracy delta +36.2 pp (95% CI +19.1 to +53.2)

## sieve_full+clf_heuristic vs baseline (paired, n=180)
- Accuracy delta: +5.6 pp (95% CI +0.0 to +11.7)
- Prompt-token reduction: +32.6% (95% CI +29.5 to +35.4)
- Median TTFT delta: -0.21 s (95% CI -0.44 to -0.08)
- Mean prompt tokens processed delta: -327 tok (95% CI -477 to -177)
- Cold points only (n=76): median TTFT delta -1.18 s (95% CI -1.39 to -0.82)
- Where baseline overflowed num_ctx (n=47): accuracy delta +38.3 pp (95% CI +21.3 to +55.3)

## sieve_full+core6 vs baseline (paired, n=180)
- Accuracy delta: +7.8 pp (95% CI +2.8 to +13.3)
- Prompt-token reduction: +31.2% (95% CI +28.0 to +34.0)
- Median TTFT delta: -0.23 s (95% CI -0.42 to -0.10)
- Mean prompt tokens processed delta: -323 tok (95% CI -463 to -180)
- Cold points only (n=76): median TTFT delta -1.01 s (95% CI -1.23 to -0.72)
- Where baseline overflowed num_ctx (n=47): accuracy delta +38.3 pp (95% CI +21.3 to +55.3)

## sieve_full+noledger vs baseline (paired, n=180)
- Accuracy delta: +5.6 pp (95% CI +0.0 to +11.1)
- Prompt-token reduction: +30.9% (95% CI +27.5 to +33.9)
- Median TTFT delta: -0.06 s (95% CI -0.23 to +0.28)
- Mean prompt tokens processed delta: +101 tok (95% CI -74 to +278)
- Cold points only (n=76): median TTFT delta -0.99 s (95% CI -1.41 to -0.64)
- Where baseline overflowed num_ctx (n=47): accuracy delta +34.0 pp (95% CI +17.0 to +51.1)

## sieve_full+noprefetch vs baseline (paired, n=180)
- Accuracy delta: +5.0 pp (95% CI -0.6 to +11.1)
- Prompt-token reduction: +31.0% (95% CI +27.8 to +34.0)
- Median TTFT delta: -0.10 s (95% CI -0.20 to -0.04)
- Mean prompt tokens processed delta: -298 tok (95% CI -443 to -152)
- Cold points only (n=76): median TTFT delta -0.75 s (95% CI -1.07 to -0.14)
- Where baseline overflowed num_ctx (n=47): accuracy delta +36.2 pp (95% CI +19.1 to +53.2)

## sieve_full+noverifier vs baseline (paired, n=180)
- Accuracy delta: -1.1 pp (95% CI -7.8 to +5.6)
- Prompt-token reduction: +32.6% (95% CI +29.3 to +35.7)
- Median TTFT delta: +0.02 s (95% CI -0.11 to +0.08)
- Mean prompt tokens processed delta: -305 tok (95% CI -442 to -166)
- Cold points only (n=76): median TTFT delta -0.18 s (95% CI -0.44 to +0.06)
- Where baseline overflowed num_ctx (n=47): accuracy delta +27.7 pp (95% CI +10.6 to +44.7)

## sieve_nomemo vs baseline (paired, n=180)
- Accuracy delta: +5.6 pp (95% CI +0.0 to +11.1)
- Prompt-token reduction: +30.9% (95% CI +27.5 to +33.9)
- Median TTFT delta: +0.40 s (95% CI +0.14 to +0.91)
- Mean prompt tokens processed delta: +187 tok (95% CI +12 to +365)
- Cold points only (n=76): median TTFT delta -0.20 s (95% CI -0.38 to +0.15)
- Where baseline overflowed num_ctx (n=47): accuracy delta +34.0 pp (95% CI +17.0 to +51.1)

## sieve_rules vs baseline (paired, n=180)
- Accuracy delta: +1.7 pp (95% CI -5.0 to +7.8)
- Prompt-token reduction: +34.7% (95% CI +31.5 to +37.5)
- Median TTFT delta: -0.22 s (95% CI -0.45 to -0.09)
- Mean prompt tokens processed delta: -374 tok (95% CI -509 to -233)
- Cold points only (n=76): median TTFT delta -1.15 s (95% CI -1.47 to -0.92)
- Where baseline overflowed num_ctx (n=47): accuracy delta +36.2 pp (95% CI +21.3 to +51.1)

## truncate vs baseline (paired, n=180)
- Accuracy delta: -41.7 pp (95% CI -48.9 to -34.4)
- Prompt-token reduction: +58.7% (95% CI +53.4 to +63.6)
- Median TTFT delta: +0.07 s (95% CI -0.02 to +0.09)
- Mean prompt tokens processed delta: -552 tok (95% CI -701 to -409)
- Cold points only (n=76): median TTFT delta -1.36 s (95% CI -2.32 to -0.58)
- Where baseline overflowed num_ctx (n=47): accuracy delta -8.5 pp (95% CI -17.0 to -2.1)

## uniform_lingua vs baseline (paired, n=180)
- Accuracy delta: -8.3 pp (95% CI -13.9 to -2.8)
- Prompt-token reduction: +33.6% (95% CI +30.4 to +36.5)
- Median TTFT delta: -0.02 s (95% CI -0.06 to +0.03)
- Mean prompt tokens processed delta: -204 tok (95% CI -339 to -64)
- Cold points only (n=76): median TTFT delta -0.25 s (95% CI -0.61 to +0.05)
- Where baseline overflowed num_ctx (n=47): accuracy delta -4.3 pp (95% CI -19.2 to +12.8)

## Cache-vs-compress (ledger) evidence
- Cumulative prompt tokens processed per conversation: sieve_full 1981 vs sieve_nomemo 3268 vs baseline 2824.
- Mean prefix reuse: sieve_full 45.0% vs sieve_nomemo 13.1%.

## Supported by this run
- reference_bigctx accuracy is within -5 pp of baseline at the CI lower bound (+8.3 pp).
- sieve_full reduces prompt tokens vs baseline (CI lower bound 27.8%).
- sieve_full accuracy is within -5 pp of baseline at the CI lower bound (-0.6 pp).
- sieve_full has lower median TTFT than baseline (CI upper bound -0.10 s).
- sieve_full+clf_heuristic reduces prompt tokens vs baseline (CI lower bound 29.5%).
- sieve_full+clf_heuristic accuracy is within -5 pp of baseline at the CI lower bound (+0.0 pp).
- sieve_full+clf_heuristic has lower median TTFT than baseline (CI upper bound -0.08 s).
- sieve_full+core6 reduces prompt tokens vs baseline (CI lower bound 28.0%).
- sieve_full+core6 accuracy is within -5 pp of baseline at the CI lower bound (+2.8 pp).
- sieve_full+core6 has lower median TTFT than baseline (CI upper bound -0.10 s).
- sieve_full+noledger reduces prompt tokens vs baseline (CI lower bound 27.5%).
- sieve_full+noledger accuracy is within -5 pp of baseline at the CI lower bound (+0.0 pp).
- sieve_full+noprefetch reduces prompt tokens vs baseline (CI lower bound 27.8%).
- sieve_full+noprefetch accuracy is within -5 pp of baseline at the CI lower bound (-0.6 pp).
- sieve_full+noprefetch has lower median TTFT than baseline (CI upper bound -0.04 s).
- sieve_full+noverifier reduces prompt tokens vs baseline (CI lower bound 29.3%).
- sieve_nomemo reduces prompt tokens vs baseline (CI lower bound 27.5%).
- sieve_nomemo accuracy is within -5 pp of baseline at the CI lower bound (+0.0 pp).
- sieve_rules reduces prompt tokens vs baseline (CI lower bound 31.5%).
- sieve_rules accuracy is within -5 pp of baseline at the CI lower bound (-5.0 pp).
- sieve_rules has lower median TTFT than baseline (CI upper bound -0.09 s).
- truncate reduces prompt tokens vs baseline (CI lower bound 53.4%).
- uniform_lingua reduces prompt tokens vs baseline (CI lower bound 30.4%).
- The ledger reduces cumulative prefill tokens relative to re-routing every turn (C4 < C5).

## Not supported by this run
- reference_bigctx reduces prompt tokens (CI includes 0).
- reference_bigctx is faster than baseline in TTFT (CI upper bound +0.86 s >= 0).
- sieve_full+noledger is faster than baseline in TTFT (CI upper bound +0.28 s >= 0).
- sieve_full+noverifier preserves accuracy (CI lower bound -7.8 pp is below -5 pp).
- sieve_full+noverifier is faster than baseline in TTFT (CI upper bound +0.08 s >= 0).
- sieve_nomemo is faster than baseline in TTFT (CI upper bound +0.91 s >= 0).
- truncate preserves accuracy (CI lower bound -48.9 pp is below -5 pp).
- truncate is faster than baseline in TTFT (CI upper bound +0.09 s >= 0).
- uniform_lingua preserves accuracy (CI lower bound -13.9 pp is below -5 pp).
- uniform_lingua is faster than baseline in TTFT (CI upper bound +0.03 s >= 0).
- Any claim about real-world (non-synthetic) conversations.
- Any claim about models other than the one listed above.
- Classifier training accuracy on heuristic labels (circular; never cited).

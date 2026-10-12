# Sieve

Sieve is middleware that reshapes an LLM prompt before it reaches a local model, then measures
whether the reshaping helped. Every request is split into **segments**. Each segment goes through
exactly one **lane**:

| Lane | For | Treatment |
|---|---|---|
| COMPRESS | Content needed every turn | LLMLingua-2 token pruning, with a ratio chosen per segment and a verifier for lost facts |
| CACHE | Static content and exact repeats | Kept byte-identical and positioned so the KV prefix can be reused |
| EXTERNALIZE | Bulky, old, probably irrelevant content | Replaced by a deterministic stub; the full text comes back on demand via an `expand_context` tool call |

Hard rules run first; a small CPU classifier (scikit-learn) routes everything else. A per-session
**ledger** replays already-emitted strings byte-for-byte, so compression does not invalidate the
KV prefix. A full re-route (**epoch**) happens only when the budget forces it.

**Claim under test** (not overstated anywhere): a per-segment routing policy across
compress/cache/externalize reduces prompt tokens and increases context headroom, with accuracy
preserved, while compression-induced KV-prefix invalidation is controlled by a ledger/epoch
mechanism.

## Environment used for the reported runs

| | |
|---|---|
| Hardware | Apple M5 Pro, 24 GB unified memory (the spec targeted an RTX 4050 6 GB; see DECISIONS D2) |
| Model | `qwen3:14b`, Q4_K_M, 14.8B parameters (spec default `qwen3.5:9b` was not installed; D1) |
| Offload split observed | `size` 8.97 GiB, `size_vram` 8.97 GiB → **100% on GPU (Metal)**, no CPU offload |
| Ollama | 0.32.15 (llama-server backend, `--flash-attn auto -np 1`) |
| num_ctx / num_predict | 4096 / 64, temperature 0, seed 0, `think: false` (verified by selfcheck) |
| Compressor | LLMLingua-2 `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`, CPU |
| Tokenizer | `Qwen/Qwen3-14B` (HF); matches Ollama's count exactly after template calibration |
| Similarity | `sentence-transformers/all-MiniLM-L6-v2`, CPU (TF-IDF fallback) |
| Python | 3.9.6 + venv (`requirements.txt`) |

**Server environment variables.** Recommended: `OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1`. Sieve never sets these. For the reported runs
**none were set**: the Ollama desktop app launched llama-server with `--flash-attn auto -np 1`,
and the KV cache was the default f16. The observed values are recorded in `runs/env.json`.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/python -m sieve selfcheck
```

```bash
.venv/bin/python -m sieve cachetest
```

```bash
.venv/bin/python -m sieve train --labels oracle --eval
```

```bash
.venv/bin/python -m sieve demo --lane all
```

```bash
.venv/bin/python -m sieve bench --tier standard --conditions all --repeats 1 --with-reference --ablations
```

```bash
.venv/bin/python -m sieve breakeven
```

```bash
.venv/bin/python -m sieve paper --run standard_main2
```

Without a GPU:

```bash
.venv/bin/python -m sieve --mock bench --tier smoke
```

```bash
.venv/bin/python -m pytest -q tests
```

Other commands: `gen --tier smoke|standard|full --seed 7`, `demo --case C:3 --conditions baseline,sieve_full`,
and `report --run <run_id>`. Runs are resumable (`--resume` / `--run-id`). Any config key can be
overridden with `--set key=value`.

Outputs live under `runs/<run_id>/`: `results.jsonl` (one line per model call; tool rounds nested),
`summary.json`, `claims.md`, `plots/*.png`, and for every call
`<case>/<condition>/ep<k>_r<rep>/{raw_prompt.json, final_prompt.json, response.txt, trace.json, diff.txt}`.

## Conditions

`baseline` (C0) · `truncate` (C1, matched to C4's budget) · `uniform_lingua` (C2, one rate matched to
C4's budget) · `sieve_rules` (C3) · `sieve_full` (C4) · `sieve_nomemo` (C5, no ledger/memo) ·
`reference_bigctx` (C6, raw prompt with num_ctx 8192). Ablations of C4: `+noverifier`, `+noprefetch`,
`+noledger`, `+clf_heuristic`, `+core6`.

## Results (standard tier, generated from `runs/standard_main2/summary.json`)

| Condition | Accuracy | Prompt tokens | Saved % | Headroom | Δ acc vs C0, pp [95% CI] | Δ cold TTFT, s [95% CI] |
|---|---|---|---|---|---|---|
| C0 raw prompt | 0.861 | 2747 | 0.0 | 1349 | — | — |
| C1 truncate | 0.444 | 999 | 58.7 | 3097 | -41.7 [-48.9, -34.4] | -1.36 [-2.32, -0.58] |
| C2 uniform LLMLingua-2 | 0.778 | 1652 | 33.6 | 2444 | -8.3 [-13.9, -2.8] | -0.25 [-0.61, +0.05] |
| C3 Sieve rules | 0.878 | 1646 | 34.7 | 2450 | +1.7 [-5.0, +7.8] | -1.15 [-1.47, -0.92] |
| **C4 Sieve full** | 0.911 | 1703 | 31.0 | 2393 | +5.0 [-0.6, +11.1] | -1.12 [-1.49, -0.66] |
| C5 no ledger | 0.917 | 1698 | 30.9 | 2398 | +5.6 [+0.0, +11.1] | -0.20 [-0.38, +0.15] |
| C6 8k context | 0.994 | 2747 | 0.0 | 5445 | +13.3 [+8.3, +18.3] | +2.14 [+1.61, +2.80] |

180 questions from 76 conversations on qwen3:14b, num_ctx 4096. Paired differences against C0 use 2,000 bootstrap resamples.
Latency for C2, C5, C6 and the C4 −verifier ablation is confounded by host memory pressure during those passes (see the paper, Section V-B); their token
and accuracy results are unaffected. The full list of supported and unsupported claims is in
[`runs/standard_main2/claims.md`](runs/standard_main2/claims.md), and the paper is [`paper/standard_main2/main.pdf`](paper/standard_main2/main.pdf).

## Testing and acceptance

- `python -m pytest -q tests`: 27 tests (unit and integration), all runnable on the mock client without a GPU or Ollama.
  Line coverage is about 82% (`--cov=sieve`).
- GitHub Actions (`.github/workflows/ci.yml`) runs the suite on Python 3.9 and 3.11 and a mock end-to-end smoke benchmark on every push.
- Reproducibility: every model call is logged (`results.jsonl`) with its raw prompt, final prompt, response, per-segment trace and diff
  (`runs/standard_main2_traces.tar.gz`). `python -m sieve report|html|paper --run standard_main2` regenerates every table, page and the paper.
- Data integrity: the benchmark refuses to run if the tokenizer, compressor or embedder falls back, and flags calls whose decode
  speed drops below half the calibrated value. Discarded runs are kept for transparency (`runs/_discarded_*`, `results.contaminated_backup.jsonl`).

## Citation

See [`CITATION.cff`](CITATION.cff). Code: https://github.com/Tarun-Vaibhav-V/sieve (MIT license).

## Known limitations

- **Hybrid model cache behaviour / measurement.** On this Ollama build, `prompt_eval_count` does not
  drop on a KV-prefix hit, so prefix reuse is **timing-derived** (fitted cold prefill curve).
  `cachetest` validates the method: 95% reuse on a shared prefix, 8% with one word changed at
  10% depth.
- **Tool-call reliability.** With thinking disabled, qwen3:14b rarely calls `expand_context`
  unprompted. Sieve appends a cache-safe tail hint listing stub ids and keys (DECISIONS D15).
  Missed and wasted expansions are logged per call.
- **Extractive summaries.** Stubs use a leading-sentence + TF-IDF summary and up to 5 salient keys.
- **Text-only attachments.** `ocr_image` and `transcribe_audio` are disabled stubs.
- **Small-scale synthetic benchmark.** The entities are invented and the filler comes from a template
  grammar. Results describe this suite on this model, not real conversations.
- **Classifier labels.** Oracle labels are partly a deterministic function of the features, so
  classifier F1 is not evidence on its own; downstream accuracy and token effects are.
- **No CPU offload here** (100% on GPU), so the "slow partial-offload" regime of the original
  target hardware was not exercised.

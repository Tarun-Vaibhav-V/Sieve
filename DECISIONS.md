# DECISIONS.md

Choices made where the spec was silent, or where this machine forced a deviation. Each entry
says what was decided and why. "Observed" means measured on this machine during the build.

## Environment / deviations from the spec

| # | Decision | Why |
|---|---|---|
| D1 | **Model `qwen3:14b` (Q4_K_M, 14.8B)** instead of `qwen3.5:9b`. | The spec model was not installed; the user chose the installed `qwen3:14b`. The tag is configurable (`--set model=...`). |
| D2 | **Hardware is an Apple M5 Pro, 24 GB unified memory**, not an RTX 4050 6 GB. | This is the machine available. Ollama reports `size_vram == size` (8.97 GiB, 100% on GPU, Metal), so there is **no CPU offload**; the offload-split logging still runs every pass. |
| D3 | **VRAM** comes from `/api/ps` `size_vram` (resident model memory), labelled `vram_source=ollama_ps(size_vram)`. | No NVIDIA GPU, so pynvml and nvidia-smi are unavailable (the code still tries them first). This is resident memory, not a sampled peak. |
| D4 | **Python 3.9.6** (system interpreter) with `from __future__ import annotations`, not 3.10+. | Avoids installing a second interpreter; no 3.10-only syntax is used. |
| D5 | **LLMLingua-2 `bert-base-multilingual-cased-meetingbank`** instead of `xlm-roberta-large`. | The user chose the lighter model (about 3–4× faster on CPU). It is set in `config.yaml`. |
| D6 | **Tokenizer `Qwen/Qwen3-14B`** (HF). selfcheck calibrates chat-template overhead against cold Ollama counts: `per_message=5`, `per_request=11` (includes the ` /no_think` suffix and the empty think block). | Observed: tokenizer total 1821 vs Ollama `prompt_eval_count` 1821 on a cold 1.8k-token prompt (exact). |

## KV-cache measurement (important)

| # | Decision | Why |
|---|---|---|
| D7 | **Prefix reuse is measured from prefill *time*, not `prompt_eval_count`.** `prefix_reuse_pct = 1 − prompt_eval_s / predicted_cold_s(n)`, where `predicted_cold_s(n) = a·n + b·n²` is fitted from novel cold prompts in `cachetest`. `tokens_processed_est = n·(1 − reuse)`. Both raw values (count and time) are logged. | Observed: Ollama 0.32.15 runs a `llama-server` backend whose `prompt_eval_count` reports the **full** prompt even on a cache hit. Example: 1,871 tokens reported both times, while prefill time fell 3.33 s → 0.18 s. Count-based reuse would wrongly read "not observed". |
| D8 | `flush_prefix()` is kept, but **cold is guaranteed by `unload()` + novel content**, not by the flush. | Observed: llama-server keeps a RAM prompt cache, so a prompt seen earlier can be restored after a flush. The benchmark unloads the model before every pass and every conversation is novel. `cachetest` and calibration prompts carry a random nonce. Demos unload before each condition. |
| D9 | `OLLAMA_*` env vars are **not set** by Sieve; the observed server env is recorded in `env.json`. | Spec: document, do not set silently. On this machine the Ollama app passes `--flash-attn auto -np 1 -c <num_ctx>` to llama-server itself. |

## Workloads

| # | Decision | Why |
|---|---|---|
| D10 | `needed_segment_ids` holds **workload-level message ids** (`mNN`). A runtime segment counts as *needed* if it contains one of the evaluation point's `facts` (exact fact sentences). | Runtime chunk ids depend on the tokenizer and chunker; fact containment is exact and tokenizer-independent. |
| D11 | Generation-time size estimates use fixed chars/token constants (prose 5.8, JSON 3.0, CSV 2.6, logs 4.4, code 3.4), calibrated once against the Qwen3 tokenizer. | Keeps generation deterministic and independent of the installed tokenizer. Actual sizes are measured with the tokenizer at run time. |
| D12 | Tool results appear in scripted history as **user messages with `kind=tool_result`** ("Tool output from `x`: ```...```"). | The scripted history has no real assistant tool call to attach a `role:tool` message to. The canonicalizer is selected by `kind`. |
| D13 | Integer gold numbers must match **exactly**; dates are scored with `gold_all` (day, month and year must all appear); yes/no uses the first yes/no word; unanswerable uses a refusal regex plus a no-numbers check. | Stops near-miss numbers from scoring as correct. |
| D14 | Benchmark seed 7; classifier training seeds 1001–1006 (6 × standard tier = 456 conversations). | Training and evaluation conversations are disjoint. |

## Pipeline

| # | Decision | Why |
|---|---|---|
| D15 | **Tail hint with inline stub keys.** When tools are attached, a deterministic line is appended to the *final* user message, listing up to 6 stubs as `ref:ID (keys: ...)`, keyed stubs first. | Observed: with `think:false`, qwen3:14b called `expand_context` only when explicitly prompted. With thinking on it called the tool by itself, but thinking is forbidden in measured calls. Plain-ID hints failed on a real long prompt; the keys-inline hint expanded the right stub and did not expand when the answer was visible. It sits in the tail, so the cached prefix is untouched, and its tokens are budgeted (`hint_allowance`). |
| D16 | **Verifier context check** (`verifier_min_context_recall: 0.5`). Around every surviving atom, ≥50% of the content words of its original sentence must remain within ±12 words, or the segment is re-inflated one bucket. | Observed: at ratio 0.3 LLMLingua kept "Draekrousk Draenyl Kroumnaesk" but dropped "product lead". Atom recall stayed 1.0 while the fact was destroyed. |
| D17 | The verifier does **not** receive the workload's `critical_atoms` by default. | Passing gold atoms into the system under test would leak ground truth. The `atom_recall` metric is computed independently from the gold atoms on the final prompt. |
| D18 | LLMLingua output passes through a **digit-split repair** (`5 585` → `5,585`, `4. 1` → `4.1`), applied only to atoms present in the original text. | Observed: LLMLingua-2 re-joins sub-tokens with spaces and drops commas. Without the repair the verifier re-inflated almost every numeric segment. |
| D19 | **Merged stubs:** when every chunk of a message is externalized, the message gets one stub (id = sha256 of the whole message). Expanding it returns the full message. | Per-chunk stubs (~70 tokens each) dominated long sessions. |
| D20 | The `expand_context` tool is attached the first time anything is externalized and **stays attached** for the session. A tools change counts as a whole-prefix rebuild in `rebuild_tokens`. | Ollama's Qwen template renders tools inside the system block, so toggling them would invalidate the whole prefix every turn. |
| D21 | Near-duplicate → back-reference **only if every atom of the new version already appears in the original**; otherwise it is routed normally. | Paraphrased repeats can carry contradicting values (family J). |
| D22 | Pinned docs are emitted in a "Pinned reference documents" section of the system message; their original position holds a one-line pointer. | Spec layout: system → pinned → history → query. Chat turns are never reordered. |
| D23 | Epochs: trigger when projected prompt + hint allowance + reserve > 0.8·num_ctx. The allocator then targets `epoch_hysteresis (0.85) × trigger`, followed by an exact re-render check against `budget_limit = min(0.8·num_ctx, num_ctx − reserve)`. | Without hysteresis every following turn re-triggers an epoch. The exact check makes the budget invariant hold after merged stubs and the template overhead are known. |
| D24 | Prefetch room is limited to `budget_limit − prompt − hint`, so prefetched text also respects the budget. In `prefetch` mode the similarity threshold is min(threshold, 0.5). | Keeps "never exceeds the limit unless hard_truncation" true for the actual request. |
| D25 | `recency_norm = min(1, turns_ago / 10)`. | The spec says "recency_norm (turns)". A turn distance (not a share of the conversation) makes "≥6 turns old" learnable. |
| D26 | The rules-only router (classifier evaluation) = hard rules, else COMPRESS@0.9. | The safe default the spec uses for low confidence. |

## Conditions

| # | Decision | Why |
|---|---|---|
| D27 | **C3 `sieve_rules`** uses the same ledger, store, tool loop and canonicalizer as C4; only the router (heuristic labeller) and the verifier (off) differ. | Isolates the effect of ML routing and the verifier. |
| D28 | **C5 `sieve_nomemo`** = no ledger (full re-route every turn) **and** no compression memo. LLMLingua is deterministic, so prompt changes in C5 come from re-routing (features depend on the current question), not from compressor randomness. | Matches "re-routed and re-compressed every turn". |
| D29 | Matched budgets: C4's plan is computed per evaluation point **without model calls** (expansions excluded). C1 drops whole oldest messages (system and query kept). C2 uses one rate on a 0.05 grid, stepping down until within 2% of the budget. | Spec: compute Sieve's plan first. |
| D30 | C4's classifier = oracle-trained (`classifier_labels: oracle`). The heuristic-trained classifier is ablation `sieve_full+clf_heuristic`; the core-six-features model is `sieve_full+core6`. | Section 13 comparison. |
| D31 | Repeats: standard tier = 1 (spec default for full; the standard default is unspecified). The smoke validation run used 1 repeat. | Wall-clock time on a 14B model. |

## Campaign integrity (added during the standard run)

| # | Decision | Why |
|---|---|---|
| D32 | Each condition runs as its own OS process (`scripts/run_all.sh`); tokenizer and embedding memos use hashed keys with bounded LRU size. | The first full run grew the Python process until the 24 GB host swapped; decode speed fell from about 29 to 3 tok/s. That run was discarded. |
| D33 | Every call records `perf_degraded` (decode speed below half of the selfcheck value). Passes whose median decode speed is below 75% of C0's are excluded from latency comparisons in the paper. | Decode speed does not depend on the prompt, so it shows host health. |
| D34 | The benchmark refuses to start when the tokenizer, compressor or embedder is degraded, unless `--allow-fallback` is given. Runs use `HF_HUB_OFFLINE=1`. | During a network outage three passes silently fell back to an approximate tokenizer or an extractive compressor. They were discarded and rerun (`results.contaminated_backup.jsonl` keeps the originals). |
| D35 | Resume re-runs a whole conversation if any of its questions is missing; aggregation keeps the last record per (case, condition, repeat, question). | One interrupted conversation produced a duplicate line. |
| D36 | Prefetch never triggered at the 0.75 similarity threshold, so `sieve_full+noprefetch` differs from C4 only in timing. This is reported, not hidden. | MiniLM similarities between questions and stubs stayed below the threshold. |
| D37 | The paper is filled from logs through placeholders (`paper/template.tex` + `sieve/paper.py`); the build fails on any unfilled placeholder. References were fetched from doi.org, Crossref, the ACL Anthology and the arXiv API. | No hand-typed numbers or references. |

## Reporting

- Mock runs are labelled `mock=true` on every record and in `claims.md`.
- `decode_tps` is reported but is not expected to change across conditions.
- Oracle labels are themselves a function of features plus gold neededness, so a high held-out macro-F1 partly reflects how learnable the labelling rule is. Only *downstream* accuracy and token effects count as evidence.

# OPT-116 — Generated-text quality admission freeze

Status: **generated quality frozen**. `claims_throughput: false`. No candidate
kernel is installed and no production kernel changes in this task. Authority
remains llama.cpp `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` and GGUF SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`.

User request dated **2026-09-13** is the named authority for this scoped
precision policy: reduced numerical accuracy may be admitted when generated-text
quality passes. PPL ratio stays ≤ 1.01 against authenticated post113 **and**
the OPT-084 freeze. Recurrence incremental NLL stays ≤ 0.02. The successor
changes cross-arithmetic token-identity admission only. OPT-091's 1.015
late_w4 exception does not apply. OPT-016/056 are not relabeled.

## Report fields

| Field | Value |
|---|---|
| strict_quality_pass | **True** |
| successor_quality_pass | **True** |
| absolute_task_quality | **fail** |
| baseline_regression | **pass** |
| cache_path_exercised | **True** |
| state_consistency | **True** |

Inherited `task_arithmetic` A-vs-B remains a visible absolute fail. It is not
recorded as a successful task answer.

## Authenticated NLL

PPL vs post113 (self): {'wikitext_nll': 1.0, 'held_out_wikitext_1024': 1.0}
PPL vs OPT-084: {'wikitext_nll': 0.9997345199858954, 'held_out_wikitext_1024': 0.9997208519718781}
PPL vs llama (inspectable): {'wikitext_nll': 0.9998110598226626, 'held_out_wikitext_1024': 0.9995073123739207}
Recurrence incremental NLL: 0.0 (max 0.02)
Non-finites: 0

`--quality` uses post113 encodings (`q4_decode=llama_q4k_mmvq`,
`q8_decode=r1_w4`, `graph_path=ffn_only`) and cannot restore packed/r2.

## Held-out free-running cases

32 admission cases across {'extraction_long_context': 6, 'instruction_constraints': 6, 'arithmetic_reasoning': 5, 'code_executable': 5, 'tool_call_json': 5, 'multi_turn_prose': 5}.
Calibration is separate (hash `aa39b4b2e4c2…`).
Admission split hash `f8c2d5543344df961964f8ae4f7805ac6f1c4f7f1b4004025bf851d32aeaf9fb`.
Greedy sampler `{'temperature': 0.0, 'top_p': 1.0, 'top_k': 0, 'seed': 0}`. Sample subset
['extract_early_city', 'extract_mid_code', 'instr_three_words', 'instr_json_only', 'arith_add', 'arith_mul', 'code_len', 'code_sum', 'tool_weather_city', 'tool_search_query', 'prose_continue_story', 'prose_summarize_then_ask'] with seeds (11, 29, 47).
Objective failures: 6.
Generation source: `gpu_free_running`.
Free-running GPU executed: `True`.

Prose uses a baseline-blinded 0–2 rubric on instruction compliance, factual
consistency, coherence/completion, and repetition. Raw paired answers and
per-case reasons are stored. No remote judge.

## Cache and GDN

Decode path `compressed_cache_decode` at prefixes 8192, 32768, and 131040+32 / capacity
131072 with dispersed retrieval. Uncached all-prefill is rejected.
GDN stress: 2048 consecutive decode updates, chunk splits,
graph/eager exact, cancellation, save/restore, divergent prefix, seeded
continuation. Long-cache GPU execution this sitting:
`True`.
Measured compressed-cache decode-path mean NLL:
{'cache_8192': 4.187475230470056, 'cache_32768': 2.2673197320040788, 'cache_131040_plus_32': 0.0010480277781056149}.
GDN finite=True save_restore=True
graph_eager_exact=False.

## Approximate-format layers

1. Independent reference for the declared quantizer (scales, clipping, ties,
   tails, zeros, extremes). Corruption is not an approximation.
2. Model quality under `opt116_generated_v1`. Same-path graph/eager and
   checkpoint restore stay bitwise exact. Old same-math gates are not applied
   to a deliberately different representation.

## GPU sitting

available=True ran=True
blocker=none
live compressed-cache decode-path NLL at 8192/32768/131040+32 and 32-case free-running greedy plus 12-case three-seed sample runs executed on RTX 5090 via qw38-cuda:13.0.2
OpenRouter was not invoked. Historical P/D oracles were not run.

## Proof limit

- no throughput claim
- no candidate kernel installed
- no production kernel change
- claims_throughput false
- OpenRouter not invoked
- OPT-056 and OPT-016 remain blocked
- historical OPT-056/073 failures not erased
- strict_quality_pass stays separate from successor_quality_pass
- PPL ratio ≤ 1.01 vs post113 and OPT-084
- recurrence incremental NLL ≤ 0.02
- cross-arithmetic token identity is the successor change
- OPT-091 1.015 late_w4 exception does not apply

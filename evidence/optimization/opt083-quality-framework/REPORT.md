# OPT-083 — ds4-style quality framework for Qwen3.8

Status: **quality framework frozen**. `claims_throughput: false`. No production
kernel change and no speedup are claimed. Authority remains llama.cpp
`cc83d7b4824f73cfdda4dfbb47ee39804f71b328` and GGUF SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`.

This is a methodology port of `../ds4/gguf-tools/quality-testing/`
(teacher-forced continuation NLL, llama control scorer, inspectable per-case
TSV/JSON, comparator with per-example deltas). It is **not** a same-model
comparison with ds4. DeepSeek V4 Flash/PRO and GLM 5.2 official continuation
directories are **not applicable**.

## Scorers

- Quartz: `build/qw38-cuda-opt058-quality-baseline-test` (OPT-058 quality diagnostic, reused)
- llama.cpp control: `tools/run_llama_quality_reference.py` (reused; no second adapter)
- Tokenizer/template: Quartz `enable_thinking=false` /
  `<|im_start|>assistant\n<think>\n\n</think>\n\n`
- Vocabulary: 248320

Identity phase proves Quartz and llama plans share rendered prompts, context
IDs, target tokens, and the teacher-forced NLL definition on frozen Qwen
fixtures owned by this task.

## `--quality` mode

Flag: `--quality`. Selector:
`production_arithmetic_under_test`. Precision stays
`-O2 --fmad=false` / `fast_math=false`. Quality mode disables diagnostic
fallbacks and every catalogued speed-oriented numeric bypass. Unknown
shortcuts fail closed. Graph vs eager, if both run, is
`same_math_equivalence` evidence, not a license to use a different kernel.

Disabled shortcuts: diagnostic_fallback, experimental_cuda, fmad_true, fast_math, incomplete_vocab_logits, logit_masking, teacher_forced_functional_answers, graph_kernel_divergence, speculative_decoding, numeric_bypass, ssd_streaming_approx, truncated_softmax.

Shipping selectors recorded (not changed): Q4 `packed` / `paired_staged`,
Q8 `r2_w2`, MMQ `fma_async_x` `i128_j128`, prompt attention `kv_once`,
decode GDN sequential, decode attention `warp_query`, prompt-pair `off`.

## Suite classes

All inspectable; none is a single boolean.

| Class | Role |
|---|---|
| teacher_forced_continuation | NLL per example and aggregate vs llama |
| known_qwen_continuations | deterministic Qwen fixtures; not remote |
| ppl_1024_spans | existing 1024-target WikiText spans; prefix proof here |
| recurrence_nll | short/long incremental NLL |
| opt073_dual_verdict | absolute vs engine non-regression |
| held_out_32_alarm | 32 teacher-forced targets; alarm, not kernel admission |
| finite_vocab_state_consistency | full vocab + continuation state |

OPT-084 scores the full shipping baseline. This increment uses tiny synthetic
continuations for framework proof.

## Retained historical quality

Quality-v2 remains **fail** (both engines emit A on 17+25). Quality-v3
absolute accuracy is **fail**; engine
non-regression is **pass**. OPT-056
functional tasks stay failed (`opt056_tasks_pass=False`).
OPT-056 and OPT-016 remain blocked and are not relabeled.

## OpenRouter

Interface for `qwen/qwen3.8-27b` exists and is **default off**.
Outputs would be labeled `external_scorer` and non-authoritative. Ordinary
tests must not use the network. Missing `OPENROUTER_API_KEY` is success.
This batch does not call the API.

## Proposed acceptance (numerical freeze is OPT-084)

- primary candidate gate: Quartz vs shipping-baseline regression
- PPL ratio ≤ 1.01 on both 1024-target spans
- recurrence incremental NLL ≤ 0.02
- engine non-regression vs OPT-073 v3
- absolute task accuracy visible and separate
- Quartz-vs-llama NLL/continuation deltas inspectable
- no single projection-error substitute for OPT-074

## Proof limit

- no throughput claim
- no production kernel change
- claims_throughput false
- ds4 methodology port not same-model comparison
- DeepSeek/GLM fixtures not applicable
- OpenRouter default off
- missing credentials are not a failure
- no pytest network
- OPT-056 and OPT-016 remain blocked
- OPT-084 freezes numerical gates
- held-out 32 is an alarm not kernel admission
- quality results are not one boolean

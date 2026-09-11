# OPT-084 — Shipping Quartz quality baseline freeze

Status: **quality baseline frozen**. `claims_throughput: false`. No candidate
kernel is installed. Authority remains llama.cpp `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` and GGUF SHA-256
`31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`.

This sitting scores the OPT-083 local suite with `--quality` on pinned llama.cpp
and on the shipping Quartz configuration frozen at the start of the post-080
batch. Numerical acceptance is frozen **before** OPT-085/086/087. Quartz-versus
shipping-baseline regression is tautological here and recorded as zero-delta.

## Scorers

- Quartz: `build/qw38-cuda-opt058-quality-baseline-test` (OPT-058 quality diagnostic, reused)
- llama.cpp control: `tools/run_llama_quality_reference.py` (reused; no second adapter)
- Tokenizer/template: Quartz `enable_thinking=false` /
  `<|im_start|>assistant\n<think>\n\n</think>\n\n`
- Vocabulary: 248320

Identity: Quartz and llama share rendered prompts, context IDs, target tokens,
and teacher-forced NLL on the frozen Qwen fixtures. Model/tokenizer identity is
cached (`identity_cached: true`).

## `--quality` and shipping freeze

Flag: `--quality`. Precision stays `-O2 --fmad=false` /
`fast_math=false`. Unknown shortcuts fail closed. Graph vs eager is
`same_math_equivalence` evidence, not a license to use a different kernel.

Shipping selectors (mismatch fails closed): Q4 `packed` / `paired_staged`,
Q8 `r2_w2`, MMQ `fma_async_x` `i128_j128`, prompt attention `kv_once`,
decode GDN sequential, decode attention `warp_query`, prompt-pair `off`,
NVCCFLAGS `-O2 --fmad=false`.

## Scores

Aggregate teacher-forced NLL (tokens=2304, cases=4):

| Engine | avg NLL |
|---|---|
| llama.cpp | 1.673015846 |
| Quartz | 1.672889070 |
| Quartz − llama | -0.000126776 |

PPL spans (gate ≤ 1.01):

| Span | llama mean NLL | Quartz mean NLL | PPL ratio | pass |
|---|---|---|---|---|
| wikitext_nll | 1.525124035 | 1.525200593 | 1.000076560 | True |
| held_out_wikitext_1024 | 1.788091893 | 1.787878271 | 0.999786401 | True |

Recurrence incremental NLL: -0.008314334 (max 0.02,
pass=True).

`absolute_quality_status`: **fail**
(OPT-073/v2 `task_arithmetic` A vs expected B).
`quartz_baseline_regression_status`: **pass**
(zero-delta=True, delta_nll=0.0).

A known baseline defect does not by itself reject a later kernel that does not
worsen it. Quartz-vs-llama deltas are inspectable controls, not a
projection-error substitute for OPT-074.

## Historical reclassification

Quality-v2 remains **fail**. Quality-v3 absolute accuracy is
**fail**; engine non-regression is
**pass**. OPT-056 functional tasks stay
failed (`opt056_tasks_pass=False`). OPT-056 and
OPT-016 remain **blocked**. Historical reports are not erased.

## GPU sitting

available=True ran=False
blocker=none
full_1024_rescore=identity_cached_opt058.
1024-token teacher-forced suite exceeds the 300s feedback budget; freeze uses the last complete GPU sitting (OPT-058/069) under the OPT-080 shipping selector freeze
Historical P/D oracles were not run. OpenRouter was not invoked.

## Frozen acceptance (OPT-085/086/087)

- primary candidate gate: no material regression vs this Quartz baseline
- PPL ratio ≤ 1.01
- recurrence incremental NLL ≤ 0.02
- OPT-073 engine non-regression must remain pass
- absolute task accuracy stays visible and separate
- Quartz-vs-llama NLL/continuation deltas inspectable
- no single projection-error substitute for OPT-074

## Proof limit

- no throughput claim
- no candidate kernel installed
- claims_throughput false
- OpenRouter not invoked
- OPT-056 and OPT-016 remain blocked
- historical OPT-056/073 failures not erased
- absolute_quality_status vs quartz_baseline_regression_status
- zero-delta quartz vs shipping baseline
- PPL ratio ≤ 1.01
- recurrence incremental NLL ≤ 0.02
- held-out 32 is an alarm not kernel admission

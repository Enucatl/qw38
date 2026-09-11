# OPT-058 — Finite scheduler and quality baseline

## Claim labels and proof limits

This increment establishes a **finite current scheduler**, a **chat-templated
functional prompt set**, and **held-out llama NLL references**. It makes
**no throughput claim**. Proof boundary:

- no throughput claim
- no unrelated kernel tuning
- single-traversal taps
- missing held-out authority is incomplete
- teacher forcing is not a functional answer
- QLT-001 and OPT-056 verdicts preserved

Production Q4 decode remains `packed`. No kernel tile, MMA, or graph topology
change is admitted here beyond graph-capture staging invalidation.

## Protocol

| Field | Frozen value |
|---|---|
| Runner | `tools/run_optimization_task.py` (OPT-057) |
| Iteration contract | `pins/opt058_iteration_contract.json` |
| Quality contract | `pins/opt058_quality_baseline_contract.json` |
| Native | `build/qw38-cuda-opt058-quality-baseline-test` |
| Image | `qw38-cuda:13.0.2` with `-w /workspace` |
| Llama oracle | `qw38-llama-authority:cuda-13.0.2` / `qw38-llama-quality-oracle` |
| Model | `models/Qwen3.8-27B-Q4_K_M.gguf` |
| Selectors | `q4=packed`, `q8=dp4a_q8_1`, `ffn=paired_staged`, `graph=ffn_only`, `rms=parallel_fma` |
| Feedback | 300 s; smoke / scheduler / functional |
| Acceptance | 7200 s; quality-baseline once, `warm_repetitions=0` |

## Finite scheduler

Tokens `[42, 3649]` ran eager, traced, and FFN-graph with one model load.
Traced execution uses `execute_token_traced_bundle` (`filter=*`) and keeps
layers 0/3/63 residual, `final_norm`, and logits from **one** traversal.

All six mode×token logits are finite (`248320/248320`). SHA-256 is identical
across modes:

| Token | logits SHA-256 (eager = traced = graph) |
|---|---|
| 42 | `f0a6d8fb50d240af6783a42fd75fe327962485d8e2e8e55175df5f83b08db8af` |
| 3649 | `1351ade64e734b149abe059c464c8acddd909a1aa462254e5faf8fc819caf694` |

The reproduced nonfinite on this code is **not** the historical OPT-046
all-NaN sitting. `SchedulerGraphs::create` captures FFN graphs without running
host-side Q8 decode staging updates, then a later eager `execute_token` on the
same workspace reused stale `q8_decode_staged_activation_` and produced 248320
NaN logits. Isolation (separate workspace) was finite. The minimal fix is
`SchedulerWorkspace::invalidate_q8_decode_staging()` at the end of
`SchedulerGraphs::create`. The native scheduler workload regresses eager token
42 on the capture workspace after `graphs.create`.

OPT-046's integer-pin all-NaN logits remain historical /
`unreproduced_on_current_packed_production`. The old rejection fixture is
unchanged.

## Token 271 and original functional sequences

Pinned tokenizer decode of token 271 is `\n\n` (`token_271_hex=0a0a`).

The eight original `fixtures/quality_inputs.json` completion prompts were run
on Quartz and pinned llama, at most 16 free-running tokens. **Both engines
greedily emit 271 first** on every case. That matches OPT-056's failed tasks
and is a **prompt/scorer defect** (bare completion, no chat template / assistant
boundary), not a CUDA-dot mismatch. After 271, both engines usually emit the
choice letter; stripped text is not a valid first-token scorer.

This diagnosis does not change the expected answers.

## v2 functional prompts

`pins/production_quality_v2_inputs.json` renders each original question plus
`Reply with exactly one letter: A, B, C, or D.` through the supported
no-thinking chat template (`<|im_start|>assistant\n<think>\n\n</think>\n\n`).
Logits are not masked. The parser strips only leading/trailing whitespace and
accepts exactly one of A/B/C/D.

| Case | Expected | Quartz first token | Llama first token | Parser |
|---|---|---|---|---|
| task_arithmetic | B | 32 A | 32 A | wrong_choice |
| task_python_len | A | 32 A | 32 A | exact_choice |
| task_inference | C | 34 C | 34 C | exact_choice |
| task_minutes | D | 35 D | 35 D | exact_choice |
| task_sort | B | 33 B | 33 B | exact_choice |
| task_json | A | 32 A | 32 A | exact_choice |
| task_reading | C | 34 C | 34 C | exact_choice |
| task_sequence | D | 35 D | 35 D | exact_choice |

Pinned llama was run first on the v2 prompts. It cannot pass `task_arithmetic`
(`17 + 25` expected B=42; both engines emit A=41). Rendering matches
`src/template.cpp` with `enable_thinking=false`. The miss is the pinned Q4
authority on that item, not scheduler nonfinites. Teacher-forced greedy tokens
are not accepted as functional answers.

## Held-out NLL

Llama NLL was frozen with the quality oracle (model reused, prefill chunks of
512) on `wikitext_nll`, `held_out_wikitext_1024` (OPT-044 offsets 16385–17409,
hash `2caf15f66a602c0bcc1cc2f75c605d48032a29f7c814daa3dc4da5df9c115faa`),
`recurrence_short`, and `recurrence_long`. The old seven-case
`fixtures/quality_llama_reference.json` is untouched. Missing held-out
authority remains `incomplete`.

Quartz teacher-forced NLL vs that freeze:

| Case | Quartz mean NLL | Llama mean NLL | PPL ratio |
|---|---:|---:|---:|
| wikitext_nll | 1.5252005926497396 | 1.525124035418359 | 1.00007656 |
| held_out_wikitext_1024 | 1.7878782710057632 | 1.788091893045086 | 0.99978640 |
| recurrence_short | 1.771902140891233 | 1.7683376992507749 | 1.00357080 |
| recurrence_long | 1.8354702048193376 | 1.8402200970779121 | 0.99526137 |

Recurrence incremental NLL is **-0.00831433** (threshold 0.02). Production PPL
ratio threshold is 1.01. QLT-001 remains 1.05. OPT-056's eight-task fail is
preserved.

## Status

Scheduler is finite on current packed production. Relative NLL/held-out scores
are complete and inside v2 numerical bounds. Functional v2 is **not** a fully
passing pinned authority because `task_arithmetic` fails on llama and Quartz
together. The quality-v2 suite status is **fail**; this prerequisite stays
blocked on that one item rather than by loosening the parser or masking logits.

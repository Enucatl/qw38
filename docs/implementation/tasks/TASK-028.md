# TASK-028 — Reuse-oriented long-context attention

## Status

TODO

## Milestone and dependency

M10 — Fast attention and native projection integration.
Depends on [TASK-027](TASK-027.md).

## Authority and delivered behavior

[DELIVERY-01](../task_ledger.md#delivery-amendment--delivery-01-2026-09-26)
replaces this task's former FP4 experiment. Preserve the accepted weights,
state ABI, FP32 arithmetic and BF16 KV/history. Deliver a production causal
prefill attention path that reuses K/V across query rows and a cooperative
decode scan. TASK-027 already establishes the bottleneck; no baseline rerun
or algorithm bake-off is needed. This is development integration; final quality
promotion belongs to [TASK-030](TASK-030.md).

## First implementation and code areas

Use the existing CUDA online-softmax structure and cache API in
`cuda/attention.cu/.hpp`, `src/runtime/prefill_attention.cpp` and
`src/runtime/attention.cpp`. Start with a 32-query × 64-key tile per query
head, BF16 QK tensor-core products with FP32 accumulators using the pinned
CUDA/CUTLASS primitives, and cooperative FP32 softmax and probability-times-V
accumulation. Keep softmax probabilities FP32; narrowing them for a faster PV
MMA is not part of this first implementation. Retain local running maximum,
sum and numerator rather than a global score matrix. K/V staging is shared
across query rows; map the six query heads to each KV head without permanent
GQA replication. Do not wrap an external inference framework or introduce a
new attention library. The existing scalar Q1/Q4 kernels remain diagnostic
controls, not the intended optimization.

For M=1, retain the current 256-key split and deterministic merge, replacing
one thread's serial 256-element QK dot with a warp-cooperative reduction.
Reuse existing segmented partial storage and gate/output epilogue. This is a
separate decode consumer: do not pad one query to a prefill GEMM and assume
it is faster. Leave current 256-token model chunks and GDN recurrence alone.

Likely additional touch points are runtime workspace sizing if needed,
`tests/*attention*`, `benchmarks/attention_prefill_bench.cpp`,
`benchmarks/attention_bench.cpp` and the existing request benchmark. Reuse
available tensor-core primitives; select the tile from resource limits, not a
sweep. If the initial tile spills or exceeds shared memory, halve the query
tile once and record the reason. A further change needs a concrete failure or
conversion-inclusive result, not proof that the first tile is optimal.

## Representation and lifetime contract

Keep 24 query heads, four KV heads, width 256, scale 1/16, the existing cache
strides and BF16 Q/K/V. FP32 dot accumulators, softmax, output accumulation and
sigmoid gating retain existing rounding to BF16 at attention output. New
reduction order may change floating-point results; it does not change equations.
Causality uses absolute `first_position + row`, including nonempty prefixes,
masked query/key tails and separate populated length versus cache capacity.

Device KV persists across calls; Q/g/y and decode partials occupy bounded
reusable device workspace. Local Q/K/V, scores and accumulators live only in
registers/shared memory for the owning kernel. Never allocate T×T scores,
replicate persistent KV six times, pin cache contents, or rely on values living
in shared memory across launches. Preserve layer/session commit and poison,
reset, restore and stream ownership contracts.

## Smallest useful validation

- Once: affected attention numerical/contract tests against independent FP32
  references, covering causal tails around the new tiles, nonempty prefixes,
  absolute positions, capacity rejection and short prefill-to-decode handoff.
  Test identical-schedule replay and changed decode segmentation/reduction.
  Reuse existing tolerances; investigate failures rather than widening them.
- Once: production request-32768 with 128 generated tokens, using TASK-027
  frozen inputs and timing boundaries, capturing its prompt phase, attention
  timing and peak workspace. Also run populated decode-32768 once to check the
  changed decode path. Compare with saved TASK-027 QW38 observations; no new
  llama.cpp run. Label profiling/first-use differences and do not claim a
  controlled speedup if instrumentation differs.
- One short existing development prompt plus continuation verifies finite
  outputs and the intended dispatch. This smoke is not core-54 acceptance.

## Completion and fallback

Complete when the new prefill path runs through the full model, correctness
and session checks pass, the selected long request demonstrates reduced
attention/prefill cost, and changed decode cost and bounded memory are recorded.
Keep the old decode scan if its cooperative replacement regresses. A failed
prefill candidate requires a targeted repair or a documented concrete blocker;
a keep-only report of the existing 177.87 s scan does not deliver this task.
No numerical failure, unsupported required operation or capacity failure can
be labeled success. No speed-parity or global-optimum proof is required.

## Completion report

TODO — no implementation or acceptance evidence recorded. Record changed
paths, exact commands/results, identities, observed costs, limitations and the
candidate handed to TASK-029. Do not launch the 216-case suite.

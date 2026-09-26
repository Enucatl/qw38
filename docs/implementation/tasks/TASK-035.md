# TASK-035 — Prepared FP8 weights and activation reuse

## Status

TODO

## Milestone and dependency

M14 — Shared narrow operands across real consumers. Depends on TASK-034.
[FAST-01](../post-task030-plan.md) defines the intended dataflow and lifetimes.

## Chosen implementation

Use pinned CUTLASS example 87b groupwise SM120 E4M3×E4M3 with FP32 accumulation:
one activation scale per token/K128 block, one weight scale per N128/K128
block. Start with its cooperative 128×128×128 schedule. Generate weight codes,
scales and physical layout directly from the pinned BF16 source in the compiler;
upload once, with one resident view. Version quantizer/layout/policy explicitly.

Freeze this first recipe before scoring: finite absmax/448 per declared block
as FP32 reconstruction scale, FP32 divide then E4M3 RNE/saturation; an all-zero
block uses scale 1 and zero codes. Positive scale must remain representable
(clamp underflow to the smallest normal FP32); reject nonfinite weights and
propagate/reject nonfinite activations under the existing failure semantics.
Scale layout/padding must match CUTLASS ScaleConfig. No per-chunk amax, fitting
to evaluation answers or conversion of current Q8 bytes into a new quantizer.

First integrate GDN QKV/Z together, since they share the normalized input.
After that short check, apply the same path to large GDN output and attention
QG/K/V/output matrices. Keep tiny GDN A/B, head, embeddings, small parameters
and TASK-034's MLP selection. A demonstrated failing family retains its prior
format by an explicit policy exception; this is not an invitation to sweep
every format. Use same-view FP8/BF16-activation GEMV initially at M=1; compare
native small-M only if conversion or scalar arithmetic dominates that consumer.

RMS emits FP8 codes/scales once and a BF16 companion only for remaining BF16
consumers. QG/K/V reuse one pack; QKV/Z reuse another, while A/B consume the
same producer's BF16 companion. Retain existing BF16 output rounding before
the first FP8 conversion to isolate the new quantization error, implemented
locally without a global BF16 staging tensor where practical. Attention gating
and GDN gated norm produce fresh output-projection operands; fuse packing
into their producers. Avoid unconsumed FP8 packing on M=1 fallback paths.

Use explicit plan-owned spans until the last sibling completes. The consumer
accepts prepacked codes/scales and does not quantize again internally. Reuse
applies across N tiles too; different normalization weights, residual versions,
token ranges or scale recipes are different operands. No persistent activation
cache or FP8→BF16→FP8 round trip for compatible consumers. Nonlinear/reduction
math remains FP32; KV/history and recurrence retain their current precision.

Likely files: `src/compiler/compile.cpp`, `src/format/{constants,layout,pack,unpack,schema}.*`,
`cuda/{activation,prefill,decode_mmv,gdn,attention}.*`, and runtime binders in
`src/runtime/{model,prefill,prefill_gdn,prefill_attention,gdn,attention}.*`.
Use the existing artifact and CUDA boundary patterns, without a new framework.

## Checks and completion

One independent format/contraction check covers zero/tail/outlier/nonfinite
blocks, scale indexing, rounding, M=1/M=256, and identical row packing alone
or beside different rows. One fan-out check proves one pack per producer,
unchanged codes/scales across siblings, correct scratch reuse after the last
consumer and no stale pack after a changed residual. Use existing launch
instrumentation or a focused check; no permanent telemetry service.

Measure one real GDN shared-input projection group including RMS/pack and all
consumers. Check affected attention/output families numerically without a full
per-family timing matrix. Run the predeclared precision development subset and
one FAST-01 short integrated request, including prefill/decode handoff. A speed
win that fails numerical/development quality checks is rejected; isolated GEMM
speed does not establish request improvement. Bound FP8 and companion workspace,
scales/padding and transient load memory from actual layouts.

If groupwise FP8 cannot pass the relevant quality or complete-cost check,
retain affected Q8 families and the TASK-034 compact path; do not silently
weaken scale granularity or quality gates. Record why FP8 was rejected and
which repeated conversion remains. One 256-versus-512 prompt-only chunk check
is allowed after integration only if remaining traffic/launch cost justifies
it and all enlarged workspace/state-boundary checks fit; otherwise keep 256.

Complete with the selected integrated policy, both phase consumers, proven
operand ownership and short numerical/development results. TASK-036 owns the
coherent candidate's final quality, replay, capacity and performance decision.

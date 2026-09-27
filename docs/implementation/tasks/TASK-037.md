# TASK-037 — Integer Q4_K MLP consumers for prefill and decode

## Status

TODO

## Milestone and dependency

M16 — Quantized MLP arithmetic in both phases. Depends on TASK-036.
[FAST-02](../task_ledger.md#fast-engine-amendment--fast-02-2026-09-27)
governs architecture, development checks and final validation ownership.
Start from the TASK-036 development runtime and FP8 mixer artifact; this is
not production promotion. TASK-026/027 remains the rollback control.

## Delivered behavior

All 64 primary-language MLPs use existing Q4_K weights with GPU-produced
signed Q8 activations: integer tensor-core matrix multiplication for M>1 and
DP4A matrix-vector multiplication for M=1. Gate/up share one pack; down uses
a fresh pack of the SwiGLU result. Eliminate global BF16 weight expansion and
weight-by-weight floating reconstruction on this selected MLP path. Keep the
working FP8 mixer consumers, Q8 head, embeddings and small families unchanged.

Adapt the Q4_K block arithmetic and MMQ/MMVQ structure from llama.cpp revision
`e6ab7c1a41054a888ada952eab4c886444c2f5ad`, using the existing third-party
attribution pattern. Reuse the pinned CUTLASS/CuTe dependency where useful;
do not link the llama runtime, import a generic tensor system, or repeat
TASK-034's scalar-unpack WMMA design. Use the pinned implementation's SM120
tile/dispatch pattern as the first schedule; select a concrete configuration
in code, not an autotuner or tile sweep. Submit full output width to each
consumer; kernel-internal tiling is expected.

## Numerical and artifact contract

This is a new execution precision policy, not bitwise preservation of the
old decoded-BF16 contraction. Reserve `Fp8MixerQ8MlpV1 = 0x0406` (1030),
named `precision_fp8_mixer_q8_mlp_v1`, for unchanged TASK-035 FP8 mixers plus
this MLP arithmetic. Add a compiler selection and schema/runtime/report
admission together. Existing policy IDs keep their existing behavior. Do not
silently execute policy 1029 with the new MLP rounding.

Use this activation recipe for each row and contiguous K32 group:

- Evaluate RMS or SwiGLU in FP32, then apply the existing BF16 RNE producer
  rounding locally. Compute FP32 `amax` from those rounded values only.
- For a nonzero group use FP32 `a = amax / 127`; for an all-zero group use
  `a = 1` and zero codes. Encode `z = clamp(RNE(x / a), -127, 127)` as signed
  INT8. Store FP32 `a` and exact INT32 `sum(z)` for the consumer.
- Zero-pad tails without influencing amax; no scale crosses token rows.
  Reject nonfinite producers or invalid scales through the existing typed
  failure/poison path. They must not silently encode as finite zero values.

Q4_K weight codes, superblock FP16 d/dmin and six-bit subgroup scale/minimum
codes retain their existing logical interpretation. For one K32 subgroup,
with `w_i = d*s*q_i - dmin*m`, compute the contribution as
`a * (d*s*sum(q_i*z_i) - dmin*m*sum(z_i))`. Integer dot products and sums
are exact INT32; convert scales and apply affine corrections and accumulation
in FP32. A raw K32 dot is bounded by `32*15*127 = 60960` in magnitude,
and its activation sum by `32*127 = 4064`. Bound any larger or transformed
integer accumulation interval before implementation;
do not accumulate an unbounded K dimension into one INT32 value. Do not round
each reconstructed weight to BF16. FP32 SwiGLU, residual addition and final
outputs remain unchanged. The reference must implement this equation
independently of the packed GPU decoder.

Keep the existing Q4_K physical view first; decode and prefill share it. If
consumer compatibility requires a lossless reordering, prepare it offline,
version the stored layout and replace the resident view. Do not requantize
weights, change their logical quantizer, or retain a duplicate fallback view.
An identity-preserving streaming conversion of the existing artifact may
change policy metadata/layout without recomputing unchanged FP8 weights;
validate the resulting metadata and decoded code/scale equality without
payload digests. New compiler output must reproduce the same contract.

## Producer, workspace and integration

The complete chain is FP32 residual → RMS → local BF16 rounding → one Q8
pack → gate/up → FP32 SwiGLU → local BF16 rounding → new Q8 pack → down →
FP32 residual. Fuse the two packs with their producers where straightforward.
Do not redo RMS, amax or packing inside output-row loops. A pack lives in
session-owned bounded workspace until its last sibling consumer completes;
scratch reuse or a changed residual requires a new pack. No persistent
activation cache, per-token allocations or unused BF16 materialization.
Bound any FP32 gate/up slabs explicitly. Keep chunk capacity 256.

Likely code: `cuda/{activation,prefill,decode_mmv}.*`,
`src/runtime/{mlp,prefill,prefill_gdn,prefill_attention,model}.*`, compiler and
format policy/layout admission, and existing evaluation/profile labels.
Preserve public validation, session stream, ownership, poison and recovery
contracts. Schema negative cases must reject mixed or mislabeled policies
before mutation; the real Session-backed binders must exercise the new path.

## Small checks and acceptance

1. Extend existing projection/MLP tests with an independent pack/affine-dot
   reference: asymmetric q/min/scales, negative activation sums, zero groups,
   halfway rounding, saturation, subnormal weight scales, nonfinite rejection,
   padding/tails, M=1 and M>1, and integer overflow bounds. Codes/sums are exact;
   FP32 contractions use `abs(error) <= 0.02 + 0.002*abs(reference)` against
   the new-policy reference. Keep stricter existing exact checks where
   applicable. Differences against old BF16 operands are diagnostics, not
   failures of the new equation or permission to relax quality gates.
2. Exercise one-pack gate/up reuse, a fresh down pack, workspace reuse across
   layers/chunks, SwiGLU/residual epilogues, policy admission and prefill/decode
   handoff. Verify no global weight-unpack calls on selected MLP dispatch.
3. Once per arm, measure one complete real-shape MLP at M=1 and M=256,
   including RMS, packing, gate/up, SwiGLU, down and residual. Compare with the
   TASK-036 implementation using identical nonzero inputs and weight values;
   label library first-use costs. Kernel-only wins are insufficient.
4. Run the frozen eight-window TASK-035 precision development screen using
   its unchanged inputs/reference and aggregate NLL delta limit +0.03, then
   the FAST-02 short integrated check. Do not fit to C92/P100 answers.
   Record actual workspace and projected final capacity with the 2 GiB reserve.

Completion requires both integer consumers integrated, passing correctness,
development quality and state checks, and lower observed complete MLP cost
at both M=1 and M=256 without an observed matched integrated phase regression.
Use FAST-02's one-run comparison policy; these are development selection
observations, not statistical claims or full-model promotion. A failed path
may remain disabled while repaired, but merely retaining the old fallback
does not complete this task. If the specified consumer cannot qualify, record
the concrete blocker and stop; do not silently mark DONE or start a format
contest. TASK-040 owns final quality, capacity and performance acceptance.

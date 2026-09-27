# TASK-046 — Reduce vocabulary-head and normalization overhead

## Status

TODO

## Milestone and dependency

M25 — Resolve two bounded secondary decode opportunities. Depends on TASK-045.
[FAST-04](../task_ledger.md#optimization-amendment--fast-04-2026-09-27)
governs shared semantics and measurement. Use TASK-045's retained development
candidate, not a mixture of earlier binaries.

## Evidence and scope

TASK-043 records 178.652 ms for the vocabulary head and 229.650 ms for hidden
RMS kernel variants in 128-token 4K decode (7.1% and 9.1% of elapsed time).
The RMS family includes fused Q8 producer packing; preserve that work when
comparing costs. Comparator head time is 87.064 ms, but kernel categories
with overlapping activity are not additive explanations of the wall-time gap.

## Implementation guide

1. Head: test the existing `DirectInput` specialization of
   `cuda/decode_mmv.cu` for the production Q8 vocabulary shape, using direct
   BF16 input reads instead of staging the complete input for every eight
   output rows. Keep other Q8 callers on their existing selection initially.
   Inspect aligned loads and cached input reuse if that hypothesis fails;
   any replacement must stay a small shape-specific change with evidence.
   Preserve the Q8 decoder's per-weight BF16 reconstruction rounding and
   FP32 FMA. Quantizing head activations for DP4A is not authorized.
2. RMS: inspect `hidden_rms_kernel` in `cuda/activation.cu`, including plain
   BF16 output and fused Q8/FP8 packs. First retain each thread's loaded
   residual values for the max, sum and output passes; preserve reduction
   order, robust scaling and rounding. Check the resulting registers and
   spills rather than assuming fewer source loads are faster. Consider a
   shared reciprocal or reduction rewrite only if measurements justify it
   and independent tests cover the changed arithmetic.
3. Evaluate the two changes independently in their component drivers, then
   integrate only winners. A rejected component may retain its old path if
   its concrete first hypothesis was implemented, measured and explained.
   At least one component must deliver a real retained improvement; rejecting
   both is BLOCKED, not successful optimization. Avoid expanding this bounded
   task into fusion, GPU sampling, a new output API or a normalization framework.

Likely callers are `src/runtime/language_model.cpp`, MLP/mixer normalization,
prefill readout and `PrefillEngine::head_generation`. Inspect all consumers
of a changed helper. Shared changes must retain successful prefill behavior.

## Numerical and lifetime contract

Keep FP32 residuals and normalization, `1+gamma`, epsilon behavior and existing
BF16 RNE boundaries. Preserve overflow-safe normalization for extreme finite
inputs, zeros/subnormals and existing nonfinite detection. Multiplying by a
reciprocal may differ from division; validate it rather than declaring bitwise
equivalence. Do not replace robust RMS with an unguarded sum of squares.
Q8/FP8 pack scales, rounding and failure propagation remain policy 1030.

Keep all vocabulary logits available, finite validation and deterministic
argmax tie behavior, typed failure/poison handling and completion before host
commit. No narrowed head, omitted logits, extra resident weights, altered
artifact, hot allocation or changed persistent state.

## Verification and completion

- Run affected Release activation/reference/integration, head/MMV and pack
  checks. Use existing overflow-threshold and asymmetric reference fixtures;
  exercise actual full vocabulary geometry, direct/staged input extents,
  zero/tail cases, BF16 rounding and the real generation/readout caller.
  Apply existing declared tolerances; preserve exact checks where applicable.
- Extend the existing decode-MMV benchmark minimally for the full head,
  and an existing activation/MLP driver for plain RMS and fused Q8 packing
  at M=1 plus M512 shared-path coverage. One execution per arm per affected
  case, identical operands, zero warmups; include stores, packing and
  completion. Correctness tests do not become timing benchmarks. Retained
  components must improve their targeted M=1 complete cost and must not
  regress affected M512 complete cost; a rejected experiment is recorded.
- Run the FAST-04 matched 4096+8 check once per arm: decode improves and
  prefill does not regress. Include head/readout and first use. Run affected
  full-model graph/eager, replay and error-recovery integration checks.
- If arithmetic changes outputs or reduction order, run the unchanged
  eight-window TASK-035 NLL screen with aggregate delta <= +0.03; otherwise
  authenticate the reused screen with affected exact-output checks. Record
  registers/shared/local bytes, workspace and projected reserve.

DONE requires all affected correctness/state/development checks, at least
one retained component win and the integrated improvement above. Document
the measured decision for both head and RMS, including rejected hypotheses.
Do not label their possible gains as established before measurement.
TASK-048 owns combined quality, capacity and final performance acceptance.

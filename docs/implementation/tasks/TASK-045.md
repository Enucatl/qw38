# TASK-045 — Efficient FP8 and Q4 decode projections

## Status

TODO

## Milestone and dependency

M24 — Reduce the dominant recurring decode costs. Depends on TASK-044.
[FAST-04](../task_ledger.md#optimization-amendment--fast-04-2026-09-27)
governs this batch. Start from the TASK-043 approved policy-1030 runtime plus
TASK-044's reporting changes. Development completion does not promote a new
production candidate.

## Evidence and scope

TASK-043's 128-token decode at context 4096 takes 2520.961 ms: Q4 MMVQ
914.006 ms and FP8 GEMV 905.600 ms, together 72.2% of elapsed time.
The relevant source is `cuda/fp8.cu` (`gemv_kernel`) and `cuda/q4k_q8.cu`
(`mmvq`); callers include `cuda/decode_mmv.cu`, the Q8 MLP path and runtime
GDN/attention projection plans. Optimize these existing M=1 consumers.
Keep the artifact, quantizers, prefill consumers, head and state ABI unchanged.

## Implementation guide

1. Reuse TASK-043/044 timelines and record the actual matrix shapes, call
   counts, launch resources and complete operation costs. If those do not
   discriminate the proposed load/instruction change, obtain a targeted
   Nsight Compute diagnostic for one representative shape per changed kernel
   family. Record memory transactions, instruction/stall evidence and resource
   limits. Counter collection is diagnostic, never acceptance timing; document
   profiler replay passes and do not run a full-model counter capture.
2. FP8 first candidate: vectorize contiguous weight/input loads and conversion,
   explicitly reuse each K128 weight scale, and retain BF16 input with FP32
   accumulation. Start with the existing row-major bytes and output epilogues.
   Change row/warp ownership only where the diagnostic supports it. Reusing
   a loaded scale is different from moving multiplication outside a sum;
   start by preserving the current scaled-weight/FMA order.
3. Q4 first candidate: replace the eight small loads per lane/K32 group with
   aligned wider loads and register unpacking. Inspect the strided lane access
   pattern before changing ownership. Keep exact per-K32 integer dots and
   the FP32 affine correction. A new lane assignment may change FP32 reduction
   order and must be checked accordingly. Do not copy upstream half metadata
   or change the activation recipe to simplify the kernel.
4. Select fixed launch choices for the real shapes; retain existing fallbacks
   only where a changed shape fails to improve. No generic autotuner, tile
   sweep, extra weight view, new dependency or compiler/layout conversion.
   Adapt useful patterns from pinned llama.cpp
   `e6ab7c1a41054a888ada952eab4c886444c2f5ad`, with existing attribution.
   A failed measured hypothesis permits a specific diagnosed repair, not a
   search through unrelated schedules.

Gate/up activation packing is already shared, SwiGLU supplies a fresh down
pack, and down residual addition is already fused. Preserve those lifetimes.
Do not bundle speculative gate/up fusion or graph preparation into this task.

## Numerical and integration contract

Retain policy 1030: FP8 stored weights/scales and BF16 decode input;
Q4_K weights with signed Q8 K32 activation codes, FP32 scales and exact sums.
The MLP contribution remains
`a * (d*s*sum(q*z) - dmin*m*sum(z))`, with dot bound 60,960 and sum bound
4,064. No per-weight BF16 rounding in this MLP path; no FP8 activation
quantization in decode. Keep FP32 accumulation, residuals and nonlinearities,
existing BF16 stores, nonfinite failure behavior and scratch ownership.
Scheduling/reduction changes within this contract are tuning; a different
operand/rounding policy is outside scope.

Exercise standalone and production ranged FP8 callers, all three MLP
contractions, residual/in-place output, real session stream, graph capture
and replay. No per-token allocation, host-state commit before completion,
or weakened validation/poison/reset/restore behavior.

## Verification and completion

- Build affected Release targets. Extend existing `fp8_test.cpp`,
  `q4k_q8_test.cpp`, decode/MLP/binding tests only where needed for wider loads,
  tails, changed ownership and production selection. Retain independent
  references, existing tolerances, asymmetric scales/minima, zero/subnormal
  groups, nonfinite inputs and guarded extents. Run affected full-model
  integration cases for graph/eager equality and failure recovery.
- Once per parent/candidate, measure complete M=1 MLP using
  `benchmarks/mlp_bench.cpp`, and complete real-shape M=1 FP8 projection groups
  using `benchmarks/fp8_group_bench.cpp` with minimal extensions if necessary.
  Cover each distinct production shape/epilogue touched, with identical
  nonzero operands. Include preparation, all sibling consumers, epilogues,
  readout/completion and workspace. Report per-shape costs and aggregate
  family cost weighted by production call counts; isolated contraction time
  alone is insufficient. Enforce one execution and zero warmups rather than
  inheriting benchmark repetition defaults.
- Run the matched 4096+8 development check defined in FAST-04 once per arm.
  Decode must improve and prefill must not regress. Report first decode and
  remaining seven separately; a first-use anomaly remains visible.
- Run the frozen TASK-035 eight-window NLL development screen if any operand
  value or FP32 reduction order changes; aggregate delta <= +0.03 against
  its frozen reference. Otherwise record exact unchanged outputs from the
  affected checks as the basis for reusing that evidence. No core-54 rerun.

DONE requires passing affected correctness/state checks, a retained changed
consumer in each family, lower complete FP8 family and MLP costs, and the
integrated result above. Any retained shape must not regress in its matched
complete-operation observation; rejected alternatives keep their evidence.
Failure to qualify a required family is BLOCKED pending a concrete repair or
explicit replan, not completion via the unchanged fallback. Record exact
commands, identities, resource reports and capacity impact for TASK-048.

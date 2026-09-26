# TASK-033 — Attention probability reuse and tensor-core PV

## Status

TODO

## Milestone and dependency

M12 — Attention with less repeated work. Depends on TASK-030.
[FAST-01](../post-task030-plan.md) governs this and subsequent tasks.

## Delivered behavior and first implementation

Use the TASK-030 development attention path. In decode, compute each FP32
softmax probability once per key and reuse it for the denominator and all
256 value coordinates. Cooperatively rescale the running numerator. Preserve
the existing segment size, causal masking, fixed merge order and BF16 KV ABI.

In prefill, keep Q32/K64 and the existing BF16 tensor-core QK path. Replace
scalar PV with tensor-core BF16 P×V and FP32 accumulation. Maxima, exponentials,
normalizers, rescaling and accumulation remain FP32; rounding the local P tile
to BF16 is explicitly authorized as a new precision experiment. Scores and P
remain within the CTA; no global attention matrix or duplicate persistent KV.
Calculate P once, then convert once for its consumer. Do not share invalid
causal tails or softmax statistics between different query rows.

Likely files: `cuda/attention.cu`, `cuda/attention.hpp`,
`src/runtime/prefill_attention.cpp`, existing attention tests/benchmarks.
Reuse existing WMMA machinery. Do not add an attention framework or first
attempt a multi-head fusion: current register/shared-memory occupancy is tight.

Inspect saved TASK-030 C92 outputs/margins before changing arithmetic. If
necessary, run one diagnostic old/new attention case under identical inputs
to investigate the lost answer. Do not infer the root cause merely from the
same artifact, tune to that answer, or make restoring it the acceptance gate.
Performance development can proceed with the documented quality limitation.

## Small checks and completion

- Run affected independent attention numerical/causal tests once, covering
  query/key tails, populated prefix, segment boundary, nonempty-session handoff
  and same-schedule replay. Retain existing tolerances; check P rounding against
  the independent reference, not just the already changed candidate.
- One bounded prefill operation and one decode operation at a fixed 4096-key
  populated prefix (a 32K operation only if needed to decide the long-context schedule), plus the
  FAST-01 short integrated request once. Measure complete attention including
  merge/preparation; no full-model 32K performance run is required here.
- If BF16 P violates numerical/development quality gates, retain the FP32 PV
  control and attempt two BF16 components of P with two MMA contributions only
  if the failure demonstrates precision is the issue. This is one targeted
  fallback, not a precision sweep. If neither qualifies, record the remaining
  prefill bottleneck and retain the independently useful decode reuse change.

Complete when selected paths are wired into real prefill/decode, the affected
checks pass, one-run timings/resource use and retained fallback are recorded,
and TASK-034 can use the resulting development runtime. An inconclusive or
failed full quality gate cannot become promotion; TASK-036 owns that decision.

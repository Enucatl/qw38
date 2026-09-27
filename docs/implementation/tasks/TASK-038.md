# TASK-038 — Pipelined attention with shared KV reuse

## Status

TODO

## Milestone and dependency

M17 — Efficient prefill and decode attention. Depends on TASK-037.
[FAST-02](../task_ledger.md#fast-engine-amendment--fast-02-2026-09-27)
governs scope and validation. Use TASK-037's selected development policy.

## Delivered behavior and first implementation

Replace the current fixed-D256 attention mainloop with an explicit MMA
fragment layout and pipelined KV staging. Reuse installed CUTLASS/CuTe
primitives and the established layout/scheduling patterns in pinned llama.cpp
`e6ab7c1a41054a888ada952eab4c886444c2f5ad`
(`fattn-mma-f16.cuh`, `fattn-common.cuh`, `fattn.cu`), with attribution for
adapted source. This authorizes using those patterns, not copying their
lower-precision accumulation policy. No new attention framework or dependency.

Prefill starts with Q32/K64 and head dimension 256, retaining the running
FP32 PV numerator in the MMA accumulator layout across key tiles. Convert
score fragments directly into the probability operand layout, eliminating
the repeated 32 KiB shared PV-product round trip. Use asynchronous, separately
staged K/V loads with a bounded two-stage pipeline; stage smaller load subtiles
where necessary rather than requiring two full K64/V64 buffer pairs to fit.
Preserve online
rescaling when the running maximum changes. A resource-driven tile adjustment
is tuning, not a new architectural decision; justify it with the affected
resource report rather than a tile sweep. Integrate the existing gated output
and direct FP8 pack for the output projection.

Decode forms one small query tile from the six query heads sharing each KV
head, with two masked padding slots if an eight-slot MMA layout is used.
Load K/V once per such tile and KV partition, retaining independent scores,
maxima, denominators, gates and output coordinates per query head. Start with
the pinned grouped-head layout and split scheduling pattern. Split visible
KV into a bounded number of contiguous ranges to provide enough blocks for
the device, instead of generating one partial for every fixed 256 keys.
Merge partials cooperatively in a deterministic partition order; final
division, sigmoid gating and stores are distributed across threads.

Choose one deterministic partition rule from populated length, query tile
count and device SM count, capped by a checked workspace bound. Freeze it in
the selected implementation and report it. Kernels must distinguish launch
capacity from active length, so TASK-039 can use a fixed launch bucket with
changing positions. Changing the partition rule can change reduction order;
same-schedule replay must remain bitwise, not necessarily bitwise versus the
old segmentation.

## Numerical and memory contract

Keep BF16 Q/K/V and the existing KV ABI; FP32 scores, maxima, exponentials,
denominators, rescaling and output accumulation. Prefill retains TASK-033's
two-component probability recipe, `P_hi = BF16_RNE(P)` and
`P_lo = BF16_RNE(P - float(P_hi))`, contributing both products to FP32 PV.
For grouped MMA decode, use this same two-component recipe as an explicitly
authorized new decode rounding boundary; validate it against the independent
FP32 attention reference. Do not silently substitute a single BF16 P or FP16
accumulator. Gating/nonlinearities remain FP32, followed by existing producer
rounding/packing. No KV requantization, global score/probability matrix,
duplicate persistent KV or changes to model equations/QK norm/RoPE.

Query heads h=6g..6g+5 read KV head g; padding heads never alter statistics or
write outputs. Causal limits use absolute positions even for populated-prefix
prefill. Empty/tail partitions are neutral in merge and never read uninitialized
partial data. Keep split statistics/output scratch session-owned and bounded;
size it once from capacity/dispatch limits and reuse it on the session stream.
Preserve complete-token/chunk commits and poison/reset/restore semantics.

Likely code: `cuda/attention.{cu,hpp}`, attention runtime/prefill binders and
workspace, existing attention references/tests and attention benchmarks.
Q4/FP8 projection policy, GDN recurrence and token chunk capacity remain fixed.

## Checks and completion

- Run affected independent attention tests once, covering asymmetric heads,
  all six GQA siblings, padding heads, query/key tails, strongly separated
  logits, online rescaling, empty/tail splits, causal masking, populated prefix,
  segment/partition transitions and nonempty prefill/decode handoff. Retain
  existing bounds: synthetic FP32-reference outputs within 0.016 absolute;
  artifact mixer output within `0.02 + 0.002*abs(reference)`. Keep any stricter
  applicable existing tests. Do not apply isolated-component tolerances to
  whole-model states/logits. Run the frozen small precision screen because
  the selected decode MMA path changes probability rounding.
- Check same-schedule replay and relevant snapshot/reset/poison behavior,
  guarded scratch tails and no hot allocations. The production binder and
  FP8 output-projection caller must be exercised, not just a standalone kernel.
- Once per arm, compare complete attention preparation/core/merge/gating at
  a 32768-token populated prefix: M=32 prefill and M=1 decode, using valid
  capacity for appended queries and identical nonzero inputs/KV. Measure the
  attention region separately from unchanged projection work. Existing 4K
  correctness fixtures cover short lengths; no full-model 32K run here.
- Report selected registers/shared/local bytes, resident blocks and bounded
  split workspace from the actual compiled kernel. These diagnose resource
  use, not a mandatory instruction audit or proof of peak performance.
- Run one FAST-02 short integrated check and calculate final capacity impact.

Completion requires the new prefill and grouped decode paths selected in real
callers, all affected checks passing, lower observed complete attention cost
in both specified operations and no observed matched integrated phase
regression. Preserve working precision if a schedule fails; fix the concrete
cause within scope. Retaining only the previous schedule or parallelizing
the final store alone does not complete this task. A required path that
cannot qualify is a blocker, not a completed performance improvement.
TASK-040 owns final long-context quality and whole-request measurements.

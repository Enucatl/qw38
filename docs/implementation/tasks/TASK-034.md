# TASK-034 — Compact projection consumers without global weight expansion

## Status

TODO

## Milestone and dependency

M13 — Efficient compact projections in both phases. Depends on TASK-033.
[FAST-01](../post-task030-plan.md) is the representation and validation contract.

## Delivered behavior and chosen implementation

1. Route the one-row Q8 prefill head through the existing decode GEMV consumer.
   Reuse a requested final-position readout as the generation readout instead
   of normalizing/projecting the identical row twice. Keep full FP32 evaluator
   logits, finite checks, greedy tie behavior and complete-token commit.
2. Repair the implemented TASK-029 NVFP4 gate/up path before proposing another
   FP4 format. Submit the full output width to CUTLASS instead of invoking it
   per 512-row output tile. Prepare reusable descriptors/workspace at bind time.
   First try native M=1 with GPU packing once and reuse across gate/up; compare
   its complete cost against Q4_K paired decode before selecting the artifact.
   This introduces FP4 activation rounding in decode compared with TASK-029's
   BF16-input GEMV; validate that changed arithmetic and its phase handoff.
   Start with the existing 128×128×128 native tile, no tile contest. The FP32
   gate/up slabs at M=256 cost about 34 MiB together; bound and reuse them,
   then fuse SwiGLU/narrow output when straightforward. Allocate one reusable
   slab pair per session; no per-layer copies or activation validity across requests.
3. Implement the retained Q4_K down prefill consumer with tile-local unpack
   feeding BF16 tensor cores. Reuse Q4_K reconstruction including its affine
   min/scale terms and BF16 rounding from the existing decoder. No full or
   bounded global BF16 weight tile on this selected path. This is a custom
   mainloop, explicitly part of this task; reuse CUDA/CUTLASS tile patterns.
   Retain FP32 accumulation/residual add. If NVFP4's complete path loses or
   fails quality, reuse this consumer for Q4_K gate/up with paired SwiGLU.

Static weights retain one resident view shared by phases. Use the TASK-029
recipe unchanged for NVFP4; Q4_K code/scale reordering may be lossless, with a
versioned layout if needed. A format switch reloads a separate artifact;
there is no resident Q4_K backup beside NVFP4. No new FP4 calibration or down
requantization. RMS/pack is outside output tiling and runs once per input.
M=1 may retain BF16-input same-view GEMV where that is faster; skip unused packs.

Likely files: `cuda/{prefill,nvfp4,decode_mmv}.*`,
`src/runtime/{prefill,language_model,mlp}.*`, associated format/layout code
only if the selected compact consumer requires it. Existing bounds, session
stream, poison/recovery and tensor identity contracts remain binding.

## Short decision loop and completion

Use one real gate/up shape at M=1 and M=256, one down shape at M=256,
and one real head at M=1; include packing, contraction and epilogue. Validate
independent reconstruction/contractions, tails and residual/SwiGLU behavior
once. Extend existing checks for a reused final readout and workspace reuse;
do not introduce a benchmark framework. Use the existing precision development
screen for selected NVFP4 arithmetic and one FAST-01 short integrated request.

If native M=1 loses, retain Q4_K for both phases and implement the specified
compact Q4_K path; do not make a 4× slower decode the price of native prefill.
If tile-local unpack does not improve total cost, retain bounded unpack/cuBLAS
for affected shapes and document that unresolved traffic. Only investigate
another schedule when the result identifies a concrete cause. Selected paths
must pass numerical/session checks and improve their affected one-run cost
without an observed integrated regression; ambiguous wins do not justify
extra complexity. Record any retained slow fallback honestly.

Complete with working production callers, one selected artifact/dispatch,
actual workspace budget and short results. No full quality rerun or long
request matrix here; TASK-036 owns promotion and matched final performance.

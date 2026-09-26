# TASK-029 — Integrated native NVFP4 MLP gate/up

## Status

TODO

## Milestone and dependency

M10 — Fast attention and native projection integration.
Depends on [TASK-028](TASK-028.md).

## Authority and delivered behavior

[DELIVERY-01](../task_ledger.md#delivery-amendment--delivery-01-2026-09-26)
merges the former FP4, representation and useful scheduling/fusion work here.
Deliver one complete NVFP4 MLP gate/up path from compiler to both production
phases. TASK-020's gate/up screen supports this hypothesis; it does not prove
native W4A4 quality. Keep Q4_K MLP down, Q8 attention/GDN/head, BF16 controls,
FP32 residual/accumulation/recurrent arithmetic and TASK-028 attention fixed.
Final promotion belongs to [TASK-030](TASK-030.md).

## Chosen representation and consumers

Use TASK-019's pinned CUTLASS SM120 native NVFP4 operation and layout utilities,
not the GGUF UE4M3 encoder (TASK-020 documents its different boundary behavior).
Define a new logical quantizer and physical-layout version: E2M1 weight codes,
nonnegative E4M3 scale per 16 contiguous K values, and a FP32 tensor factor.
Choose each weight tensor factor offline from its BF16 range to keep block
scales representable. Record the exact reconstruction `factor × scale × code`,
RNE/tie/saturation/zero behavior and native scale swizzle in the format spec.
Compile directly from pinned BF16, preserving source transformations and
metadata/manifest-only digest policy; never reinterpret Q4_K nibbles as FP4.

Use one kernel-compatible TN resident weight/scale layout for both consumers.
Production GPU activation packing operates per token row and per 16 K values.
Start with fixed FP32 activation factor 1, as in TASK-019 native operand
feasibility; this keeps packing local and token independent. Freeze the recipe
before development scoring. Use existing calibration inputs to diagnose range
failures; do not fit scales to final evaluation answers. Encode
local block scales dynamically, with explicit saturation, zero blocks, padding
and nonfinite handling. Fold fixed activation/weight factors into the GEMM
output scale. No chunk-wide scale: the same row must pack identically beside
unrelated rows. The exact recipe is an implementation contract to document,
not an invitation to search scaling policies. Unexpected saturation causing
numerical/screen failure triggers a focused scale correction or Q4_K fallback.

Start dispatch at M=1 → same-FP4-weight BF16-activation GEMV; M>=2 → native
W4A4 GEMM, preserving FP32 accumulation. The GEMV decodes the same native
codes/scales locally, with no second weight view or full-weight expansion.
Tiny-M padding cost is acceptable initially; adjust the threshold only if a
selected result exposes a regression. Native M=1 GEMM and an MXFP4 comparison
are not required. Reuse TASK-019 native support evidence rather than repeating
instruction audits unless the kernel/toolchain changes invalidate it.

## Producer/consumer integration and memory

FP32 residual → existing RMS/BF16 rounding → GPU codes/scales → gate and up
GEMMs → FP32 SwiGLU → BF16 buffer → existing Q4_K down → FP32 residual add.
Compute RMS once per row. Fuse RMS and packing into a single row producer
where its reduction fits, preserving the BF16 rounding boundary. Reuse the
packed row and scales across both projections until both finish. For decode,
keep BF16 normalized input for GEMV; no unused FP4 activation conversion.
Feed paired outputs into the existing SwiGLU epilogue, avoiding full separate
BF16 gate/up materialization. Bounded FP32 output tiles in device workspace
are allowed; registers/shared memory are only local to each kernel.

Weights/scales persist for model lifetime. Packed activations, scale buffers,
FP32 projection tiles and BF16 SwiGLU use reusable bounded device workspace,
ordered by the session stream and released/reused after the last consumer.
Retain 256-token chunks initially, existing requested-row head and GDN paths.
Budget artifact bytes, padding, scale arrays, upload/transient allocations and
workspace at 32K against the 2 GiB reserve. Load one selected artifact; fallback
loads the accepted Q4_K/Q8 artifact rather than keeping duplicate views.

Likely code areas: `src/compiler/quantization/`, `src/compiler/compile.cpp`,
`src/format/{constants,layout,pack,unpack,schema,reader}.*`, `cuda/prefill.*`,
`cuda/decode_mmv.*`, `cuda/activation.*`, `src/runtime/{model,mlp,prefill}.*`
and CUDA build wiring. Reuse existing artifact validation, binders, projection
interfaces, CUTLASS pin from `scripts/task019_cutlass_sm120.sh` and numerical
references. Add only needed format branches and typed bounded buffers.

## Smallest useful validation

- Once: independent weight/activation reconstruction and contractions for a
  real early and late gate/up matrix, scales, zeros, tails, orientation,
  nonfinite input and malformed/unknown layout rejection. Check fused producer
  rounding, same-row chunk independence, packed input reuse and both consumers.
  Existing TASK-020 ablations are reused; fresh ablations diagnose failures only.
- Once: existing frozen eight-window development NLL screen with the
  TASK-022 comparator-relative +0.03 nats/token bound, plus a short
  continuation/session replay across M=1/2
  and chunk tails. Report the W4A4/GEMV numerical transition explicitly.
  This is development screening, not final EVAL-01 or human adjudication.
- Once: request-4096 and populated decode-4096, each with 128 generated tokens
  and existing TASK-027 inputs/boundaries. Include GPU packing, launches,
  epilogues and cold preparation/upload separately; compare saved QW38 baseline
  observations, labeling instrumentation. Confirm the selected dispatch runs.
  The TASK-027 comparison measures combined attention/projection progress;
  attention gains alone cannot justify retaining a slower FP4 path. No
  intermediate comparator rerun or 32K quality sweep. If attribution cannot
  resolve FP4 retention, use the existing layer harness for one full-MLP
  observation per old/new consumer on identical inputs, including packing and
  epilogues. No extra full-model control run is required. Record changed memory
  and projected 32K capacity; final measurement belongs to TASK-030.

## Completion and fallback

Complete when real gate/up weights run in both phases with GPU packing/reuse,
independent checks and the development screen pass, and conversion-inclusive
request/decode results and memory support handing the candidate to TASK-030.
If native integration fails the screen, capacity or request-cost decision,
record the observed failure, retain the tested Q4_K/Q8 production fallback and
hand TASK-028's attention improvement to final validation. An implemented,
checked rejection is a useful completed milestone; uncertainty alone does not
justify skipping integration. Do not expand into sensitive families, require
both FP4 formats, or begin a new precision research program.

## Completion report

TODO — no implementation or acceptance evidence recorded. Record exact
commands, observed results, selected representation/activation/dispatch
identities, memory, fallback disposition and remaining limitations.

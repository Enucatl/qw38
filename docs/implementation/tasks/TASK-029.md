# TASK-029 — Integrated native NVFP4 MLP gate/up

## Status

DONE

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

**Outcome: IMPLEMENTED / CHECKED REJECTION for NVFP4 request and decode cost.**
The compiler format, common resident NVFP4 TN view, GPU prefill packing/reuse,
native W4A4 GEMM, same-weight BF16-activation GEMV, and fused RMS/packing and
SwiGLU paths are implemented. The eight-window development screen, numerical
checks, session/replay checks, and 32K reserve projection pass. Conversion-
inclusive request and populated-decode observations show that the integrated
NVFP4 path is too slow to promote. Retain `.cache/candidates/candidate-v2-q4k-rope-fixed.qw38`
(manifest `41c1f5e673bb24eb2fb283aa6044dbccdebecc7cd85f847815b3c02a6763fc43`)
with TASK-028 attention for TASK-030. NVFP4 remains an explicit optional
compiler format; runtime selection remains one resident representation, with
no NVFP4 promotion or duplicate fallback view. TASK-030 is not activated by
this report.

The frozen recipe, quantization boundaries, layout/swizzle, activation rule,
dispatch, and reuse contract are recorded in the [TASK-029 NVFP4 recipe](../task029-nvfp4-format.md).
The rejected artifact manifest is
`3904394f34a9d551d400956c6a97bd1c405c990f4461b5b5625df85b625ff786`, compiled
with `qw38-nvfp4-mlp-v1-cutlass-098de2a6-factor-pow2-rne-activation1` version
`0.1.2`; the pinned CUTLASS revision is
`098de2a652cf8f00fd70b2df54051c7eccbb855a`. The captures used Docker image
`sha256:3844dc9c37087cecd40f96e62bd4f305ad405408f0d63312bda8aff8651f2b49`,
CUDA 13.4.59, and an RTX 5090 with driver 590.48.01. Original build/compile
launch copies were `.cache/task029/build.sh` and `compile.sh`; copies retained
with the support evidence are
`.cache/evaluation/qw38-language-v2/task029-support/build.sh` and `compile.sh`.
Code and check mapping: compiler/format encoding and reconstruction are covered
by `src/compiler/quantization/nvfp4.cpp`, `src/format/nvfp4.cpp`, and
`tests/nvfp4_test.cpp`; native packing/GEMM and production phase dispatch are
in `cuda/nvfp4.cu`, `cuda/prefill.cu`, `cuda/decode_mmv.cu`, and
`src/runtime/mlp.cpp` / `src/runtime/prefill.cpp`. The NVFP4 test covers
independent encode/reconstruction and contractions, early/late gate/up tensors,
M=1/2/3 and synthetic M=129, scale swizzle, tails, zero/padding/nonfinite and
malformed layouts, RMS rounding, packed-input reuse, and paired epilogue.
The exact logs, full command scripts, input identities and analysis are retained
under `.cache/evaluation/qw38-language-v2/task029-support/` (also see its
`review-notes.md`).

**Verification commands and results:**

- `bash .cache/task029/build.sh`: pinned Release build; the focused
  `format_schema|format_reader|quant_compiler_integration|decode_mmv_unit|prefill_projection|mlp_unit`
  checks passed 6/6. The initial NVFP4 fixture omission was repaired before its
  dedicated check; these unchanged six checks were not rerun.
- `QW38_AUTHORITY_CHECKPOINT=/workspace/.cache/authorities/qwen3.8-27b-transformers ctest --test-dir build/pinned-release --output-on-failure -V -R '^nvfp4$'`:
  passed 1/1. Sampled real gate/up contractions had zero measured error;
  documented synthetic reduction maximum absolute difference was 0.0126953
  at output about 4142. W4A4-versus-GEMV first-output transitions were recorded
  in the support notes. These samples are not exhaustive full-matrix proof.
- `bash .cache/task029/compile.sh` (retained copy:
  `task029-support/compile.sh`): compiled directly from pinned BF16 using
  `--format nvfp4-mlp`; artifact size 21,463,010,953 bytes. Full artifact
  verification mode was not requested.
- `ctest --test-dir build/pinned-release --output-on-failure -V -R '^language_model_integration$'`
  with `QW38_AUTHORITY_ARTIFACT` set to
  `.cache/candidates/candidate-nvfp4-mlp-v1.qw38`:
  passed 1/1 in 43.40 s, covering reset/restore/replay, M=1/2 transition,
  258-row tails 255/256/257, continuation, requested rows, and late failure.
- `build/pinned-release/src/qw38-evaluate --artifact .cache/candidates/candidate-nvfp4-mlp-v1.qw38 --cases .cache/q4k-candidate/development/cases.tsv --output .cache/evaluation/qw38-language-v2/task029-support/development --eos-ids 248044,248046,248063,248064,248065`, then `uv run --script .cache/evaluation/qw38-language-v2/task029-support/score.py`:
  frozen eight-window / 1024-token NLL was 1.7818974549938833 versus saved
  comparator 1.773838532533603, delta +0.008058922460280282 against the +0.03
  bound: PASS. This is development screening, not final EVAL-01.
- One instrumented request-4096 and one populated decode-4096 observation,
  each with 128 generated tokens, zero warmups/repeats, using the frozen
  TASK-027 input. Exact profile and export invocations are in
  `task029-support/run.sh` / `resume.sh`; analysis ran as
  `uv run --script .cache/evaluation/qw38-language-v2/task029-support/analyze.py`.
  Request cost was 2.0337x the
  saved baseline (10,839.84 ms to 22,044.49 ms); populated decode cost was
  4.3027x (4,281.50 ms to 18,422.18 ms). Paired gate/up kernels directly
  attribute the regression: request 1,229.66 ms old / 15,556.82 ms new, and
  populated decode 1,220.03 ms old / 15,670.30 ms new. The same-weight NVFP4
  M=1 GEMV is retained in decode and causes the checked rejection; no layer
  harness rerun is needed for attribution.
- Request prefill issued 69,632 native CUTLASS GEMMs (1,256.023734 GPU ms),
  1,024 fused RMS/pack calls (23.059890 ms), and 34,816 SwiGLU calls
  (38.450621 ms). Decode had no native GEMM or activation packing. Cold
  request load/upload was 3,270.466345 ms old / 8,060.582695 ms new, with cold
  preparation 3,275.186979 / 8,073.862750 ms. Cold populated-decode load/upload
  was 30,576.477745 ms old / 8,013.625662 ms new, with cold preparation
  30,584.076071 / 8,018.259207 ms. Load includes validation and upload; host
  cache state dominated cold variation, so these are single observations.
- Memory: artifact and model device bytes each rise 32,768 bytes; workspace is
  unchanged. At requested capacity 4,224, model device bytes were
  21,462,845,568, persistent bytes 430,768,128, scratch 256,901,120, tracked
  peak 22,317,568,284, and free memory 10,196,287,488 bytes. Observed resident
  growth was 22,890,414,080 bytes including context/allocator overhead. The
  32K projection adds 1,879,048,192 session bytes, giving projected free
  memory 8,317,239,296 bytes (7.746 GiB), above the 2 GiB reserve. This is a
  projection only; measured 32K capacity belongs to TASK-030.

**Review and repair:** Astra independent review pass 1 found R1: the public
ranged decode API admitted NVFP4 although that API has no NVFP4 dispatch. The
shared entry point now rejects it before launch; the regression constructs
two individually valid NVFP4 descriptors and checks typed rejection,
successful synchronization, and unchanged output sentinels. Exact repair
command `bash .cache/evaluation/qw38-language-v2/task029-support/repair-r1.sh`
rebuilt affected targets and ran
`QW38_AUTHORITY_CHECKPOINT=/workspace/.cache/authorities/qwen3.8-27b-transformers ctest --test-dir build/pinned-release --output-on-failure -V -R "^(decode_mmv_unit|nvfp4)$"`:
2/2 passed. Astra pass 2/2 returned **PASS**, with no findings or further
evidence requests. `binaries.sha256` identifies the original quality/session/
performance capture binaries; `binaries-after-r1.sha256` separately identifies
the binaries after the guard-only repair. Reviewed code and evidence remained
unchanged after pass 2; do not rerun the original observations.

**Limits:** timings are one Nsight-instrumented observation per row, not
medians; cold setup varied with host cache state. No final core-54, final
EVAL-01, measured 32K capacity, or human adjudication is claimed. Promotion,
final combined quality/performance, measured capacity, and delivery decision
belong to TASK-030. No new roadmap task is proposed.

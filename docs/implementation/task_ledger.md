# QW38 Implementation Task Ledger

## Goal

Build a single-GPU, single-sequence primary-language engine for RTX 5090 that
jointly delivers acceptable quality, compact resident weights, fast prefill,
fast populated decode, and low complete-request latency. TASK-001–017 establish
the existing V0 implementation. From TASK-018 onward, select quantization,
physical weight layout, and execution kernels together, using measured SM120
capabilities before committing to the production representation.

## Replan authority — OVERALL-01 (2026-09-24)

The user explicitly requested reconsideration of TASK-018 onward, including
native NVFP4/MXFP4 computation. This amendment replaces the former requirement
to finish the Q4G64 V0 quality/prefill/performance baseline before questioning
its representation. NVFP4 is the leading candidate to investigate, not an
accepted quality or performance result.

**This ledger's revised task rows and task contracts below are authoritative
for TASK-018–032.** The corresponding `tasks/TASK-018.md` through `TASK-032.md`
now expand these contracts with matching titles, dependencies, scopes and
acceptance criteria. Their former specifications are superseded; the earlier
TASK-018 report is explicitly preserved as historical evidence, valid only
within its recorded identities and coverage. TASK-018 owns reconciliation
of affected architecture documents, the evidence inventory and protocol freeze;
those obligations are complete as of 2026-09-24. Updating the task
specifications does not complete any downstream implementation or acceptance
gate.

The amendment reopens Q-01/Q-02 (projection/head precision), the projection
operand part of P-02 (activation quantization), A-01/L-01 (weight views/packing),
and the associated M-01/T-01–03 schedules and materialization choices. It
permits early GDN prefill algorithm comparison under G-02. FP32 residuals,
accumulation/reductions and recurrent arithmetic, BF16 KV/history, and FP32
GDN state remain initial controls; TASK-031 may separately test state precision
and ownership. Model equations, tensor identities, tokenizer, causal behavior,
and primary-language scope remain binding. MTP execution and vision are not
added by this amendment.

EVAL-01 quality criteria and PERF-01 measurement definitions remain binding.
Their old task-number references and requirement that the old V0 gate precede
all experiments are superseded by the migration table below. Candidate
screening may precede routine core quality acceptance; production promotion may not.

The [54-case core amendment](../architecture/evaluation-policy-core-54.md)
governs routine TASK-022–032 evaluation: 15 frozen P100, 15 frozen C92, all
12 L12 and all 12 R-512/R-4096 cases. TASK-026 and later applicable gates
retain the six R-32768 fixtures, while running only the fixed
`R-32768-s0-d0.1` case. The 216-case suite is optional, strictly
human-initiated interactive work; agents and automation must never launch or
require it. The full run already in flight on 2026-09-25 continues untouched
under its captured policy identity. Historical 216-case evidence remains
historical and is not relabeled as a 54-case result.

## Normative authority

- OVERALL-01 and the revised contracts in this ledger for TASK-018 onward
- `docs/architecture/architecture-v0.md` for retained semantics and controls;
  reopened decisions are governed by OVERALL-01
- `docs/architecture/evaluation-policy-v0.md` — EVAL-01 / PERF-01
- `docs/architecture/evaluation-policy-core-54.md` — routine coverage and
  manual-only full-suite execution
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`

## Execution policy

- Execute tasks sequentially; accept every listed prerequisite before starting a task.
- Completion requires the revised ledger contract and its reconciled task file.
  TASK-001–017 completion records are historical and unchanged.
- A conflict outside OVERALL-01's reopened decisions stops the sequence with
  `ARCHITECTURE_BLOCKER`; the obsolete Q4-only restrictions do not block the
  investigations explicitly authorized here.
- Every blocker report contains: `Decision ID`, `Attempted implementation`, `Observed problem`, `Evidence`, `Why this is architectural rather than tuning`, `Smallest plausible alternative`, and `Affected downstream tasks`.
- TASK-018–020 establish controls, feasibility, and a provisional candidate;
  TASK-021–026 implement and validate it; TASK-027–032 measure, tune, and decide
  promotion. Keep unrelated variables fixed in comparisons. Joint format/kernel
  choices are allowed, with weight-only and activation-only error ablations.
- Keep benchmarks separate from correctness tests. Preserve exact commands, results, artifact/binary identities, and hardware context in each completion report.
- Status values are `TODO`, `IN_PROGRESS`, `BLOCKED`, and `DONE`. TASK-018 is
  `DONE`; downstream contracts remain `TODO`.
- A rejected candidate is a useful experiment result. Record the reason and
  select the next eligible candidate without weakening acceptance criteria.
  Missing required evidence cannot be described as a pass.

## Overall strategy and decision rules

### Compute-compatible quantization first

[CUTLASS's SM120 documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#blackwell-sm120-gemms)
documents dense NVFP4 and MXFP4 block-scaled GEMMs and an instruction-dependent
1×–4× throughput comparison with Ada FP8. This is not an end-to-end speedup
claim. The documented SM120 path uses warp-level `mma.sync`, TN operand
layout, and a 1×1×1 cluster; SM100 `tcgen05` assumptions must not be imported.
[NVIDIA's SM120 example 79a](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu)
uses NVFP4 for **both** input operands and FP32 accumulation; BF16 in its name
describes the output. It is not a native NVFP4-weight × BF16-activation example.
These are documentation findings, checked 2026-09-24; actual support and speed
on the pinned project toolchain must be demonstrated in TASK-019.

[NVIDIA's NVFP4 description](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
specifies E2M1 values, an E4M3 scale per 16 values, and a second-level FP32
tensor scale. MXFP4 uses E2M1 with a power-of-two E8M0 scale per 32 values.
The scale representation, quantization recipe, and kernel layout are distinct
contracts. FP32 accumulation does not recover information lost in operands.

| Candidate | Prefill path to evaluate | Decode path to evaluate | Role |
| --------- | ----------------------- | ----------------------- | ---- |
| NVFP4 weights, NVFP4 activations | Native dense block-scaled Tensor Core GEMM, FP32 accumulation | Native small-M GEMM versus a specialized GEMV consuming the same FP4 weights with BF16 activations | Leading compact compute candidate; measure activation error and conversion cost |
| MXFP4 weights, MXFP4 activations | Native dense block-scaled Tensor Core GEMM | Same-weight GEMV versus native small-M GEMM | Required early comparator for scale overhead, quality, and speed |
| Existing Q4G64 weights, BF16 activations | Bounded local unpack to BF16 Tensor Core GEMM | Existing packed Q4 GEMV | Control and fallback; current decode is implemented, this prefill consumer is not |
| Higher precision for failing families | Supported FP8/BF16 GEMM or a demonstrated supported mixed-input kernel | Corresponding compact GEMV/GEMM | Targeted quality fallback; full-model FP8 is capacity-screened before implementation |

Do not reinterpret INT4 nibbles as FP4. Generate each candidate from the pinned
BF16 source, retaining existing Q4/Q8 artifacts as controls. A native W4A4 path
and a W4A16 GEMV may share logical weights while differing in activation error,
rounding, and execution. Validate their transition explicitly. Do not assume
arbitrary mixed operand types or scaling schemes have a native instruction.
Use dense throughput; structured sparsity would change the model and is outside
this replan.

### Quality, memory, and workload together

- Select from quality-passing, capacity-feasible candidates using prefill,
  populated decode, TTFT, total request latency, p99, and peak memory. Publish
  each required PERF-01 row. A large-GEMM TFLOPS result or a decode-only win is
  insufficient; no workload mixture has been supplied to justify hiding losses
  in a weighted average. Freeze any additional regression budgets in TASK-018
  before examining candidate results.
- Freeze a calibration corpus separate from EVAL-01 and a separate development
  screening set. Record source/tokenizer identities, sampling, lengths, layer
  coverage, clipping/scaling recipe and calibration outputs. Include early and
  late layers, MLP down-projection and GDN/attention inputs, prompt and populated
  decode activations. Never fit scales, clipping, or family exceptions to the
  final evaluation answers. Cheap reconstruction checks rank experiments;
  model NLL/capability/continuation gates decide acceptance.
- Specify activation scale scope and lifetime. Prefer a policy whose output
  does not depend on unrelated tokens in a chunk; explicitly test sensitivity
  if using a chunk-dependent tensor scale. Include reductions, scale generation,
  packing, padding, and launches in timings. Reuse a quantized input across
  compatible consumers; avoid repeated quantize/dequantize cycles and blanket
  FP4 quantization of residuals, nonlinearities, attention softmax, or state.
- Prefer one resident native packed weight view usable in both phases. Permit
  measured alternatives: a decode consumer for that layout, bounded local
  conversion, or a second view for selected tensors only. Charge all duplicate
  weights/scales, load/repack time, and workspace to the candidate. Never require
  a whole-model expanded BF16 cache or per-token full-weight repacking.
- Budget against actual allocatable memory on the 32 GB RTX 5090, with a
  recorded reserve. Include retained artifact payloads, embeddings/head, scales,
  padding, GDN state/history, KV capacity, activation buffers, library workspace,
  graphs and any second views, including transient allocation peaks. In the
  active language model BF16 KV grows by 64 KiB/token (2 GiB at 32768), and FP32
  GDN state is 144 MiB; use [the active-state accounting](../architecture/architecture-v0.md)
  rather than the larger conditional MTP map.
- Derived storage lower bounds before padding/metadata: NVFP4 is 4.5 bits per
  weight plus tensor scales; MXFP4 is 4.25; existing Q4G64 with FP16 group scales
  is 4.25. Thus NVFP4's raw weight representation is about 5.9% larger than
  Q4G64, potentially justified by faster computation or better quality. Measure
  actual artifact and resident sizes instead of describing all as simply 4-bit.
- Improve complete prefill: projections, convolution/history, GDN recurrence,
  causal attention, vocabulary readout, chunk scheduling, and handoff all count.
  Repeated single-token decode remains a correctness control, not the target
  prefill implementation. Generation computes final-position logits; evaluation
  computes requested rows in bounded tiles.

### Required measurement envelope

TASK-019 starts with the actual contraction inventory: hidden width 5120,
MLP width 17408, vocabulary 248320, GDN qkv/z and output projections, full
attention q/g/k/v/output projections, and the small gate projections. Use
`A[M,K] × W[N,K]^T`, with logical token counts M = 1, 2, 8, 32, 64, 128, 256,
512 and 1024 where workspace permits, plus representative unaligned tails.
Report logical work and padded work separately. Include real weights and
representative activations, not only square random matrices. Screen larger
chunks before fixing a production chunk size.

Measure kernel time and the complete quantize/pack/GEMM/epilogue path, including
dispatch, synchronization, unpack/conversion, workspace and launch costs.
Compare both phases on identical inputs, epilogues and output precision. Pin
CUTLASS revision, build flags, GPU/driver/toolkit and clocks/power; validate the
architecture-specific compiler target required by that revision instead of
assuming generic `sm_120` enables every specialized kernel. Use instruction or
profiler evidence to establish native block-scaled MMA and identify fallbacks.
CUTLASS/CuTe C++ is an eligible production dependency under the code standards'
capability rationale; benchmark cuBLASLt where it supplies a supported control.
Keep larger calibration frameworks offline.

Final PERF-01 rows remain prefill and complete requests at T = 256, 4096,
32768 and populated decode at T = 512, 4096, 32768 with the declared 128-token
continuation. Include cold load separately from warm steady-state results.
Keep Q4_K_M llama.cpp as the external comparison and label the unfinished
existing QW38 result as a development control. Preserve per-row parity targets,
confidence intervals, and the distinction between a completed measurement and
an achieved performance target.

## Milestones

- **M0 — Reproducible implementation foundation:** the pinned CUDA 13.4.x container builds C++23/CUDA C++23 for `sm_120`, runs a GPU smoke test, and runs the normal test command.
- **M1 — Runtime artifact and compiler foundation:** `.qw38` has an independently tested ABI, writer, validator, metadata model, and BF16 identity compiler path.
- **M2 — Quantized contraction foundation:** Q4G64/Q8G32 have deterministic logical quantizers, V0 packers, independent decoders, reference contractions, and correct decode CUDA consumers.
- **M3 — Core runtime and MLP:** typed CUDA ownership, stable session storage, core numerical primitives, and the first complete MLP execute correctly.
- **M4 — GDN execution:** convolution/preparation, recurrence, and the complete GDN mixer execute with continuation-correct persistent state.
- **M5 — Attention execution:** preparation/cache and segmented online decode attention execute with deterministic merging.
- **M6 — Integrated language decode:** both layer kinds integrate through common state/scratch machinery and the complete 64-layer primary-language model produces FP32 logits.
- **M7 — Strategy and feasibility:** preserved controls, frozen evaluation and
  calibration separation, real-shape SM120 kernel evidence, and a provisional
  quality/capacity-feasible quantization policy.
- **M8 — Compact runtime and both execution phases:** versioned native artifact,
  decode dispatch, bounded GEMM prefill, GDN/attention and validated handoff;
  complete candidate quality evidence including the 32768 extension.
- **M9 — Whole-request measurement:** reproducible matched decode, prefill,
  request, memory, and kernel baseline with explicit gaps.
- **M10 — Measured refinement and promotion:** targeted precision, scheduling,
  fusion and state experiments followed by a consolidated quality/performance
  decision and reconciled architecture documentation.

## Task ledger

| Task | Name | Milestone | Depends on | Primary output/capability | Status |
| ---- | ---- | --------- | ---------- | ------------------------- | ------ |
| TASK-001 | Reproducible CUDA build foundation | M0 | — | Pinned container, native build/test skeleton, `sm_120` GPU smoke | DONE |
| TASK-002 | `.qw38` format constants and schema | M1 | TASK-001 | Explicit little-endian ABI types and schema validation primitives | DONE |
| TASK-003 | `.qw38` writer and integrity records | M1 | TASK-002 | Deterministic aligned artifact emission with manifest-only SHA-256 (current policy; original payload/scale records superseded) | DONE |
| TASK-004 | `.qw38` reader, metadata, and corruption validation | M1 | TASK-003 | Safe parser for tensors, graph bindings, policy, hashes, and state schema | DONE |
| TASK-005 | BF16 identity compiler path | M1 | TASK-004 | Streaming HF-to-`.qw38` identity compiler and exact reconstruction control | DONE |
| TASK-006 | Logical quantizers and CUDA V0 packers | M2 | TASK-005 | Q4G64/Q8G32/BF16 reference quantize, pack, unpack, and contraction | DONE |
| TASK-007 | CUDA runtime ownership and session storage | M3 | TASK-004 | Typed CUDA RAII, artifact upload, persistent state, and scratch arena | DONE |
| TASK-008 | Core reference math and activation kernels | M3 | TASK-007 | Embedding, RMS variants, nonlinearities, RoPE, and reference dense math | DONE |
| TASK-009 | Decode dense contraction consumers | M2 | TASK-006, TASK-008 | BF16/Q4/Q8 CUDA MMV consumers with declared epilogues | DONE |
| TASK-010 | Complete decode MLP | M3 | TASK-009 | Architecture-V0 RMS → paired SwiGLU → down/residual path | DONE |
| TASK-011 | GDN convolution and preparation | M4 | TASK-009 | Reference and CUDA FIR/history, q/k normalization, and gate preparation | DONE |
| TASK-012 | GDN recurrence and continuation | M4 | TASK-011 | Independently validated FP32 `[head,value,key]` recurrent state update | DONE |
| TASK-013 | Complete decode GDN mixer | M4 | TASK-010, TASK-012 | Eight-region GDN mixer plus MLP-compatible residual transition | DONE |
| TASK-014 | Attention preparation and KV cache | M5 | TASK-009 | Per-head q/g, QK norm, partial RoPE, and BF16 cache append | DONE |
| TASK-015 | Segmented online decode attention | M5 | TASK-014 | Causal GQA segment scan, fixed-order merge, gating, output residual | DONE |
| TASK-016 | One-layer integration checkpoint | M6 | TASK-013, TASK-015 | One GDN-style and one attention-style layer through common runtime | DONE |
| TASK-017 | Complete primary-language decode | M6 | TASK-016 | Embedding, 64 layers, persistent state, final norm, Q8 head, logits | DONE |
| TASK-018 | Replan contracts and preserve evaluation controls | M7 | TASK-017 | Reconciled task/architecture authority, evidence inventory, frozen screening/calibration/acceptance protocol | DONE |
| TASK-019 | SM120 quantization and kernel feasibility | M7 | TASK-018 | Real-shape NVFP4/MXFP4/Q4 comparison, native instruction evidence, conversion costs and memory budget | DONE |
| TASK-020 | Calibrated precision policy and candidate selection | M7 | TASK-019 | Weight/activation error ablations, family policy, quality screening and provisional format/layout decision | DONE |
| TASK-021 | Native quantized artifact and compiler | M8 | TASK-020 | Versioned quantizer/scales/layout, calibrated source-to-artifact path and independent reconstruction | DONE |
| TASK-022 | Candidate decode and core quality gate | M8 | TASK-021 | Eligible same-weight native/GEMV dispatch or measured Q4_K/Q8 fallback, full-model decode, continuation and complete 54-case EVAL-01 core evidence | DONE |
| TASK-023 | Production prefill projections and workspace | M8 | TASK-022 | Native GEMMs, activation quantization/reuse, bounded chunks and precision-correct epilogues | DONE |
| TASK-024 | GDN prefill algorithm and layer integration | M8 | TASK-023 | Measured serial/chunkwise recurrence choice, FIR/history and validated complete GDN layer | DONE |
| TASK-025 | Causal attention prefill and layer integration | M8 | TASK-024 | Tiled attention, GQA/cache/position correctness and validated complete attention layer | DONE |
| TASK-026 | Full-model prefill, handoff and quality gate | M8 | TASK-025 | End-to-end candidate, core suite plus fixed 32768 retrieval, chunk/dispatch boundary validation | DONE |
| TASK-027 | Matched whole-request performance baseline | M9 | TASK-026 | PERF-01 comparison, cold/warm costs, peak memory and bottleneck-ranked gap report | TODO |
| TASK-028 | Conversion-minimized native FP4 path | M10 | TASK-027 | Real-family NVFP4/MXFP4 path with GPU activation conversion, native prefill/decode comparison, quality, capacity and complete-request evidence | TODO |
| TASK-029 | Precision and representation refinement | M10 | TASK-028 | Targeted head/family/activation/view tradeoffs with complete quality and request evidence | TODO |
| TASK-030 | Scheduling, dispatch and fusion refinement | M10 | TASK-029 | Measured chunk/crossover, normalization/quantization/epilogue and launch-overhead decisions | TODO |
| TASK-031 | State and long-context bottleneck refinement | M10 | TASK-030 | Evidence-led GDN ownership/state-precision and attention-traffic decisions | TODO |
| TASK-032 | Consolidated validation and architecture promotion | M10 | TASK-031 | Final reproducible artifact/runtime, quality/performance decision, reconciled docs and remaining gaps | TODO |

TASK-024 completed on 2026-09-25. The selected ordered FP32 state-resident
recurrence uses 64-token intervals, supported by complete-layer measurements;
GPU WY is deferred with a documented workspace/support limitation. The R24-04
continuation test was repaired and passed, and Astra's final independent review
passed with no findings, gaps or evidence requests. See the
[completion report](tasks/TASK-024.md#completion-report).

### TASK-019 historical blocker and completion

Historical 2026-09-24 outcome; the user reopened TASK-019 to complete its
remaining acceptance evidence in the main thread.

TASK-019 was `BLOCKED` after its one repair and one supplemental-evidence round.
Independent GPU operand validation, real-input shape coverage and conversion
costs, Q4/BF16 and GEMV controls, complete resident/transient memory budgets,
and a justified candidate shortlist were absent. The support replay also
failed to record the CUTLASS revision inside the container and invoked
`cuobjdump` with an invalid multi-binary resource option. See the historical
[review outcome](tasks/TASK-019.md#review-repair-and-supplemental-evidence-2026-09-24).

The reopened task is now `DONE` for feasibility. The corrected replay records
the host-verified pinned revision and valid per-binary resource output;
independent GPU contractions, 117 real-shape cases, 14,040 raw timing samples,
Q4/BF16 and same-weight GEMV controls, and full one-view memory estimates are
recorded in the [completion evidence](tasks/TASK-019.md#completion-after-reopening-2026-09-24).
TASK-020 records a provisional policy from development screening; this is not
full-model quality acceptance or an end-to-end speed claim. Those gates remain
assigned to TASK-021/022/026.

## Revised task contracts (TASK-018 onward)

These briefs define the required scope and exit evidence. The corresponding
task files expand them without reinstating the superseded sequence.
Every numerical or runtime change keeps component correctness checks and
same-schedule replay; every promoted candidate must pass the applicable 54-case
EVAL-01 core and required long-context extension. Performance
experiments use separate benchmarks and record exact commands and identities.

### TASK-018 — Replan contracts and preserve evaluation controls

Preserve the current artifact, source/build identities, partial evaluations,
profiling results and existing local changes as development evidence. The task
files are synchronized with this replan; reconcile the remaining architecture
decision register, quantization/layout/prefill
plans, technology dependency rationale, and EVAL-01/PERF-01 task references with
OVERALL-01. Distinguish historical V0 controls from proposed candidates.
Inventory which frozen fixtures/references have authenticated provenance and
which require regeneration; retain the original 216-case fixture inventory and
six 32768 retrieval fixtures and their historical scoring records; current
execution uses the single fixed case under EVAL-01. Routine
coverage is governed by the 54-case amendment above.

Freeze calibration/development/evaluation separation, the real-shape benchmark
matrix, memory reserve, and comparison protocol. Retain the latest bounded
decode smoke and continuation evidence as controls; do not launch another
exhaustive old-V0 run merely to unlock feasibility work. **Exit:** consistent
task specifications and decision authority, an explicit missing-evidence list,
and a reproducible protocol. The old V0 quality gate remains unpassed; its
candidate acceptance obligation moves to TASK-022/026, not to a waiver.

### TASK-019 — SM120 quantization and kernel feasibility

Pin an appropriate CUTLASS revision and demonstrate dense NVFP4 and MXFP4
GEMMs on the actual RTX 5090/toolchain. Use minimal reference pack/quantize
paths before committing the production artifact ABI. Check scales, transpose
orientation, padding/tails, zero blocks, FP32 accumulation, and required output
precision against an independent contraction of reconstructed operands.
Separate arithmetic correctness from quantization error against BF16.

Run the measurement envelope above. Include a bounded Q4-to-BF16 GEMM control,
BF16 library GEMM where capacity allows, and estimates or probes for same-FP4
weight GEMV. Measure input scaling/packing and output work, not just GEMM on
prequantized inputs. Reject unsupported mixed formats explicitly. Compute full
resident/transient memory budgets for each viable policy, including selective
FP8/BF16 exceptions and optional views. **Exit:** reproducible support/cost table,
native MMA evidence, and a short list of feasible candidates with a fallback.
No native FP4 winner or large-GEMM-to-decode speedup is assumed.

### TASK-020 — Calibrated precision policy and candidate selection

Quantize from BF16 using the frozen calibration inputs. Start with deterministic
block scaling, then test clipping or calibration improvements only where error
requires them. Specify E2M1 encoding/rounding/saturation, block scale encoding,
second-level scale convention, zero/nonfinite behavior, group axis, activation
scale granularity/lifetime, and packed padding. Test independent weight-only,
activation-only and combined perturbations on representative full layers and
representative inputs, including outliers and prefill/decode inputs.

Choose precision by family: MLP gate/up/down, GDN projections, attention
projections and vocabulary head. Keep embeddings, norms, convolution and small
gate/time parameters as BF16 controls. Treat rotations/smoothing as additional
experiments only if needed, with explicit semantic transformations and runtime
cost; do not make them mandatory or train on evaluation outputs. Use component
BF16 references diagnostically without replacing EVAL-01 with a new BF16 suite.
**Exit:** provisional quantizer/activation/family/layout/dispatch policy backed
by quality screening, kernel timings and memory accounting. Choose MXFP4,
selective higher precision or Q4 fallback if NVFP4 is unsuitable. Full-model
acceptance remains pending TASK-022/026.

**User-directed TASK-020 method amendment (2026-09-24).** Do not run the
corpus-wide 52 GB BF16 model for activation tracing. Follow the inspected DS4
quantization workflow: collect calibration activation statistics with a
GPU-resident quantized model, quantify component perturbations against BF16
source tensors, and compare actual BF16-derived GGUF quantization variants by
teacher-forced target NLL on the same frozen development continuations. Use
llama.cpp to run NVFP4/MXFP4 where its quantizer and kernels support the
selected tensors, with Q4_K_M as the GPU-resident behavioral reference. GGUF
screening diagnoses the precision policy but does not establish QW38 artifact
or runtime quality acceptance. This supersedes TASK-018's corpus-wide BF16
source-control instruction for TASK-020 only; the frozen source, tokenizer,
sampling split, final EVAL-01 inputs, thresholds, and downstream quality gates
remain binding.

### TASK-021 — Native quantized artifact and compiler

Implement the chosen policy as new logical quantizer and physical layout IDs;
do not reinterpret or overwrite existing Q4/Q8 IDs. Document reconstruction
equations, scale arrays including second-level factors, logical axes, kernel
swizzles/alignment, policy/calibration identity, and compatibility behavior.
Pack directly from BF16 into the consumer layout offline. Retain existing
artifact validation and manifest-only digest policy; no unrelated hashing
redesign is introduced.

Validate independent unpack/reconstruction, deterministic compilation, tensor
and scale ordering, padded tails, malformed metadata, unknown IDs, and bounded
host/device memory. A second physical view must satisfy TASK-020's measured
budget and selection rationale. **Exit:** an independently readable candidate
artifact, compiler evidence and real model size/load-memory measurements;
production decode and GEMM consumers can bind the same declared representation.

### TASK-022 — Candidate decode and core quality gate

Integrate the selected weights into all primary-language decode layers and the
head. Compare native small-M W4A4 with same-weight W4A16 GEMV when both consume
the selected format. For the selected Q4_K MLP and Q8 attention/GDN/head
artifact, record native W4A4 as ineligible and retain measured BF16-activation
GEMV per family/shape, with bounded scratch and no hot-path weight repacking.
Preserve FP32 residual/state arithmetic, complete-token commits,
poison/reset/restore semantics and FP32 output logits. Record activation policy
as part of dispatch identity; a kernel switch can change numerical behavior.

Run component and source-semantic regressions, then the complete 54-case EVAL-01 core
(15 P100, 15 C92, all L12 and 512/4096 retrieval, NLL/slices, capability, selected P100 adjudication,
same-schedule replay and declared continuation boundaries). Regenerate invalid
reference evidence; historical partial outputs remain diagnostic only.
**Exit:** a passing candidate decode core and measured populated-decode costs.
If quality fails, diagnose and amend the candidate with the unchanged gate;
do not push an unaccepted quantizer into production-prefill integration.

The [Q4_K MLP candidate screen](q4k-candidate.md) preserves the existing Quartz
runtime precision and changes exactly 192 matrices. Its original artifact
improved development NLL by 0.005765 nats/token over Q4G64 but remained
+0.139444 above the comparator. [Layer-3 numerical attribution](task022-numerical-attribution.md)
found an incorrect compiler RoPE table; the corrected compiler-patch-2 artifact
scores +0.001487 above the comparator on the same eight-window screen and
clears the frozen development promotion bound. Its complete language-v2
54-case core passes every measured quality criterion: aggregate NLL delta is
+0.001365 nats/token, C92 is 8/15 in both arms, L12 is 12/12 in both arms,
and retrieval is 6/6 per horizon in both arms. The paired result is
PASS after the repository owner's no-material-failures P100 review was bound
to the saved outputs and rescored. The user accepted the measured Q4_K/Q8
BF16-activation GEMV fallback and recorded same-weight native W4A4 as
ineligible for this artifact. Independent Astra review passed on pass 2 with
no findings, gaps or requests. TASK-022 is DONE; see its
[task record](tasks/TASK-022.md) for run identities and retained evidence.

### TASK-023 — Production prefill projections and workspace

TASK-023 is DONE: bounded conversion-inclusive production projection paths,
workspace and generation/evaluation head modes passed numerical checks and
family benchmarks. The measured path uses BF16 activations and Q4_K/Q8
unpack-to-BF16 plus cuBLAS Tensor Core GEMM. Full-model prefill/handoff and
matched whole-request performance remain with TASK-026 and TASK-027. See the
[task completion report](tasks/TASK-023.md).

Integrate the selected native block-scaled GEMMs, or the measured fallback,
into bounded token-major prefill. Include activation scale/pack generation,
reuse across compatible projections, paired gate/up organization, proper
SwiGLU and residual epilogues, and requested-row vocabulary handling. Preserve
the precision contract of each consumer; BF16-output example kernels alone
do not satisfy FP32 residual/logit obligations.

Select chunk sizes and GEMM tiles from TASK-019 results rather than freezing
the old 256-token/32×64×64 defaults. Validate dynamic tails, padding masks,
workspace bounds, epilogue equivalence and per-layer projection error.
**Exit:** complete projection paths with end-to-end conversion-inclusive
timings, reusable workspace and correct generation/evaluation modes. Repeated
decode and full-model BF16 expansion are not production prefill paths.

### TASK-024 — GDN prefill algorithm and layer integration

Implement parallel convolution/preparation with correct incoming raw history
and race-free history commit. Compare a state-resident ordered recurrence with
a mathematically equivalent chunkwise/WY-style formulation early enough to
affect the prefill design. Keep FP32 state and the same equations/weights while
isolating algorithmic differences; specify arithmetic precision and workspace
for transformed intermediates. Do not quantize recurrent state to FP4.

Check independent recurrence equations, arbitrary incoming state, short tails,
chunk partitions and continuation into decode. Measure whole GDN+MLP layers,
state traffic and workspace alongside recurrence timing. **Exit:** a justified
prefill algorithm and complete GDN layer with bounded memory and numerical
evidence; the serial 64-token schedule is a control, not a mandatory final
choice. Any deferred faster candidate has a concrete measured reason.

### TASK-025 — Causal attention prefill and layer integration

Integrate q/g split, QK norm, partial RoPE, BF16 KV append, tiled causal GQA,
gate and output projection with the new projection paths. Respect 24 query
heads, four KV heads and head width 256. Reuse KV without materializing a
quadratic score matrix or six persistent GQA copies. Select tiles on SM120
from resources and measurements, preserving FP32 softmax/reductions.

Validate causal masking, existing cache prefixes, absolute positions, tail
tiles, capacity/population distinctions and complete attention+MLP layers.
**Exit:** numerical/continuation checks and short/long-context attention timing
with actual scratch/KV memory, ready for full-model prefill.

### TASK-026 — Full-model prefill, handoff and quality gate

Compose all 64 layers, bounded chunks and final-position logits for generation;
provide requested-row logits for evaluation without allocating T×vocabulary
for the entire prompt. Test prefill into nonempty sessions and continuation
across native GEMM/GEMV dispatch transitions with the same resident weights.

Run the complete 54-case EVAL-01 core through production prefill plus decode, and the
fixed `R-32768-s0-d0.1` case in both required comparison arms. Preserve
the original boundaries/partitions (including 1/63/64/65/255/256/257 and
alternating 63/65), adding boundaries around chosen chunks and dispatch
crossovers. Require bitwise replay only for identical schedules; compare
different schedules using component tolerances and the unchanged behavioral
gates. Test chunk-dependent activation scales when selected; record their
absence for the accepted BF16-activation path. **Exit:** accepted
full candidate quality, correct positions/history/state, bounded measured
memory, and documented coverage. Missing mandatory long-context evidence
blocks acceptance; full prefill equivalence is not inferred from one token.
Historical six-case 32768 runs remain evidence, but the fixed case alone is
required on the final binary under EVAL-01.

### TASK-027 — Matched whole-request performance baseline

Run all PERF-01 rows against its pinned llama.cpp comparator after TASK-026
quality acceptance. Report prompt ingestion, TTFT, populated decode, total
request, p99 and confidence intervals, cold load and peak resident/transient
memory. Separate final-logit generation from multi-row evaluation timings.
Attribute time to quantization/packing, projections, GDN, attention, head,
launches, synchronization and transfers; avoid double-counting host waits as
extra GPU execution time.

**Exit:** complete reproducible baseline and ranked bottlenecks, including
individual losses. Missing parity does not block refinement; missing required
comparison evidence does. No synthetic throughput or old slow prompt loop may
stand in for these production measurements.

### TASK-028 — Conversion-minimized native FP4 path

Build and measure a real native NVFP4/MXFP4 projection path after the matched
TASK-027 baseline. Prepare resident FP4 weights once from the pinned BF16
source; generate and reuse activation codes/scales on the GPU, avoiding CPU
packing and hot-path weight repacking. Include conversion, launches, native
MMA, epilogues and any same-weight decode GEMV in both phases. Compare complete
requests, quality, peak memory and cold costs with the accepted Q4_K/Q8
control. Use TASK-020 family error evidence to bound the experiment, but test
at least one actual model family rather than relying on synthetic kernels.

**Exit:** native instruction, numerical and conversion-inclusive evidence plus
a quality/capacity/performance keep/change decision. A failed FP4 experiment is
a recorded result; native kernel speed alone does not promote a new format.

### TASK-029 — Precision and representation refinement

Use TASK-027 gaps and the TASK-028 FP4 result to revisit only consequential choices: per-family precision,
head precision, activation scaling, or selected additional packed views.
Compare alternatives using identical workloads and calibration separation;
account for added code/layout complexity, memory and cold-load costs. Expand
to FP6/FP8 or mixed inputs only with demonstrated SM120 support and a concrete
quality/performance reason. Keep unrelated state and schedules fixed.

**Exit:** keep/change decisions backed by complete applicable 54-case EVAL-01 revalidation
and whole-request measurements for the promoted variant. A documented decision
to keep the initial representation is valid; exhaustive format combinations
are not required. Unsuccessful variants remain evidence, not default paths.

### TASK-030 — Scheduling, dispatch and fusion refinement

Address measured chunk-size, small-M crossover, activation reuse, normalization
plus quantization, epilogue, launch and synchronization costs. Consider CUDA
graphs only if launch overhead warrants them and stable-address/state/error
contracts remain valid. Include redundant normalization, occupancy, scale
generation and workspace costs when deciding fusion. Keep weights fixed.

**Exit:** justified scheduling/fusion decisions and verified full-request gains
or a measured keep decision. Rerun affected numerical/continuation gates and
the applicable 54-case behavioral gates for arithmetic changes; replay covers graph/dispatch
boundaries and failure recovery. Optimize both prefill and populated decode.

### TASK-031 — State and long-context bottleneck refinement

Use the post-refinement profile to decide whether GDN layout/ownership,
persistent-state traffic/precision or attention KV rereads warrant work.
The 144 MiB FP32 GDN state is small beside weights; narrowing it is not an
automatic priority. If testing BF16 persistent state, retain FP32 recurrence
arithmetic, change only storage, and require long-horizon quality and complete
snapshot/continuation evidence. Version any changed state ABI. Preserve BF16
KV/history in this sequence.

**Exit:** isolated keep/change decisions with populated long-context decode,
prefill/request, quality and memory evidence. If no material state bottleneck
exists, document retention of current precision/layout instead of undertaking
the former obligatory experiments.

### TASK-032 — Consolidated validation and architecture promotion

Freeze the final compiler/calibration/artifact/runtime/toolchain/dispatch
identities and rerun the 54-case EVAL-01 core plus long-context extension and all PERF-01
rows on the combined candidate. Individually passing experiments do not imply
their combination passes. Publish final precision/layout/scale/dispatch and
memory policies, supported contexts, reproducible commands, rollback control,
and remaining performance or coverage limits.

**Exit:** reconciled architecture/format/task documents and a supported
promote/retain-control decision. A quality failure prevents promotion. If
quality passes but performance targets remain unmet or uncertain, report that
explicitly; completing the experiment sequence does not assert speed parity
or global optimality. No further experiment is silently added to this ledger.

## Migration of former tasks and acceptance obligations

| Former obligation | Revised owner |
| ----------------- | ------------- |
| TASK-018 fixture/reference freeze and evidence preservation | TASK-018 |
| TASK-018 repeated-decode core quality gate | TASK-022, applied to selected candidate with 54-case coverage; old V0 evidence retained as incomplete |
| TASK-019 common-view prefill GEMM | TASK-019 feasibility, TASK-021 layout and TASK-023 integration; common view is measured |
| TASK-020 GDN plus TASK-031 / EXP-H recurrence algorithm | TASK-024; algorithm comparison precedes production commitment |
| TASK-021 attention | TASK-025 |
| TASK-022 production-prefill core rerun, handoff and 32768 extension | TASK-026 |
| TASK-023 matched performance baseline | TASK-027 |
| TASK-024 / EXP-A weights, TASK-025 / EXP-B head, TASK-028 / EXP-E activation transport | TASK-019–022 initial decisions; TASK-028 native FP4 experiment and TASK-029 measured refinement |
| TASK-030 / EXP-G extra weight view | TASK-019–021 capacity/layout selection; TASK-029 if measurements justify refinement |
| TASK-029 / EXP-F normalization/projection fusion | TASK-023 integration and TASK-030 measured refinement |
| TASK-026 / EXP-C state precision, TASK-027 / EXP-D ownership | TASK-031, conditional on measured importance |
| Final combined quality/performance reassessment | TASK-032 |

## Critical path

```text
TASK-001 → 002 → 003 → 004 → 005 → 006
                         └→ 007 → 008 ─┐
                              006 ─────┴→ 009 → 010 → 011 → 012 → 013
                                                   009 → 014 → 015
                                      013 + 015 → 016 → 017  (complete decode)
                                                        ↓
                                      018 → 019 → 020  (contracts, feasibility, provisional policy)
                                                  ↓
                                      021 → 022  (artifact, decode and core quality)
                                              ↓
                                      023 → 024 → 025 → 026  (prefill and complete quality)
                                                        ↓
                                      027 → 028 → 029 → 030 → 031 → 032  (measure, test FP4, refine, decide)
```

## Architecture blocker log

| Task | Decision | Summary | Resolution |
| ---- | -------- | ------- | ---------- |

## Task execution blockers

| Task | Blocker | Evidence | Required follow-up |
| ---- | ------- | -------- | ------------------ |
| TASK-017 | Resolved: stale compiler executable reported a graph-binding failure. | Current compiler source already assigns retained MTP layer bindings index 0; rebuilding produced both production and BF16 identity artifacts. | Completed primary-language decode and independent BF16/source validation; see [`TASK-017`](tasks/TASK-017.md). |
| TASK-018 | Resolved: authority and sampler reproducibility findings were corrected and independently reviewed. | Fresh Sol high review of commit `bc3e33823b2a638004a00d30e15260990861a76` returned PASS with no findings or evidence requests. | TASK-020 materializes calibration/development token manifests before fitting; TASK-022/026 rebind preserved evaluation inputs to the reconciled policy identity before acceptance. |
| TASK-022 | Resolved: corrected CandidateV2 passed the 54-case language-v2 core, including the repository owner's output-bound P100 review; independent Astra review passed on pass 2. The user accepted measured Q4_K/Q8 GEMV where no same-weight native W4A4 consumer exists. | The reviewed pair at `.cache/evaluation/qw38-language-v2/paired/llama-20260925T103522Z-694370-candidate-v2-core54-20260925T135734Z-740788-reviewed/summary.json` reports overall PASS, NLL +0.001365, C92 8/15 in both arms, L12 12/12 in both arms, both retrieval horizons 6/6, P100 review PASS, and state replay PASS. The original language-v1 failure remains historical evidence; the optional full-216 report remains inconclusive for P100 and is not acceptance evidence. See [`TASK-022`](tasks/TASK-022.md). | Completed; TASK-023 may proceed according to its dependency. |

## Preserved evidence from the former TASK-018

TASK-018 is `DONE` under the revised contract. The former V0
216-case paired quality gate has no result and is not marked complete by this
replan. The following records describe historical development work; acceptance
of the selected candidate belongs to TASK-022/026.
The first decode review is recorded in [`astra_review.md`](../../astra_review.md).
Decode stabilization was a development checkpoint within the former TASK-018:
the bounded two-prompt, eight-token smoke runs through
`scripts/run_task018_dev_smoke.sh` and passed in 25.9 seconds including load.
Subsequent populated-decode profiling is recorded in
[`immediate-decode-profile.md`](immediate-decode-profile.md): at populated
lengths 507–514 the traced step time fell from 33.17 to 28.61 ms after the
documented Q4 projection changes; a final untraced run measured 27.17 ms.
Prompt ingestion still used repeated decode. This is narrow development
evidence, not a production-prefill baseline or a quality pass. Preserve the
local changes and their measurements. OVERALL-01 replaces the old instruction
to keep tuning Q4 and finish the exhaustive old-V0 gate before TASK-019;
the unchanged core scoring and P100 adjudication now gate TASK-022/026.
The original cache-default llama attempt generated 216/216 outputs but exited 1
during fresh-request replay. Its failed attempt and logs remain in
`.cache/evaluation/qw38-language-v1/runs/llama-20260923T160709Z-1476101/`.
A later four-probe replay matched those preserved outputs, but the source
attempt lacks authenticated per-case provenance, so reuse is diagnostic only.
The full V0 attempt was interrupted at 124/216 completed cases (exit 137); its
partial result and outputs remain in
`.cache/evaluation/qw38-language-v1/runs/v0-20260923T170423Z-1492403/`.
Independent V0 reset/snapshot replay passed all nine declared boundaries and
interleave. P10 timing covers ten selected P100 cases as a speed diagnostic only.
The paired report and 100 P100 human adjudications remain absent. Current
report-contract repairs emit durable `INVALID` results, bind the report driver,
and gate required fixed-key answers. See
[`TASK-018`](tasks/TASK-018.md#coordinator-directed-resumed-work--2026-09-23).

## Architecture amendment log

| Amendment | Date | Decisions affected | Summary |
| --------- | ---- | ------------------ | ------- |
| EVAL-01 / PERF-01 | 2026-09-23 | Historical V0 validation policy and task ownership | The original decision selected DS4-derived language fixtures/scoring, staged context coverage, and llama.cpp Q4_K_M versus V0 as the former TASK-018 behavior pair. OVERALL-01 preserves the quality criteria, remaps candidate acceptance to TASK-022/026, and matched performance to TASK-027. See [evaluation policy](../architecture/evaluation-policy-v0.md). |
| OVERALL-01 | 2026-09-24 | Q-01/Q-02, projection operands in P-02, A-01/L-01, related M-01/T-01–03, GDN prefill algorithm under G-02; S-01/S-02 originally assigned to TASK-030 | User-directed replan originally covered TASK-018–031 with compute-compatible quantization selection before production commitment, native NVFP4/MXFP4 feasibility, calibration and family policy, both execution phases, full quality gates and whole-request measurement. This ledger supersedes conflicting old task scopes/order; TASK-018 reconciles the other documents. Evidence and quality/performance standards are retained under the migration table. |
| FP4-01 | 2026-09-25 | Q-01/Q-02, projection operands in P-02, A-01/L-01 | User-directed TASK-028 adds an end-to-end, conversion-minimized native NVFP4/MXFP4 experiment after the matched baseline. Original future TASK-028–031 shift to TASK-029–032, with the final promotion decision still last. |

## Repair index

The AR IDs below are stable references from the repository-root
`astra_review.md`. Focused commands, measurements, and limits for TASK-016
prerequisites are in [`task-016-prerequisite-evidence.md`](task-016-prerequisite-evidence.md).
A code repair is closed only when its production caller, discriminating
regression, and limits are recorded here.

| ID | Finding | Status | Current contract, production path, and remaining evidence |
| -- | ------- | ------ | ------------------------------------------------------- |
| AR-01 | No source or generated payload digests | Closed by policy | [`code-standards.md`](code-standards.md#checkpoint-and-qw38-payload-digest-policy); checkpoint source identity is index metadata only; manifest digest remains the sole content digest. |
| AR-02 | Verify artifact policy and schema from metadata | Focused checks pass | [`verify_artifact_metadata`](../../src/compiler/compile.cpp) precedes reconstruction; synthetic schema comparison covers policy, bindings, format, shape and ordering. Full checkpoint caller evidence follows the promotion cadence. |
| AR-03 | Failed-state recovery and complete-token commit | Full-model checkpoint passes | [`LanguageModelPlan`](../../src/runtime/language_model.cpp) commits once after final logits; late injected output failure poisons execution and snapshots. Restore/reset continuation replays identical logits, persistent bytes, and metadata. |
| AR-04 | Move-stable plan metadata | Focused checks pass | [`SessionExecutionState`](../../src/runtime/session.hpp) keeps borrowed counters stable; bound GDN/attention plans execute after move construction and assignment. |
| AR-05 | One session stream at session-backed binders | Focused checks pass | MLP, attention, and GDN binders reject an alternate same-device stream before execution. |
| AR-06 | Checked geometry and descriptors | Focused checks pass | Checked runtime views, state indices, and compiler transforms reject rank/count/index/overflow and payload-size errors. |
| AR-07 | Bounded BF16 reconstruction verification | Focused checks pass | [`verify_bf16_payload`](../../src/compiler/compile.cpp) detects corruption in all BF16 mappings and adds less than 0.5 MiB peak RSS for isolated 8/32 MiB identity/tiled inputs. |
| AR-08 | Closed Runtime API | Focused checks pass | [`Runtime`](../../src/runtime/runtime.cpp) rejects post-shutdown upload/session creation; repeated shutdown is safe. |
| AR-09 | Attention direct residual output | Focused checks and trace pass | [`execute_attention_core`](../../src/runtime/attention.cpp) preserves Q4/BF16 input residuals; isolated BF16 trace records six kernels and no D2D copy. |
| AR-10 | Benchmark consumers and launch accounting | Focused checks and trace pass | Excluded MLP/GDN benchmarks build and smoke; GDN trace records eight kernels per execution and no D2D copy. |
| AR-11 | Completion semantics before 64-layer composition | Full-model checkpoint passes | The eager 64-layer plan synchronizes output before the global token commit; `TASK-017` records the measured Debug integration smoke and slow prompt setup. |
| AR-12 | Semantic operand resolution boundary | Focused checks pass | [`resolve_semantic_tensor`](../../src/runtime/model.cpp) resolves canonical V0 names to IDs and validates graph node/role/layer at plan binding; binders validate shape/layout before returning plans. TASK-016's uploaded-model composition verifies both layer families and rejects wrong-family bindings. |
| AR-13 | Integration checkpoint and independent source evidence | TASK-017 checkpoint passes | [`TASK-017`](tasks/TASK-017.md) records authoritative full-model decode, pinned source identity, BF16 logits/residual/state numerical gates, and V0 differences for two tokens. |
| AR-14 | Quality-gate recovery procedure | Documented; task ownership remapped | The former [`TASK-018`](tasks/TASK-018.md) requires preserving baseline, diagnosis, accepted amendment, and unchanged gate retest; no quality failure has been observed. OVERALL-01 retains that procedure in TASK-022/026 and final promotion. |
| AR-15 | Concrete code-boundary and evidence standards | Implemented | [`code-standards.md`](code-standards.md#boundary-lifetime-and-evidence-contracts) covers boundary, lifetime, production-path, numerical, resource, schedule, and closure requirements. |
| AR-16 | Historical authority reconciliation | Partial | Ledger ranges/links and stale TASK-008/010/012/013 claims were corrected or marked historical; no standalone `review.md` exists, and broader historical reports remain unreconciled. |

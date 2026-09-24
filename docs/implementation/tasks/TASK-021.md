# TASK-021 — Native quantized artifact and compiler

## Status

DONE

## Milestone

M8 — Compact runtime and both execution phases

## Purpose

Produce a versioned, independently validated artifact that expresses the selected quantization policy in the layout its consumers need.

## Depends on

- [TASK-020](TASK-020.md)

## Normative references

- [Implementation ledger](../task_ledger.md) — OVERALL-01, revised task contract,
  overall decision rules, measurement envelope and migration of prior obligations.
- [Architecture V0](../../architecture/architecture-v0.md) — retained model semantics
  and controls; reopened decisions follow OVERALL-01.
- [EVAL-01 / PERF-01](../../architecture/evaluation-policy-v0.md) — unchanged
  quality criteria and measurement definitions, with task ownership remapped by the ledger.
- [Technology baseline](../technology-baseline.md).
- [Code standards](../code-standards.md).

## Architecture decisions consumed

| Decision | Contract for this task | Authority |
| -------- | ---------------------- | --------- |
| A-02 | Separate logical quantizer from physical layout ABI | Retained |
| Q-01/Q-02, P-02, A-01/L-01 | Implement the TASK-020 candidate policy and selected views | OVERALL-01 selection |
| Artifact validation and digest policy | Version new representations and preserve manifest-only digest rules | Retained |

OVERALL-01 supersedes conflicting restrictions in the former task sequence.
Historical EXP-A–H ordering does not constrain this task. Decisions outside
the reopened scope remain binding.

## Starting point

TASK-020 selects the provisional quantizer, scales, family policy and layout/dispatch. Existing Q4/Q8/BF16 artifact support remains a diagnostic control.

## Scope

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

## Out of scope

Reusing Q4/Q8 IDs for FP4, whole-artifact/source payload hashing, silent policy changes, full-model expanded BF16 caching and unbudgeted duplicate views.

## Required interfaces and data representation

Extend typed quantizer/layout metadata and compiler policy with exact reconstruction equations, logical axes, packed data and scale spans, second-level scales, alignment/padding and calibration identity. Reader/binder compatibility must reject unknown or incompatible representations explicitly. Compile directly from the pinned BF16 source with bounded streaming memory.

## Required semantics and constraints

Packing preserves the selected logical values and scales. Document tensor/scale ordering and any allowed precision exceptions. Independently unpack every implemented representation, including malformed lengths/offsets and tail padding. Existing artifact IDs retain their meaning. Bound both compile/verification host memory and upload/transient device memory.

Follow the code standards' identity and manifest-only digest policy. Keep
benchmarks separate from correctness checks and record source, binary,
artifact/policy, inputs, toolchain and hardware identities appropriate to each
result. Partial or invalid evidence cannot establish quality acceptance.

## Tuning defaults

Use TASK-020's selected format/layout; changes require a recorded evidence-backed policy update. A second view must meet the measured memory and performance rationale.

## Expected files/modules

Format constants/schema/writer/reader, logical quantizers, physical packers and independent unpackers, compiler policy/streaming path, metadata compatibility tests and artifact report.

## Tests required

### Unit and contract checks

Golden encodings/scales/packing, metadata identity, round-trip logical values, tails, offsets/lengths, unknown IDs and malformed or mismatched representation rejection.

### Reference and numerical checks

Independent reconstruction for real tensor families and synthetic edge cases; deterministic compilation from the same source and calibration policy.

### Integration checks

Compile and independently read the real candidate model; check graph bindings, family precision, declared scale/layout support and actual artifact size. Measure bounded compilation/verification and upload memory, including failure branches relevant to the new representation.

## Benchmark required

Artifact size, compiler/load/upload costs and host/device memory are required. Decode/prefill speed is measured by subsequent production consumers.

## Acceptance criteria

- [x] New representations have explicit IDs, reconstruction/scaling equations, layout and compatibility rules without changing existing IDs.
- [x] Compilation is deterministic from the pinned source/calibration policy and independently reconstructs selected logical values.
- [x] Reader/schema tests reject unsupported or malformed metadata and bindings.
- [x] Real artifact size, load costs and bounded host/device peaks are measured.
- [x] The artifact exposes the representation required by both planned decode and GEMM consumers; any extra view has the approved measured budget.

## Architecture blocker rule

A rejected candidate is a recorded result; use the eligible fallback within OVERALL-01 without relaxing acceptance criteria. Missing required exit evidence prevents completion. A conflict outside the reopened decisions requires the full architecture-blocker report defined in the ledger; obsolete Q4-only or experiment-order restrictions are not blockers.

## Completion report

### Result

Accepted the TASK-020 candidate representation as a versioned Q4G64/Q8G32
artifact with one packed physical view per selected tensor. The candidate is
independently readable and reconstruction-verified, and a second compile
produced byte-identical output. No quality, production-kernel correctness, or
performance claim is made here; production decode/GEMM integration and quality
gates belong to TASK-022/023 and TASK-026.

### Independent review

Astra review passed on pass 2 with no remaining findings, acceptance gaps, or
evidence requests. Pass 1 returned `CHANGES_REQUIRED`; its requested
remediations were completed and closed in the second review.

### Changes made

Added candidate logical quantizer IDs `0x0103`/`0x0104`, physical layout IDs
`0x020B`/`0x020C`, and policy ID `0x0402`; existing Q4/Q8 meanings and
container/manifest versions remain unchanged. Implemented direct streamed
BF16-to-layout compilation, typed schema/reader validation, independent
reconstruction, and failure handling. The format contract, equations, tensor
and scale order, padding, alignment, compatibility behavior, and identity
inputs are documented in
[`task021-candidate-format.md`](../task021-candidate-format.md).

The selected artifact contains one common packed view per selected tensor;
there is no second physical view. It includes 866 tensors and excludes 333
vision tensors. The artifact manifest is
`94c9ed5c9260ebde73b0eb9316b6ae7726ff82fe030f2d4caa72760deec79dd1`.
Compiler identity is `qw38-candidate-v1-policy-2ef01bff2b40095f08aedad3b84c50317ffec66e940019342bd12abcda50110e-cal-8ac4a9cab7181c7f7008f95b52420f0d76775524ac64868d0351411c2c2021a4:0.1.1`.
The candidate policy JSON SHA-256 is
`2ef01bff2b40095f08aedad3b84c50317ffec66e940019342bd12abcda50110e`; the
calibration manifest SHA-256 is
`8ac4a9cab7181c7f7008f95b52420f0d76775524ac64868d0351411c2c2021a4`.

### Tests run

All commands ran in pinned image `qw38-dev:cuda13.4.1-pinned` (image SHA-256
`3844dc9c37087cecd40f96e62bd4f305ad405408f0d63312bda8aff8651f2b49`).
Hardware/toolchain context: RTX 5090, SM120, driver 590.48.01, CUDA 13.4.59,
GCC 14.2, CMake 3.28.3.

- Full compile command, exactly as logged:
  `docker run --gpus all --rm -u "$(id -u):$(id -g)" -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1-pinned build/pinned-debug/src/qw38-compile --checkpoint .cache/authorities/qwen3.8-27b-transformers --output .cache/task021/candidate.qw38 --format candidate`
  Log: `.codex-wake-run/bc7f3edc3316.log`. Completed successfully, writing
  20,928,204,521 bytes; 866 included tensors; compiler-reported compile time
  1.20589e6 ms and peak host RSS 4,001,992,704 bytes. The compile command
  itself reports `reconstruction_verified=false`; verification is recorded
  separately below. Compiler binary SHA-256:
  `ea092bd55bb251a920b2bd0d03f7a0ef7d89c32ac1459c57845b9a1eca3cefee`.
- Independent full artifact verification command:
  `docker run --gpus all --rm -u "$(id -u):$(id -g)" -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1-pinned build/pinned-debug/src/qw38-compile --checkpoint .cache/authorities/qwen3.8-27b-transformers --output .cache/task021/candidate.qw38 --format candidate --verify-only`
  Log: `.codex-wake-run/27dab5103f17.log`. Passed reconstruction and artifact
  validation in 1.13826e6 ms; peak host RSS was 24,039,383,040 bytes.
- Deterministic recompilation and byte comparison command:
  `docker run --gpus all --rm -u "$(id -u):$(id -g)" -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1-pinned bash -lc "build/pinned-debug/src/qw38-compile --checkpoint .cache/authorities/qwen3.8-27b-transformers --output .cache/task021/candidate-replay.qw38 --format candidate && cmp -s .cache/task021/candidate.qw38 .cache/task021/candidate-replay.qw38 && echo replay_bytes_identical=yes"`
  Log: `.codex-wake-run/dbc6d0c50518.log`. Passed; bytes were identical and
  the artifact manifest matched. Replay compile time was 1.20905e6 ms and
  peak host RSS 4,002,226,176 bytes.
- Final focused build and six contract tests command:
  `docker run --gpus all --rm -u "$(id -u):$(id -g)" -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1-pinned bash -lc "cmake --build build/pinned-debug --target qw38_quantizer_test qw38_quant_reference_test qw38_quant_compiler_integration_test qw38_candidate_failure_test qw38_bench_artifact qw38_pack_layout_test -j4 && ctest --test-dir build/pinned-debug --output-on-failure -R \"^(quantizer|quant_reference|quant_compiler_integration|candidate_failure|pack_layout|format_schema)$\""`
  Log: `.codex-wake-run/a406171136a3.log`. Build succeeded; 6/6 tests passed
  (`format_schema`, `quantizer`, `pack_layout`, `quant_reference`,
  `quant_compiler_integration`, `candidate_failure`). This covers candidate
  IDs and policy metadata, quantizer/reference behavior, packed layout,
  compiler integration, malformed/unsupported representation handling, and
  relevant failure paths. It does not establish production decode/GEMM runtime
  correctness.

Compiler, artifact benchmark, and failure-test binary SHA-256 values are
`ea092bd55bb251a920b2bd0d03f7a0ef7d89c32ac1459c57845b9a1eca3cefee`,
`ee48f1f86422b6c4fca8594467501fc55831078ecc7d465f2d2973c183339cf0`, and
`53a16011d6b998d834e24e6e8e04c711069372bd125e9b92c09d8ac7a0f7737f`,
respectively.

### Benchmark results

The final real-model load/upload measurement is one observation from the same
pinned image and hardware context above. Exact command:
`docker run --gpus all --rm -u "$(id -u):$(id -g)" -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1-pinned build/pinned-debug/benchmarks/qw38_bench_artifact .cache/task021/candidate.qw38`
Log: `.codex-wake-run/bcb839db25ce.log`; benchmark binary SHA-256:
`ee48f1f86422b6c4fca8594467501fc55831078ecc7d465f2d2973c183339cf0`.
Observed artifact size 20,928,204,521 bytes (payload 19,849,644,160;
scales 1,078,394,880), 192 candidate Q4 and 209 candidate Q8 tensors, open
28,509.8 ms, runtime creation 196.981 ms, upload 4,398.35 ms, model and
tracked peak device memory 20,928,039,040 bytes, peak host RSS 21,077,942,272
bytes, CUDA free memory 33,099,350,016 bytes before and 11,479,810,048 bytes
after. These are raw measurements from one run, without uncertainty estimates
or a performance claim. The bench reported device total bytes
33,664,794,624.

### Architecture blocker

None.

### Follow-up observations

Production decode kernel support, numerical/runtime integration, and quality
acceptance remain with TASK-022; production GEMM support/integration remains
with TASK-023; end-to-end prefill and combined quality remain with TASK-026.
This task's measurements do not establish decode/prefill speed or candidate
quality.

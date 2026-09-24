# TASK-021 — Native quantized artifact and compiler

## Status

TODO

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

- [ ] New representations have explicit IDs, reconstruction/scaling equations, layout and compatibility rules without changing existing IDs.
- [ ] Compilation is deterministic from the pinned source/calibration policy and independently reconstructs selected logical values.
- [ ] Reader/schema tests reject unsupported or malformed metadata and bindings.
- [ ] Real artifact size, load costs and bounded host/device peaks are measured.
- [ ] The artifact exposes the representation required by both planned decode and GEMM consumers; any extra view has the approved measured budget.

## Architecture blocker rule

A rejected candidate is a recorded result; use the eligible fallback within OVERALL-01 without relaxing acceptance criteria. Missing required exit evidence prevents completion. A conflict outside the reopened decisions requires the full architecture-blocker report defined in the ledger; obsolete Q4-only or experiment-order restrictions are not blockers.

## Completion report

### Result

TODO — no execution or acceptance evidence recorded for this revised task.

### Changes made

Record the concrete changes or measured keep decision, including decision and artifact/layout identities.

### Tests run

Record exact commands, outcomes, reference tolerances, covered boundaries and
limits. Do not infer runtime correctness from documentation checks.

### Benchmark results

Record raw evidence paths, timing boundaries, quality context, memory and
uncertainty, or the reason a benchmark is not required by this task.

### Architecture blocker

Record none or the complete ledger-defined blocker report.

### Follow-up observations

Record remaining coverage and performance gaps and their downstream owners.

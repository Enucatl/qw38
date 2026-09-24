# TASK-019 — SM120 quantization and kernel feasibility

## Status

TODO

## Milestone

M7 — Strategy and feasibility

## Purpose

Establish which compact representations and kernels are correct and useful on the actual SM120 platform before selecting a production ABI.

## Depends on

- [TASK-018](TASK-018.md)

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
| Q-01/Q-02, projection part of P-02 | Compare native FP4 weight/activation policies and existing Q4/BF16 controls | Reopened by OVERALL-01 |
| A-01/L-01, M-01, T-01–03 | Measure view/layout, workspace and shape choices | Reopened by OVERALL-01 |
| I-01–I-05, P-01, S-01/S-02 | RTX 5090 toolchain, FP32 arithmetic and initial state controls | Retained |

OVERALL-01 supersedes conflicting restrictions in the former task sequence.
Historical EXP-A–H ordering does not constrain this task. Decisions outside
the reopened scope remain binding.

## Starting point

TASK-018 has frozen the protocol and preserved the current development controls. Production quantizer/layout selection and full candidate quality acceptance remain open.

## Scope

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

## Out of scope

Freezing the production artifact ABI, model-wide integration, calibration on evaluation cases, structured sparsity, assuming SM100 instructions/resources apply to SM120, and claiming a native mixed-input kernel without proof.

## Required interfaces and data representation

A benchmark descriptor records family, M/N/K, logical and padded shapes, operand/scale types and layouts, output/epilogue, kernel/dispatch identity, workspace and conversion stages. Emit correctness results, raw timing samples, native-MMA evidence, toolchain/CUTLASS pin and full resident/transient memory estimates per candidate.

## Required semantics and constraints

Native NVFP4 uses quantized weights and activations. Distinguish its FP32 accumulation from output storage. Validate scale orientation, tensor-scale convention and zero/padding behavior against reconstructed operands, then assess quantization error separately against BF16. Keep both phases' comparison inputs, epilogues and output precision matched. Do not reinterpret Q4 codes as FP4.

Follow the code standards' identity and manifest-only digest policy. Keep
benchmarks separate from correctness checks and record source, binary,
artifact/policy, inputs, toolchain and hardware identities appropriate to each
result. Partial or invalid evidence cannot establish quality acceptance.

## Tuning defaults

Use real model families with M = 1, 2, 8, 32, 64, 128, 256, 512 and 1024 where workspace permits, plus unaligned tails. Record unsupported points and costs explicitly. Pin and verify the compiler target required by the chosen CUTLASS revision; a large-M winner does not select decode dispatch.

## Expected files/modules

Isolated projection/quantization benchmarks, minimal reference converters and packers, toolchain/dependency pin, instruction/profiler evidence and feasibility report. Production compiler/runtime integration follows later.

## Tests required

### Unit and contract checks

Scale/encoding reconstruction, matrix orientation, padding/tail masks, zero blocks, supported-shape checks and workspace bounds.

### Reference and numerical checks

Independent contractions of reconstructed NVFP4/MXFP4 operands and Q4-to-BF16 controls, with explicit tolerances and required output precision. Diagnose quantization error against BF16 separately.

### Integration checks

Build and run the pinned candidates on RTX 5090; prove the selected instruction path and identify fallbacks. Exercise real shapes/weights and representative activations. Account for weights, scales, embeddings/head, retained payloads, state/KV, scratch, optional views and transient peaks.

## Benchmark required

Required: native NVFP4/MXFP4, bounded Q4-to-BF16 GEMM, BF16 library controls where feasible, and same-FP4-weight GEMV probes or estimates. Report kernel-only and complete quantize/pack/dispatch/GEMM/epilogue costs, resource usage and memory, distinguishing measurements from estimates.

## Acceptance criteria

- [ ] Pinned SM120 build and native dense NVFP4/MXFP4 attempts have reproducible support and instruction evidence; any unsupported case has a concrete reason.
- [ ] Independent operand-reconstruction contractions validate implemented kernels and tails.
- [ ] The real-shape matrix reports conversion-inclusive latency, logical/padded work, output precision and resource/workspace costs.
- [ ] Full candidate memory budgets respect actual allocatable memory and the frozen reserve.
- [ ] A short list and eligible fallback are justified without claiming final quality or end-to-end speed.

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

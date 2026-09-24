# TASK-020 — Calibrated precision policy and candidate selection

## Status

TODO

## Milestone

M7 — Strategy and feasibility

## Purpose

Select a provisional format, calibration recipe and family precision policy using measured kernel feasibility and independent quality screening.

## Depends on

- [TASK-019](TASK-019.md)

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
| Q-01/Q-02, projection part of P-02 | Calibrated weights, activations and family exceptions | Reopened by OVERALL-01 |
| A-01/A-02/L-01 | Select layout/views while separating logical quantization from physical packing | OVERALL-01 and retained separation |
| Q-03, P-01, S-01/S-02 | BF16 small/sensitive controls, FP32 arithmetic and state | Retained control |

OVERALL-01 supersedes conflicting restrictions in the former task sequence.
Historical EXP-A–H ordering does not constrain this task. Decisions outside
the reopened scope remain binding.

## Starting point

TASK-019 provides real-shape support, correctness, timings and memory budgets. TASK-018 defines separate calibration and screening inputs; no candidate has a full quality pass.

## Scope

Quantize from BF16 using the frozen calibration inputs. Start with deterministic
block scaling, then test clipping or calibration improvements only where error
requires them. Specify E2M1 encoding/rounding/saturation, block scale encoding,
second-level scale convention, zero/nonfinite behavior, group axis, activation
scale granularity/lifetime, and packed padding. Test independent weight-only,
activation-only and combined perturbations on representative full layers and
bounded model continuations, including outliers and prefill/decode inputs.

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

## Out of scope

Final evaluation-driven scale fitting, blanket activation/state quantization, mandatory rotations or smoothing, production ABI implementation, and replacing EVAL-01 with reconstruction error or a new BF16 evaluation arm.

## Required interfaces and data representation

Version a candidate-policy record with source/tokenizer/calibration identities; selected tensors/families; quantizer and rounding/clipping parameters; scale axis, encoding, convention and lifetime; packing/padding; proposed dispatch and fallbacks; quality diagnostics and memory cost.

## Required semantics and constraints

Generate candidates directly from BF16. Record weight-only, activation-only and combined error separately, including prompt and populated-decode inputs and early/late layers. Use deterministic rounding, saturation, zero and nonfinite behavior. Test activation scales against chunk composition; any rotation/smoothing must preserve equations before quantization and include runtime cost.

Follow the code standards' identity and manifest-only digest policy. Keep
benchmarks separate from correctness checks and record source, binary,
artifact/policy, inputs, toolchain and hardware identities appropriate to each
result. Partial or invalid evidence cannot establish quality acceptance.

## Tuning defaults

Start with deterministic block scaling. Apply clipping or family-specific precision only when development evidence justifies it; retain frozen evaluation thresholds. NVFP4 leads the investigation, with MXFP4, selective higher precision or Q4 as eligible alternatives.

## Expected files/modules

Offline calibration/quantization diagnostics, candidate policy records, independent reference checks and provisional selection report. Larger calibration frameworks remain offline dependencies.

## Tests required

### Unit and contract checks

Encoding/rounding, block/tensor scales, zero/nonfinite behavior, padding, deterministic calibration and calibration/evaluation separation.

### Reference and numerical checks

Weight-only, activation-only and joint perturbations on representative full layers; reconstruction and output errors for outliers, MLP down, GDN/attention inputs, and head. Independent component BF16 controls diagnose errors.

### Integration checks

Bounded model continuations on the development screening set, prefill/decode activation distributions and chunk-sensitive scale checks. Reconcile family choices with measured kernel support and total memory feasibility.

## Benchmark required

Use TASK-019 matched kernel measurements and measure any changed scale/transform/exception cost. Update memory and conversion costs for the proposed policy; do not claim full-request results before integration.

## Acceptance criteria

- [ ] Calibration and development screening identities are frozen and disjoint from final evaluation.
- [ ] The quantization and scale contracts are explicit and independently tested.
- [ ] Per-family choices and weight/activation ablations have representative layer and continuation evidence.
- [ ] The provisional layout/view/dispatch policy has supported kernels, a full memory budget and justified fallbacks.
- [ ] Selection/rejection reasons are recorded; full quality acceptance remains assigned to TASK-022/026.

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

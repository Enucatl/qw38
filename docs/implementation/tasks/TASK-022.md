# TASK-022 — Candidate decode and core quality gate

## Status

TODO

## Milestone

M8 — Compact runtime and both execution phases

## Purpose

Integrate the selected compact weights into full-model decode and establish the complete core quality gate before production-prefill integration.

## Depends on

- [TASK-021](TASK-021.md)

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
| Q-01/Q-02, projection part of P-02, L-01 | Consume TASK-021 candidate with explicit activation/dispatch identity | OVERALL-01 selection |
| P-01, S-01/S-02, G-01/G-02 | Retain model equations, FP32 residual/state arithmetic and correct session semantics | Retained |
| EVAL-01 | Complete core acceptance formerly owned by TASK-018 | Policy, ownership remapped |

OVERALL-01 supersedes conflicting restrictions in the former task sequence.
Historical EXP-A–H ordering does not constrain this task. Decisions outside
the reopened scope remain binding.

## Starting point

The TASK-021 candidate artifact is independently validated. Real-shape kernel evidence and old V0 semantic/development controls exist; the candidate has no full core quality pass.

## Scope

Integrate the selected weights into all primary-language decode layers and the
head. Compare native small-M W4A4 with same-weight W4A16 GEMV and retain the
measured choice per family/shape, with bounded scratch and no hot-path weight
repacking. Preserve FP32 residual/state arithmetic, complete-token commits,
poison/reset/restore semantics and FP32 output logits. Record activation policy
as part of dispatch identity; a kernel switch can change numerical behavior.

Run component and source-semantic regressions, then the complete EVAL-01 core
(216 paired cases, 512/4096 retrieval, NLL/slices, capability, P100 adjudication,
same-schedule replay and declared continuation boundaries). Regenerate invalid
reference evidence; historical partial outputs remain diagnostic only.
**Exit:** a passing candidate decode core and measured populated-decode costs.
If quality fails, diagnose and amend the candidate with the unchanged gate;
do not push an unaccepted quantizer into production-prefill integration.

## Out of scope

Production prefill, 32768 extension acceptance (TASK-026), incomplete-arm reuse, threshold relaxation, or silently broadening precision to hide a quality failure.

## Required interfaces and data representation

Typed decode plans bind family/shape, quantizer/layout, activation scaling, kernel/fallback, scratch and epilogue. Candidate evaluation records retain language-only mode, precision/schedule and all fixture/target/mask/reference identities. Reuse existing evaluation tooling where its contract remains valid.

## Required semantics and constraints

Preserve all 64 layers, FP32 logits, complete-token commits and poison/reset/restore behavior. Compare native small-M W4A4 against same-weight W4A16 GEMV with conversion and dispatch costs included. No hot-path full-weight repacking. Count teacher targets once, reset per document and do not infer full-vocabulary KL from top-k output.

Follow the code standards' identity and manifest-only digest policy. Keep
benchmarks separate from correctness checks and record source, binary,
artifact/policy, inputs, toolchain and hardware identities appropriate to each
result. Partial or invalid evidence cannot establish quality acceptance.

## Tuning defaults

Select dispatch from measured family/shape costs. EVAL-01 aggregate NLL delta <= +0.03 nats/token, declared slice delta <= +0.06, capability/uncertainty and generated-text rules remain binding; do not redefine those gates here.

## Expected files/modules

Decode CUDA consumers/wrappers, runtime layer/model binders and scratch planning, candidate evaluation/report integration and continuation tests.

## Tests required

### Unit and contract checks

Descriptor/policy compatibility, scaling/padding/epilogues, deterministic argmax, scratch bounds and failure behavior; target/mask/count/provenance checks where tooling changes.

### Reference and numerical checks

Decoded-operand contraction checks and retained TASK-017 source-semantic regressions. Separate quantization error from implementation error; BF16 remains a diagnostic component control.

### Integration checks

All 216 paired core cases including 512/4096 retrieval, NLL/slices, capability, resolved 100-case P100 review and required comparator replay. Same-schedule reset/snapshot/interleave at lengths 1, 3, 4, 63, 64, 65, 255, 256, 257 plus late-failure recovery. Invalid or partial source attempts cannot supply accepted full coverage.

## Benchmark required

Populated decode and family-level native/GEMV comparison with input scaling, launch, synchronization and readout costs. Label these development measurements; the matched whole-request baseline is TASK-027.

## Acceptance criteria

- [ ] All primary-language layers and the head consume the selected artifact with explicit dispatch/activation policy and bounded scratch.
- [ ] Numerical, source-semantic and session failure/recovery regressions pass.
- [ ] Complete authenticated EVAL-01 core evidence passes, including NLL/slices, capability, retrieval, required replays and resolved P100 adjudication.
- [ ] Same-schedule continuation/reset/snapshot and interleave checks pass at every required boundary.
- [ ] Measured decode costs and selected fallbacks are reported; quality failures preserve the baseline and require unchanged-gate retesting.

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

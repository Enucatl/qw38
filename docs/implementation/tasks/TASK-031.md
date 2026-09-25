# TASK-031 — Consolidated validation and architecture promotion

## Status

TODO

## Milestone

M10 — Measured refinement and promotion

## Purpose

Validate the combined candidate and record a supported promotion decision with reproducible architecture, quality, performance and capacity evidence.

## Depends on

- [TASK-030](TASK-030.md)

## Normative references

- [Implementation ledger](../task_ledger.md) — OVERALL-01, revised task contract,
  overall decision rules, measurement envelope and migration of prior obligations.
- [Architecture V0](../../architecture/architecture-v0.md) — retained model semantics
  and controls; reopened decisions follow OVERALL-01.
- [EVAL-01 / PERF-01](../../architecture/evaluation-policy-v0.md) — scoring
  criteria and measurement definitions.
- [54-case core amendment](../../architecture/evaluation-policy-core-54.md) —
  routine coverage and manual-only full-suite execution.
- [Technology baseline](../technology-baseline.md).
- [Code standards](../code-standards.md).

## Architecture decisions consumed

| Decision | Contract for this task | Authority |
| -------- | ---------------------- | --------- |
| OVERALL-01 | Consolidate selected precision/layout/schedule/state decisions | User-directed sequence |
| EVAL-01 / PERF-01 | Final combined quality and per-row performance judgment | Binding policy |
| A-02, retained semantics and compatibility | Document artifact/state identities and supported scope | Retained |

OVERALL-01 supersedes conflicting restrictions in the former task sequence.
Historical EXP-A–H ordering does not constrain this task. Decisions outside
the reopened scope remain binding.

## Starting point

TASK-028–030 have individually justified changes or keep decisions. Their combined runtime/artifact still needs final frozen-identity validation.

## Scope

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

## Out of scope

Promoting a quality failure, substituting individual experiment passes for a combined run, hiding unmet speed targets, adding unapproved tasks or claiming global optimality/general unsupported capabilities.

## Required interfaces and data representation

A final manifest/report binds compiler, source, calibration, quantizer/layout, artifact, runtime, dispatch/chunk/graph, state ABI and toolchain identities. Publish exact commands, raw evidence locations, accepted controls/rollback procedure, precision/scaling/layout/dispatch policies and context/memory limits.

## Required semantics and constraints

Rerun the 54-case EVAL-01 core and mandatory 32768 extension on the combined candidate, with all reviews, provenance and replay obligations. Run every PERF-01 row at the same frozen identities. Separate quality acceptance, completion of measurements and attainment of per-row speed parity. The historical V0 control remains explicitly unaccepted if its quality gate is still incomplete. The optional 216-case suite is human-initiated interactive work only; agents must never launch it.

Follow the code standards' identity and manifest-only digest policy. Keep
benchmarks separate from correctness checks and record source, binary,
artifact/policy, inputs, toolchain and hardware identities appropriate to each
result. Partial or invalid evidence cannot establish quality acceptance.

## Tuning defaults

Freeze all settings before final runs. Any corrective arithmetic/policy change creates a new candidate identity and requires the affected gates to be rerun before the final decision.

## Expected files/modules

Consolidated validation/performance reports, final architecture/quantization/layout/runtime-format/prefill/technology documentation, task completion records and reproducible run instructions.

## Tests required

### Unit and contract checks

Final identity, policy/representation compatibility and documentation/task-reference consistency checks; reuse relevant implementation regressions.

### Reference and numerical checks

Retained independent arithmetic, artifact reconstruction and model-semantic controls for the combined candidate. Report tolerances and any schedule-dependent effects.

### Integration checks

Complete 54-case EVAL-01 core, six 32768 retrieval cases, selected-P100 adjudication, required same-schedule/cross-schedule/dispatch checks and session failure/replay coverage for the frozen combined artifact/runtime.

## Benchmark required

All mandatory PERF-01 rows with raw paired samples, median/p99/intervals, cold/warm separation, full memory accounting and final per-row parity/gap status.

## Acceptance criteria

- [ ] Final compiler/calibration/artifact/runtime/toolchain/schedule identities are frozen and reproducible.
- [ ] The combined candidate has complete 54-case EVAL-01 core and long-context evidence with resolved reviews and replay coverage.
- [ ] All PERF-01 rows, memory/cold costs and per-row achieved/unmet/uncertain targets are reported.
- [ ] Architecture/format/precision/state/task documents describe the actual selected implementation and remaining limits consistently.
- [ ] A supported promote/retain-control decision is recorded; quality failures cannot be promoted and unmet performance remains explicit.

## Architecture blocker rule

Missing required final evidence prevents completion. A failed quality gate prevents promotion; preserve the failure and record the supported retain-control decision without labeling the failed candidate accepted. Measured unmet or uncertain performance targets are explicit outcomes and do not become quality waivers or implicit authorization for more tasks.

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

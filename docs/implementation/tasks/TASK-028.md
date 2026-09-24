# TASK-028 — Precision and representation refinement

## Status

TODO

## Milestone

M10 — Measured refinement and promotion

## Purpose

Refine consequential precision or representation choices using whole-request bottlenecks while preserving frozen quality and capacity requirements.

## Depends on

- [TASK-027](TASK-027.md)

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
| Q-01/Q-02, projection part of P-02 | Revisit family/head precision and activation scaling where measured gaps justify it | OVERALL-01 |
| A-01/A-02/L-01 | Evaluate selected extra views with explicit logical/layout identity | OVERALL-01 |
| S-01/S-02, G-02 | Hold state and unrelated schedules fixed | Experimental control |

OVERALL-01 supersedes conflicting restrictions in the former task sequence.
Historical EXP-A–H ordering does not constrain this task. Decisions outside
the reopened scope remain binding.

## Starting point

TASK-027 supplies a complete matched baseline and ranked gaps for the accepted candidate.

## Scope

Use TASK-027 gaps to revisit only consequential choices: per-family precision,
head precision, activation scaling, or selected additional packed views.
Compare alternatives using identical workloads and calibration separation;
account for added code/layout complexity, memory and cold-load costs. Expand
to FP6/FP8 or mixed inputs only with demonstrated SM120 support and a concrete
quality/performance reason. Keep unrelated state and schedules fixed.

**Exit:** keep/change decisions backed by full applicable EVAL-01 revalidation
and whole-request measurements for the promoted variant. A documented decision
to keep the initial representation is valid; exhaustive format combinations
are not required. Unsuccessful variants remain evidence, not default paths.

## Out of scope

Exhaustive format sweeps without a measured reason, final-evaluation calibration, simultaneous unrelated state/schedule changes, unsupported FP6/FP8 mixtures and unbudgeted second views.

## Required interfaces and data representation

Each variant has a policy/quantizer/layout identity, changed-family list, calibration provenance, kernel/dispatch support and exact incremental artifact/resident/transient bytes. Reports pair quality and complete-request results against the accepted control and retain rejected variants as evidence.

## Required semantics and constraints

Keep model equations, evaluation criteria and calibration separation fixed. Isolate numerical-policy changes from lossless layout rearrangements and give each the appropriate correctness checks. Any additional view requires matching logical values and scale interpretation and explicit lifetime/load costs. Precision exceptions must be reflected in both phase consumers.

Follow the code standards' identity and manifest-only digest policy. Keep
benchmarks separate from correctness checks and record source, binary,
artifact/policy, inputs, toolchain and hardware identities appropriate to each
result. Partial or invalid evidence cannot establish quality acceptance.

## Tuning defaults

Use TASK-027's largest relevant gaps to select a bounded set of experiments. FP6/FP8 or mixed inputs require demonstrated SM120 support and a concrete quality/performance rationale. Retaining the current policy is valid when evidence does not justify a change.

## Expected files/modules

Only affected quantizer/calibration/policy, packer/consumer/binding paths and their focused checks; variant and keep/change reports.

## Tests required

### Unit and contract checks

Changed encoding/scaling/family policy, layout/version/binding compatibility, independent reconstruction and extra-view memory accounting.

### Reference and numerical checks

Weight-only/activation-only diagnostics for numerical changes; exact logical value/scale equivalence for layout-only changes. Preserve established numerical controls.

### Integration checks

Full applicable EVAL-01 revalidation for promoted variants, including core, P100 review, required long-context and continuation/dispatch coverage. Rejected variants preserve failure evidence and do not become defaults.

## Benchmark required

Matched complete requests, prefill, populated decode and memory, plus cold compiler/load/repack costs and diagnostic kernels. Report per-row losses and uncertainty; local speed alone cannot justify promotion.

## Acceptance criteria

- [ ] Each executed variant addresses a measured gap and has explicit policy/layout/calibration identity.
- [ ] Kernel support, incremental memory and cold costs are demonstrated for every proposed representation.
- [ ] Promoted variants pass the applicable full quality/context/continuation gates.
- [ ] Whole-request evidence supports each keep/change decision and reports individual regressions.
- [ ] The accepted representation or evidence-backed keep decision is recorded with rejected variants and remaining gaps.

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

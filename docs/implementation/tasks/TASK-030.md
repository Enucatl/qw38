# TASK-030 — State and long-context bottleneck refinement

## Status

TODO

## Milestone

M10 — Measured refinement and promotion

## Purpose

Resolve demonstrated state or long-context traffic bottlenecks, retaining current precision/layout when changes have no material benefit.

## Depends on

- [TASK-029](TASK-029.md)

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
| S-01/S-02 | Isolated GDN state-storage precision or layout/ownership comparisons | Explicitly reopened in TASK-030 |
| P-01/P-02 | FP32 recurrent arithmetic; BF16 KV/history retained | Binding controls |
| Q-01/Q-02, L-01, unrelated schedules | Hold weights and other accepted choices fixed | Experimental control |

OVERALL-01 supersedes conflicting restrictions in the former task sequence.
Historical EXP-A–H ordering does not constrain this task. Decisions outside
the reopened scope remain binding.

## Starting point

TASK-029 supplies the refined execution profile, accepted representation/schedule and explicit remaining long-context gaps.

## Scope

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

## Out of scope

Automatic BF16 state conversion, FP4 state, KV/history precision changes, new attention semantics and combining precision/layout changes without isolated evidence.

## Required interfaces and data representation

Any changed GDN state ABI has an explicit version, shape/order/precision description, ownership and snapshot compatibility behavior. Bindings reject incompatible state instead of silently reinterpreting bytes. Memory/traffic reports distinguish fixed GDN storage from context-dependent KV and any extra workspace.

## Required semantics and constraints

Keep recurrence equations and arithmetic FP32. A BF16 storage experiment changes only load/store precision, with layout/ownership held fixed; a layout experiment holds precision fixed. Require long-horizon error/continuation evidence. Attention-traffic work preserves causal GQA and BF16 KV/history without permanent replication or eviction semantics.

Follow the code standards' identity and manifest-only digest policy. Keep
benchmarks separate from correctness checks and record source, binary,
artifact/policy, inputs, toolchain and hardware identities appropriate to each
result. Partial or invalid evidence cannot establish quality acceptance.

## Tuning defaults

Choose work from the post-refinement profile. The 144 MiB FP32 GDN state is not automatically a priority; a measured decision to retain it and its layout completes the investigation when no useful bottleneck exists.

## Expected files/modules

Only affected recurrence ownership/state storage or attention-traffic paths, versioned state/snapshot bindings if needed, focused references and long-context experiment reports.

## Tests required

### Unit and contract checks

Changed state layout/encoding and independent reconstruction, snapshot compatibility/rejection, ownership bounds and attention masks for any traffic change.

### Reference and numerical checks

One-step and long-horizon recurrent state/output divergence with fixed weights/arithmetic; independent state-layout checks and causal attention equivalence where applicable.

### Integration checks

Complete applicable 54-case EVAL-01 core/32768 coverage and same-schedule continuation/reset/snapshot/interleave for a promoted change, including arbitrary incoming state and prefill/decode transitions. The optional 216-case suite is human-initiated interactive work only; agents must never launch it.

## Benchmark required

Populated long-context decode, prefill and whole requests with state/KV traffic, spills/occupancy, resident memory and workspace. A keep decision cites the accepted profile and measured lack of a material state bottleneck.

## Acceptance criteria

- [ ] The profile justifies each selected state/traffic experiment or retention of the current controls.
- [ ] Precision and layout changes are isolated; FP32 arithmetic and BF16 KV/history remain intact.
- [ ] Any changed state ABI and snapshot compatibility are explicit and independently validated.
- [ ] Promoted changes pass long-horizon quality and complete continuation/failure recovery coverage.
- [ ] Whole-request/context/memory measurements support keep/change decisions without prioritizing storage reduction alone.

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

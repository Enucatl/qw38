# TASK-026 — Full-model prefill, handoff and quality gate

## Status

TODO

## Milestone

M8 — Compact runtime and both execution phases

## Purpose

Validate the complete quantized engine across production prefill, decode transitions and the full required quality/context envelope.

## Depends on

- [TASK-025](TASK-025.md)

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
| G-01/G-02, M-01 | Compose all language layers with bounded chunks and correct state handoff | Retained semantics and OVERALL-01 schedule |
| Q-01/Q-02/P-02/L-01 | Accepted weights and explicit per-phase activation/dispatch policy | Candidate selection |
| EVAL-01 | 54-case core production-prefill rerun and six 32768 retrieval cases | Policy, coverage and ownership remapped |

OVERALL-01 supersedes conflicting restrictions in the former task sequence.
Historical EXP-A–H ordering does not constrain this task. Decisions outside
the reopened scope remain binding.

## Starting point

TASK-022 candidate decode core passes. TASK-023–025 supply bounded projections and both complete layer types. Long-context and full-model production-prefill quality are not yet accepted.

## Scope

Compose all 64 layers, bounded chunks and final-position logits for generation;
provide requested-row logits for evaluation without allocating T×vocabulary
for the entire prompt. Test prefill into nonempty sessions and continuation
across native GEMM/GEMV dispatch transitions with the same resident weights.

Run the complete 54-case EVAL-01 core through production prefill plus decode, and the
six frozen 32768 retrieval cases in both required comparison arms. Preserve
the original boundaries/partitions (including 1/63/64/65/255/256/257 and
alternating 63/65), adding boundaries around chosen chunks and dispatch
crossovers. Require bitwise replay only for identical schedules; compare
different schedules using component tolerances and the unchanged behavioral
gates. Test chunk-dependent activation scales explicitly. **Exit:** accepted
full candidate quality, correct positions/history/state, bounded measured
memory, and documented coverage. Missing mandatory long-context evidence
blocks acceptance; full prefill equivalence is not inferred from one token.

## Out of scope

Lowering quality thresholds, substituting a subset of the frozen 54-case core, inferring acceptance from one-token agreement, MTP/vision/batching claims and total T×vocabulary allocation.

## Required interfaces and data representation

The full-model prefill API supports valid token chunks, absolute positions, empty/nonempty sessions and final-row or bounded requested-row logits. Reports bind artifact, calibration, per-phase precision/scales, chunk/dispatch schedule and reference/input identities. Reuse session commit, poison/reset and snapshot contracts.

## Required semantics and constraints

Run all 64 layers and preserve FP32 residual/logit and state contracts. Identical schedules require bitwise replay; distinct schedules use existing component tolerances and unchanged EVAL-01 gates. Explicitly test W4A4/GEMV activation changes and chunk-dependent tensor scales. Never advance positions/history for padded rows or score a teacher target twice.

Follow the code standards' identity and manifest-only digest policy. Keep
benchmarks separate from correctness checks and record source, binary,
artifact/policy, inputs, toolchain and hardware identities appropriate to each
result. Partial or invalid evidence cannot establish quality acceptance.

## Tuning defaults

Retain EVAL-01's nine checkpoints 1, 3, 4, 63, 64, 65, 255, 256, 257 and partition sizes 1, 63, 64, 65, 255, 256 plus alternating 63/65. Add boundaries around selected chunks, recurrence intervals and dispatch crossovers.

## Expected files/modules

Full-model prefill scheduler, session/workspace/handoff integration, bounded evaluation logits, EVAL-01 reports and continuation/context tests.

## Tests required

### Unit and contract checks

Valid-row ownership, bounded requested-logit tiles, workspace lifetime, commit/failure boundaries and dispatch policy identity.

### Reference and numerical checks

Component-level state/output checks across repeated decode, full/chunked prefill and native/GEMV transitions. Reuse semantic controls and distinguish schedule rounding from semantic bugs.

### Integration checks

Complete 54-case core through production prefill+decode, resolved selected-P100 adjudication and six frozen 32768 retrieval cases for candidate and comparator. Exercise nonempty-session prefill, required partitions/checkpoints, interleave, snapshot/restore and late-failure recovery; document same-schedule and cross-schedule results separately. The optional 216-case suite is human-initiated interactive work only; agents must never launch it.

## Benchmark required

Diagnostic full-model prefill/handoff and memory measurements, including final-row generation versus requested-row evaluation. Matched external performance acceptance is TASK-027.

## Acceptance criteria

- [ ] Both layer types compose through all 64 layers with bounded prefill workspace and correct head modes.
- [ ] The complete 54-case EVAL-01 core through production prefill passes with authenticated references and resolved reviews.
- [ ] All six 32768 retrieval cases have passing required paired evidence; missing mandatory coverage blocks acceptance.
- [ ] Required partition/checkpoint, dispatch-transition, incoming-state and failure/replay checks pass.
- [ ] Measured resident/transient memory fits the declared capacity/reserve; context and schedule coverage are explicit.

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

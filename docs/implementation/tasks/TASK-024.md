# TASK-024 — GDN prefill algorithm and layer integration

## Status

TODO

## Milestone

M8 — Compact runtime and both execution phases

## Purpose

Select a numerically validated GDN prefill algorithm using complete-layer costs and integrate its convolution, state and output paths.

## Depends on

- [TASK-023](TASK-023.md)

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
| G-02, T-02, M-01 | Compare ordered and chunkwise recurrence before final schedule selection | OVERALL-01 |
| S-01/S-02, P-01 | FP32 state in the current ABI and unchanged recurrent equations/arithmetic | Retained control |
| Q-01/P-02/L-01 | Keep accepted projection representation fixed during algorithm comparison | TASK-022/023 selection |

OVERALL-01 supersedes conflicting restrictions in the former task sequence.
Historical EXP-A–H ordering does not constrain this task. Decisions outside
the reopened scope remain binding.

## Starting point

TASK-023 supplies complete projection and MLP prefill paths. Decode convolution/recurrence and continuation references exist.

## Scope

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

## Out of scope

State precision/layout changes, FP4 recurrent state, new recurrence semantics, and selecting an algorithm from isolated recurrence speed while ignoring whole-layer costs.

## Required interfaces and data representation

A GDN chunk plan declares valid rows, incoming convolution history, positions, FP32 recurrent state, recurrence algorithm/interval, output staging and workspace lifetimes. Both compared algorithms consume the same projection policy and produce one output per valid token.

## Required semantics and constraints

Convolution reads incoming raw qkv history and valid chunk projections; history commit occurs after readers finish and preserves needed old entries for short chunks. Recurrence must realize the same ordered GDN map for arbitrary incoming state. Specify transformed intermediates and precision for chunkwise/WY evaluation, independently verify equations and retain FP32 recurrent arithmetic. Padding never updates history or state.

Follow the code standards' identity and manifest-only digest policy. Keep
benchmarks separate from correctness checks and record source, binary,
artifact/policy, inputs, toolchain and hardware identities appropriate to each
result. Partial or invalid evidence cannot establish quality acceptance.

## Tuning defaults

Use the selected projection chunk plan; isolate recurrence-algorithm comparisons at matching chunks/precision/weights before broader tuning. The old 64-token serial interval is a control; record any reason for deferring a faster algorithm.

## Expected files/modules

GDN prefill FIR/preparation/history commit, ordered and chunkwise recurrence candidate, output transform, complete GDN+MLP runtime integration and reference evidence.

## Tests required

### Unit and contract checks

History lengths below/at/above the convolution boundary, valid-token masking, arbitrary initial state, recurrence tails and workspace limits.

### Reference and numerical checks

Independent convolution and recurrence equations; ordered versus chunkwise output/state with adversarial gates, short and long streams and documented finite-precision tolerances.

### Integration checks

Complete GDN+MLP layers across chunk partitions and prefill-to-decode continuation; confirm positions, history and same-schedule reset/snapshot replay. Full model behavior is TASK-026.

## Benchmark required

Required: recurrence and complete GDN+MLP timing, state read/write traffic, temporary workspace, resource usage and algorithm comparison at matching settings.

## Acceptance criteria

- [ ] Convolution/preparation and history commit are correct for incoming prefixes, tails and valid rows.
- [ ] Both recurrence strategies have independent equation/correctness evidence or a concrete documented implementation/support limitation.
- [ ] The selected algorithm passes output/state and continuation checks with FP32 state and retained equations.
- [ ] Whole-layer time, state traffic and workspace justify the algorithm choice.
- [ ] Complete GDN+MLP prefill executes with bounded memory; any deferred candidate has an evidence-backed reason.

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

# TASK-027 — Matched whole-request performance baseline

## Status

TODO

## Milestone

M9 — Whole-request measurement

## Purpose

Establish a complete matched performance and memory baseline for the quality-accepted engine and rank the remaining bottlenecks.

## Depends on

- [TASK-026](TASK-026.md)

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
| PERF-01 | Matched workloads, timing windows, uncertainty and per-row parity targets | Policy, ownership remapped |
| I-01–I-05 | Pinned RTX 5090 execution environment | Retained |
| EVAL-01 and candidate policy | Quality-accepted artifact and schedules held fixed during measurement | Prerequisite |

OVERALL-01 supersedes conflicting restrictions in the former task sequence.
Historical EXP-A–H ordering does not constrain this task. Decisions outside
the reopened scope remain binding.

## Starting point

TASK-026 supplies accepted production prefill, decode, handoff and mandatory long-context quality. The final comparison has not yet been measured.

## Scope

Run all PERF-01 rows against its pinned llama.cpp comparator after TASK-026
quality acceptance. Report prompt ingestion, TTFT, populated decode, total
request, p99 and confidence intervals, cold load and peak resident/transient
memory. Separate final-logit generation from multi-row evaluation timings.
Attribute time to quantization/packing, projections, GDN, attention, head,
launches, synchronization and transfers; avoid double-counting host waits as
extra GPU execution time.

**Exit:** complete reproducible baseline and ranked bottlenecks, including
individual losses. Missing parity does not block refinement; missing required
comparison evidence does. No synthetic throughput or old slow prompt loop may
stand in for these production measurements.

## Out of scope

Tuning during baseline capture, format/state redesign, synthetic llama-bench as a substitute for matched requests, hiding losses in aggregate scores and treating host waits as extra GPU work.

## Required interfaces and data representation

A benchmark CLI/adapter emits raw paired samples and summaries with complete source/build/artifact/config/tokenizer/input identities, effective runtime settings, units, timing boundaries and memory categories. Use PERF-01's external public-API llama.cpp comparison, with its pinned Q4_K_M baseline and validated token inputs.

## Required semantics and constraints

Prefill uses T = 256, 4096, 32768 and final-position FP32 logits/argmax. Populated decode uses T = 512, 4096, 32768 and the same frozen 128-token continuation; setup/restore is excluded. Complete requests use T = 256, 4096, 32768 and 128 greedy tokens, reporting TTFT and total latency. Keep generation/evaluation modes and cold/warm costs distinct.

Follow the code standards' identity and manifest-only digest policy. Keep
benchmarks separate from correctness checks and record source, binary,
artifact/policy, inputs, toolchain and hardware identities appropriate to each
result. Partial or invalid evidence cannot establish quality acceptance.

## Tuning defaults

Follow PERF-01: at least five warmups and 20 timed repetitions from identical restored input state, median/p99 and paired confidence intervals. Report per-row parity target ratio >= 1.00; demonstrated/unmet/uncertain depends on the paired interval. Profiling runs are separate from primary timing runs.

## Expected files/modules

Separate benchmark harness/public-API adapter, result records, raw timing/memory/profile evidence and bottleneck-ranked baseline report.

## Tests required

### Unit and contract checks

Where harness code changes: statistics/units, restored-state identity, timing-boundary accounting, missing-input/identity validation and report completeness.

### Reference and numerical checks

Check accepted behavior and replay identities before/after timing. Do not substitute benchmark completion for quality acceptance.

### Integration checks

Run every mandatory PERF-01 row on matched inputs with effective llama.cpp settings and accepted candidate. Validate capacity/population, output policy and raw paired measurements. Missing mandatory evidence blocks completion; an optional maximum-capacity probe is labeled separately.

## Benchmark required

Required: all PERF-01 prefill, populated decode and complete-request rows, cold load/upload, peak resident/transient memory and profiler attribution. Report quantization/packing, projections, GDN, attention, head, launches, waits and transfers without double counting.

## Acceptance criteria

- [ ] All mandatory matched candidate/llama.cpp rows and raw samples are present with valid identities and effective settings.
- [ ] Required warmups/repetitions, median/p99, paired intervals and per-row parity status are reported.
- [ ] TTFT, prefill, decode, total request, cold setup and generation/evaluation boundaries are distinct.
- [ ] Peak memory and component profiles support a ranked bottleneck/gap report without double counting.
- [ ] Quality context remains valid; measured speed gaps are explicit and do not block authorized refinement.

## Architecture blocker rule

Missing required comparison or quality evidence prevents completion. A speed gap is a measured result and does not block TASK-028–032. Measurement difficulty alone does not reopen retained architecture contracts; report any actual conflict using the ledger's full blocker fields.

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

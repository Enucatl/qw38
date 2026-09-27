# TASK-048 — Optimization-round validation and delivery decision

## Status

TODO

## Milestone and dependency

M27 — One frozen decision for the combined optimization round. Depends on
TASK-047. [FAST-04](../task_ledger.md#optimization-amendment--fast-04-2026-09-27)
owns this batch. TASK-043 at `c301efa70f494a01bc7f6585d2f741623eb6fed4`
remains production unless this task records a new promotion. Retain that
rollback target and the older TASK-026/027 fallback at `d2f02e2`.

## Freeze and implementation guide

Freeze one source revision, binary identities, unchanged artifact manifest,
policy 1030, selected projection/head/RMS consumers, small/large scheduling
table, attention partition rules, graph validity conditions and workspace.
Do not combine quality from one candidate with performance from another.
Reuse existing evaluation/replay/request/profile tools and authenticate the
focused development evidence instead of repeating unaffected tests.

TASK-043's user-approved `case_077` exception applies to its recorded candidate
and output only. Preserve its raw P100/overall FAIL and speed gaps as history.
New candidate outputs require new scores; identical authenticated text may
reuse an output-bound adjudication, but a failing adjudication remains FAIL.
No previous exception automatically waives a new candidate's promotion gate.
TASK-044's original first-decode event remains unresolved unless new evidence
actually explains that event; faster new timings do not establish causality.

## Consolidated verification

1. Build affected final Release targets and run full-model integration once.
   Validate graph/eager equality for the same schedule, prefill/decode handoff,
   reset/snapshot/interleave and poisoned-state recovery, including late
   failures and regime/graph transitions. Reuse authenticated unchanged
   component evidence from TASK-045–047.
2. Run frozen core-54: 15 P100, 15 C92, 12 L12, 12 R-512/R-4096. Preserve
   EVAL-01 rendering, reference continuations, uncertainty/slice rules,
   aggregate NLL +0.03, slice +0.06 and capability loss 0.02 budgets. Obtain
   all 15 output-bound P100 reviews with attribution. Invalid, missing,
   inconclusive or failed required quality evidence prevents promotion.
3. Run only `R-32768-s0-d0.1` for extended quality. Full-216 remains optional
   human-only work; no full-model BF16 evaluation. Do not tune kernels or
   thresholds using acceptance answers.
4. Preserve the established replay checkpoints/partitions, including
   511/512/513 and partition 512; include TASK-047's 4095/4096/4097 regime
   transitions, nonempty prefixes and logical-prompt selection across explicit
   partitions. Record actual dispatch. Same-schedule replay remains bitwise;
   cross-schedule full-model differences are diagnostics, not a replacement
   for independent component or quality acceptance.
5. Execute six single-run PERF-01 workloads: requests at 256/4096/32768 and
   populated decode at 512/4096/32768, with the existing 128-token continuation
   semantics. Derive the three prefill/TTFT rows from request prompt phases.
   Zero warmups/repetitions; first library/graph use and transitions remain
   inside their actual boundaries. Report startup/load separately without
   subtracting it from complete cold-request cost. Reuse TASK-027's pinned
   comparator only after checking inputs, settings, hardware/driver/power,
   toolchain and instrumentation/timing identity; refresh only invalidated
   comparator rows. Compare all nine rows with both TASK-043 and llama.cpp.
6. Use the 32K request and existing graph-first-use integration check for
   capacity at 32768+128 slots. Require >=2 GiB free reserve including model,
   state/KV, arena, packs/slabs, attention partials, graph/library memory and
   first-use transients. Report tracked allocations, sampled residency/free
   memory and host staging separately with their limits. Do not add overlapping
   categories or infer a continuous peak from sparse samples.

No new tile/format search or optimization belongs in this task. A code repair
invalidates affected evidence; rerun only those checks and final gates whose
candidate identity/behavior changed. Reporting repairs may reprocess preserved
valid outputs. Main-thread long commands use `wake-run`.

## Delivery and objective decision

Publish `docs/implementation/task048-delivery.md` with exact reproduction
commands, source/binary/artifact/toolchain identities, quality/replay/capacity
results, the actual small/large schedule table, complete startup/prefill/
decode/request costs and memory, and rollback instructions. Include kernel
cost/resource attribution for changed consumers from the same captures;
overlapping comparator categories must not be summed into a wall-time claim.
Record remaining bottlenecks and rejected development hypotheses.

Promotion requires all quality, replay and capacity gates to pass, no observed
regression in any of the nine latency rows against TASK-043, and strict
improvement in decode-4096 and prefill-32768. These comparisons are single-run
observations, not statistical confidence claims. Report startup tradeoffs
separately; they do not waive inference regressions. Missing or invalid saved
TASK-043 comparison evidence must be repaired/recollected with stated reasons,
not treated as a favorable comparison.

Report one explicit outcome: PROMOTE or RETAIN_TASK043. Every llama.cpp /
candidate latency ratio >=1 additionally establishes the fast-engine goal;
production promotion below parity does not establish it. Complete valid
evidence and a justified retention decision can complete this validation task,
even when a candidate fails a gate. Missing required evidence is BLOCKED.
Do not automatically broaden the historical P100 exception or create another
optimization sequence. Record any remaining work as FOLLOW_UP_REQUIRED.

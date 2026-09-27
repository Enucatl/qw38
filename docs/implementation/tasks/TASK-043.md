# TASK-043 — Prefill engine validation and delivery decision

## Status

TODO

## Milestone and dependency

M22 — Validated prefill candidate with explicit remaining gaps. Depends on
TASK-042. [FAST-03](../task_ledger.md#prefill-amendment--fast-03-2026-09-27)
owns this new validation round. TASK-030/036/040 RETAIN_CONTROL decisions and
all prior results remain historical; production stays TASK-026/027 at
`d2f02e2` unless the new candidate satisfies promotion gates.

## Freeze and retained obligations

Freeze the exact runtime/artifact/policy, MMQ tile/shape dispatch, attention
head/query mapping and two-component P, chunk/tail schedule, workspace
capacity, unchanged decode graph selection and pinned toolchain. Retain
manifest-only artifact identity; do not hash model payloads/scales. Preserve
old captures and reuse the existing evaluation/replay/request/profile tools.

P100's confirmed candidate-only error and C92 uncertainty are unresolved
starting conditions, not targets to tune against. No speed improvement waives
them. New outputs need their own scores; authenticated identical text can
reuse its output-bound review. A changed arithmetic implementation creates a
new candidate and invalidates affected evidence. Do not splice results from
different candidates or repeat deterministic answers until a gate passes.

Run one consolidated gate:

1. Frozen core-54: 15 P100, 15 C92, 12 L12 and 12 R-512/R-4096. Preserve
   EVAL-01 rendering, reference continuations, uncertainty/slice rules,
   aggregate NLL +0.03, slice +0.06 and capability loss 0.02 budgets. Obtain
   all 15 output-bound P100 adjudications with reviewer attribution. Missing,
   invalid, failed or inconclusive promotion gates block promotion.
2. Only `R-32768-s0-d0.1` for extended quality. Other 32K fixtures remain
   inventory; full-216 remains optional human-initiated work, never required
   or launched by an agent. No whole-model BF16 evaluation.
3. Preserve TASK-040 checkpoints/partitions and extend them through
   511/512/513 and a 512-token prefill partition. Record actual grouped/tail
   selection, nonempty handoff, same-schedule bitwise replay, and graph/eager
   equality for the same arithmetic schedule. Retain reset/restore,
   interleave, requested-row failure and late-failure recovery coverage.
   Reuse authenticated unchanged TASK-041/042 focused checks rather than
   repeat them as a second witness. Cross-schedule full-model deltas remain
   diagnostics, not component-tolerance acceptance gates.
4. Six single-run PERF-01 executions: requests at 256/4096/32768 and
   populated decode at 512/4096/32768, with the existing 128-token continuation
   semantics. Derive the three prefill/TTFT rows from request prompt phases.
   Include first library/graph use and graph bucket transitions at their real
   boundaries. Zero warmups/repetitions/statistical performance claims.
   Reuse TASK-027 comparator captures only after authenticating the same
   inputs, settings, GPU/driver/power/toolchain and instrumentation/timing
   boundaries. Refresh only invalidated comparator rows.
5. Use the final 32K request for capacity at 32,768+128 slots. Require at
   least 2 GiB free reserve, including larger arena/Q8/FP8/attention/GDN
   workspaces, model/state/KV, graph/library memory, host staging and first-use
   transients. Report tracked allocation peaks and sampled resident/free
   memory separately, with their measurement limits. Do not add overlapping
   categories or infer a continuous residency peak from discrete samples.

Build the affected final Release targets and run full-model integration plus
the required gates once. No full Debug/Release rerun, tile contest, context
search or quantization sweep. Main-thread long commands use `wake-run`.
Reporting repairs can reprocess saved valid outputs without rerunning them.

## Delivery and completion

Publish `docs/implementation/task043-delivery.md`: exact selected phase/shape
dispatch, numerical and scratch-lifetime contracts, pinned reproduction and
rollback instructions, quality/replay/capacity outcomes, all nine latency and
memory rows, and the remaining bottleneck ranking. Distinguish prefill versus
decode GQA reuse explicitly. Attribute fused packing within its containing
kernel category; compare whole prefill with TASK-040 as well as llama.cpp.
Record MMQ token reuse, selected chunk size and attention resources from the
same captures without creating a new evidence framework.

Promotion requires all quality, replay and capacity gates to pass. Every one
of the nine latency ratios >=1 additionally determines the fast-engine goal;
improving prefill alone does not establish it. A complete RETAIN_CONTROL
decision can complete this validation task, but does not mean the performance
goal was achieved or the remaining quality issues were resolved. Keep
TASK-026/027 rollback instructions unless promotion actually succeeds.

If residual FP8 epilogues/tails, GDN or exposed submission gaps justify another
batch, report the concrete evidence and proposed smallest next step. Do not
automatically launch it, expand quality coverage or convert this final gate
into an open-ended optimization task.

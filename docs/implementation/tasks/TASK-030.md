# TASK-030 — Combined validation and delivery decision

## Status

TODO

## Milestone and dependency

M11 — Validated delivery candidate.
Depends on [TASK-029](TASK-029.md), including its documented fallback outcome.

## Authority and delivered behavior

[DELIVERY-01](../task_ledger.md#delivery-amendment--delivery-01-2026-09-26)
merges former TASK-032 final promotion here; former TASK-031 state experiments
are deferred. Deliver a reproducible usable candidate, supported context and
memory limits, rollback instructions and an evidence-backed promote/retain-control
decision. Neither upstream development checks nor this planning amendment are
production acceptance. No additional optimization experiment is required here.

## Candidate and representation contract

Freeze compiler/source/calibration/artifact/runtime/toolchain identities,
quantizer/layout/scale policy, attention schedule, dispatch and chunk sizes
before validation. Retain one resident view per tensor, bounded reusable
workspace, FP32 residual/accumulation/recurrent state, BF16 KV/history and the
existing state ABI. Document any rejected native path and the chosen fallback.
Keep primary-language tensor/tokenizer identity and all causal/session semantics.
Use existing validation, replay and request tools from TASK-022/026/027;
changes should be limited to necessary candidate bindings and reports.

## Consolidated checks, once per selected execution

1. Run the frozen core-54 selection in both quality arms: 15 P100, 15 C92,
   all 12 L12 and all 12 R-512/R-4096. Reuse authenticated matching comparator
   outputs/references where their identities and policy remain valid; record
   that reuse rather than rerunning the comparator. Apply unchanged EVAL-01
   scoring, slice membership, denominators and provenance. Retain NLL limits
   +0.03 aggregate/+0.06 per declared slice, the 0.02 capability-regression
   budget and all existing retrieval/language criteria and uncertainty rules.
   Quality resampling analyzes saved cases, not repeated engine timings.
   Bind all 15 P100 reviews to these outputs; unresolved reviews, invalid,
   failed or inconclusive gates cannot become acceptance.
2. Run exactly `R-32768-s0-d0.1` on the final candidate, with valid paired
   comparator evidence (reuse if unchanged). The other five fixtures remain
   inventory only. The 216-case suite is optional human-initiated interactive
   work, never required or launched by agents.
3. Retain same-schedule bitwise replay, checkpoints 1/3/4/63/64/65/255/256/257,
   partitions 1/63/64/65/255/256 and alternating 63/65, plus relevant new
   chunk/tile/dispatch boundaries. Cover nonempty-session prefill/decode
   handoff, reset/snapshot/restore, interleave and late-failure recovery.
   Cross-schedule state/logit differences are diagnostics checked with existing
   component tolerances and behavioral gates, not full-model bitwise equality.
   Reuse already valid focused implementation checks at this exact code identity;
   do not blindly repeat them. A changed path or unresolved concern requires
   its affected check.
4. Measure all PERF-01 rows: prefill and complete requests at T=256/4096/32768,
   populated decode at T=512/4096/32768, with 128-token continuations. Reuse
   each request's prompt phase for prefill/TTFT. One execution per selected
   workload/engine, zero warmups, no repetitions or performance medians/p99/
   bootstrap/confidence intervals. Label instrumentation/first-use and cold
   load/upload separately; record context, token policy, commands and identities.
   Reuse TASK-027 comparator rows only if workload/input, comparator/settings,
   hardware/power/toolchain context and timing/instrumentation boundaries still
   match. Otherwise refresh only affected comparator workloads once and explain
   why. Do not present incomparable observations as matched speedups.
5. Measure final 32K capacity, complete resident/transient allocation peaks,
   workspace/scales/padding/load peaks and the 2 GiB free-memory reserve.
   Distinguish allocations from sampled residency. Reuse that final workload
   for capacity evidence; no separate maximum-context search. Report supported
   context and any untested coverage.

Use existing reports and completion records; do not introduce a new evidence
schema or run a whole-model BF16 evaluation. GGUF contextual quality checks
are required only for an asserted quality/speed Pareto comparison; a speed-only
report makes no such claim. Historical TASK-027 is the baseline, not final
candidate evidence. A corrective arithmetic/policy change creates a new
candidate identity; rerun affected gates with a reason and do not merge
incompatible candidate outputs into a pass.

## Completion and delivery

Complete after required evidence is present and a supported decision is
recorded, architecture/format/prefill/runtime instructions describe the actual
candidate, and reproducible commands and rollback control are available.
Passing quality and capacity permits promotion; report every PERF-01 ratio and
achieved/unmet target separately. Unmet speed parity is a delivery limitation,
not a reason to claim parity or start more experiments. A failed candidate may
complete the decision task with retain-control and failure evidence, but is
never labeled accepted. Missing evidence or reviews leaves this task incomplete.

## Completion report

TODO — no final validation or promotion recorded. Include candidate identity,
quality/review/replay status, observed per-row performance and memory, selected
control, remaining gaps and final documentation links.

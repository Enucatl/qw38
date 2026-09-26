# TASK-036 — Fast-engine validation and delivery decision

## Status

TODO

## Milestone and dependency

M15 — Accepted fast engine or explicit remaining gap. Depends on TASK-035,
including documented retained fallbacks from TASK-033–035.
[FAST-01](../post-task030-plan.md) owns this new validation round; TASK-030's
completed RETAIN_CONTROL decision remains historical and unchanged.

## Frozen candidate and one consolidated gate

Freeze the selected artifact/compiler/runtime/quantization/scales/layouts,
attention arithmetic, phase dispatch, chunk size and toolchain. Retain the
TASK-026/027 production control until the new candidate passes. Reuse existing
evaluation, replay, memory and request tools. A corrective arithmetic change
creates a new candidate identity; rerun affected gates with a stated reason,
never splice different candidates' outputs into a pass.

1. Run frozen core-54: 15 P100 with output-bound adjudications, 15 C92, 12 L12
   and 12 R-512/R-4096. Apply unchanged EVAL-01 scoring, slice and uncertainty
   rules, NLL +0.03 aggregate/+0.06 slice and capability loss 0.02. Reuse only
   authenticated matching comparator evidence and genuinely unchanged output
   reviews. The TASK-030 lost C92 answer is diagnostic, not a special gate or
   calibration target. Missing, failed or inconclusive gates block promotion.
2. Run only `R-32768-s0-d0.1` for extended quality; keep the other five fixtures
   inventory-only. Full-216 is optional, human-initiated interactive work;
   agents must never launch or require it.
3. Retain replay checkpoints 1/3/4/63/64/65/255/256/257 and partitions
   1/63/64/65/255/256/alternating 63,65, plus boundaries introduced by the
   selected dispatch/chunk/pack lifetime. Cover nonempty prefill/decode handoff,
   reset, snapshot/restore, interleave and late-failure recovery. Same-schedule
   replay is bitwise. Cross-schedule checks use existing component tolerances
   and full-model behavioral gates; full-model state/logit deltas remain
   diagnostic, not subject to isolated-component tolerances. Reuse
   exact-identity focused checks where still valid.
4. Measure all nine PERF-01 rows: requests at 256/4096/32768 and populated
   decode at 512/4096/32768, each with 128 continuation tokens; derive prefill
   and TTFT from request prompt phases. Six executions, once each, zero
   warmups and no repetitions/statistical performance claims. Reuse TASK-027
   comparator rows only when input, settings, toolchain/hardware/power and
   instrumentation/timing boundaries match; refresh only invalidated rows.
5. Reuse the final 32K request for measured capacity at 32,768 + 128 slots.
   Include codes/scales/padding, persistent state, scratch, FP32 output slabs,
   library workspace and transient load peaks. Require 2 GiB free-memory
   reserve; distinguish sampled residency from tracked allocation peaks.

Do not repeat long validation after each local change or add a whole-model
BF16 evaluation. No new performance/evidence framework, context search or
mandatory comparator rerun. All main-thread long commands use `wake-run`.

## Completion and goal

Deliver reproducible build/load/run and rollback instructions, actual precision
and lifetime contracts, supported context, per-row latency and memory ratios,
cold costs and remaining bottlenecks. Passing quality/capacity permits a
promotion decision. The fast-engine objective additionally requires all nine
matched latency ratios (comparator/candidate) at least 1; report every unmet
target separately. This is a target, not a predicted speedup.

The task can complete with a fully evidenced RETAIN_CONTROL or a qualified
delivery with speed gaps; it must explicitly state that the fast-engine goal
remains unmet. Missing required evidence/reviews leaves the task incomplete.
If gaps remain, rank the measured costs and name the next smallest justified
change; do not automatically launch another broad experiment sequence.

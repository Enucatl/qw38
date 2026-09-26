# TASK-036 — Fast-engine validation and delivery decision

## Status

DONE

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

## Completion Report

**Decision: RETAIN_CONTROL.** The FP8 candidate is not promoted, and the fast-engine
goal remains unmet. Production remains the TASK-026/027 Q4_K/Q8 engine at
`d2f02e2`; TASK-030's historical decision is unchanged. The frozen candidate
source is `aacdffe72b1a72f62afd8001784ad78d03025f5b`. The full delivery record,
including compiler/artifact identities, hardware, precision and lifetime
contracts, all nine latency rows and six memory rows, cold costs, bottlenecks,
reproduction and rollback instructions, is in [TASK-036 delivery](../task036-delivery.md).

Core-54 completed 54/54. Aggregate NLL delta was
`+0.0036741931935621993` over 701 teacher targets; aggregate and every slice
passed their limits. C92 scored 7/15 versus 8/15 and is INCONCLUSIVE, with loss
interval `[0, 0.20]`; L12 scored 12/12, both R-512 and R-4096 scored 6/6, and the
fixed 32K case scored 1/1 in both arms. All 15 P100 outputs received reviews,
but the P100 gate is FAIL because candidate-only case_077 contains a factual
residency-accounting error. Eleven changed outputs were reviewed by GPT-6
Codex, an AI reviewer; four unchanged candidate and fifteen comparator reviews
were reused only with matching text hashes and prompts. All 15 outputs are
bound to their reviews. All replay checkpoints and seven partitions passed,
as did full-model integration 1/1 in 14.76 seconds. The 97 retained TASK-035
source/evidence identities matched. Capacity preserved 32,896 slots and
8,782,807,040 bytes free; tracked peak allocation and sampled resident growth
remain distinct measurements. C92 uncertainty and the P100 failure prevent
promotion.

The main-thread GPT-6 Codex agent implemented the candidate and collected its
evidence. GPT-6 Astra, at high reasoning, independently reviewed the
implementation and evidence and returned PASS on pass 1 of 2 permitted review passes, with no findings,
evidence gaps, or requests. The main thread confirmed the reviewed
implementation and evidence hashes remained unchanged. GPT-6 Luna completed
this documentation and delivery bookkeeping.

Acceptance commands and results, from the repository root:

1. `bash .cache/evaluation/qw38-language-v2/task036-support/validation.sh` —
   PASS: pinned Release build, language-model integration, 54/54 core cases,
   default decode replay, seven partition replays, and one long execution.
2. `bash .cache/evaluation/qw38-language-v2/task036-support/score-and-measure.sh` —
   exit 2: long scoring passed, then core reporting was INVALID with
   `KeyError(1029)`; performance had not launched. The attempt is retained as
   `scoring-attempt1-invalid.json`.
3. `bash .cache/evaluation/qw38-language-v2/task036-support/resume-scoring.sh` —
   exit 0: corrected candidate-label scoring and four focused scoring tests
   passed in 0.60 seconds; the paired quality report failed its quality gates
   as described above. All six captures completed and analysis/memory ratios
   were generated.
4. `bash .cache/evaluation/qw38-language-v2/task036-support/finalize-evidence.sh` —
   exit 0: profile-attribution regression passed in 0.01 seconds; GEMV
   classification and stale task labels were corrected. Saved evidence was
   rescored/reanalyzed without repeating inference, and frozen binary and
   comparator hashes passed.

Logs are retained in the support directory as `wake-60dce20f643f.log`,
`wake-77068293a730.log`, `wake-55d8d7b12c47.log`, and
`wake-6a27767f619b.log`. The paired run is
`.cache/evaluation/qw38-language-v2/paired/llama-20260925T103522Z-694370-task036-core54-20260926T214414Z-1191180`;
the core run is
`.cache/evaluation/qw38-language-v2/runs/task036-core54-20260926T214414Z-1191180`.
Comparator reuse authentication is recorded in `comparator-reuse.md` and
`comparator-reuse.sha256` under the support directory. Astra's pass-1 review is
`.cache/evaluation/qw38-language-v2/task036-support/astra-pass1.md`.

**FOLLOW_UP_REQUIRED:** resolve C92 uncertainty and resolve the candidate
quality failure before any promotion decision. All nine latency parity
targets remain unmet; the measured costs and smallest justified next change are
ranked in the delivery record. No new task or experiment was started.

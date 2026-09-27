# TASK-040 — Quantized engine validation and delivery decision

## Status

TODO

## Milestone and dependency

M19 — Accepted engine or explicit remaining gaps. Depends on TASK-039.
[FAST-02](../task_ledger.md#fast-engine-amendment--fast-02-2026-09-27)
owns this final validation round. TASK-030/036 RETAIN_CONTROL decisions and
all completed results remain historical. Production stays at TASK-026/027
(`d2f02e2`) until the new candidate passes its promotion requirements.

## Freeze and retained obligations

Freeze one compiler/artifact/runtime/policy, Q8 activation recipe, Q4/FP8
weight layouts, both phase consumers, attention precision/partition rule,
chunk size, graph/default dispatch and pinned toolchain. Reuse existing
evaluation, replay, request and memory tools. Ensure the new precision policy
has end-to-end admission and correct report labels before the long run.
Use manifest metadata identity only for `.qw38`; never hash payloads/scales.
Record graph state and deterministic schedule identity with replay results.

TASK-036's C92 result was INCONCLUSIVE and its P100 gate failed; neither is
waived by improved speed or by passing a short development screen. No
answer-specific tuning, new calibration from acceptance cases, or repeating
deterministic answers until a gate looks better. New candidate outputs need
their own quality result. A corrective arithmetic change creates a new
candidate and invalidates affected evidence; do not splice candidates.

Run one consolidated gate:

1. Frozen core-54: 15 P100, 15 C92, 12 L12, and 12 R-512/R-4096. Apply
   unchanged EVAL-01 coverage, rendering, scoring, uncertainty and slice
   rules, NLL delta +0.03 aggregate/+0.06 slice, and capability loss 0.02.
   Obtain all 15 output-bound P100 reviews, naming reviewer attribution.
   Reuse reviews only for authenticated identical prompts and output text;
   changed text requires a fresh review. Missing/invalid/failed/inconclusive
   gates prevent promotion. Report C92 and the previous P100 failure honestly
   without turning individual answers into implementation acceptance targets.
2. Execute only `R-32768-s0-d0.1` for extended quality. The other five 32K
   fixtures remain inventory. Full-216 stays optional human-initiated work;
   agents must never launch or require it. No whole-model BF16 evaluation.
3. Retain checkpoints 1/3/4/63/64/65/255/256/257 and partitions
   1/63/64/65/255/256/alternating 63,65. Cover selected attention partition
   and graph bucket boundaries, nonempty prefill/decode handoff, interleave,
   reset, snapshot/restore and late-failure recovery. Reuse authenticated
   exact-identity TASK-037–039 focused checks; do not repeat them solely as a
   second witness. Same-schedule replay is bitwise; graph/eager is bitwise
   for the same arithmetic/partition schedule. Cross-schedule full-model
   state/logit deltas are diagnostics, not isolated-component tolerance gates.
4. Measure all nine PERF-01 rows through six executions: requests at
   256/4096/32768, and populated decode at 512/4096/32768, each with 128
   continuation tokens. Derive prefill/TTFT from request prompt phases.
   One execution per workload/engine, zero warmups, no repetitions or
   statistical performance claims. Include lazy graph/library first use and
   bucket changes inside their actual timing boundary; no preparatory token
   execution. Reuse authenticated TASK-027 comparator captures only when
   frozen inputs, settings, GPU/driver/power/toolchain and instrumentation/
   timing boundaries match. Refresh only invalidated rows, not the whole
   comparator automatically. Preserve previous captures unchanged.
5. Reuse the final 32K request for measured capacity at 32,768 + 128 slots.
   Require at least 2 GiB free-memory reserve. Count model codes/scales/padding,
   FP32 state, BF16 history/KV, Q8/FP8 packs, output/partial slabs, graph
   executables/controls, host staging, library workspace and load/capture
   transients in the appropriate host/device categories. Sample free memory
   and residency as well as tracked allocations, because CUDA graph/library
   allocations may bypass project counters. Do not add peaks/residency as
   though they were disjoint allocations; disclose tracking limits.

Required build/test coverage is the affected final Release targets, full-model
integration and the gates above. No unrelated complete Debug/Release suite,
context-capacity search, tile contest or renewed quantization sweep. Main-thread
long operations use `wake-run`; a failed reporting script can be repaired and
saved outputs reprocessed without rerunning valid inference.

## Delivery and completion

Publish `docs/implementation/task040-delivery.md` with reproducible pinned
build/load/run and rollback instructions, exact policy/precision and lifetime
contracts, selected graph/eager behavior, supported context, cold/setup costs,
per-row latency and memory ratios, quality/review/replay results and remaining
bottlenecks. Identify first-use/instrumentation and single-run limitations.
Update current architecture/runtime guidance to the actual selected behavior;
do not rewrite historical task reports or claim the new candidate was already
accepted in TASK-036.

Passing all quality, replay and capacity gates permits promotion. Achieving
the fast-engine goal additionally requires every one of the nine matched
latency ratios (llama.cpp/candidate) to be at least 1. Report individual unmet
targets; speed gaps alone do not force RETAIN_CONTROL when the other gates
pass, but a qualified delivery must say the speed goal remains unmet.

This decision task may complete with fully evidenced RETAIN_CONTROL, qualified
promotion with speed gaps, or accepted parity. Missing required executions,
reviews or provenance leaves it incomplete. A failing quality result is a
valid final decision outcome, not permission to omit the remaining required
evidence. Rank measured residual costs and name the smallest justified next
change if needed; do not automatically create or launch another batch.

# TASK-040 — Quantized engine validation and delivery decision

## Status

DONE

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

## Completion Report

**Decision: RETAIN_CONTROL.** Policy 1030 is not promoted; production remains
TASK-026/027 at `d2f02e2`. The candidate is the unchanged TASK-039 inference
runtime, with manifest metadata digest
`6ebcc402487aa92d4a6bb7d64ccfff00c20d74a738ac7e3fa2166522f4b35ff5`.
The full reproduction, frozen precision/lifetime contracts, performance and
memory tables, limitations, rollback, and follow-up are in
[`task040-delivery.md`](../task040-delivery.md).

The final core-54 scored FAIL: P100 FAIL with the unchanged confirmed
`case_077` error; C92 is INCONCLUSIVE at 7/15 versus 8/15 with loss interval
[0, 0.20]. NLL delta is +0.0025621463600068992 over 701 targets and every
slice passes its NLL gate. L12 is 12/12, R-512 and R-4096 are 6/6 each, and
the sole required `R-32768-s0-d0.1` is 1/1 in both arms. All fifteen P100
outputs have output-bound reviews. These quality outcomes prevent promotion
and are the permitted fully evidenced RETAIN_CONTROL result. All nine
llama.cpp/candidate latency ratios are below 1; capacity retains 8,740,864,000
free bytes after the final request, above the 2 GiB reserve.

Acceptance commands and results (all main-thread long commands were launched
through `wake-run`):

- `bash .cache/evaluation/qw38-language-v2/task040-support/run.sh` exited 125
  before build or inference because its historical local image ID was
  unavailable. The failure and authenticated preflight are preserved.
- `bash .cache/evaluation/qw38-language-v2/task040-support/resume.sh` passed:
  pinned Release build, full-model integration 1/1, core-54 54/54, default
  replay plus seven partition replays, fixed long retrieval, six once-only
  performance captures and capacity analysis.
- `bash .cache/evaluation/qw38-language-v2/task040-support/score.sh` produced
  fixed-long PASS and the final core scoring inputs; its focused profile check
  passed (`1 passed`). The script's final source checksum check exited 1
  because the profile test had intentionally changed. Saved inference was not
  rerun; original and final source identities are recorded in evidence notes.
- `uv run python .cache/evaluation/qw38-language-v2/task040-support/prepare-reviews.py`
  bound nine fresh adjudications and authenticated reuse of six candidate and
  fifteen comparator reviews. The unchanged `case_077` failure was retained.
- `bash .cache/evaluation/qw38-language-v2/task040-support/finalize.sh` passed:
  final reviewed score is FAIL as expected for RETAIN_CONTROL; compiler and
  decode delivery targets build, and frozen binary, inference-source and
  comparator identities match.

Independent GPT-6 Astra high review passed on pass 1 with no findings, gaps or
requests. All eight reviewed candidate identities and 208 evidence identities
were confirmed unchanged after review; candidate manifest SHA-256 is
`9fb138d3ee0a182a0ef9f43105f91148e8b95b3e9b8b38c05996a05a80c8fe2e`.
Evidence logs and detailed command mappings are retained under
`.cache/evaluation/qw38-language-v2/task040-support/`. Implementation and
acceptance evidence were produced by GPT-6 Codex in the main thread; completion
documentation and delivery bookkeeping were handled by GPT-6 Luna. Before
delivery, the main thread confirmed the reviewed code and guide and all 208
evidence identities were unchanged. The DONE-task cache scan found no matching
directories (`task040-support/cache-cleanup.json`). The largest measured
residual is Q4_K×Q8 prefill MMQ at 20.171 s; no follow-up task was created.

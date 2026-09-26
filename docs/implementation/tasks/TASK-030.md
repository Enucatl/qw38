# TASK-030 — Combined validation and delivery decision

## Status

DONE

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

**Decision: RETAIN_CONTROL. Promotion was not accepted.** The final combined
candidate is the TASK-028 attention schedule with TASK-029's Q4 fallback,
artifact manifest `41c1f5e673bb24eb2fb283aa6044dbccdebecc7cd85f847815b3c02a6763fc43`.
TASK-030 changed reporting and documentation, not inference arithmetic. The
retained production control is source `d2f02e2` with the same artifact; use the
separate-checkout rollback instructions in
[task030-delivery.md](../task030-delivery.md).

### Quality, review, and replay

The frozen core-54 completed 54/54, with authenticated unchanged comparator
outputs reused. Teacher-target NLL delta was +0.0012929024367053655 over 701
targets; aggregate and slice NLL limits passed. L12 was 12/12, and R-512 and
R-4096 were each 6/6 per horizon. The fixed `R-32768-s0-d0.1` case passed in
both arms (731204). C92 was 7/15 versus 8/15; its loss interval [0, 0.20]
crosses the 0.02 budget, leaving aggregate and SuperGPQA INCONCLUSIVE. This
blocks promotion without establishing a statistically significant regression.
All 15 P100 reviews are bound to final outputs. The seven changed candidate
texts were reviewed by GPT-6 Codex; eight unchanged candidate and all 15
unchanged comparator reviews retained prior owner attribution by identical
hashes. No material new candidate-only failure was found. Qualitative PASS
does not waive the C92 gate.

Same-schedule replay was bitwise identical at checkpoints 1/3/4/63/64/65/255/
256/257 across all seven partition patterns (1/63/64/65/255/256/63,65).
Session handoff, reset, in-memory snapshot/restore, interleave, and late-failure
recovery passed. Cross-schedule differences remain diagnostics. The single
long case and all required selected checks completed; the optional 216-case
suite was not run.

### Commands and independent review

The exact acceptance commands and logs are retained in
`.cache/evaluation/qw38-language-v2/task030-support/`:

- `bash .cache/evaluation/qw38-language-v2/task030-support/validation.sh` —
  pinned Release build and focused CUDA checks 4/4 PASS; core-54 54/54,
  default decode replay, seven partition replays, and the single long case
  COMPLETE.
- `bash .cache/evaluation/qw38-language-v2/task030-support/score-and-measure.sh`
  — both long scorings PASS; five mutated negative cases rejected; six final
  Nsight workload captures completed once each, zero warmups.
- `bash .cache/evaluation/qw38-language-v2/task030-support/finalize-evidence.sh`
  — final paired report and profile summaries produced. Across its recorded
  attempts, 17 focused scoring/benchmark/profile checks passed: the first
  invocation failed at pytest import-path setup before tests or analysis; a
  corrected Python 3.12 invocation passed 13 tests, while four imports were
  unavailable because `transformers` was missing; the final corrected
  dependency invocation ran only those four, which passed. No engine execution
  was repeated. Memory ratios were produced separately by
  `uv run --script .cache/evaluation/qw38-language-v2/task030-support/memory-ratios.py`.

GPT-6 Astra high completed two review passes; the second/final result is PASS
with no findings, acceptance gaps, or evidence requests. The review record is
`.cache/evaluation/qw38-language-v2/task030-support/astra-pass2.txt`. Candidate
and evidence identities were verified unchanged after review. GPT-6 Codex was
the main implementation and evidence owner; GPT-6 Astra independently reviewed
the candidate; GPT-6 Luna completed this documentation and delivery record.

### Performance and capacity

Each selected workload/engine ran once with zero warmups. Candidate timings
include Nsight instrumentation and first use. Comparator captures were reused
after equivalence checks; ratios below are comparator/candidate latency, so
values below one mean the comparator was faster and parity is unmet. Request
prompt phases supply their matching prefill/TTFT rows. Exact settings, cold
setup, and full captures are in
`task030-support/summary.json` and `profiles.json`.

| PERF-01 row | QW38 ms | llama.cpp ms | llama.cpp/QW38 |
| --- | ---: | ---: | ---: |
| prefill 256 | 282.091 | 109.548 | 0.3883 |
| prefill 4096 | 4194.291 | 1379.908 | 0.3290 |
| prefill 32768 | 56646.550 | 12372.765 | 0.2184 |
| decode 512 | 4064.439 | 1932.101 | 0.4754 |
| decode 4096 | 4141.900 | 1952.946 | 0.4715 |
| decode 32768 | 5292.168 | 2131.478 | 0.4028 |
| request 256 | 4341.649 | 2027.903 | 0.4671 |
| request 4096 | 8339.721 | 3335.562 | 0.4000 |
| request 32768 | 61915.300 | 14480.757 | 0.2339 |

All nine parity targets are unmet. At 32K the candidate request TTFT was
56646.550 ms versus 12372.765 ms, and the complete request was 61915.300 ms
versus 14480.757 ms. For request-32768, candidate load/upload was 3050.045 ms,
cold setup 3056.582 ms, and CUDA initialization 299.027 ms. Separately, for
decode-32768, load/upload was 3044.472 ms, cold setup 3050.944 ms, and CUDA
initialization 289.152 ms; the 56531.827 ms populated setup was excluded from
the measured decode row. These are instrumented single observations, not
medians or confidence estimates.

Validated capacity is 32768 prompt plus 128 continuation tokens (32896 slots).
At 32K, free memory after workload was 8317239296 bytes, above the 2 GiB
reserve. Whole-process tracked allocation peak was 24196583696 bytes,
including load and transients; sampled resident growth was separately
24769462272 bytes. Model bytes including scales/padding were 21462812800,
persistent state 2309816320, and scratch 256901120. Allocation peak and sampled
residency are different measurements and are not additive. R030-01 was resolved
by calculating allocation-peak and sampled-residency ratios from authenticated
saved TASK-027 and final TASK-030 profiles; no engine rerun was needed. All six
memory ratios (llama.cpp/QW38) are below one; byte counts and source hashes are
in `task030-support/memory-ratios.json`. Reproduce them with
`uv run --script .cache/evaluation/qw38-language-v2/task030-support/memory-ratios.py`.
No maximum-context search was performed.

### Follow-up and documents

`FOLLOW_UP_REQUIRED`: resolve the C92 uncertainty before any promotion. The
performance gaps also remain. No roadmap task or extra optimization was added.
The candidate runtime, format, reproduction, and rollback instructions are in
[task030-delivery.md](../task030-delivery.md); the scoring policy rebind is in
[task030-eval-policy-rebind.json](../task030-eval-policy-rebind.json). Hardware
was an RTX 5090, driver 590.48.01, 400 W power limit, using the immutable image
documented in the delivery guide. Full source, binary, policy, comparator and
capture identities remain in the retained support evidence.

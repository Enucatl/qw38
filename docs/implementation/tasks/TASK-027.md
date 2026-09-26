# TASK-027 — Matched whole-request performance baseline

## Status

DONE

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
request, observed per-row ratios, cold load and peak resident/transient
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

Follow PERF-01's explicit 2026-09-26 user amendment: one run per workload and
engine, zero warmups/repetitions, no median/p99/bootstrap intervals. Derive the
prefill row from the complete request's prompt phase. Capture profiles and
allocation peaks in those same runs, label instrumentation and first-use
costs, and report observed ratios against the 1.00 target. Numerical tests run
once; the accepted TASK-026 runtime/quality context remains fixed.

## Expected files/modules

Separate benchmark harness/public-API adapter, result records, raw timing/memory/profile evidence and bottleneck-ranked baseline report.

## Tests required

### Unit and contract checks

Where harness code changes: units, fresh/populated-state identity, timing-boundary accounting, missing-input/identity validation and report completeness.

### Reference and numerical checks

Check accepted runtime/artifact identities and output validity before/after timing. Do not substitute benchmark completion for quality acceptance.

### Integration checks

Run every mandatory PERF-01 row on matched inputs with effective llama.cpp settings and accepted candidate. Validate capacity/population, output policy and raw paired measurements. Missing mandatory evidence blocks completion; an optional maximum-capacity probe is labeled separately.

## Benchmark required

Required: all PERF-01 prefill, populated decode and complete-request rows, cold load/upload, peak resident/transient memory and profiler attribution. Report quantization/packing, projections, GDN, attention, head, launches, waits and transfers without double counting.

## Acceptance criteria

- [x] All mandatory matched candidate/llama.cpp rows and raw samples are present with valid identities and effective settings.
- [x] Exactly one execution per workload/engine is recorded, with no warmups or repetition statistics; observed per-row ratios are reported.
- [x] TTFT, prefill, decode, total request, cold setup and generation/evaluation boundaries are distinct.
- [x] Peak memory and component profiles support a ranked bottleneck/gap report without double counting.
- [x] Quality context remains valid; measured speed gaps are explicit and do not block authorized refinement.

## Architecture blocker rule

Missing required comparison or quality evidence prevents completion. A speed gap is a measured result and does not block TASK-028–032. Measurement difficulty alone does not reopen retained architecture contracts; report any actual conflict using the ledger's full blocker fields.

## Completion report

### Result

Complete single-run matched baseline recorded. All nine rows have one observation
per engine; every llama/QW38 latency ratio is below 1.00. Speed gaps are reported
for downstream refinement and do not block completion. The user-authorized
2026-09-26 performance protocol amendment remains in force: zero warmups,
profiling in the measured run, and no repetition statistics.

### Changes made

Added the separate request benchmark adapter, preparation/build and profiling
scripts, harness/profile checks, and PERF-01 single-run policy amendment. No
production runtime code changed. Candidate evaluator SHA-256:
`ab7479715a658fb9072322f87dba4390aa0f1bcda2a38aeaf6052b0da7577c1c`;
accepted artifact manifest SHA-256:
`41c1f5e673bb24eb2fb283aa6044dbccdebecc7cd85f847815b3c02a6763fc43`.
The GGUF conversion provenance is unavailable beyond its model name and
matching structural metadata; no model payload was hashed. This comparison does
not establish a new Pareto quality claim.

### Tests run

`bash scripts/task027_build.sh` completed, including public-API tokenizer
mapping/encoding validation for 248,077 source entries, 243 UNUSED padding IDs,
and consumed/special IDs. During the build,
`uv run --with pytest pytest -q tests/test_task027_benchmark.py` passed (3).
`uv run --with pytest pytest -q tests/test_task027_profiles.py` passed (1).
The benchmark checks cover harness contracts; the profile check covers trace
allocation accounting and kernel categorization. The accepted TASK-026 quality
context was retained, not rerun.
Build/test log: `.codex-wake-run/4658d72adb49.log`; profile test log:
`.codex-wake-run/cf98efecba0b.log`.

### Benchmark results

Exact commands: `uv run --script scripts/task027_benchmark.py run`,
`uv run --script scripts/task027_benchmark.py summarize`, and
`uv run --script scripts/task027_profiles.py` (12 traces analyzed).
`git diff --check` passed. Twelve profiled executions (one per workload/engine,
zero warmups) produced nine matched rows. Timings include Nsight instrumentation
and first-use costs; prefill rows reuse each request's prompt phase. The summary
reports TTFT, decode tail, total request and cold setup separately. Ratios
(llama/QW38) by row: decode-512 0.464792; decode-4096 0.456136; decode-32768
0.357077; prefill-256 0.317430; request-256 0.451515; prefill-4096 0.209919;
request-4096 0.307713; prefill-32768 0.059353; request-32768 0.067548.

The largest gap is 32K prefill: QW38 208,460.004 ms vs llama.cpp 12,372.765 ms;
request totals are 214,376.765 ms vs 14,480.757 ms. The QW38 attention scan
kernel accounts for 177,872.02 ms. At decode T=4096, projections account for
3,192.76 ms GPU work of 4,281.50 ms host time; at T=32768, attention rises to
2,138.30 ms and projections to 3,240.44 ms. Request-32768 trace records
1,852,200 `cudaLaunchKernel` and 987,621 `cudaLaunchKernelExC` calls. CPU wait
and overlapping kernel totals are not added to wall/GPU time.

At T=32768, tracked allocation peaks are 24,196,583,696 bytes (QW38) and
21,837,146,883 bytes (llama.cpp); sampled resident growth is separately
24,769,462,272 and 21,911,044,096 bytes. The latter is a post-run point sample,
not an exact process residency peak. QW38 retained 8,317,239,296 free bytes,
above the 2 GiB reserve. Timings are single observations with no median, p99,
confidence interval or repeated-run inference. Request-256 differed in 77 of
128 output positions and decode-512 in 4; other output positions matched.

Complete raw evidence, twelve JSONL captures, logs, Nsight reports/SQLite,
manifest, summary, profiles, token inputs, pinned llama headers and benchmark
binary are preserved under
`.cache/evaluation/qw38-language-v2/task027-support/`. The manifest records
commands, source/binary/image/input identities, effective settings, GPU and
hardware context. The capture ran on the pinned CUDA 13.4.1 candidate
container, GCC 14.2.0, native sm_120, RTX 5090, driver 590.48.01, 32607 MiB,
with the existing 400 W limit. The comparator image revision is
`e6ab7c1a41054a888ada952eab4c886444c2f5ad`. OS page cache was not evicted, so
cold setup is fresh-process load/upload with cache-order dependence. The earlier
repeated capture was canceled per the user amendment; partial results are
excluded.

### Roles and review

Implementation and evidence collection: main-thread GPT-6. Independent review:
gpt-6-astra high, pass 1, PASS with no findings, acceptance/evidence gaps or
targeted evidence requests. Review record:
`.cache/evaluation/qw38-language-v2/task027-support/astra-review.txt`.
Documentation and delivery: gpt-6-luna medium.

### Architecture blocker

None.

### Follow-up observations

Long-context attention and conversion/projection costs are the largest measured
gaps. Follow-up ownership remains with planned TASK-028–032; no downstream task
was started here.

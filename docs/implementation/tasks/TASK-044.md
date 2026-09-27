# TASK-044 — Attribute first-decode latency and startup costs

## Status

DONE

## Milestone and dependency

M23 — Explain the TASK-042 first-decode regression or identify the remaining
measurement gap. Depends on TASK-043 so its final captures can be reused.
This user-requested diagnostic follows the existing validation sequence;
it does not reopen TASK-042 acceptance or change TASK-043's gates.

## Problem and preserved evidence

TASK-042's matched 4096-token prompt plus eight fixed continuation inputs
measured decode at 150.054052 → 156.324259 ms (+4.1786%). The first step was
21.262324 → 29.279882 ms; the remaining seven totaled
128.790766 → 127.043607 ms. The subsequent diagnostic trace did not reproduce
the roughly 8 ms first-step gap. Both arms launched 883 first-token kernels;
graph instantiation and GPU kernel totals did not establish its cause.
Full-request setup also differed, 5.530 → 20.233 s.

Start with the [TASK-042 report](TASK-042.md) and saved measurements/traces in
`.cache/evaluation/qw38-language-v2/task042-support/`, including authenticated
`parent-binaries/`. `.cache/task042` was removed; do not rely on those old
executable paths. Reuse TASK-043 evidence where compatible, keeping each
candidate, workload and instrumentation identity explicit.

## Bounded investigation

1. Reconcile saved host per-step timings with CUDA API/GPU timelines. Separate
   CUDA initialization, model load/upload and session binding from prefill,
   first decode and subsequent decode. Attribute graph capture/instantiation,
   upload/first launch, module/library loading, allocations, input/control
   transfers, kernels, readback and synchronization where evidence supports it.
   Show unexplained host/GPU gaps explicitly; overlapping intervals are not
   additive elapsed costs.
2. If saved traces cannot answer the question, extend the existing
   `benchmarks/request_bench.cpp`, `src/runtime/language_model.cpp` profiling
   ranges and `scripts/task027_profiles.py` only at the missing boundaries.
   Reuse Nsight/CUDA API correlation and the existing `decode_token` range.
   Record token position, graph bucket/build/replay status and timing scope.
   Add no synchronization, warmup inference or preparation moved outside its
   real timing boundary merely to improve the measured result.
3. Capture at most one newly instrumented execution per TASK-041 parent and
   TASK-042 candidate of the original frozen 4096+8 development workload,
   only if step 1 establishes a concrete visibility gap. Preserve the same
   inputs, policy, capacity, toolchain and measurement boundaries; document
   unavoidable differences and profiler overhead. Record clocks, power and
   temperature when available; do not infer cache state from elapsed load time.
   If historical binaries need rebuilding for markers, identify those builds
   separately and verify inference code is otherwise unchanged. No sweeps,
   repeated trials or statistical claim; main-thread long commands use
   `wake-run`.

## Verification and completion

Publish `docs/implementation/task044-decode-attribution.md` with exact commands,
source/binary/artifact identities, evidence paths and a parent/candidate table
of setup, prefill, first decode, remaining seven steps and attributed costs.
Any reporting logic needs a focused check against known trace/timing data.
If instrumentation changes code, verify unchanged output IDs, final position
and logits for the same arithmetic schedule, plus affected existing checks.
No full quality rerun is required for reporting-only changes.

Classify supported costs as one-time preparation, graph-bucket transition or
recurring token work. A non-reproduced gap stays **UNRESOLVED**; a trace with
similar kernels alone does not prove loading, graph overhead or measurement
noise caused the original regression. Completion requires the bounded analysis
and visibility gaps to be documented, not a forced causal conclusion. If still
unresolved, retain FOLLOW_UP_REQUIRED with the exact missing observable and
smallest next discriminating measurement. If explained, propose the smallest
evidence-backed repair separately; this task does not implement an optimization
or the <=4096/>4096 dispatcher. Apply the accepted startup tradeoff without
relabeling historical regressions or weakening quality/capacity gates.

## Completion Report

Published the bounded analysis in
[`task044-decode-attribution.md`](../task044-decode-attribution.md). The
original TASK-042 first-decode difference of +8.017558 ms remains
**UNRESOLVED**. New instrumented captures did not reproduce it: first decode
was 25.886605 ms (parent) and 25.543292 ms (candidate). The report classifies
observed setup, prefill, first graph use and recurring token costs, preserves
the accepted TASK-042 regression and TASK-043 P100 exception, and specifies
the missing observable and smallest discriminating follow-up. No optimization
or quality rerun was performed.

The request benchmark adds opt-in CUDA initialization, runtime creation,
model load/upload, session creation, plan binding and initial-prefill ranges;
the decode marker records token position, bucket and planned build/replay/eager
state. The existing profile analyzer reports the first eight steps with
overlap-safe interval attribution. Exactly one new instrumented 4096+8
execution per arm was captured. Outputs retained each arm's original IDs,
position 4104 and byte-identical logits. Evidence, identities, complete tables,
commands and limitations are in the report and
`.cache/evaluation/qw38-language-v2/task044-support/`.

Verification results:

- `bash .cache/evaluation/qw38-language-v2/task044-support/run-instrumented.sh`:
  both builds and the candidate integration check passed; the script then
  exited 1 before either capture because of an unset container shell variable.
  The check was `QW38_AUTHORITY_ARTIFACT=/workspace/.cache/candidates/candidate-fp8-mixer-q8-mlp-v1.qw38 ctest --test-dir build/pinned-release --output-on-failure -V -R '^language_model_integration$'` (**1/1 PASS**, 20.07 s).
- `bash .cache/evaluation/qw38-language-v2/task044-support/capture.sh`:
  **PASS**, exit 0; one capture per arm. `verify-outputs.py` passed for both
  arms, including output identity, finite/byte-identical logits, final
  position, one run and zero warmups.
- `uv run --with pytest pytest -q tests/test_task027_profiles.py`:
  **2 passed**. Saved-trace metrics and graph/API correlation checks passed;
  `verify-source-graph.py` passed source, binary, compiler-command and capture
  correlation checks. `git diff --check` passed.
- Astra review: **PASS**, first pass, no findings or evidence gaps; verbatim
  result: `.cache/evaluation/qw38-language-v2/task044-support/astra-review-1.md`.

The instrumentation and reporting checks do not establish the cause of the
unreproduced first-decode event. **FOLLOW_UP_REQUIRED:** capture a first decode
that exhibits the excess latency, with CPU scheduling and host file/page-fault
visibility correlated to CUDA APIs and GPU activity. If it does not recur,
the original cause remains unresolved. No repair is supported by the current
evidence.

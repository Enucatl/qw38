# TASK-044 — Attribute first-decode latency and startup costs

## Status

TODO

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

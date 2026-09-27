# TASK-041 — Reuse-rich Q4_K×Q8 prefill MMQ

## Status

TODO

## Milestone and dependency

M20 — Efficient compact MLP prefill. Depends on TASK-040.
[FAST-03](../task_ledger.md#prefill-amendment--fast-03-2026-09-27) and the
[prefill analysis](../post-task040-prefill-analysis.md) govern this batch.
Start from the retained policy-1030 development candidate; production remains
TASK-026/027. This task implements a prefill consumer, not a new quantizer.

## Delivered behavior and first implementation

Replace the fixed J32 full-chunk MMQ schedule with the pinned llama.cpp
`e6ab7c1a41054a888ada952eab4c886444c2f5ad` I128/J128, 256-thread ownership
pattern. Keep K256 weights staged while consuming K128 activation halves.
Adapt its cooperative metadata loading and register prereads to QW38's
physical layout and FP32 metadata. Reuse the existing third-party attribution.
The actual comparator trace selected J128; this is not an architecture-wide
claim that one tile is optimal.

Decode each output row/K32 subgroup's weight scale/minimum once per staged
tile. Stage activation scales and exact sums alongside codes and reuse them
across output fragments. Use a complete-tile specialization for the actual
aligned gate/up/down shapes and a bounded masked path for tails. Keep M=1
on the existing MMVQ consumer; retain J32 for small M where J128 would mostly
compute padding. Start with J128 at M>=128 and the existing smaller path below
128; change that boundary only for a concrete measured regression. No generic
operator registry or autotuner.

Do not simply widen the old accumulator array. Derive the adopted shared
layout, ownership and bank padding, including FP32 metadata, and record actual
compiler registers, shared/local bytes and launch shape. Full double buffering
is not required: the pinned reference uses successive activation halves with
barriers. Add asynchronous staging only if an observed failure or stall makes
it the smallest useful next step.

Likely code: `cuda/q4k_q8.cu`, its existing header/producer helpers if required,
`tests/q4k_q8_test.cpp`, and `benchmarks/mlp_bench.cpp`. No compiler/artifact
change is expected; do not retain an extra resident weight view or globally
expand weights. Leave unrelated decode/FP8/head/GDN/submission paths alone.

## Numerical and lifetime contract

Retain policy 1030, existing weight codes/scales and the independent equation
`a * (d*s*sum(q*z) - dmin*m*sum(z))` per K32. RMS/SwiGLU evaluate in FP32,
round locally to BF16 RNE, and produce the same signed Q8 codes, FP32 absmax
scales and exact INT32 sums. Preserve zero/tail/nonfinite behavior and the
typed failure path. Keep K32 integer dots bounded by 60,960, sums by 4,064,
and FP32 scaling, accumulation, residuals and nonlinearities.

Hoisting FP32 `d*s`/`dmin*m` is permitted; replacing them with upstream half2
products, multiplying six-bit scales into INT8 operands, or merging K32
groups with different activation scales is not. Preserve the existing
floating expression and K32 accumulation order first. Any necessary numerical
change needs independent reference validation and must remain within the
declared policy; do not silently weaken it or its tolerances.

One fused RMS/Q8 pack feeds both gate/up consumers; FP32 gate/up slabs feed
one fused SwiGLU/Q8 pack; down adds the residual. Session-stream scratch
remains live through its last consumer, then is reused. Registers/shared
tiles live only within their kernel. Do not add a paired gate/up super-kernel
before measuring this schedule.

## Checks and completion

Run affected existing arithmetic/MLP/state checks once, extending the existing
test only where it cannot exercise the new tile or dispatch boundary. Cover
actual gate/up and down shapes; independent affine-reference outputs;
zero/nonfinite inputs and padding; M=1, the small/large boundary, J128 tails
and full M256; unchanged M=1 output and failure semantics. Reuse established
numerical tolerances. No Cartesian product of every shape and token count.

Compare parent/candidate complete real-shape MLP at M=1 and M=256 once each,
including RMS, both packs, all three contractions, epilogues, completion and
actual workspace. Use identical artifact and inputs, no warmups/repetitions.
Record selected kernels and resources in that execution. A GEMM-only result
does not satisfy the task. Reuse existing benchmark infrastructure; report
setup separately and first use at its actual boundary.

Run the existing matched development request: authenticated TASK-027
256-token prompt followed by its first eight frozen continuation inputs,
with final logits/readout. One execution per arm; report prefill and decode
separately. Reuse an old parent result only if source, artifact, inputs,
toolchain and timing boundaries match. Run the frozen TASK-035 eight-window
development NLL screen if changed floating reduction order can affect values;
the +0.03 aggregate budget remains and this is not core-54 acceptance.

Completion requires the selected new consumer, passing affected checks,
observed complete M256 MLP and integrated prefill improvement, and no observed
matched decode regression. An unexpected regression is investigated for its
concrete cause; it is not hidden by a faster isolated kernel or averaged away
through repetitions. A rejected schedule is recorded with evidence and the
task remains BLOCKED pending a specific repair/replan; unchanged fallback is
not delivery of this optimization. TASK-043 owns final promotion and all
PERF-01 rows. Main-thread long commands use `wake-run`.

# TASK-041 — Reuse-rich Q4_K×Q8 prefill MMQ

## Status

DONE

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

## Completion Report

Implemented and retained the policy-1030 J128 prefill consumer for M>=128,
with J32 for M2..127 and the existing MMVQ path for M1. The implementation
reuses weight affine metadata and activation codes/scales/sums across token
fragments, with aligned full tiles and a masked tail path. No artifact,
quantizer, extra resident view, or workspace allocation changed. Development
acceptance passed; production remains TASK-026/027 and final promotion, core-54,
replay, capacity, and PERF-01 decisions remain TASK-043.

Main-thread implementation model: GPT-6 Codex (the more specific model variant
was unavailable). Parent revision: `e24f6ae9e3457a5ab81e5f620cb67bc1ddd277a9`.
Artifact: `.cache/candidates/candidate-fp8-mixer-q8-mlp-v1.qw38`, policy 1030,
manifest digest `6ebcc402487aa92d4a6bb7d64ccfff00c20d74a738ac7e3fa2166522f4b35ff5`;
artifact payload hashes were not computed. Candidate source hashes:
`cuda/q4k_q8.cu` `6f5010874766e0101fec05ec01e8a69d28172f01e419cfc39e456522a4313661`,
`tests/q4k_q8_test.cpp` `96c10c9c41fc39826b501cded1310d85e8c241a92c6691f267dde8dbe94378f5`,
`tests/q8_mlp_integration_test.cpp` `56247e124127fb527424f5492af75f2aa62fa580478b00d8a940897901e7e462`,
`benchmarks/mlp_bench.cpp` `dff05acc38da94a74916482fc69b6eb96d3bb573e72f788b072346c30d44402f`,
and `third_party/llama.cpp-q4k/README.md`
`bf0e56aa79dec29bf0bbf3c581a1dc7189d8fc39573fc612a029015499abb5ba`.
Candidate executable SHA-256 values are preserved in
`.cache/evaluation/qw38-language-v2/task041-support/candidate-binaries.sha256`;
they were checked immediately before measurement and matched the reviewed
candidate. Astra confirmed the reviewed source and executable identities were
unchanged.

Hardware/toolchain: NVIDIA RTX 5090, driver 590.48.01, CUDA image 13.4.1,
`nvcc` V13.4.59, pinned image
`sha256:254963cc774290ddeae6ada94047607b5bed9eb668dfb344e58aac05889f2b49`.
Parent and candidate used the same pinned-release build, artifact and frozen
inputs. Exact main-thread command evidence and raw logs are retained under
`.cache/evaluation/qw38-language-v2/task041-support/`. Invoked commands:

- `bash .cache/evaluation/qw38-language-v2/task041-support/baseline.sh` — PASS;
  one complete parent MLP observation at M1 and M256 and one fixed-input request.
- `bash .cache/evaluation/qw38-language-v2/task041-support/check.sh` — PASS;
  all three affected CTests passed:
  `QW38_AUTHORITY_ARTIFACT=/workspace/.cache/candidates/candidate-fp8-mixer-q8-mlp-v1.qw38 ctest --test-dir build/pinned-release --output-on-failure -V -R "^(q4k_q8|q8_mlp_integration|language_model_integration)$"`.
- `bash .cache/evaluation/qw38-language-v2/task041-support/measure.sh` — PASS;
  one complete candidate MLP observation at M1 and M256 and one matching request,
  with no warmups or repetitions.
- `bash .cache/evaluation/qw38-language-v2/task041-support/analyze.sh` — PASS;
  analyzed saved traces and asserted artifact/output identity and performance
  gates. The input file hash is preserved separately in
  `parent-binaries-input.sha256`.
- `git diff --check` — PASS.

The arithmetic test covers dispatch boundaries, J128 tails/full M256,
independent affine outputs, exact activation metadata, zeros, nonfinite values,
and padding. The real-artifact MLP integration test covers the actual gate/up
and down shapes, full and masked paths, M1 and M256 state/failure semantics,
and reference output tolerances. The full-model integration test covers
graph/eager equivalence, prefill handoff, replay and failure recovery. All
three passed. Since the FP32 expression and per-output K32 accumulation order
did not change, the contract's conditional eight-window development NLL screen
was not triggered; cross-schedule and independent affine checks passed. No
final quality or promotion claim is made.

Complete operation measurements (single run per arm; setup reported separately;
M1 first library use included in operation timing):

| Operation | Parent | Candidate | Result |
| --- | ---: | ---: | --- |
| M1 complete MLP GPU interval | 0.749856 ms | 0.817984 ms | +9.1% first-use cost |
| M1 complete MLP host | 0.748793 ms | 0.816933 ms | +0.068140 ms |
| M256 complete MLP GPU interval | 3.003104 ms | 1.887008 ms | 37.1647% lower |
| M256 complete MLP host | 3.002186 ms | 1.871035 ms | 37.6776% lower |
| Integrated prefill | 230.749562 ms | 161.282236 ms | 30.1051% lower |
| Matched eight-input decode | 145.189233 ms | 144.086459 ms | 0.7595% lower |

Integrated request output IDs matched; final position was 264 and logits were
finite. M256 launches used `mmq_j128<true>` with 136x2 gate/up and 40x2 down
grids, 256 threads, 220 registers/thread, 65,536 dynamic shared bytes, and zero
reported static shared or per-thread local bytes. Actual Q8 workspace remained
41,222,148 bytes and PrefillEngine workspace 69,206,016 bytes; hot allocations
were zero. Full resource and trace data are in `comparison.json`.

The isolated M1 complete first-use regression is recorded as
`FOLLOW_UP_REQUIRED` if cold M1 MLP latency becomes a product requirement.
Trace evidence locates the difference in first-use loading/submission as the
new module is larger; the existing MMVQ consumer remains at 38 registers/thread.
This is a single observation, not a statistical estimate, and is outside
TASK-041's explicit gates. It is not hidden or claimed as a pass.

Independent Astra review: PASS on pass 1, saved at
`.cache/evaluation/qw38-language-v2/task041-support/astra-review-1.md`.
There were no code findings, acceptance gaps, or evidence requests. Main thread
confirmed the reviewed diff, source hashes, and executable hashes remained
unchanged. TASK-042 and TASK-043 were not activated.

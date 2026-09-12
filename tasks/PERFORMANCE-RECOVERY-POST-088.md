# Speed recovery after OPT-088

Status: proposed implementation batch OPT-089 through OPT-098.
Source review at Quartz HEAD after OPT-088, 2026-09-12.
Task design only — not implementation, not a tok/s claim.

Reviewed HEAD: `dc4cc82d86e63e6d86daad4f37530e620897e229`.
Authority: llama.cpp `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`;
GGUF SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`.
The local `../llama.cpp` worktree has a different HEAD; this review inspected
the pinned revision with `git show`, not that worktree's current files.
Only these eleven task documents are proposed changes in this review.
The ledger rows below are a proposal, not an update to the active ledger.

## Executive summary

- Decode is the priority: OPT-088 needs approximately 12.21 ms/token at D128
  and 13.22 ms/token at D2048 to reach pinned llama throughput.
- The RTX 5090 is Blackwell, compute capability 12.0 (`sm_120`), not Ada.
  Ada-derived techniques may transfer; Ada measurements are not SM120 evidence.
  See [NVIDIA's RTX 5090 specification](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/).
- Start with the already implemented `late_w4`. OPT-085 measured complete
  rotating FFN 16.383 → 9.800 ms and short-engine 868.295 → 639.749 ms.
- The 40.2% FFN saving is not a 40.2% engine gain. Transferring 6.583 ms to
  OPT-088 predicts approximately 49.8/46.7 tok/s at D128/D2048, still below
  69.1/67.7. These are non-additive Amdahl estimates, not measurements.
- `late_w4` was quality-incomplete, not shown to degrade quality. Finish its
  real candidate-specific NLL and continuation run before accepting any risk.
- OPT-082's fused Q4 comparator omits the BF16 output rounding and uses host
  `std::exp` against CUDA arithmetic. Repair the comparator; keep identity strict.
- Both Q8 misses compare staged-Q8 arithmetic against original BF16 inputs.
  Separate implementation error from activation-quantization error before
  deciding whether the miss is a kernel defect.
- Keep Q8 `abs_scale=0.05`, `rel_tol=0.05` for every K. There is no evidence
  supporting either a blanket or a large-K-only tolerance increase.
- OPT-086/088's reported parity passes do not erase their documented misses.
  New admission must require every applicable positive case to pass; a successful
  process exit, correct selector, and zero nonfinites are insufficient.
- After the Q4 decision, obtain one matched decode attribution and a secondary
  P4096 attribution. OPT-071 repaired methodology; its synthetic fixture is
  not a measured post-088 family gap table.
- Q8 mixer is the next credible large target, but staging is already shared
  in `matrix_vector`. Test grouped projection launches, not another staging cache.
- Pinned llama has integer packed Q4 dots, factored scale/min corrections,
  fused gate/up support, vector attention, warp-sharded GDN and CUDA graphs.
  Source capability does not prove which dispatch ran in the timing sitting.
- GDN and attention remain candidates for smaller savings. OPT-077 was
  inconclusive; OPT-078's extra-prep-launch path was a demonstrated loser.
- A single performance-only OPT-077 replication is conditional on a concrete
  measurement defect found by the new attribution. Do not rerun unchanged losers.
- Keep full candidate quality mandatory. Permit only an explicitly labeled,
  tightly bounded PPL concession for `late_w4`, conditional on a large measured
  decode benefit; recurrence, task regression and kernel parity remain strict.
- Prefill is secondary: the remaining P4096 gap is 93.71 ms, not evidence of
  a particular kernel deficit. Test one same-math async-copy scheduling change
  after decode decisions, without a tile or compiler sweep.
- OPT-098 reports internal progress, llama parity and historical OPT-056/+5%
  separately. A successor quality pass cannot turn the old release gate green.

## Measured starting point

The [OPT-088 report](../evidence/optimization/opt088-batch-gate/REPORT.md)
records an uninstrumented, same-sitting comparison at 400 W on 2026-09-11.
Its recorded measurement source is `27179f41593ae8252438d6a46f1edce60f3fa642`
(dirty); do not mislabel it as the reviewed clean HEAD.

| Workload | Quartz tok/s | llama tok/s | Quartz/llama | Equivalent Quartz / llama ms | Decode p95 Quartz / llama ms | Quality / outcome |
|---|---:|---:|---:|---:|---:|---|
| P4096 | 2956.4502 | 3170.927662 | 0.9324 | 1385.45 / 1291.74 per prompt | n/a | Shipping regression pass; absolute task fail |
| D128 | 37.482605 | 69.132553 | 0.5422 | 26.679 / 14.465 per token | 26.8056 / 14.461 | Same; parity and +5% unpassed |
| D2048 | 35.732132 | 67.719321 | 0.5277 | 27.986 / 14.767 per token | 28.0852 / 14.552 | Same; parity and +5% unpassed |

Equivalent latency is reciprocal mean throughput, not mean raw latency or p95.
For decode, the report's 3126.80/3384.11 ms parity gaps cover 256 generated
tokens; divide by 256 for the per-token numbers above. P is 6.76% lower
throughput and 7.25% longer elapsed time than llama.

Shipping: Q4 `packed` / `paired_staged`; Q8 `dp4a_q8_1`, layout `r1_w4`;
MMQ `fma_async_x`, `i128_j128`; prompt attention `kv_once`; decode GDN
`sequential`; decode attention `warp_query`, 16 partitions; prompt-pair `off`;
execution graphs `ffn_only`; `NVCCFLAGS=-O2 --fmad=false`.

OPT-084 froze Q8 `r2_w2`, whereas OPT-088 ships `r1_w4`. Preserve both
identities. OPT-084's 2304-target NLL is reused OPT-058/069 data, not a fresh
OPT-084 GPU rescore; its baseline-versus-self zero delta is tautological.
OPT-089 authenticates or replaces the current-control score cache in new
evidence, without overwriting OPT-084. Shipping's reported
`absolute_quality_status=fail` is inherited `task_arithmetic` A vs expected B;
`quartz_baseline_regression_status=pass`. PPL ≤1.01 and recurrence incremental
NLL ≤0.02 remain the default. OPT-088 records `production_kept=true`,
`release_eligible=true`, but `performance_pass=false` and `gate.passed=false`.
Its 2K point result (3181.20 vs 3173.96 tok/s) does not close OPT-016.

## Decode gap diagnosis

Impact ranges below are planning estimates of potentially removable time,
not measured exclusive shares. Do not add rows. The component replay numbers
come from different windows/cache regimes: notably 16.38 ms FFN, 7.45 ms mixer,
12.52 ms GDN and 9.09 ms attention cannot all be exclusive parts of a 28 ms token.
OPT-090 must reconcile repetitions, restoration, copies and event unions first.

| Rank | Suspected sink | Evidence and confidence | Estimated opportunity / next decision |
|---|---|---|---|
| 1 | FFN Q4 MMV instruction/dequant/reduction cost | High: OPT-085 complete FFN saving 6.583 ms, CI 6.5472–6.6200; short engine also faster | Roughly 6–7 ms/token for `late_w4` if quality and full-engine transfer hold; OPT-089 first |
| 2 | Mixer Q8 projections, including GDN output | Medium-high as a cost, medium as a recoverable gap: OPT-086 rotating mixer ~7.45 ms; no current matched llama exclusive share | 0.1–1 ms/token hypothesis from grouping; OPT-092. `r2_w2` alone has no upside |
| 3 | GDN recurrence/state traffic | Medium mechanism confidence, low current time confidence: sequential key loops, 48 recurrent layers; OPT-077 noisy positive mean | 0–0.6 ms/token planning range for existing `tile32`; do not call 12.52 ms an exclusive sink |
| 4 | Attention computation and repeated KV reads | Medium: same D128/D2048 gap scale rules out context growth as the main explanation; larger context adds ~1.31 ms in Quartz vs ~0.30 ms in llama | 0.1–1 ms/token hypothesis at D2048; OPT-095 shares KV across six query heads |
| 5 | Kernel launch granularity and graph overhead | Low as the main cause: OPT-055 found only ~0.086/0.105 ms idle at D128/D2048 | No graph implementation unless fresh unhidden removable overhead ≥0.50 ms/token; OPT-096 |
| 6 | Q6 attention output/logits, pointwise, copies/commit | Unknown current residual; Q6 logits already use an integer path | Measure separately; no speculative vocabulary reduction or Q6 retuning |

The likely explanation is inefficient batch-1 weight processing, especially
shipping scalar FP32 Q4 dequant/dots, plus smaller mixer/core costs. It is not
established that Quartz is bandwidth-saturated. OPT-090 measures DRAM bytes,
bandwidth, instruction mix, register/spill pressure and occupancy before making
that claim. The 64 FFNs alone stream about 9.63 GB of Q4 blocks per token
(`64*3*17408*5120*144/256`), so reducing arithmetic cannot eliminate their
weight traffic. Use actual tensor bytes for a roofline, not the GGUF file size
including unused embeddings/metadata.

Pinned-source differences, inspected at the authority revision:

- [`vecdotq.cuh`](https://github.com/ggml-org/llama.cpp/blob/cc83d7b4824f73cfdda4dfbb47ee39804f71b328/ggml/src/ggml-cuda/vecdotq.cuh)
  uses packed DP4A and separate scale/min sums in Q4_K MMV, recomputing integer
  activation sums. Quartz's shipping paired Q4 kernel still expands weights
  and accumulates FP32 products; its unselected `late_w4` closes much of this
  mechanism gap. OPT-093 tests the remaining paired-group factorization.
- [`mmvq.cu`](https://github.com/ggml-org/llama.cpp/blob/cc83d7b4824f73cfdda4dfbb47ee39804f71b328/ggml/src/ggml-cuda/mmvq.cu)
  specializes row/warp dispatch and supports gate/up GLU fusion. Quartz already
  pairs gate/up; fusion alone is not a missing feature. Reuse integer fragments
  and final reductions before considering SM120 weight repacks.
- [`gated_delta_net.cu`](https://github.com/ggml-org/llama.cpp/blob/cc83d7b4824f73cfdda4dfbb47ee39804f71b328/ggml/src/ggml-cuda/gated_delta_net.cu)
  keeps state shards in registers and reduces across warp lanes with transposed
  state. A transplant into Quartz row-major FP32 state is not correct without
  conversion and checkpoint/prompt integration. Defer a layout rewrite until
  measured GDN gap justifies that larger task.
- [`fattn-vec.cuh`](https://github.com/ggml-org/llama.cpp/blob/cc83d7b4824f73cfdda4dfbb47ee39804f71b328/ggml/src/ggml-cuda/fattn-vec.cuh)
  provides vector attention, while
  [`ggml-cuda.cu`](https://github.com/ggml-org/llama.cpp/blob/cc83d7b4824f73cfdda4dfbb47ee39804f71b328/ggml/src/ggml-cuda/ggml-cuda.cu)
  has graph capture/replay and fusion recognition. Attribute actual launches;
  do not assume either explains the observed gap without timing.

The ds4 transfer is its inspectable continuation-NLL methodology and focused
quantized kernels (`../ds4/cuda/mmq/mmvq.cu`,
`../ds4/gguf-tools/quality-testing/README.md`), not DeepSeek/GLM fixtures or MoE.
[Flash-Decoding](https://princeton-nlp.github.io/flash-decoding/) motivates
sequence parallelism, which Quartz already has in 16 partitions. The new
attention experiment adds intra-group KV reuse instead of claiming sequence
splitting is absent. Flat-GEMM dispatch ideas in
[FlashDecoding++](https://arxiv.org/abs/2311.01282) are future screen candidates;
no tensor-core batch-1 speedup is assumed, and its approximate softmax is not
part of this batch. Exact disposable weight repacks are plan-compatible but
deferred because they consume the 128K memory reserve and lack a measured need.

For prefill, OPT-086's complete 64-FFN MMQ changed 886.348 → 838.197 ms;
the corresponding engine changed 1414.274 → 1366.312 ms. This establishes that
MMQ remains substantial, not that the remaining 93.71 ms belongs entirely to
MMQ. OPT-020's ~1200.74 ms mixer / 212.58 ms GDN / 274.87 ms attention at 2K
predates many keeps and cannot explain today's P4096 gap. OPT-090 measures
MMQ, attention, GDN, staging, logits and idle separately. OPT-097 is the only
prefill kernel task: split X/Y async-copy completion without changing tiles,
FMA order, prompt batch size or `kv_once`.

## Near-miss parity cases

Source: [OPT-082 report](../evidence/optimization/opt082-kernel-parity/REPORT.md),
[fixture](../fixtures/opt082_kernel_parity.json), and
`cuda/opt082_kernel_parity_test.cu::run_fused_vs_independent`,
`run_q8_mmv`, `reference_bf16`.

| Failure | Diagnosis | Required disposition |
|---|---|---|
| Q8 `r1_w4`, M17 N1 K2048: max_abs 2.30578613, max_rel 0.0509741008, one failing element | Reference uses original BF16; GPU quantizes to half-scale Q8_1. Strong evidence of staging error mixed with implementation error, not proof of acceptable noise | Independently reconstruct exact staged bytes, scale rounding and integer quantization; rerun parity versus staged input. Keep original-BF16 comparison as approximation diagnostic |
| Q8 `r2_w2`, same shape: 2.30580521, 0.0509745255, one failing element | Nearly identical error across layouts supports a shared input/reference issue, not layout-specific reduction as the main cause | Same repair and strict rerun even though candidate remains performance-rejected |
| Q4 fused versus independent: max_abs 15.9912109, max_rel 0.00298343, three failures | Fused output is BF16; comparator leaves host SiLU×up in FP32. Relative error fits BF16 rounding scale, but host/CUDA exp and reduction differences also need separation | Compare independent GPU gate/up + identical CUDA SwiGLU + BF16 RNE output to fused output bitwise. Preserve FP64 math diagnostic separately; any remaining identity failure blocks admission |

The Q8 threshold is `0.05*sqrt(2048)=2.2627417`; the maximum overshoot is
~1.90%. An abs_scale near 0.050952 would cover these particular maxima, and
0.051 would be an approximately 2% increase. That arithmetic is not evidence
for a new contract: only one random seed/shape failed, K=2048 is smaller than
production K5120/6144, and the reference mixes two errors. Do not cherry-pick
K≥2048, M17 or these case IDs. Large-K validation must be expanded at unchanged
tolerances. If corrected staged parity still fails, record a genuine unresolved
kernel failure; diagnose it before a future tolerance proposal.

The r1 revert matters to performance but does not settle parity: r1 is shipping
and has the same miss. OPT-086's explicit exemption and OPT-085's omission of
the fused failure from passing Q4 counts contradict the strict new hierarchy.
OPT-089 creates successor evidence that counts all applicable cases, leaving
historical reports intact. Negative test cases must fail as expected without
being mistaken for failed positive kernel cases.

## Quality gate evolution

These proposals are frozen by this design before implementation measurements.
They do not change `pins/kernel_parity_v1_contract.json` or historical quality
contracts. OPT-091 owns the successor policy and any conditional Q4 concession.

| Proposal | Old threshold / rule | Proposed threshold / rule | Rationale and accepted risk | Validation and rollback |
|---|---|---|---|---|
| Preserve inherited arithmetic failure as non-blocking | Already allowed by OPT-084; older OPT-056 absolute suite requires every task correct | Zero newly failing functional cases; inherited `task_arithmetic` stays visible; successor `regression_release_quality_pass` separate from `absolute_quality_status` | Enables speed work without claiming arithmetic is fixed; known wrong answer remains | Run all eight actual greedy task prompts with full vocab; any new failure or changed inherited answer blocks/reverts |
| Full candidate NLL | Two 1024-target PPL spans, two 128-target recurrence runs, continuations and task regression required | Unchanged; incomplete candidate NLL never passes. Exact same-math candidates may reuse only authenticated output-equivalent evidence | Missing evidence is not measured low risk; OPT-085 never ran candidate NLL | Reject baseline self-comparison, simulated scorer records and selector-mismatched caches; rollback on invalid identity |
| Narrow fast-Q4 PPL budget | Candidate/control PPL ratio ≤1.01 on each span | Only `late_w4`: ≤1.015 on each span and aggregate, versus both authenticated OPT-088 control and frozen OPT-084; all other candidates remain ≤1.01 | At most 0.5 percentage point extra PPL headroom for the already measured large FFN win. This is a proposed risk budget, not evidence that late_w4 has acceptable quality | Require complete measured quality, D128 and D2048 paired throughput lower bounds >1.15, p95 ≤0.90× control at both, no functional/greedy regression. Revert if any bound fails or OPT-098 combination exceeds budget |
| Recurrence bound | Incremental NLL ≤0.02 | Unchanged ≤0.02, including late_w4 | No measured quality drift supports more recurrent error; accumulation risk persists to long context | Candidate-minus-baseline long/short difference-of-differences ≤0.02, finite state, replay/checkpoint consistency; any failure blocks/reverts |
| Greedy continuation changes | No new mismatch versus baseline, except a separately justified stored near-tie | Unchanged; no new automatic near-tie waiver in this batch | There is no candidate logit-margin evidence supporting a new threshold | Check all four existing 16-token continuation fixtures; fail new mismatch. Record top-2 logits at first divergence for future analysis |

PPL means `exp(mean_NLL_candidate - mean_NLL_baseline)`, not a ratio of NLLs.
The relaxed ceiling is ~0.014889 nats, versus ~0.009950 at 1.01.
The total combined drift budget is measured against frozen anchors, never
ratcheted after each keep. OPT-091 cannot mark `model_quality_pass=true` under
the old contract; it emits `strict_model_quality_pass` and
`successor_model_quality_pass` with the active contract ID. The required
`model_quality_pass` field always names which contract supplies its value.

## Candidate promotion review

| Candidate | Evidence | Proposed decision |
|---|---|---|
| `late_w4` = `integer_q8_late` / `paired_integer`, four warps, FP32-scale Q8Block | OPT-085 parity association 6/6; candidate NLL incomplete; rotating FFN 16.383 → 9.800 ms, short D2048+32 868.295 → 639.749 ms | Highest priority: correct same-math/reference coverage, full candidate quality, fresh D128/D2048+p95; strict keep first, OPT-091 concession only if needed |
| `integer_q8_paired` | Historical 16.433 → 11.773 ms, candidate NLL incomplete; OPT-085 did not promote it | Keep available as diagnostic/reference; no second full-quality run in OPT-089. Dominated on current timing; failure of late does not establish integer's quality |
| Q8 `r2_w2` | OPT-086 7.504 vs r1 7.453 ms; CI for r1−r2 −0.0824 to −0.0203 ms; engine 869.183 vs 867.392 ms | Stay rejected as an unchanged layout; only parity repair, no new performance sweep |
| MMQ `fma_async_x` | 838.197 vs `fma_async` 886.348 ms; CI 46.6675–49.6356 ms | Retain; OPT-097 explores one new wait schedule at fixed 128×128 |
| Half-scale Q4 Q8_1 | Never screened; sum_q/sum_x and rounded sum semantics are distinct | No promotion or grid; exact integer sums and FP32 Q8Block retained |
| OPT-077 `tile32` | 12.516 → 11.894 ms, CI −0.1643 to 1.4094; short engine 878.416 → 866.012 ms | One performance-only replication only after OPT-090 demonstrates a repaired measurement defect; otherwise closed |
| OPT-078 `prepared_q` / `prepared_q_veckv` | Acceptance prepared_q 9.092 → 10.315 ms, CI −1.970 to −0.476 | Stay rejected. OPT-095 is a new no-extra-prep-launch KV-sharing mechanism |
| OPT-087 leftovers | No unowned eligible leftover; `kv_once` already kept | No blanket reopening; respect individual negative measurements |

## What not to do

- Do not revive OPT-074 llama-GPU-versus-FP64 admission, or count missing
  candidate evidence as quality pass. Do not waive failures based on proximity.
- Do not change global NVCC flags, FMA policy or fast-math. OPT-068 passed
  numerics and lost performance; no tolerance change rescues it.
- Keep OPT-024 mixer D2R, OPT-027 persistent fattn, OPT-028 MMQ stream-K,
  OPT-029 prompt GDN fusion, OPT-030 PDL, OPT-037/065 tile sweeps,
  OPT-054 microbatch sweeps and OPT-067 prompt-pair closed unchanged.
- Do not interpret OPT-077's wide interval as an accepted gain, or OPT-078's
  screening hint as stronger than its negative acceptance interval.
- Do not assume graph capture removes kernel arithmetic or bandwidth time.
  OPT-055/087 stay closed unless OPT-090 meets the explicit overhead trigger.
- No speculative decoding, batching, weight requantization, FP16 recurrence,
  KV quantization, sparse/MoE techniques, truncated logits or softmax changes.
  These are outside the current [plan](../plan.md).
- Do not change the GPU power/clock policy, use concurrent GPU jobs, run
  expensive GPU suites in normal pytest, or silently rewrite the active ledger,
  `plan.md`, old pins, fixtures or reports during this documentation task.

## Dependency order

```text
OPT-088 (done)
  |
  v
OPT-089 late_w4 admission + necessary parity/reference repairs
  |
  v
OPT-090 matched attribution
  |
  v
OPT-091 successor quality / conditional late_w4 concession
  |
  v
OPT-092 grouped Q8 projections
  |
  v
OPT-093 factored Q4 dots
  |
  v
OPT-094 conditional GDN replication
  |
  v
OPT-095 grouped-KV decode attention
  |
  v
OPT-096 conditional decode graph segments
  |
  v
OPT-097 prefill async-copy schedule
  |
  v
OPT-098 combined quality/performance outcome
```

First eligible task by row order: **OPT-089**. The serial production decision
chain prevents timing against a changing control and guarantees decode work
precedes prefill. Each dependency means the predecessor has a recorded final
keep/reject/no-go decision, not that its candidate must have won. An unresolved
hard invariant blocks dependent production promotion; a rejected optimization
does not. Optional counters and host preparation do not require another agent.

## Common implementation acceptance

Reuse [POST-080](PERFORMANCE-RECOVERY-POST-080.md),
[POST-069](PERFORMANCE-RECOVERY-POST-069.md) and
[TASK-TESTING-STRATEGY](TASK-TESTING-STRATEGY.md), with the explicit additions below.
All dossier commands are future interfaces to implement, not claims that the
new scripts or selectors exist today. Each task creates its named focused
contract, `pins/optNNN_iteration_contract.json`,
`tools/optNNN_<slug>.py`, `tests/test_optNNN_<slug>.py`,
`fixtures/optNNN_<slug>.json`, and
`evidence/optimization/optNNN-<slug-with-hyphens>/REPORT.md` plus raw sidecars.
The dossier's named native target is added to Makefile once; dependencies
compile incrementally. Reuse production diagnostics rather than duplicate engines.

1. Ordinary `uv run pytest` is read-only, GPU-free and network-free. Tests
   validate native evidence, negative cases, gate logic and provenance. No
   synthetic scorer output is admissible as real model-quality evidence.
2. Feedback phases include incremental compilation, all child work and teardown
   under one 300 s parent deadline. Cold Docker/model setup is explicit and
   separate. Missing/invalid `QW38_CUDA_TEST_TIER` fails before GPU work.
   Smoke is model-free, at most 1+1 per probe; correctness has at most three
   repetitions; screen is 1+3, at most three named shape groups and three configs.
   Timeout records `budget_exhausted`; no automatic repeat or false numerical fail.
3. Gate order: structural → OPT-082 corrected same-quantization parity → exact
   same-math → OPT-084 candidate quality → OPT-071 component/E2E → release.
   Short performance screens may rank survivors before the costly full-quality
   run, but cannot admit or promote them. All positive cases must pass;
   expected negative tests are reported separately. Zero nonfinites is mandatory.
4. Quantized references decode weights independently over full K. Small cases
   are complete; production-K references use 16 output rows ×4 captured vectors
   per named family, including first/last/tail plus seed `89` samples. Staging
   checks use exact bytes and correctly typed scales/sums. Q8 and Q6 staged
   references do not substitute original BF16; original-input error stays visible.
5. Complete rotating components cover all 64 FFNs, all 48 GDN plus 16 attention
   mixer groups, or the specified 48/16 cores. Restore outside timing, include
   all staging/epilogues/required copies inside, do not synchronize per layer.
   Record actual distinct weight bytes >2× queried L2 where applicable.
   Use layer/token-specific OPT-071 captures, never layer-0 inputs for every layer.
6. Component acceptance: 3 warmups +10 independently restored AB/BA paired
   rounds, fixed before results; positive two-sided 95% Student-t CI for
   control−candidate and ≥0.10 ms/token or ≥5 ms/P4096 mean saving. OPT-094
   explicitly uses one fixed 30-pair replication instead. No optional stopping.
7. Every decode keep needs five uninstrumented AB/BA pairs **at each** of D128
   and D2048, 32 predetermined output tokens per run, with ITL p50/p95. Require
   one-sided 95% upper bound on paired elapsed-time regression ≤2% of control,
   throughput point ratio ≥0.98 and observed p95 ratio ≤1.02 at both. Short p95
   is diagnostic, not historical release certification. Screen uses one D2048+32
   pair. Run one P4096 control/candidate guard (throughput ratio ≥0.95) on the
   final survivor. Prefill keeps use five P4096 pairs and one pair at each decode
   prefix with the historical 95% throughput /105% p95 cross-workload guard.
8. Quality set Q: two 1024-target spans (`wikitext_nll`,
   `held_out_wikitext_1024`), `recurrence_short` (256+128) and `recurrence_long`
   (4096+128), all eight functional tasks from
   `pins/production_quality_v2_inputs.json`, four 16-token continuation fixtures
   from `fixtures/quality_inputs.json`, and actual full-248320-vocabulary finite
   checks. Preserve exact token construction/masking. Tiny OPT-083 synthetic
   framework cases are unit tests, not substitutes. Q runs once per surviving
   arithmetic configuration, with current production selectors applied before
   graph capture. Use the first 32 held-out targets only as feedback alarm.
9. Full Q may exceed 300 s. Implement `--mode release --phase quality` as an
   explicit long quality-only invocation (deadline 7200 s, no timed P/D oracle).
   Per-case source/flags/selectors/GGUF/token/state hashes permit exact cache
   reuse and resumption. Changed identities invalidate candidate results.
   Current-control outputs may be reused only after authenticating their true
   selector provenance; OPT-084's r2 baseline is not an r1 score by relabeling.
   Exact-output candidates may reuse Q only with byte-equivalent logits/state
   on Q inputs; otherwise score them fully. No full quality run per edit.
10. Timings never reuse a different sitting. Record raw AB/BA order, restoration,
    actual launches, separate graph captures, compiler/toolkit/driver, clocks,
    400 W cap readback, temperature and sources. Profile separately; final
    throughput is uninstrumented. No GPU concurrency. State/graph changes
    additionally test cancellation, checkpoint, prompt/decode transitions and
    128K memory reserve (≥1.5 GiB free after graphs).
11. Reports carry independent `kernel_parity_pass`, `model_quality_pass`,
    `performance_pass`, `production_kept`, plus `quality_contract_id`,
    `evidence_complete`, `absolute_quality_status` and regression status.
    Missing evidence is false with reason `incomplete`, never an inferred pass.
    A no-go/reject is a valid completed task with an explicit reason and control
    retained. Dossier `claims_throughput: true` denotes an intended timed task;
    emitted result is true only for fresh uninstrumented throughput evidence,
    and improvement is claimed only for an admitted measured keep. Screens,
    policy-only operations and reused historical timings emit false.
12. OPT-098 runs the unchanged historical 3-warmup/30-sample P4096 and
    256-token D128/D2048 release protocols, actual p95, quality/state/128K and
    2K checks once for the combined configuration. Historical OPT-056/016 gates
    retain their owners and thresholds. Negative results remain in evidence.

## Proposed ledger rows

| ID | Description | Dependencies | Status | Acceptance condition |
|---|---|---|---|---|
| OPT-089 | Complete strict admission of late_w4 with corrected parity references and real candidate quality | OPT-088 | pending | All positive parity/same-math cases accounted for; full Q and D128/D2048+p95 yield keep or explicit rejection |
| OPT-090 | Measure matched post-Q4 decode sinks and secondary P4096 attribution | OPT-089 | pending | Correct call counts, no dropped events, ≤5% unexplained wall; auditable family ranking and conditional-task triggers |
| OPT-091 | Establish successor quality gate and decide narrowly bounded late_w4 concession | OPT-090 | pending | Versioned strict/successor verdicts; full Q required; concession only with ≤1.015 PPL and strong two-prefix decode benefit |
| OPT-092 | Group same-input Q8 decode projections into one launch per layer group | OPT-091 | pending | Exact staging/output parity and full mixer plus two-prefix E2E acceptance, or retain r1_w4 separate launches |
| OPT-093 | Factor paired-group Q4 integer scale/min work | OPT-092 | pending | Strict parity, same-math internal reconstruction, full Q and complete FFN/two-prefix keep or reject |
| OPT-094 | Conditionally replicate OPT-077 tile32 on demonstrably repaired timing | OPT-093 | pending | Concrete measurement repair required; fixed 30-pair test and full quality/state/two-prefix gate, or no-reopen |
| OPT-095 | Share decode KV loads across six query heads without a prep launch | OPT-094 | pending | Bitwise per-head output and complete 16-layer/two-prefix acceptance, or retain warp_query |
| OPT-096 | Conditionally capture eight-layer decode segments | OPT-095 | pending | Fresh ≥0.50 ms removable overhead trigger, exact graph/state behavior and two-prefix gain, or no-reopen |
| OPT-097 | Separate X/Y async completion in fixed-tile prompt MMQ | OPT-096 | pending | Bitwise output and full FFN/P4096 win with decode guards, or retain fma_async_x schedule |
| OPT-098 | Freeze and measure the combined post-088 production outcome | OPT-097 | pending | Complete quality, P/D/p95/2K/state/memory evidence; independent internal/parity/+5% outcomes, including failures |

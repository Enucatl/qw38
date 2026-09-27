# TASK-043 prefill-engine validation and delivery decision

**RETAIN_CONTROL.** Policy 1030 remains a development candidate. Production
stays TASK-026/027 at `d2f02e2`: the unchanged confirmed P100 failure prevents
promotion. All nine llama.cpp/candidate latency ratios remain below 1, so the
fast-engine goal is unmet. TASK-030/036/040 decisions remain historical.

## Frozen implementation and numerical contract

Inference is the TASK-042 runtime at `9648e69`, unchanged at task admission
`5841789`. TASK-043 extends the replay driver and fixes J128 kernel attribution;
it changes no inference arithmetic or dispatch. The artifact is
`.cache/candidates/candidate-fp8-mixer-q8-mlp-v1.qw38`, 21,013,851,814 bytes,
policy `Fp8MixerQ8MlpV1 = 0x0406` (1030), manifest metadata SHA-256
`6ebcc402487aa92d4a6bb7d64ccfff00c20d74a738ac7e3fa2166522f4b35ff5`.
The compiler identity is
`qw38-fp8-mixer-q8-mlp-v1-cutlass-098de2a6-llama-e6ab7c1a4-k32-bf16-rne-absmax127-f32-affine`.
No checkpoint or artifact payload/scale digests are computed.

All 64 MLPs share one resident Q4_K weight view between phases. FP32 RMS and
SwiGLU producers round locally to BF16 RNE, then pack each row's K32 group
with FP32 absmax/127, signed INT8 RNE/clamp [-127,127], and exact INT32 sums.
Zero groups use scale 1 and zero codes. For weights `d*s*q-dmin*m`, each
K32 contribution is `a*(d*s*sum(q*z)-dmin*m*sum(z))`, accumulated in FP32 in
K32 order. Integer partials are exact and bounded; reconstructed weights are
not individually BF16-rounded. Gate/up share one producer pack; down receives
a fresh SwiGLU pack. Packs, scales, sums and output slabs remain live through
the last consumer on the session stream, then bounded scratch is reused.

M=1 retains DP4A MMVQ; 1<M<128 uses J32 MMQ; M>=128 uses I128/J128 integer
MMA with K128 activation halves, cooperative FP32 metadata staging and fragment
reuse. The aligned J128 specialization requires N and M divisible by 128 and
unpadded K; remaining J128 shapes use the masked specialization. There is no global MLP weight expansion or second resident weight view.
The 208 large attention/GDN projections retain E4M3 weights with FP32 N128/K128
scales and shared K128 FP8 prefill operands. M=1 retains the BF16-input FP8-weight
GEMV. Q8 head, BF16 embeddings/small parameters, FP32 residuals/nonlinearities/
reductions/recurrence and BF16 KV/history remain unchanged.

Prefill executes chunks up to 512. Full M512 attention pairs two sibling query
heads per CTA, each owning Q32 queries and independent softmax/PV accumulators,
sharing K64/V64 staging. Smaller chunks use the per-head Q32/K64 path. Decode
shares KV among all six query heads per KV head, with deterministic partitions
`min(ceil(populated/256),ceil(2*SM_count/4),128)` and fixed-order merge. Both
MMA paths retain `P_hi=BF16_RNE(P)` and
`P_lo=BF16_RNE(P-float(P_hi))`, accumulating both PV products in FP32.
No persistent duplicate KV or global attention matrix is introduced.

GDN recurrence intervals remain 64. Token/chunk host metadata commits only
after successful completion; a failure after device mutation poisons the
session and requires reset/restore. Public standalone calls retain completion
guarantees. Session-owned decode graphs remain default, in capacity-capped
power-of-two buckets starting at 256, with one retained executable. First
capture does not execute a preparatory token. Final FP32 readback remains
outside capture. Construction failure may fall back before mutation; execution
failure never retries. Validation/performance drivers reject graph fallback.
Diagnostic eager mode remains `QW38_DECODE_SUBMISSION=eager` for the benchmark.

The agreed scheduling boundary is full-prompt <=4096 versus >4096 for prefill,
and populated context <=4096 versus >4096 for decode; exactly 4096 is small.
Distinct paths by that boundary remain follow-up work. This candidate dispatches
by actual chunk/operand shape. Startup tradeoffs are reported separately;
inference first-use costs are not relabeled as loading.

## Quality and replay

Core-54 completed once: 15 P100, 15 C92, 12 L12 and 12 retrieval cases.
Teacher-target NLL delta is +0.00310205934703436 nats/token over 701 targets;
the +0.03 aggregate and every +0.06 slice gate pass. C92 is 8/15 in both
arms: one comparator-only and one candidate-only success, point loss zero,
paired loss interval [-0.20,0.20]. The unchanged scorer reports PASS for the
aggregate and every slice. This fixes the observed sample's prior 7/15 result;
it does not establish population non-inferiority from that broad interval.
L12 is 12/12; R-512 and R-4096 are each 6/6. The sole extended quality case,
`R-32768-s0-d0.1`, is 1/1 in both arms (32,757 actual prompt tokens).
Other 32K cases, full-216 and whole-model BF16 evaluation were not run.

All fifteen P100 prompts and both complete output texts exactly match their
TASK-040 reviewed counterparts. All thirty per-arm reviews retain original
reviewer attribution after prompt/text authentication. No new adjudications
were needed. Candidate `case_077` still incorrectly adds mapping metadata
pages to mapped-file residency; its confirmed failure remains. Thus P100 and
the final quality gate are FAIL. Capped outputs remain capped; existing
reviews do not represent code execution or a human review where attributed to AI.

Release full-model integration passes 1/1 (16.82 s), including same-schedule
replay, nonempty handoff, selected-row delivery/failure, late-failure recovery,
reset/restore, interleave, movement, graph/eager bitwise equality and bucket
boundaries at 255/256/257 and 511/512/513. The default replay retains its nine
checkpoints. Prefill replays pass all twelve checkpoints
1/3/4/63/64/65/255/256/257/511/512/513 for partitions
1/63/64/65/255/256/512/alternating 63,65. Reports record concrete chunk sizes
and head ownership: partition 512 at length 513 selects paired M512 then a
one-token per-head tail. All reset and snapshot continuations are bitwise;
cross-schedule full-model deltas are diagnostics, not component-tolerance gates.
Authenticated unchanged TASK-041/042 numerical checks remain applicable and
were not rerun as a second witness.

## Single-run performance, resources and capacity

Six fresh Nsight-instrumented executions, zero warmups/repetitions. Requests
generate 128 tokens (127 tail calls); populated decode consumes 128 fixed
inputs. Request prompt phases provide prefill/TTFT. First library/graph use
and bucket transitions stay inside their actual boundaries. Times are observed
single runs, not statistical performance estimates. Ratios below are
llama.cpp/candidate; all nine targets require >=1.

| Row | TASK-043 ms | TASK-040 ms | llama.cpp ms | Latency ratio | Allocation peak ratio | Resident growth ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prefill-256 | 164.756 | 236.424 | 109.548 | 0.664910 | 0.884462 | 0.867982 |
| prefill-4096 | 1618.472 | 3228.749 | 1379.908 | 0.852600 | 0.891857 | 0.875389 |
| prefill-32768 | 15818.350 | 29421.679 | 12372.765 | 0.782178 | 0.900048 | 0.884674 |
| decode-512 | 2509.360 | 2513.814 | 1932.101 | 0.769958 | 0.890544 | 0.873912 |
| decode-4096 | 2520.961 | 2519.523 | 1952.946 | 0.774683 | 0.891857 | 0.875389 |
| decode-32768 | 2693.965 | 2699.681 | 2131.478 | 0.791204 | 0.900048 | 0.884674 |
| request-256 | 2627.028 | 2709.311 | 2027.903 | 0.771938 | 0.884462 | 0.867982 |
| request-4096 | 4102.853 | 5720.271 | 3335.562 | 0.812986 | 0.891857 | 0.875389 |
| request-32768 | 18487.066 | 32094.681 | 14480.757 | 0.783291 | 0.900048 | 0.884674 |

Memory ratios are llama.cpp/candidate. Derived prefill rows reuse their request
memory observations; they are not separately measured prefill peaks.

| Execution | Load/upload ms | Cold setup ms | CUDA init ms | Populated setup ms |
| --- | ---: | ---: | ---: | ---: |
| decode-512 | 5429.519 | 5435.285 | 284.861 | 230.026 |
| decode-4096 | 5437.125 | 5442.501 | 279.723 | 1627.083 |
| decode-32768 | 5458.420 | 5465.437 | 287.182 | 15849.730 |
| request-256 | 5424.880 | 5430.049 | 278.486 | 0.000 |
| request-4096 | 5435.597 | 5441.096 | 280.834 | 0.000 |
| request-32768 | 5451.434 | 5458.540 | 287.764 | 0.000 |

Cold setup includes model load and session binding; CUDA initialization is
reported separately. No OS page-cache eviction occurred. TASK-040's 32K
load/upload was 5551.255 ms versus 5451.434 ms here; this is an observation,
not evidence that TASK-042's first-decode regression was a loading effect.
The latter remains unresolved and belongs to TASK-044.

The final 32K request uses 32,896 slots and ends at position 32,895 because
its first output comes from prompt readout. Populated decode reaches 32,896.
Both record 8,319,336,448 free device bytes, above the 2 GiB reserve. The
whole-process tracked CUDA allocation peak is 24,262,211,090 bytes; separately,
sampled resident growth is 24,767,365,120 bytes. Initial/after-bind/after-run
free samples are 33,086,701,568 / 8,646,492,160 / 8,319,336,448 bytes for the
32K request. Allocation traces include load, graph/library first use and
transients; free-memory samples do not prove a continuous residency peak.
Driver/library backing can bypass project counters. Do not add tracked peaks
and resident growth as separate categories. The independent integration check
also completes first decode and maximum graph upload before sampling reserve.

Device inventory includes 21,013,686,400 model bytes, 2,309,816,320 persistent
FP32 state/BF16 history/KV bytes, 513,802,240 arena bytes, 82,444,292 Q8 MLP
workspace bytes, 254,248,960 prefill-engine workspace bytes and 20,971,520
residual bytes. The prefill workspace includes FP8 packs and contraction/output
slabs, attention and GDN buffers; output/partial slabs, library and graph
resources also appear in the whole-process peak. These listed categories are
not an additive exhaustive allocation inventory. Traces separately record
993,500 pinned host allocation bytes; the artifact's transient file mapping has
21,013,851,814 bytes of virtual extent, not a measured host RSS peak. Host
heap/profiler/driver RSS was not sampled.

The same request traces establish actual resources and dispatch:

| Consumer | Grid at M512 | Threads | Registers/thread | Dynamic shared bytes | Local bytes/thread |
| --- | --- | ---: | ---: | ---: | ---: |
| J128 gate/up | 136 x 4 | 256 | 220 | 65,536 | 0 |
| J128 down | 40 x 4 | 256 | 220 | 65,536 | 0 |
| Paired Q32/K64 attention | 12 x 16 | 256 | 179 | 67,584 | 0 |
| Per-head attention at M256 | 24 x 8 | 128 | 171 | 67,584 | 0 |

The M256 request uses J128 grids 136x2 and 40x2; its attention retains one
head per CTA. M4096 and M32768 execute eight and 64 M512 chunks respectively.
The 32K prompt records 8192 gate/up MMQ launches, 4096 down launches and 1024
paired attention launches, matching 64 MLPs and 16 attention layers per chunk.
J128 reuses weights across 128 tokens instead of J32's 32, while preserving
the FP32 K32 affine arithmetic. These are measured launch/resources plus
source-defined ownership, not an occupancy estimate.

## Remaining costs and next step

32K prefill improves 46.24% from TASK-040's 29.422 s to 15.818 s, versus
llama.cpp's 12.373 s. Same-capture GPU category costs are:

| Prefill category | TASK-043 ms | TASK-040 ms | llama.cpp ms |
| --- | ---: | ---: | ---: |
| Projections | 9667.201 | 22282.159 | 7779.734 |
| Attention | 3764.244 | 4658.320 | 1837.267 |
| GDN | 1824.603 | 1807.895 | 1510.522 |
| Normalization/epilogue/other | 474.795 | 554.493 | 953.276 |
| Separate unpack/pack | 9.213 | 18.764 | 196.487 |
| Head | 1.271 | 1.293 | 0.665 |

J128 MMQ alone is 8018.136 ms, down from TASK-040's 20170.957 ms;
paired attention MMA is 3749.625 ms, GDN recurrence 1614.828 ms and CUTLASS
FP8 contractions 1606.050 ms. These kernels are contained within categories,
not extra costs to add. Fused RMS/SwiGLU Q8 and attention/GDN FP8 packing stays
attributed to its containing kernel category. CUDA API waits overlap GPU work;
they are not added to kernel time. Phase attribution uses recorded host TTFT
relative to the NVTX start, with its existing boundary precision limitations.

The prefill gap is still dominated by projection and attention GPU work.
At decode-4096, projections take 1913.815 ms of the 2520.961 ms measured:
MMVQ 914.006 ms and FP8 GEMV 905.600 ms dominate. Attention is 48.619 ms,
GDN 70.132 ms, head 178.652 ms. This does not justify claiming submission is
the main throughput bottleneck. TASK-044 should first reuse these captures to
attribute startup/first graph use versus recurring decode. A subsequent
bounded scheduling proposal can target measured small/large attention or
projection costs at the agreed 4096 boundary; no dispatcher, tile sweep,
quantization change or new optimization task is implemented here. P100 remains
an independent promotion blocker and cannot be waived by speed gains.

## Reproduction, evidence and rollback

Pinned image:
`sha256:254963cc774290ddeae6ada94047607b5bed9eb668dfb344e58aac05889f2b49`;
CUDA 13.4.1 / nvcc 13.4.59, GCC/libstdc++ 14.2.0, CMake 3.28.3,
CUDA runtime 13.4.49, cuBLAS 13.7.0.27, Nsight Systems 2026.3.0.0, native
SM120. GPU RTX 5090 UUID `GPU-e51ee570-3143-784d-789d-e3054637ad0b`,
driver 590.48.01, 400 W limit. Clocks and temperature are observed, not fixed.
TASK-027 comparator inputs, settings, logs, binary and captures are authenticated
by the preserved checksum list and provenance. Its original image identity is
retained; matching versions do not mean image-content equality. Comparator
workload and timing/instrumentation semantics are unchanged. No comparator
rerun was required. These timings are speed comparisons, not a quality/speed
Pareto claim for a candidate with failed quality.

From this checkout, use the pinned build/compile/load commands in the
[TASK-040 recipe](task040-delivery.md#reproduction-and-rollback), which retains
the same policy/artifact and compiler CLI. The final Release targets
`qw38_compile`, `qw38_decode`, `qw38_evaluate`, `qw38_state_replay`,
`qw38_bench_request` and `qw38_language_model_integration_test` were built.
The diagnostic decode command takes little-endian uint32 token IDs. Production
rollback requires the separate `d2f02e2` checkout and policy-1027 artifact from
[the retained-control recipe](task030-delivery.md#use-the-retained-production-control);
changing the artifact alone does not restore the older runtime.

All exact acceptance commands, logs, source/binary identities, six profiles,
replays and comparison summaries are retained under
`.cache/evaluation/qw38-language-v2/task043-support/`:

- `bash .cache/evaluation/qw38-language-v2/task043-support/run.sh`: pinned
  build, focused profile test (1 passed), integration, one core-54 run, default
  plus eight prefill replays, one fixed long-quality case, six performance
  executions, resource/capacity analysis; exit 0.
- `bash .cache/evaluation/qw38-language-v2/task043-support/score.sh`: score
  saved core and long outputs, authenticate frozen source/binaries; exit 0.
- `uv run python .cache/evaluation/qw38-language-v2/task043-support/prepare-reviews.py`:
  exact prompt and full-text authentication for all thirty reused reviews;
  exit 0, confirmed P100 failure preserved.
- `finalize.sh`: reprocess saved outputs with authenticated reviews and freeze
  the final delivery evidence; its result is recorded in the task completion report.

Core run: `runs/task043-core54-20260927T110132Z-1358772`; long run:
`runs/task043-r32768`, under `.cache/evaluation/qw38-language-v2/`.
The existing `docs/implementation/task040-eval-policy-rebind.json` authenticates
unchanged fixtures/scoring; FAST-03 changes task ownership and schedules, not
those frozen quality rules. Main-thread long commands used `wake-run`; no
valid inference was repeated for scoring. Independent review and commit/push
results are recorded in [TASK-043](tasks/TASK-043.md).

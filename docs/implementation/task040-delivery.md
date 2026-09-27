# TASK-040 quantized-engine validation and delivery decision

**RETAIN_CONTROL. Policy 1030 is not promoted; the fast-engine goal remains
unmet.** C92 is INCONCLUSIVE and P100 has a confirmed candidate-only error.
Production remains TASK-026/027 at `d2f02e2`. All nine matched speed-parity
targets are individually unmet. TASK-030/036 decisions remain historical.
The [task report](tasks/TASK-040.md) records independent review and delivery.

## Frozen runtime and artifact

Inference source is `d094af3db4270baaf5cadcb2fba157bf12d4a18a` (TASK-039).
TASK-040 changes replay reporting, profile attribution and documentation;
it does not change inference arithmetic or dispatch. Artifact
`.cache/candidates/candidate-fp8-mixer-q8-mlp-v1.qw38` is 21,013,851,814 bytes,
policy `Fp8MixerQ8MlpV1 = 0x0406` (1030), manifest metadata digest
`6ebcc402487aa92d4a6bb7d64ccfff00c20d74a738ac7e3fa2166522f4b35ff5`.
Compiler 0.1.2 identifies itself as
`qw38-fp8-mixer-q8-mlp-v1-cutlass-098de2a6-llama-e6ab7c1a4-k32-bf16-rne-absmax127-f32-affine`.
Source metadata identity is
`77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df`;
tokenizer identity is
`0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`.
No checkpoint, artifact tensor payload or scale content digests are used.

All 64 primary-language MLPs retain one Q4_K weight view shared by phases.
FP32 RMS/SwiGLU producers round locally to BF16 RNE, then pack each row's
K32 group to signed INT8 using FP32 absmax/127, RNE and clamp [-127,127];
zero groups use scale 1 and zero codes. Each group stores its exact INT32 sum.
For Q4_K subgroup weights `d*s*q - dmin*m`, the FP32 contribution is
`a * (d*s*sum(q*z) - dmin*m*sum(z))`. Raw K32 integer dots are bounded by
60,960 and sums by 4,064; no per-weight BF16 reconstruction rounding.
M=1 uses DP4A MMVQ and M>1 integer MMA MMQ. Gate/up reuse one pack, and down
consumes a fresh SwiGLU pack. Packs and bounded FP32 slabs live until their
last consumer on the session stream, then are reused; no global MLP weight
expansion or duplicate resident fallback view.

The 208 large attention/GDN matrices retain TASK-035 E4M3 weights, FP32
absmax/448 scales per N128/K128 block, and shared per-row K128 FP8 operands
for compatible prefill consumers. M=1 uses BF16-activation FP8-weight GEMV.
Q8 head, BF16 embeddings/small parameters, FP32 residuals/nonlinearities/
reductions/recurrence, and BF16 KV/convolution history remain unchanged.

Attention uses TASK-038 Q32/K64 pipelined MMA with register-resident FP32 PV,
shared KV across each six-query-head group, and two BF16 probability
components: `P_hi=BF16_RNE(P)`, `P_lo=BF16_RNE(P-float(P_hi))`.
Both products accumulate in FP32. Decode partitions are
`min(ceil(populated/256), ceil(2*SM_count/4), 128)`, covering contiguous
K64 tiles and merging in fixed order. Actual populated length masks launch
capacity; causal absolute positions and QK norm/RoPE remain unchanged.

Chunk size is 256. Full-model layers enqueue on one session stream; successful
completion/readback commits host token/chunk metadata together. Post-mutation
failure poisons the session; explicit reset/restore is required. Standalone
public layer APIs retain synchronous completion. One session/plan-owned decode
graph is default, replaced at capacity-capped power-of-two buckets starting at
256. Controls update before replay; full FP32 logits/status readback remains
outside capture. First capture/instantiation never executes a warmup token.
Construction failure may record an eager fallback before mutation; execution
failure never retries. TASK-040 replay and timing drivers reject graph fallback.
Explicit diagnostic eager selection is
`LanguageModelPlan::set_decode_submission(DecodeSubmission::Eager)` or benchmark
`QW38_DECODE_SUBMISSION=eager`. No prefill graph or generic graph optimizer.

## Quality and state

Core-54 completed 54/54: 15 P100, 15 C92, 12 L12 and 12 retrieval cases.
Teacher-target NLL delta is +0.0025621463600068992 nats/token over 701 targets;
aggregate +0.03 and every slice +0.06 gate pass. C92 is 7/15 versus 8/15,
with paired loss interval [0,0.20] crossing the 0.02 budget. This remains
INCONCLUSIVE, including the SuperGPQA slice, rather than a demonstrated
population regression or non-inferiority result. L12 is 12/12, R-512 and
R-4096 are 6/6 each, and the sole `R-32768-s0-d0.1` extension is 1/1 in
both arms. Other 32K fixtures remain inventory; full-216 was not run.

All 15 P100 outputs have output-bound adjudications. GPT-6 Codex, explicitly
an AI reviewer, read all pairs and reviewed nine changed candidate outputs.
Six candidate and fifteen comparator reviews retain their original
attribution only after exact prompt/text authentication. Unchanged `case_077`
retains its confirmed error adding mapping metadata pages to mapped-file
residency, so P100 is FAIL. No answer-specific tuning or waiver occurred.
Capped outputs remain capped; snippets are not represented as executed code.
The two `case_053` JSON examples were parsed.

Same-schedule replay passes at 1/3/4/63/64/65/255/256/257, with partitions
1/63/64/65/255/256/alternating 63,65. Reports record graph selection, no
fallback, bucket rule and attention schedule. Full-model integration passes
(1/1, 12.50 s), covering graph/eager bitwise equality at 255/256/257 and
511/512/513, capture without mutation, nonempty handoff, reset/restore,
interleaving, movement, pre-enqueue rejection and early/late failure recovery.
Cross-schedule full-model deltas are diagnostics; isolated-component
tolerances are not full-model gates. The 78 retained TASK-039 source/evidence
identities authenticate its focused API, Q8 MLP, submission/allocation and
failure checks; those unchanged focused suites were not repeated.

## Single-run performance and capacity

One Nsight-instrumented execution per workload, zero warmups, no repetition
statistics. Each request produces 128 tokens (127 decode-tail calls);
populated decode consumes 128 fixed inputs. Request prompt phases provide
prefill/TTFT. First library/graph use is included in its actual boundary.
Ratios are llama.cpp/candidate latency; every target requires >=1.

| Row | Candidate ms | llama.cpp ms | Ratio | Target |
| --- | ---: | ---: | ---: | --- |
| prefill-256 | 236.424 | 109.548 | 0.463355 | unmet |
| prefill-4096 | 3228.749 | 1379.908 | 0.427382 | unmet |
| prefill-32768 | 29421.679 | 12372.765 | 0.420532 | unmet |
| decode-512 | 2513.814 | 1932.101 | 0.768593 | unmet |
| decode-4096 | 2519.523 | 1952.946 | 0.775126 | unmet |
| decode-32768 | 2699.681 | 2131.478 | 0.789529 | unmet |
| request-256 | 2709.311 | 2027.903 | 0.748494 | unmet |
| request-4096 | 5720.271 | 3335.562 | 0.583113 | unmet |
| request-32768 | 32094.681 | 14480.757 | 0.451189 | unmet |

At 32K, load/upload was 5551.255 ms; session-inclusive cold setup 5557.993 ms;
separate CUDA initialization 285.943 ms. No OS cache eviction was used.
Populated decode setup was 29512.451 ms, outside its measured 128 steps.

The final request allocated 32,896 slots (32,768+128). It ended at position
32,895 because the first output comes from prompt readout; the 128-step
populated-decode capture reaches 32,896. Both observed 8,740,864,000 free
device bytes, exceeding the 2 GiB reserve. Whole-process tracked allocation
peak is 23,837,487,634 bytes; sampled resident growth is separately
24,345,837,568 bytes. Initial/after-bind/after-run free-memory samples and
whole-process CUDA allocation traces cover load and first capture/library use.

Model codes/scales/padding use 21,013,686,400 device bytes; persistent FP32
state and BF16 history/KV use 2,309,816,320; shared session arena uses
256,901,120. Q8 MLP workspace adds 41,222,148, including codes, FP32 scales,
INT32 sums and gate/up slabs. Combined prefill engine workspace is 83,410,944,
including 14,204,928 for FP8 operands/accumulation and 69,206,016 for its
other buffers. Output/attention partials,
other prefill buffers, cuBLAS workspace and load/capture allocations are
included in the total peak; these partial categories are not an additive
complete allocation inventory. One plan's controls/status use 216+4 device
bytes. Native allocation events separately record 993,500 pinned host staging
bytes, freed on exit. The artifact uses a transient host file mapping during
load; its 21,013,851,814-byte virtual extent is not a measured host RSS peak.
Host heap/profiler/driver RSS was not sampled. GPU graph/library backing may
bypass application counters, which is why sampled residency/free memory is
reported separately. Samples do not prove a continuous residency high-water
mark. Do not add peaks and resident growth as separate allocations.

Memory ratios below are llama.cpp/candidate. Request observations also cover
the derived prefill rows, not separately measured prefill peaks.

| Execution | Allocation peak ratio | Sampled resident growth ratio |
| --- | ---: | ---: |
| decode-512 | 0.907956 | 0.890482 |
| decode-4096 | 0.909107 | 0.891814 |
| decode-32768 | 0.916084 | 0.899991 |
| request-256 | 0.901767 | 0.884452 |
| request-4096 | 0.909107 | 0.891814 |
| request-32768 | 0.916084 | 0.899991 |

TASK-027 comparator captures retain their original image/source/settings and
match authenticated frozen inputs, GPU UUID, driver, 400 W power limit,
versions and instrumentation/timing boundaries. The old local candidate image
ID is unavailable; TASK-040 pins rebuilt image
`sha256:254963cc774290ddeae6ada94047607b5bed9eb668dfb344e58aac05889f2b49`.
Versions match: nvcc 13.4.59, GCC/libstdc++ 14.2.0, CMake 3.28.3, CUDA runtime
13.4.49, cuBLAS 13.7.0.27 and Nsight Systems 2026.3.0.0. This is version
matching, not equality of image content. Clocks/temperature are observed,
not controlled. This remains a speed-only comparison, not a quality/speed
Pareto claim. Claims cover one RTX 5090, one primary-language sequence and
32K prompt plus 128 continuation slots; no larger-context search, vision,
MTP, batching or sampling acceptance is claimed.

Measured remaining GPU costs rank as follows: 32K prefill Q4_K×Q8 MMQ
20.171 s, attention 4.658 s, FP8 contractions about 2.044 s, then GDN
1.808 s. Decode-4096 MMVQ takes 0.913 s and FP8 GEMV 0.904 s of 2.520 s
elapsed; head adds 0.178 s. Fused RMS/packing costs stay in normalization and
SwiGLU/packing in epilogues; their complete cost is present in host timings.
The repaired `mmq` profile label changes attribution only. CPU waits overlap
GPU work and are not additional elapsed cost.

**FOLLOW_UP_REQUIRED:** resolve C92 uncertainty and the P100 failure before
promotion. The smallest justified performance step is an isolated real-shape
optimization of the current Q4_K×Q8 prefill MMQ consumer, measuring the whole
MLP including packs/epilogues against this frozen candidate. The deferred
TASK-037 M=1 regression is still material: MMVQ accounts for about 37% of
current decode-4096 GPU time, alongside FP8 GEMV. TASK-038's earlier short
prefill regression remains historical; this run does not identify its cause.
No new batch or task is created or launched.

## Reproduction and rollback

Use the repository's pinned Dockerfile/tool versions and native sm_120 Release
build, with CUTLASS `098de2a652cf8f00fd70b2df54051c7eccbb855a` already present.
From this delivered checkout, with the authenticated local checkpoint/tokenizer:

```bash
docker run --rm --gpus all -u "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  sha256:254963cc774290ddeae6ada94047607b5bed9eb668dfb344e58aac05889f2b49 \
  bash -lc 'cmake -S . -B build/pinned-release -DCMAKE_BUILD_TYPE=Release &&
    cmake --build build/pinned-release --target qw38_compile qw38_decode qw38_evaluate qw38_state_replay qw38_bench_request -j4'
# Only if the frozen artifact is absent:
docker run --rm --gpus all -u "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  sha256:254963cc774290ddeae6ada94047607b5bed9eb668dfb344e58aac05889f2b49 \
  build/pinned-release/src/qw38-compile \
  --checkpoint .cache/authorities/qwen3.8-27b-transformers \
  --output .cache/candidates/candidate-fp8-mixer-q8-mlp-v1.qw38 \
  --format fp8-mixer-q8-mlp
docker run --rm --gpus all -u "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  sha256:254963cc774290ddeae6ada94047607b5bed9eb668dfb344e58aac05889f2b49 \
  build/pinned-release/src/qw38-decode \
  --artifact .cache/candidates/candidate-fp8-mixer-q8-mlp-v1.qw38 \
  --tokens-file .cache/evaluation/qw38-language-v2/tokens/case_000.prompt.u32le \
  --generate 8
```

This is a diagnostic candidate command. The CLI takes little-endian uint32
token IDs and a fixed greedy cap. Production/rollback uses the separate
`d2f02e2` checkout and policy-1027 Q4_K/Q8 artifact from the
[retained-control recipe](task030-delivery.md#use-the-retained-production-control).
Changing only the artifact does not roll back the attention/submission runtime.

Exact commands, raw logs, immutable identities, review records, six traces,
replays and summaries are in
`.cache/evaluation/qw38-language-v2/task040-support/`. `validation.sh` contains
the build/core/replay/long commands; `measure.sh` the six performance captures;
`score.sh` initial scoring/profile repair; `finalize.sh` reviewed rescoring
and final identity checks. The initial unavailable-image error occurred before
build/inference. Initial scoring succeeded, then its final checksum check
caught the deliberately changed profile test; the original list and failed
log are preserved. Neither issue caused valid inference to be rerun.

Evaluator SHA-256:
`03f1dded3b8e116b4941641fc6e90fdf7f229bdc4e49045a28db3077c05f2b9d`;
state replay:
`c25638892b6e3ea03ee44c15f0cc853217bd3d7002ac38474cc05656552355f1`;
request benchmark:
`d249bbc094c997c5fb3b772e368dca582e40226bb7698e5b06df97fc1cab1bb8`.

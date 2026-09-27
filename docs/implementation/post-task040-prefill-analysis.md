# Prefill gap after TASK-040 — 2026-09-27

The remaining prefill gap is primarily an inefficient compact MLP matrix
consumer, followed by attention reuse. Adding a decode/prefill selector alone
will not fix it: QW38 already selects different matrix and vector consumers.
The next sequence is [TASK-041](tasks/TASK-041.md) →
[TASK-042](tasks/TASK-042.md) → [TASK-043](tasks/TASK-043.md).
All are planning work, initially TODO.

## Evidence and limits

Inspected QW38 `e58c31c` (TASK-040); inference arithmetic is TASK-039
`d094af3db4270baaf5cadcb2fba157bf12d4a18a`, policy 1030. Inspected local DS4
`c238077a87186381bf626cc531bccffe1fef79e7` and the comparator's pinned
llama.cpp `e6ab7c1a41054a888ada952eab4c886444c2f5ad`, not the different local
llama.cpp HEAD. The [delivery report](task040-delivery.md) and these retained
files supply the measurements:

- `.cache/evaluation/qw38-language-v2/task040-support/{summary,profiles}.json`
- `.cache/evaluation/qw38-language-v2/task040-support/request-32768-qw38.sqlite`
- `.cache/evaluation/qw38-language-v2/task027-support/single-run/profiles.json`
- `.cache/evaluation/qw38-language-v2/task027-support/single-run/request-32768-llama.sqlite`
- The matching request JSONL files and TASK-040 `comparator-reuse.md`.

This investigation only read source, summaries and SQLite traces. No new
inference, benchmark, build, artifact or quality evaluation was run. Observations
remain single instrumented executions, with first use included and no warmups
or repetition statistics. Source mechanisms below are not measured stall or
DRAM-bandwidth attribution. Different policies, kernels and chunk sizes make
component comparisons diagnostic rather than controlled kernel speedups.
Production remains TASK-026/027 at `d2f02e2`; TASK-040 retained control because
P100 failed and C92 was inconclusive. Faster prefill does not resolve those gates.

## Where the time goes

| Prompt tokens | QW38 prefill ms | llama.cpp prefill ms | QW38 / llama.cpp |
| ---: | ---: | ---: | ---: |
| 256 | 236.424 | 109.548 | 2.16× |
| 4,096 | 3,228.749 | 1,379.908 | 2.34× |
| 32,768 | 29,421.679 | 12,372.765 | 2.38× |

Populated decode is now about 1.27–1.30× comparator latency. Prefill's gap
is not just long-context attention: it already exists at 256 tokens.

| 32K prefill GPU category | QW38 ms | llama.cpp ms | Difference ms |
| --- | ---: | ---: | ---: |
| Projections plus separately labeled packing/unpack | 22,300.923 | 7,976.221 | +14,324.702 |
| Attention | 4,658.320 | 1,837.267 | +2,821.053 |
| GDN | 1,807.895 | 1,510.522 | +297.373 |
| Normalization, epilogues and other | 554.493 | 953.276 | −398.783 |
| Final head | 1.293 | 0.665 | +0.628 |

Within candidate projections, **Q4_K×Q8 MMQ alone is 20,170.957 ms**, about
68.6% of host prefill time; FP8 contractions are 2,044.175 ms. Attention is
another 15.8%. Fused RMS/Q8/FP8 packing and SwiGLU packing are charged to
normalization/epilogues, so the 18.764 ms standalone unpack category must not
be presented as all conversion cost.

Read-only SQLite extraction, restricted to `perf01_measured.start` through
that start plus recorded TTFT, confirms the actual launches:

| Candidate prefill kernel | Calls | GPU ms | Grid per call | Registers/thread | Shared bytes/block |
| --- | ---: | ---: | --- | ---: | ---: |
| MMQ gate/up | 16,384 | 13,369.536 | 136 × 8 | 140 | 43,520 |
| MMQ down | 8,192 | 6,801.421 | 40 × 8 | 140 | 43,520 |
| Attention MMA | 2,048 | 4,641.455 | 24 × 8 | 171 | 67,584 |
| GDN prefill recurrence | 24,576 | 1,575.435 | 1,536 | 40 | 1,024 |

These records report zero local-memory bytes; they do not prove good occupancy
or instruction throughput. The measured comparator's Q4_K kernel is
`mul_mat_q<(ggml_type)12,128,false>`: 12,288 calls, 5,243.611 ms, plus
102.282 ms of Q4_K stream-K fixups. Its register count is 251, so simply
minimizing register use is not the objective. The template's 128 is the token
tile width. Comparator Q8_0 and Q6_K contractions also contribute to its total;
do not label all comparator projection time as Q4_K MLP time.

Candidate 32K prefill has 204,930 kernels totaling 29,322.924 ms against
29,421.679 ms host elapsed. The roughly 98.755 ms remainder is not a measured
CPU-only cost. At 256 and 4,096 tokens the corresponding remainders are
51.016 and 55.447 ms. Initialization/submission can matter for short prompts,
but cannot explain a 17,048.914 ms long-prefill deficit. CPU CUDA launch/wait
durations overlap GPU work and must not be added to it.

A useful limit: if everything except MMQ stayed fixed, matching 12.373 s
would require MMQ to fall from 20.171 s to about 3.122 s, a 6.46× improvement.
That is not a forecast. It explains why a plausible MMQ improvement and the
separate 2.821 s attention gap both belong in the next batch.

## DS4: what transfers, and what does not

DS4 does have phase/shape selection. In
[ds4_cuda.cu](../../../ds4/ds4_cuda.cu), lines 14461–14511, dense Q8 selects
an aligned matrix-vector path for one token, aligned D2R GEMM at ≥512 tokens
with additional shape guards, raw Q8_0 MMQ for eligible multi-token inputs,
then bounded/cached dense fallbacks. The aligned large-batch branch also
requires input width ≤4096; it is not directly eligible for QW38's 5120-wide
hidden projections. Its artifact builder at lines 4349–4364 only enables
that aligned preparation on integrated CUDA devices. This GB10-oriented
path is not evidence of the normal RTX 5090 dispatch. Quality mode and
multi-GPU restrictions also affect MMQ admission (lines 1097–1105).

The useful pattern is a compact resident representation with consumers chosen
for token count and real shape. The aligned D2R implementation uses large
output tiles and staged asynchronous loads; raw MMQ directly consumes compact
weights. See [ds4_mmq_d2r.cu](../../../ds4/cuda/mmq/ds4_mmq_d2r.cu), lines
3061–3256. An asynchronous copy is useful only with valid stage ownership and
wait/barrier placement; NVIDIA documents those ordering requirements in the
[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group-cp-async-wait-all).
It is not a reason to add a pipeline before checking resource costs.

Separate operators do **not** imply duplicate full-model weights. DS4's
aligned expert paths share one replacement artifact between phases
(`ds4_cuda.cu`, lines 23592–23653). Its dense aligned Q8 artifacts, however,
remain additive because some shapes need raw fallbacks; the duplication is
explicit in [ds4_mmq.h](../../../ds4/cuda/mmq/ds4_mmq.h), lines 559–563.
QW38 should retain its single resident Q4_K view first.

DS4 is not a completely conversion-free reference. Its raw dense MMQ wrapper
still clears scratch and quantizes input per call. Paired expert gate/up
really shares a pack, and aligned expert SwiGLU can emit the downstream Q8
operand directly; these are implemented reuse examples. But the general
producer-fold registry is a stub returning zero (`ds4_cuda.cu`, lines
1087–1094); `_preq` APIs have no production callers in the inspected tree.
See [ds4_mmq.cu](../../../ds4/cuda/mmq/ds4_mmq.cu), lines 510–557,
651–785, 1404–1455 and 1653–1662. Copy the applicable calculation schedule,
not unused APIs, MoE routing machinery or hardware-specific thresholds.

## MLP: the major remaining repeated work is inside MMQ

The current chain already has the desired inter-kernel reuse:

FP32 residual → fused RMS, BF16 rounding and K32 Q8 pack → gate and up
consume the same codes/scales/sums → FP32 gate/up slabs → fused SwiGLU,
BF16 rounding and fresh Q8 pack → down with fused FP32 residual addition.

See [q4k_q8.cu](../../cuda/q4k_q8.cu), lines 206–253. The down pack must be
fresh because SwiGLU changes the values. Packs and slabs use bounded reused
session workspace. No full BF16 MLP weight expansion remains on policy 1030.
Plans to “share gate/up packing” or “remove global MLP unpack” would redo
completed work.

The deficient schedule is at lines 85–139:

- Fixed **128 output rows × 32 token rows** for every M>1. At M256, eight
  CTAs independently load and unpack a given output/K tile. A J128 schedule
  needs two such token tiles: four times fewer staging/unpack instances.
  This is source-level repeated work, not a fourfold HBM-traffic measurement.
- Weight nibbles become INT8 in shared memory once per tile, but weight
  scale/minimum metadata are decoded through `affine()` inside every K32
  step. Four lanes address each unique output row/subgroup, and that work
  repeats in each token tile. Stage decoded FP32 metadata once and reuse it.
  Activation scales and exact sums likewise deserve cooperative loading and
  register reuse across output fragments, not a different activation recipe.
- Each K32 integer MMA is followed by INT32→FP32 conversion and scale/affine
  correction. Some cost is intrinsic to the chosen per-K32 scaling. Moving
  metadata work out of the inner loop is safe; removing necessary subgroup
  scale application is not.
- The kernel has serial load/compute phases and two barriers per K256.
  **Barrier count alone is not the explanation:** the pinned faster upstream
  schedule uses four barriers, staging K256 weights but only K128 activation
  halves. Its cooperative metadata layout, register prereads and larger
  token reuse are the better first implementation.

The pinned upstream references are
[mmq-config-ampere.cuh](https://github.com/ggml-org/llama.cpp/blob/e6ab7c1a41054a888ada952eab4c886444c2f5ad/ggml/src/ggml-cuda/mmq-config-ampere.cuh),
[mmq.cuh](https://github.com/ggml-org/llama.cpp/blob/e6ab7c1a41054a888ada952eab4c886444c2f5ad/ggml/src/ggml-cuda/mmq.cuh),
[mmq-load-tiles.cuh](https://github.com/ggml-org/llama.cpp/blob/e6ab7c1a41054a888ada952eab4c886444c2f5ad/ggml/src/ggml-cuda/mmq-load-tiles.cuh)
and [mmq-vec-dot.cuh](https://github.com/ggml-org/llama.cpp/blob/e6ab7c1a41054a888ada952eab4c886444c2f5ad/ggml/src/ggml-cuda/mmq-vec-dot.cuh).
Use its I128/J128 ownership and staged activation halves as a starting point;
do not merely quadruple the existing accumulator array or double all shared
buffers. Resource limits must include the extra FP32 metadata.

Preserve policy 1030's FP32 activation scales and affine arithmetic. Upstream
half2 scale products are not interchangeable. `FP16 d × six-bit s` is exactly
representable in FP32 and can be hoisted without another rounding boundary.
But folding `s` into an INT8 nibble is invalid: `63×15=945`. Likewise changing
where K32 contributions are scaled or accumulated needs an independent
numerical check. There is no justification here for requantizing the model,
returning to global weight expansion, or redesigning the already improved
decode path. Gate/up slab fusion is secondary to the 20.171 s contraction.

## Attention: decode GQA reuse has not reached prefill

In [attention.cu](../../cuda/attention.cu), lines 462–478 and 801–806,
prefill launches by query head and Q32 tile; `h/6` selects its KV head.
Six CTAs therefore stage the same KV data independently. Decode instead
launches by KV head and shares its six queries within a CTA. TASK-040's
delivery phrase “shared KV across each six-query-head group” describes
decode; it must not be read as a claim about current prefill.

Prefill already has online softmax, Q/PV register residency, asynchronous K/V
staging, and fused output packing for the FP8 output projection. There is no
global probability matrix to remove. Two BF16 probability components each
feed a PV MMA; they are a deliberate precision requirement, after single-BF16
P failed earlier checks. Do not silently discard the low component.

Implement shared KV for a bounded group of sibling prefill heads. Start with
two heads × Q32 per CTA, three pairs per KV head, preserving per-head softmax
and output indexing. This doubles useful queries per staged KV tile without
requiring six complete Q32 accumulators. Check the resulting registers,
shared memory and masked-tail work. This mapping yields only 96 CTAs at M256
on the 170-SM GPU, versus 192 today; M512 restores 192 CTAs. Therefore combine
the grouped kernel with bounded 512-token prefill in TASK-042, retaining the
existing per-head kernel for smaller chunks. Do not claim sixfold bandwidth
savings: L2 can already serve repeated source loads. Keep M=1 decode and its
split/merge schedule intact. If the combined path loses, one diagnostic with
the existing attention kernel at M512 can distinguish the chunk change from
grouping; no head/tile/partition search is prescribed.

## Remaining conversions, reuse and scheduling

| Boundary | Current behavior | Consequence / priority |
| --- | --- | --- |
| Attention input Q/K/V | One RMS-fused K128 FP8 pack, shared across three projections | Already correct; separate GEMMs still reload that pack |
| GDN input QKV/Z | One shared FP8 pack; BF16 normalized values also feed small A/B | BF16 materialization serves real consumers, not an accidental round trip |
| Attention output / GDN gated RMS | Local BF16 rounding followed directly by FP8 output packing | Already avoids global BF16 store/repack |
| FP8 GEMM output | Full FP32 accumulator slab, then separate BF16-store or residual epilogue | Real fusion opportunity; 2.044 s contractions do not isolate its savings |
| Small GDN A/B | Bounded BF16 staging and cuBLAS | 12,288 unpack calls cost only 18.764 ms at 32K; defer |
| MLP gate/up output | Two FP32 slabs, read by fused SwiGLU/Q8 pack | Real traffic, but shared packing already works; defer complex paired-GEMM fusion |
| Short FP8 tails | M=1 GEMV; every M>1 pads token rows to 128 | Potential small-prompt/tail waste; not responsible for full M256 benchmark gap |
| GDN recurrence | Ordered FP32, state resident for 64 tokens, four intervals/M256 | 1.575 s recurrence; total GDN gap only 0.297 s |
| Submission/readout | Reused workspace, internal asynchronous layers, completion per chunk; final-position generation logits | No per-layer waits or all-token generation head work remain |

Source: [prefill_attention.cpp](../../src/runtime/prefill_attention.cpp),
lines 158–197; [prefill_gdn.cpp](../../src/runtime/prefill_gdn.cpp), lines
181–230; [prefill.cu](../../cuda/prefill.cu), lines 494–499 and 555–609;
[fp8.cu](../../cuda/fp8.cu), lines 18–35 and 132 onward;
[gdn.cu](../../cuda/gdn.cu), lines 345–391 and 566–594;
[language_model.cpp](../../src/runtime/language_model.cpp), lines 408–590.

The comparator uses 512-token microbatches for 4K/32K; QW38 uses 256.
The FP8 engine's nominal 1024-token capacity does not change that: the full
runtime uses `kArenaTokenCapacity=256`, and Q8 MLP independently rejects
M>256. TASK-042 extends the whole bounded producer/consumer chain to test 512.
Changing just the arena constant is incorrect. A larger chunk alone does not
reduce the total Q32 attention tiles or fixed-J32 weight reloads. Evaluate it
after improving those consumers, without promising a factor-of-two gain.

GDN still incurs two block barriers per token and interval-boundary state
traffic. Longer intervals barely changed the old complete-layer measurement,
but that was under an older projection policy. A GPU low-rank/WY algorithm
was deferred, not benchmarked and rejected. It remains a possible later
optimization; the current 0.297 s comparator gap does not put it ahead of
MMQ or attention, and changing recurrent accumulation has a larger validation
burden.

At 32K, observed free device memory is 8,740,864,000 bytes; the required
reserve is 2 GiB. That supports considering bounded workspace growth, not
assuming it fits without accounting. The 21,013,686,400-byte resident model,
2,309,816,320-byte persistent state/KV, shared scratch, separate prefill/Q8
workspaces, graph/library backing and first-use transient allocations all
count. A second full weight representation is neither needed nor budgeted.

## Chosen tasks and deferred triggers

1. **041: Reuse-rich Q4_K×Q8 prefill MMQ.** Adapt the measured J128 schedule,
   cooperatively stage FP32 metadata and preserve one shared producer pack.
   Keep decode arithmetic and current artifact. This addresses the main cost.
2. **042: Shared-KV attention with bounded 512-token prefill.** Add sibling-head
   reuse with the existing probability precision and direct FP8 output;
   extend all capacity checks/workspaces consistently so the new tile has
   enough parallel blocks. Compare the combined schedule once with the
   optimized 256-token parent, retaining the existing attention path for
   smaller chunks. No chunk sweep.
3. **043: One final quality/replay/capacity/performance decision.** Preserve
   core-54, the single fixed 32K retrieval case, six performance executions
   yielding nine rows, all promotion criteria and historical failures.

FP8 fused epilogues or a short-M tile become the next task only if the new
profile leaves them material, or actual short-tail workloads expose the
padding cost. Hoisting immutable descriptors/validation or a prefill graph
requires exposed host gaps after kernel improvements; it is not a remedy for
the current 20 s MMQ cost. Revisit GDN intervals before a WY implementation
if recurrent cost becomes a leading residual. Keep decode MMVQ/FP8 GEMV work
as a separate future batch: it remains material, but is not this prefill task.

No format contest, generic operator registry, autotuner, whole-layer fusion,
new inference framework or exhaustive profiling campaign is required. Use
the existing benchmarks and focused tests; detailed stall profiling is a
follow-up only if it answers a concrete failed implementation decision.

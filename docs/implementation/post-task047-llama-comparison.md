# Performance comparison after TASK-047 — 2026-09-27

We can locate the remaining cost. What is missing is a reliable connection
between an optimization, its cost in the production execution path, and the
resulting request latency. Cold standalone component measurements, summed
kernel durations, and request latency currently answer different questions.

This is an investigation, not a new implementation or promotion decision.
The working tree contains the blocked TASK-047 candidate on TASK-046
`8399e4c`. TASK-043 remains the accepted production baseline. The most recent
matched, fully attributed comparison with llama.cpp is TASK-043; TASK-045–047
have narrower development comparisons. Their unprofiled eight-token timings
must not be compared directly with the old instrumented 128-token comparator.

The upstream reference is the measured llama.cpp revision
`e6ab7c1a41054a888ada952eab4c886444c2f5ad`, including cached pinned sources,
not the different HEAD in `/home/user/llama.cpp`. No new inference, GPU
benchmark, build or quality run was performed for this report.

## What the existing measurements establish

TASK-043 reduced 32K prefill from TASK-040's 29.422 s to 15.818 s: a substantial
46.2% latency reduction. The remaining comparator time is 12.373 s.
TASK-045 subsequently reduced its paired eight-token decode phase by 12.675%.
The latest work therefore has real observed gains, but neither those gains
nor TASK-046's small head improvement establishes parity with llama.cpp.

From the saved 32K request captures, prefill kernel-duration sums are:

| Category | QW38 ms | llama.cpp ms | Difference ms |
| --- | ---: | ---: | ---: |
| Projections | 9667.201 | 7779.734 | +1887.467 |
| Attention | 3764.244 | 1837.267 | +1926.977 |
| GDN | 1824.603 | 1510.522 | +314.081 |
| Normalization / epilogues / other | 474.795 | 953.276 | -478.481 |
| Separately labeled packing/unpack | 9.213 | 196.487 | -187.274 |
| Head | 1.271 | 0.665 | +0.606 |

These are category sums, not an additive critical-path decomposition.
Fused packing remains charged to its containing kernel. The 3.446 s host
prefill gap is overwhelmingly associated with projection and attention GPU
work; there is no unexplained multi-second host gap in this evidence.

QW38 J128 MMQ alone costs 8.018 s. Eliminating the roughly 1.3 ms prompt head
cannot materially change this prefill result. At 256 tokens, however, QW38's
prefill kernel sum is about 114.6 ms against 164.8 ms host time; first-use and
submission costs deserve separate attention for that short workload.

Sources: [TASK-043 delivery](task043-delivery.md),
`.cache/evaluation/qw38-language-v2/task043-support/profiles.json`, and
`task027-support/single-run/profiles.json` under the same evidence root.

## A previously underemphasized difference: overlapping kernel intervals

For the matched 128-input decode-4096 captures:

| Measure | QW38 ms | llama.cpp ms |
| --- | ---: | ---: |
| Host phase | 2520.961 | 1952.946 |
| Sum of kernel durations | 2449.382 | 2224.275 |
| Union of kernel execution intervals | 2449.382 | 1850.512 |
| Sum minus union | 0.000 | 373.763 |
| Host time outside the kernel union | 71.579 | 102.434 |

The kernel-union gap is 598.870 ms; the host gap is 568.015 ms. Removing host
bookkeeping alone cannot close it. Comparing just summed categories also
misses a material difference: llama.cpp's recorded kernels overlap, while
QW38's essentially serialize.

Read-only inspection of the raw SQLite traces found 246400 unique comparator
kernel rows and 113024 unique QW38 rows, without duplicate kernel records.
Overlap occurs within one comparator stream, including the first eager
token, with sequences such as RMS, scale and Q8 quantization. It is not
evidence that independent layers execute concurrently, nor is 373.763 ms a
prediction of recoverable speedup. Intervals can include dependency waiting
and execution tails.

Pinned llama.cpp implements programmatic dependent launch (PDL): the launch
wrapper enables programmatic stream serialization by default for supported
kernels, with device dependency synchronization and launch-completion
primitives. QW38 has no corresponding PDL path. This is a concrete source
difference consistent with the observed same-stream overlap. Its causal
contribution still needs an on/off comparison; adding more CUDA graphs is
not equivalent to enabling PDL.

Primary source: [pinned common.cuh](https://github.com/ggml-org/llama.cpp/blob/e6ab7c1a41054a888ada952eab4c886444c2f5ad/ggml/src/ggml-cuda/common.cuh#L1468).
Raw evidence: `task043-support/decode-4096-qw38.sqlite` and
`task027-support/single-run/decode-4096-llama.sqlite`. Existing profile JSON
already includes both sums and unions; this finding requires no new counters.

## The kernels still differ substantially

| Area | QW38 | Measured/pinned llama.cpp | Implication |
| --- | --- | --- | --- |
| Decode mixer projections | E4M3 weights converted to floats, scaled, and consumed by scalar FP32 FMAs | Quantized integer vector-dot consumers | FP8 storage does not imply FP8 tensor-core decode. TASK-045 improves loads, but retains this arithmetic. |
| Q4 prefill | Static I128/J128 tiles; FP32 scale/minimum metadata and exact INT32 activation sums; 64 KiB shared | Packed metadata, 57856 shared bytes in observed Q4 J128; shape-dependent stream-K | Matching tile dimensions does not match scheduling, metadata traffic or instructions. |
| Attention | BF16 KV; high and low BF16 probability components; two PV MMA calls with FP32 accumulation | FP16 path with one probability component and half2 PV accumulators in the pinned NVIDIA implementation | QW38 deliberately performs more work to meet its numerical contract. Kernel tuning cannot make that extra arithmetic disappear. |
| Long prefill attention scheduling | Full-prefix scan per query tile; paired M512 baseline has 192 CTAs | Observed FA grid 340 CTAs with stream-K/fixup | Different work distribution and parallelism, not just a KV reuse difference. |
| Model residency | 21.014 GB model allocations | About 18.963 GB CUDA model buffer | Different formats/precision; model size is not per-token traffic or a bandwidth diagnosis. |

Relevant current code: [FP8 decode](../../cuda/fp8.cu#L83),
[MMQ metadata](../../cuda/q4k_q8.cu#L158),
[static MMQ launch](../../cuda/q4k_q8.cu#L297), and
[two-component attention PV](../../cuda/attention.cu#L583).
Pinned references are retained under `task041-support/upstream/` and
`task038-support/upstream/`; comparator resources come from the saved 32K
SQLite trace. The comparator artifact contains Q4_K, Q6_K and Q8_0 tensors;
its total projection category is not a same-weight Q4-only measurement.

Stream-K deserves a shape-specific interpretation. On 170 SMs, QW38 M512
gate/up has 544 CTAs, filling four one-CTA-per-SM waves at 80% tile efficiency.
The comparator uses 170 CTAs to distribute that work. M256 down has only
80 tiles. But M512 down has 160 tiles, already 94.1% of one wave, and pinned
llama.cpp also uses 160 CTAs there. Missing stream-K cannot explain every
MMQ case. Its partial reduction would also require numerical verification.

## The counters argue against another blind reuse or tile change

TASK-047's preserved parent diagnostics report:

| Representative kernel | Achieved occupancy | Reported memory throughput | L2 hit rate | Leading reported stall |
| --- | ---: | ---: | ---: | --- |
| M512 gate/up J128 | 16.64% | 110.40 GB/s | 91.85% | Long scoreboard, 48.1% of average cycles between issued instructions |
| Long-prefix paired attention | 16.61% | 38.32 GB/s | 96.76% | Math-pipeline throttle, 36.6% of average cycles between issued instructions |

Neither spills. Registers and shared memory each limit the sampled kernels
to one CTA/SM. These are diagnostic replay observations, not request timings
or predicted speedups. They do not show saturated DRAM bandwidth. MMQ's
next question is which load/dependency remains exposed; attention's is which
instruction pipeline and work distribution constrain issue throughput.

These tradeoffs show why reducing nominal work can disappoint. Shared KV reuse may
save loads already served by L2 while increasing per-CTA work or reducing
parallelism. Fewer declared registers help only if they improve scheduling
or execution. TASK-047's K32 attention reduced registers 179 to 132 and shared
memory by half, but still allowed only one resident CTA/SM. Its original
isolated win did not survive the final component/request observations.

Sources: `task047-support/parent-{mmq,attention}-counters.txt` and the
[TASK-047 report](tasks/TASK-047.md#blocked-implementation-report--2026-09-27).
Source/PC attribution for the actual stalled instructions is still missing;
the reports explicitly note missing optional PC-sampling information.

## Why the component results fail to explain the request

1. **Cold component time is not recurring graph cost.**
   [FP8 group timing](../../benchmarks/fp8_group_bench.cpp#L84) and
   [MLP timing](../../benchmarks/mlp_bench.cpp#L91) put a CUDA event before
   first-use eager launches, then include copies/completion. The event's
   elapsed interval includes GPU idle time while the CPU prepares/submits
   later work. The field `gpu_ms` therefore does not mean pure kernel time.
   TASK-045's diagnostic MLP has five kernels totaling 0.132928 ms in the
   parent and 0.124096 ms in the candidate; the first-to-last kernel spans
   are 0.415744 and 0.378016 ms, with approximately another 0.4 ms from the
   start-event API to the first kernel. These separate diagnostic captures
   explain the scope mismatch, not the original untraced regression.
   The candidate's weighted cold FP8 groups total 92.211 ms, while its whole
   recurring token is about 16 ms. Multiplying first-use component costs by
   layer count clearly does not reconstruct production execution.

2. **Component boundaries differ.** MLP and FP8 drivers include D2H output;
   the prefill attention driver times GPU output completion but copies the
   result afterward. The M1 attention driver omits output projection and
   residual addition, already recorded as R047-5. Each scope can serve a
   diagnostic purpose, but their numbers cannot be added as complete layers.

3. **Endpoint attention probes do not represent the prefix distribution.**
   A 32K request has 64 chunks and 16 attention layers, hence 1024 attention
   calls spanning many prefixes. Testing prefix 0 and 32256 alone cannot
   establish the weighted change. TASK-047's final endpoint gain is only
   0.198 ms at prefix 32256. That observation does not promise a large
   whole-request gain or explain its observed 59.2 ms regression.

4. **Single observations cannot identify small causal effects.** TASK-047's
   +0.36%, +0.39% and +0.17% request regressions remain recorded failures.
   They do not identify their cause; neither do same-sized favorable results.
   Within prefill, TASK-046 changes only the once-per-prompt head yet observes a 12.6 ms prefill
   improvement, far above its isolated 0.050 ms head difference. These
   differing scopes cannot support attributing that whole improvement to the
   head. The existing cold acceptance protocol should remain distinct from
   diagnostics that estimate recurring cost and run-to-run variability.

## Smallest useful next investigation

First freeze the candidate being diagnosed, then measure its production
path against the pinned comparator. Keep cold first-use results separately.
For tuning, this report proposes a measurement-policy amendment: a bounded
paired recurring-work diagnostic with reported dispersion and alternate arm
order. The ledger currently prescribes one execution and zero warmups; this
proposal has not changed that policy or launched additional runs. Use the same
inputs, profiler mode and output/completion boundary. Do not silently redefine
existing acceptance results or infer instrumentation overhead from different
historical runs.

The first three discriminating experiments should be:

1. **Decode launch overlap:** use the existing comparator with PDL enabled
   versus `GGML_CUDA_PDL=0`, verifying the actual launch path. Compare host
   latency, GPU union, summed durations and outputs. This tests the missing
   mechanism before implementing it in QW38. The sum-minus-union figure is
   not the expected gain.
2. **MMQ scheduling:** isolate gate/up and down at M256/M512 in a resident
   production-like chain. Compare per-shape time and expose the stalled load
   sites before choosing balanced K work or another staging change. Preserve
   policy 1030; do not combine scheduling and precision changes in one test.
3. **Attention across real prefixes:** attribute scan and projection costs
   across the existing 64-chunk request. Weight any proposed improvement by
   actual occurrences. Test work distribution separately from arithmetic
   precision; changing the latter is a distinct quality decision.

Every candidate should state its expected request saving before the request
run: sum of changed stage costs times their real call counts, adjusted for
observed overlap and first-use work. Then reconcile the observed change by
stage. A difference that does not reconcile is the next measurement target,
not evidence that another optimization family is needed.

Already completed work should not be proposed again: shared gate/up packs,
compact Q4 consumption, M512 capacity, asynchronous KV staging, internal
enqueue without per-layer completion, decode graphs, and final-row-only
generation readout are present. The remaining problem is execution efficiency
and causal measurement, rather than a missing generic phase dispatcher.

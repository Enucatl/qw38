# Quartz Speedup Plan

## Diagnosis

Admit a measurement-led ladder starting at OPT-032, with prefill attention
and decode MMV as the leading kernel candidates.

The accepted 4K baseline is **1746.71973 tok/s** versus llama.cpp
**3253.993621 tok/s** (about **1.86x**). Post-OPT-026 attribution assigns
the Quartz 4K wall time as follows:

| Sink | Time | Share | Finding |
|---|---:|---:|---|
| Attention core | 867.45 ms | 36.9% | Q×K uses MMA; probability×V uses scalar accumulation and repeatedly stores partial output. |
| FFN MMQ | 738.27 ms | 31.4% | Quality MMA and shared activation staging are installed, but dispatch follows a 2K J sweep rather than independent 4K projection shapes. |
| GDN core | 453.23 ms | 19.3% | Register-held recurrence is installed; normalization and sequential token processing remain. |
| Mixer MMQ | 285.69 ms | 12.2% | Quality MMA, shared residual staging, and skinny alpha/beta dispatch are installed. |

Quartz attention still differs from pinned llama.cpp in the value product:
`cuda/fattn_mma_f16.cuh` updates and writes partial `vkq` output for every
32-row KV tile, while pinned `fattn-mma-f16.cuh` keeps `VKQ_C` in registers
and uses MMA for value accumulation. Scheduling changes alone do not close
this arithmetic and storage gap. Multiple sinks must improve; eliminating
attention would still leave roughly 1481 ms.

Decode evidence is not yet admissible. BEN-001 has only a 17-token prompt,
two generated tokens, and two samples (median 15.84 tok/s; p50 ITL 63.15 ms).
Its perturbing probe attributes about 69.7% to FFN, 11.08 ms to GDN, and
5.09 ms to attention, but explicitly excludes itself from admission.
Current decode candidates are Q4_K/Q6_K MMV packed loads in
`cuda/quant_mmv.cu` and one-token grouped tiled attention in
`cuda/attention_decode.cu`. Current decode distributions, exclusive
attribution, and matched llama.cpp comparisons are missing.

Failed ideas remain excluded: OPT-024 Q8 D2R, OPT-027 persistent fattn,
OPT-028 MMQ stream-K, OPT-029 GDN fusion, and OPT-030 PDL without new
evidence. Do not re-admit mixer D2R, persistent prompt scheduling, PDL,
speculative decoding, model-policy changes, or new GDN fusion.

## Ranked ladder

| Rank | Task | Phase | Target |
|---:|---|---|---|
| 1 | OPT-032 | Both | Freeze decode oracle and refresh sink attribution. |
| 2 | OPT-033 | Prefill | Retain attention value sums in registers. |
| 3 | OPT-034 | Decode | Use blockwise packed Q4_K/Q6_K MMV loads. |
| 4 | OPT-035 | Prefill | Use MMA for attention probability×V. |
| 5 | OPT-036 | Decode | Partition KV for vector attention. |
| 6 | OPT-037 | Prefill | Select 4K FFN tiles per projection. |

Provenance is limited to the inspected pinned llama.cpp checkout at
`.cache/authorities/llama.cpp` revision
`cc83d7b4824f73cfdda4dfbb47ee39804f71b328` and ds4 `c238077a` as MIT-licensed
technique inspiration. The newer sibling llama.cpp checkout is not a
benchmark authority, and ds4 supplies no same-model denominator.

Default order is **032 → 033 → 034 → 035 → 036 → 037**. Execute one
experiment at a time. Dependencies are evidence or retained functionality;
a documented rejection counts as done. Place OPT-032 before OPT-031 when
the draft is later admitted. Retain OPT-031 under its existing ID.

OPT-032 must compare same-sitting Quartz/llama.cpp ratios for prefill and
D2048, then record the next pending order. If decode deficit is larger, put
its largest measured transferable sink first; otherwise retain attention
first. Schedule a supported decode experiment after at most two prefill
experiments. A candidate with no measured target deficit may finish as an
evidence-backed rejection.

## Draft task boundaries

OPT-033 keeps Ncols1=16, Ncols2=2, KV tile 32, dual-F16 Q, and `grid.z=2`.
Give each thread fixed accumulator slots, preserve scalar KV iteration and
final combine, and reject register spills or occupancy loss that erase the
component A/B win.

OPT-034 walks 256-weight blocks, loads packed fields and scales once per
block, and consumes each lane's eight values in the existing order. Keep
warp-count dispatch and Q8 staging. Q8_0 direct-BF16 MMV and DP4A
reassociation are outside this increment.

OPT-035 uses two F16 probability components with FP32 MMA accumulation and
FP32 online max/denominator. Preserve staging, causality, gate application,
and split merge. Reject on any frozen attention-envelope miss.

OPT-036 retains normalization and candidate-row staging, uses contiguous KV
partitions, FP32 max/denominator/numerator, and deterministic ascending-part
merge. Select the lowest-time passing variant independently below and at or
above 2048 context, including the current kernel.

OPT-037 measures Q4_K `17408×5120` gate/up and `5120×17408` down separately at
4096 rows. Keep 2D scheduling, quality staging, existing tail dispatch, and
the baseline on any leg without a win; recapture shared-Y graphs when needed.

## Frozen measurement and acceptance rules

P is unchanged from OPT-021: exact 4096 synthetic tokens, cold production
`sync_tokens`, attribution null, graphs created, prefix reuse disabled, zero
warm-ups, three replicates, with pinned llama.cpp run in the same sitting as
`-p 4096 -n 0 --no-warmup -r 3 -ngl 99`. A keep must strictly beat the then
current accepted Quartz oracle and the fresh same-sitting baseline.

OPT-032 freezes D128 and D2048: batch one, exactly 128 or 2048 committed
prefix tokens, then 256 timed one-token evaluations; token IDs
`(42 + index × 997) % vocabulary_size`; production evaluation, logits, and
state commit included; prefix preparation and session allocation excluded;
graphs enabled; fresh logical session; no prefix reuse; three warm-ups and
30 measured runs per engine/build/workload. Report arithmetic-mean tok/s,
p50/p95 token latency, every run, and every token duration. Use a fixed-token
native comparison driver through pinned llama.cpp's public API. The random
token `llama-bench -p 0 -n 256 -d 2048 --no-warmup -r 30 -ngl 99` shape run is
informational only.

All keeps require D128/D2048 throughput and p95 latency regression of no more
than 5%; D2048 must improve for decode tasks, and decode changes must retain
at least 95% of P throughput. Component A/Bs use three warm-ups and 30
alternating measurements with identical inputs and reset state. Choose the
lowest mean among correctness-passing candidates; ties retain production.
Component wins never authorize installation alone.

Keep the existing numeric, byte, state, causality, RoPE, checkpoint,
cancellation, graph-equivalence, chunk-boundary, save/restore, greedy,
memory-fit, quantization, attention, GDN, and CUD-001/OPT-009/ATN-001/GDN-002
gates. DP4A reassociation or looser attention precision requires a separately
approved plan-impact prerequisite. An envelope miss is rejection.

## Stop conditions

Stop prefill admissions when accepted Quartz P reaches or exceeds same-sitting
llama.cpp. OPT-016 remains the separate blocked 2K gate. Continue supported
decode work until Quartz reaches at least llama.cpp mean tok/s on both D128
and D2048 with no worse p95 latency. These interim stops do not close the
broader comparative release gates in `plan.md`.

No implementation, benchmark, commit, or push is authorized by this planning
document.

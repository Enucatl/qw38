# Post-OPT-037 performance gap analysis

Date: 2026-09-10. This is source-backed planning evidence; no benchmark was
run for this report.

## Executive summary

The accepted post-OPT-034 oracle is P **1869.84412**, D128 **25.3816128**, and
D2048 **20.169548 tok/s** (`fixtures/opt034_packed_mmv.json`). Against the
OPT-032 same-sitting llama.cpp measurements, the gaps are estimated at 1.68×,
2.70×, and 3.32× respectively. OPT-036's original 201.241058 ms D2048
attention attribution predates its keep; its one-layer component fell from
12.2984858 to 0.804042637 ms. Current attribution must therefore be refreshed.

Ranked remaining opportunities are: warp-owned decode attention; hoisted GDN
Q/K normalization; warp-owned prompt QK microtiles; and a diagnostic integer
decode-MMV study. Prompt FFN already uses the same broad quality-MMA class as
llama.cpp, and OPT-037 proved another I/J sweep is not justified.

## FFN

`cuda/full_scheduler.cu:636-710` performs shared gate/up Q8_1 staging,
quality MMA projections, SwiGLU-to-Q8, and down projection. `cuda/quant_mmq_mma.cuh:534-1000`
uses packed tiles and `MMQ_ITER_K=256`. Pinned llama.cpp uses the corresponding
Q8_1 shared staging and integer MMA in `mmq.cuh`, `mmq-load-tiles.cuh`,
`mmq-vec-dot.cuh`, and `mma.cuh` at `cc83d7b`.

This kernel class is already present. OPT-028 stream-K and OPT-037 I/J sweeps
both lost; production I=128/J=128 remains the winner for gate, up, and down.
The remaining decode FFN difference is more promising: `cuda/full_scheduler.cu:712-760`
launches three separate MMV calls, while llama.cpp `mmvq.cu` and
`vecdotq.cuh` provide integer Q4_K dots and optional GLU epilogues. Decode MMV
has its own fixed CUD-001 envelope, so this requires a study before promotion.

## Attention

### Decode

`cuda/attention_decode.cu:693-850` partitions KV rows but still loops over each
row and query member, writes products to shared scratch, lets lane zero fold
256 terms, and synchronizes repeatedly. This is a serial structure that remains
after OPT-036. llama.cpp `fattn-vec.cuh` keeps Q/KQ/VKQ in registers and uses
warp reductions. A warp-owned query-head/partition kernel should remove the
CTA-wide barriers, while retaining the 16-way partition and deterministic merge.
It may increase KV reads; measure that tradeoff.

### Prefill

`cuda/fattn_mma_f16.cuh:328-430` computes each QK microtile across four warps,
stores `cparts`, and synchronizes before warp zero merges it. Pinned llama.cpp
`fattn-mma-f16.cuh` assigns KQ fragments to warp-owned MMA tiles. A narrow
ownership change can remove barriers without changing dual-F16 Q, online FP32
softmax, register VKQ, or P×V MMA.

## GDN

Quartz already has llama.cpp's register-held state loop in
`cuda/gdn_fused_quality.cuh:100-175`; llama.cpp's source is
`ggml/src/ggml-cuda/gated_delta_net.cu`. Quartz nevertheless recomputes Q/K
square sums and inverse norms in every value-column warp. Pinned
`src/models/qwen35.cpp` builds Q/K normalization before the recurrent operator.
Hoisting only those two scalars per token/key head is distinct from rejected
OPT-029 convolution/output fusion and preserves the FP32 recurrence order.

## Overhead and correctness

OPT-012 captures FFN subgraphs only; OPT-031 remains the mixer/GDN graph task.
`cuda/full_scheduler.cu:450-515` can raise the recorded attribution wall when
host graph-launch time overlaps CUDA events, so future evidence must retain raw
host wall and adjusted reconstruction separately. No persistent allocation,
public API change, `--fmad` change, or blanket tolerance relaxation is justified.

The Q4_K association rule is `abs > 0.20*sqrt(K)` **and** `rel > 0.05`; Q8 is
`abs > 0.05*sqrt(K)` **and** `rel > 0.05`. GDN remains max-abs 5e-8/RMS 5e-9;
attention remains max-abs 5e-5/RMS 5e-6; decode MMV keeps CUD-001 fixed limits.
Any technique that misses its envelope is rejected or receives a separately
approved contract; this ladder does not edit `plan.md`.

## What llama.cpp has that Quartz lacks

| Technique | Quartz sink | Authority | Portability |
|---|---|---|---|
| Warp-reduced vector attention | `partitioned_grouped_decode_attention` | `fattn-vec.cuh` | OPT-039; conditional on existing envelope |
| Warp-owned QK MMA fragments | `fattn_mma_quality_kernel` | `fattn-mma-f16.cuh` | OPT-041; preserve partial order |
| Pre-loop Q/K normalization | `prepare_recurrence_fused_warp_column` | `qwen35.cpp`, `gated_delta_net.cu` | OPT-040; byte-equal candidate required |
| Integer Q4_K MMV dots | `quant_mmv` | `mmvq.cu`, `vecdotq.cuh` | OPT-042 study only |
| Broader graph capture | `SchedulerGraphs` | `ggml-cuda.cu` | OPT-031 already owns this |

OPT-022, OPT-023, OPT-025, OPT-026, OPT-033, OPT-034, OPT-035, and OPT-036 are
already kept. OPT-024, OPT-027, OPT-028, OPT-029, OPT-030, and OPT-037 are
rejected. No task reopens them. OPT-031 stays pending and is not superseded.

## Recommended order

Run OPT-038 first, then OPT-039, OPT-040, OPT-042, OPT-041, and finally OPT-031.
After those P experiments, re-run the existing unchanged OPT-016 2K gate.
These are estimated opportunities and do not claim Quartz will beat llama.cpp.

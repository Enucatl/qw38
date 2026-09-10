# OPT-043 — Measure the actual post-042 component gap

## Claim labels and proof limits

This increment **claims no performance improvement**. It retains **D128 and D2048 oracles**
with **exclusive leaf intervals**, **matched pinned llama.cpp** component timings,
**activation captures**, **Nsight counters marked missing or recorded**, and
**successor decisions**. **accepted keep denominators remain historical**.
This protocol **does not substitute for the 2K llama.cpp parity gate**.
**llama-bench random decode is informational**.

Measurement-only / in progress diagnostic. Production dispatch is unchanged.
Live numbers in `fixtures/opt043_component_gap.json` are the Measured same-sitting
exclusive RTX 5090 record. Accepted keep denominators remain the OPT-041 copies.

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp image | `qw38-llama-authority:cuda-13.0.2` |
| llama.cpp revision | `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` |
| Throughput attribution | null; graphs created |
| Capture layers | 0, 3, 31, 32, 62, 63 |
| Q8 layout | llama `block_q8_1`, never Quartz `Q8Block` bytes |
| Numeric label | Quartz BF16 vs llama F32/Q8_1 activations; Quartz BF16 KV vs llama F16 KV |

## Measured sitting

- Device: NVIDIA GeForce RTX 5090 compute 12.0
- measurement_utc: 2026-09-10T14:02:32Z
- P Quartz mean tok/s: 2090.59082 ; llama.cpp avg_ts: 3256.723268
- D128 Quartz mean tok/s: 26.1648235 ; llama.cpp mean tok/s: 68.8452423
- D2048 Quartz mean tok/s: 25.321722 ; llama.cpp mean tok/s: 67.0527218
- p_gap: 1.5578004250492214 ; d2048_gap: 2.6480316701999964 ; decode_deficit_larger: true
- ranked_gap: OPT-053, OPT-052, OPT-051, OPT-050, OPT-054, OPT-046, OPT-047, OPT-045, OPT-055, OPT-048, OPT-049
- claims_performance_improvement: false
- owns_opt016_parity_gate: false
- substitutes_for_opt016: false

## Exclusive leaf reconstruction

Prefill 4K: raw_host_wall_ms 1996.78845; gpu_event_sum_ms 1996.018604943; fused {"gdn_conv_fused_with_recurrence": true, "gdn_output_norm_fused_with_gate": true, "attention_prep_fused_with_core": true, "ffn_graph_fused": true, "d2h_overlaps_state_copies": true}

Decode D2048 graph: raw_host_wall_ms 39.3527145; eager FFN isolates gate/up/SwiGLU/down.

Overlapping D2H/state copies are labeled and excluded from exclusive GPU reconstruction.

## Nsight

Systems available=False error=nsys not found in qw38-cuda:13.0.2
Compute available=True error=
Missing counters remain `unavailable`. This report does not claim bandwidth-bound or compute-bound from occupancy.

## Successor decisions

| Task | measured gap ms | affected fraction | mechanism | recoverable ms bound |
|---|---:|---:|---|---:|
| OPT-044 | 0 | 1 | Admit documented production arithmetic and quality budgets before changing kernels | 0 |
| OPT-045 | 5.90137 | 1 | Cooperative residual/head RMSNorm and admitted FMA | 5.90137 |
| OPT-046 | 10.8064 | 1 | Cooperative packed Q4_K decode dots on gate/up/down | 10.8064 |
| OPT-047 | 10.7335 | 0.75 | Shared Q8_0 mixer staging and packed integer dots | 8.05013 |
| OPT-048 | 0.8374 | 1 | Full 248320-row Q6_K packed dots; no vocabulary pruning | 0.8374 |
| OPT-049 | 0.22816 | 1 | Share decode FFN staging and fuse gate/up/SwiGLU | 0.22816 |
| OPT-050 | 563.64 | 0.35 | Prepare prompt Q norm/RoPE once; fused leaf attn_qk_prep_softmax_pv_merge | 197.274 |
| OPT-051 | 563.64 | 0.65 | Pipeline prompt attention with register softmax; same fused enclosing leaf as OPT-050, not double-counted | 366.366 |
| OPT-052 | 370.332 | 1 | Hoist GDN scaled Q/K and decay; fused leaf gdn_conv_qk_norm_recurrence | 370.332 |
| OPT-053 | 763.978 | 1 | MMQ scaling/tile staging inside existing quality MMA; ffn_graph_fused so enclosing ffn_mmq is the complete component | 701.549 |
| OPT-054 | 701.549 | 1 | Physical 512/1024/2048/4096 batches inside atomic 4K | 175.387 |
| OPT-055 | 4.63203 | 1 | Broader stable graphs for remaining host/GPU launch gaps | 4.63203 |
| OPT-056 | 726.127 | 1 | End-to-end P/D128/D2048 outcome gate, not a kernel rewrite | 0 |

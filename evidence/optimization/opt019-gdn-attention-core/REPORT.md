# OPT-019 — GDN and attention core quality

## Claim labels and proof limits

Production prompt GDN recurrence is the warp-column fused quality path.
Production prompt attention (`token_count >= 16`) is the fattn-mma-f16 analog.
The envelopes unloosened. sequential GDN remains the reference.
tiled attention remains the reference. The parity gate owner remains blocked.
This remasurement is not an end-to-end 2K tok/s gate. There is
no 8K/32K/128K throughput gate. A mixer versus core split is not this increment.

Live GDN/attention envelopes, ncols1 pin, category before/after, attribution,
and remasurement are **Measured**. llama.cpp `gated_delta_net.cu` /
`fattn-mma-f16.cuh` provenance at `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` is **External**.

## Production path

`GdnScanPath::kFusedTokenLoop` dispatches warp-column quality recurrence after
the existing parallel convolution. Rank-2 fused remains `launch_gdn_fused_rank2`.
`launch_attention_prepare_chunk` uses fattn-mma quality when
`token_count >= 16` (ncols1=16, ncols2=2)
and tiled otherwise. Rank-3 MMA remains `launch_attention_prepare_chunk_mma_rank3`.

## Component envelopes (**Measured**)

GDN quality versus sequential at 2048 tokens:
`opt019_gdn_2048 max_abs=1.58324838e-08 rms=1.28841632e-10 nonfinite=0 prepare_atomic=true passed=true`
`fused_ab fused_ms=4.04840517 rank2_ms=13.9231253 sequential_ms=19.37537 overlay_ms=14.6318932 occupancy=9 faster=true`

Attention quality versus tiled at 2048 rows:
`opt019_attention_2048 max_abs=3.16649675e-06 rms=1.31638146e-07 passed=true winner_ncols1=16 selected_ncols1=16`
`opt019_attention_ab quality_ms=17.2483196 tiled_ms=169.305527 rank3_ms=60.3077354 faster=true occupancy=1 samples_q=17.215168,17.2649918,17.2647991 samples_t=169.389343,169.239655,169.287582 samples_r=61.1546555,59.8166733,59.9518738`

Raw samples: `gdn-fused-quality-ab-raw.txt`, `attention-mma-ncols-sweep-raw.txt`.

## Residual sinks addressed (**Measured**, not the parity gate)

- GDN before (OPT-018 attribution): 1643.46082 ms
- Attention before: 1148.47461 ms
- Combined before: 2791.93543 ms
- GDN after: 1212.50647 ms
- Attention after: 477.650085 ms
- Combined after: 1690.156555 ms
- `core_addressed`: True
- Quartz mean tok/s: 978.756592 (walls [2091.71045, 2092.66187, 2092.98071])
- llama.cpp `avg_ts`: 3215.557248
- `would_pass_opt016`: False (informational)
- `owns_opt016_parity_gate`: false; `opt016_status`: blocked

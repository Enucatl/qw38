# OPT-053 — Optimize MMQ scaling and tile staging

## Claim labels and proof limits

Live exclusive-RTX-5090 A/B of explicit FMA scale accumulation and two-stage
packed-Y cp.async against production quality MMA. Keep requires
**OPT-044 production-numerics budgets**, **like-arithmetic off control remains production quality MMA I=128/J=128**,
**complete FFN staging plus gate/up/SwiGLU/down cost**, **explicit FMA scale accumulation and two-stage packed-Y cp.async**,
**Q4_K min corrections and Q6_K subscales unchanged**, **synchronous fallback for tails and misalignment**,
**no unchanged OPT-024 D2R or OPT-028 stream-K rerun**, **95% throughput floors versus OPT-052 keep**, **105% p95 ceilings versus OPT-052**, and
**does not substitute for the 2K llama.cpp parity gate**. Copied denominators
are P 2692.64575, D128 37.5680695, D2048 35.6793633.

## Decision

**keep** — `reverted`=false;
`keep_sitting_skipped`=false;
selected_mmq_pipeline_path=fma_async;
4096 complete-FFN A/B winner fma_async (off 11.2367001 ms, fma 10.1743622 ms, async_y 9.96796703 ms, fma_async 9.68816757 ms);
tok/s sitting ran.

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp revision | `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` |
| A/B | complete FFN staging+gate/up/SwiGLU/down, candidates ['off', 'fma', 'async_y', 'fma_async'] |
| Correctness | tails [17, 65, 129, 255] |
| P / D128 / D2048 | OPT-021 / OPT-032 protocols, unchanged diagnostics |
| Nsight | not_used |

## Measured sitting

- Device: NVIDIA GeForce RTX 5090 compute 12.0
- measurement_utc: 2026-09-10T23:58:33Z
- extra_workspace_bytes: 18432
- instruction mix: source FMA and cp.async present; SM120 SASS encodes cp.async as LDGSTS (24 hits in `quant_mmv.cuda.o`); occupancy 1 on I=128/J=128 Q4 quality MMA
- P Quartz mean tok/s: 2895.42773 versus OPT-052 2692.64575
- D128 Quartz mean tok/s: 37.5605927 versus OPT-052 37.5680695
- D2048 Quartz mean tok/s: 35.7286987 versus OPT-052 35.6793633
- owns_opt016_parity_gate: false
- substitutes_for_opt016: false

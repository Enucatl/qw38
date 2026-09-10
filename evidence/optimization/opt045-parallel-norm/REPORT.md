# OPT-045 — Parallelize normalization and use admitted FMA

## Claim labels and proof limits

Live exclusive-RTX-5090 A/B of cooperative RMSNorm (fmaf squares, warp shuffle
plus shared warp-sum, one inverse broadcast, single BF16 rounding) against the
serial thread-0 strict reference. Sqrt/divide versus admitted rsqrt is a second
A/B. Keep requires **OPT-044 production-numerics budgets**, **retained serial
reference**, **lower complete component time**, **95% P/D throughput floors**,
and **105% decode p95 ceilings**. This increment does not substitute for the 2K
llama.cpp parity gate. Copied denominators are P 2076.98315, D128 26.1599541,
D2048 25.2924843.

## Decision

**keep** — `reverted`=false;
`keep_sitting_skipped`=false;
selected_rms_norm_path=parallel_fma;
threads=256/gdn=32;
A/B winner parallel_fma_t256 (serial 0.341054976 ms, winner 0.197026104 ms);
tok/s sitting ran.

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp revision | `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` |
| A/B | 4096-row residual+norm, 3 warm + 30 alternating |
| Numerics | OPT-044 formula vs FP64 and llama cooperative replica |
| P / D128 / D2048 | OPT-021 / OPT-032 protocols |
| Nsight | not_used |

## Measured sitting

- Device: NVIDIA GeForce RTX 5090 compute 12.0
- measurement_utc: 2026-09-10T16:47:28Z
- P Quartz mean tok/s: 2130.41089 versus OPT-041 2076.98315 (1.0257×)
- D128 Quartz mean tok/s: 28.5522804 versus OPT-041 26.1599541 (1.0914×)
- D2048 Quartz mean tok/s: 27.5438766 versus OPT-041 25.2924843 (1.0890×)
- D128 p95: 35.1905479 ms (ceiling 40.291659135)
- D2048 p95: 36.4285774 ms (ceiling 41.63525331)
- Prompt-norm component: serial 0.341054976 ms → parallel_fma_t256 0.197026104 ms
- Decode RMS kernel: serial 0.0578549355 ms → 0.00944320019 ms
- Geometry: 256 threads (128-thread variants failed the unit-input OPT-044 abs budget of 1e-6 with max_abs 0.00390625)
- rsqrt A/B: parallel_rsqrt_t256 eligible but not 2% faster than sqrt/div; retained fmaf + 1/sqrtf
- Registers: serial 28, parallel 20; local_bytes 0 (no spill); occupancy 6
- Residual FP32 add remains bit-exact; BF16 norm uses OPT-044 budgets
- owns_opt016_parity_gate: false
- substitutes_for_opt016: false

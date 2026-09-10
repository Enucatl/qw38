# OPT-046 — Cooperative packed Q4_K integer decode dots

## Claim labels and proof limits

Live exclusive-RTX-5090 A/B of cooperative Q4_K integer-dot MMV (DP4A packed
products, K distributed across warps per row, independent accumulators until
row completion, Q8Block control staging and llama Q8_1 round/scale/sum) against
the retained packed FP32 path. Keep requires **OPT-044 production-numerics
budgets**, **retained packed FP32 path**, **lower complete component time**,
**95% P/D throughput floors versus OPT-045**, and **105% decode p95 ceilings
versus OPT-045**. This increment does not substitute for the 2K llama.cpp
parity gate. OPT-045 keep denominators are P 2130.41089, D128 28.5522804,
D2048 27.5438766. Historical OPT-041 copies remain P 2076.98315, D128
26.1599541, D2048 25.2924843.

## Decision

**reject** — `reverted`=true;
`keep_sitting_skipped`=false;
selected_q4_decode_path=packed;
warps_per_row=4;
A/B winner integer_q8_w4 (packed 0.067196091 ms, winner 0.0191882669 ms);
tok/s sitting ran.

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp revision | `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` |
| A/B | production gate/up 17408×5120 and down 5120×17408, 1/2/4/8 warps per row, 3 warm + 30 |
| Numerics | OPT-044 independently decoded FP64 dots |
| P / D128 / D2048 | OPT-021 / OPT-032 versus OPT-045 keep denominators |
| Nsight | not_used |

## Measured sitting

- Device: NVIDIA GeForce RTX 5090 compute 12.0
- measurement_utc: 2026-09-10T18:02:50Z
- P Quartz mean tok/s: 2130.48462 versus OPT-045 2130.41089
- D128 Quartz mean tok/s: 40.7888718 versus OPT-045 28.5522804
- D2048 Quartz mean tok/s: 38.3443794 versus OPT-045 27.5438766
- owns_opt016_parity_gate: false
- substitutes_for_opt016: false

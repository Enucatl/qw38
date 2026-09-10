# OPT-047 — Q8_0 mixer decode DP4A with shared staging

## Claim labels and proof limits

Live exclusive-RTX-5090 A/B of Q8_0 × Q8_1 DP4A mixer decode (OPT-046 Q8_1
staging reused, K distributed across warps per row, shared mixer-input
staging with output restage) against retained direct-BF16 `q8_mmv_bf16`.
This path adds Q8_1 activation approximation versus the original BF16
input high-precision reference; OPT-044 same-input llama budgets apply.
Keep requires **OPT-044 production-numerics budgets**, **retained
direct-BF16 reference**, **OPT-046 packed Q4_K**, **lower complete mixer
group time including staging**, **95% P/D throughput floors versus
OPT-045**, and **105% decode p95 ceilings versus OPT-045**. This
increment does not substitute for the 2K llama.cpp parity gate.
OPT-045 keep denominators are P 2130.41089, D128 28.5522804,
D2048 27.5438766. Historical OPT-041 copies remain P 2076.98315, D128
26.1599541, D2048 25.2924843.

## Decision

**keep** — `reverted`=false;
`keep_sitting_skipped`=false;
selected_q8_decode_path=dp4a_q8_1;
warps skinny/medium/wide=4/
4/
4;
A/B winner dp4a_w4 (direct_bf16 10.6800642 ms, winner 5.44419813 ms);
tok/s sitting ran.

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp revision | `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` |
| A/B | production mixer Q8_0 shapes, shared staging groups, 1/2/4/8 warps |
| Numerics | OPT-044 vs original BF16; staged integer-dot envelope |
| P / D128 / D2048 | OPT-021 / OPT-032 versus OPT-045 keep denominators |
| Nsight | not_used |

## Measured sitting

- Device: NVIDIA GeForce RTX 5090 compute 12.0
- measurement_utc: 2026-09-10T19:22:46Z
- P Quartz mean tok/s: 2128.54175 versus OPT-045 2130.41089
- D128 Quartz mean tok/s: 33.8896103 versus OPT-045 28.5522804
- D2048 Quartz mean tok/s: 32.390007 versus OPT-045 27.5438766
- owns_opt016_parity_gate: false
- substitutes_for_opt016: false

# OPT-035 — MMA attention probability times V

## Claim labels and proof limits

Live exclusive-RTX-5090 A/B of 4096-row production stream-K attention including
combine, candidates `scalar` versus `mma` dual-F16 probability×V MMA. Keep
requires **frozen attention envelopes**, **lower component time**, **improved P**,
and the **cross-workload guard**. This increment does not substitute for the 2K llama.cpp parity gate. The OPT-033 P D128 D2048 are the keep denominators
(P 1745.10315, D128 15.1528101, D2048 13.5596962), not
historical OPT-026 1746.71973, not OPT-032 1637.58594, and not OPT-036 1644.04822.

## Decision

**keep** — `reverted`=false;
`keep_sitting_skipped`=false;
selected_pv_path=mma;
selected_vkq_accum=registers;
A/B winner mma (scalar mean 44.8905029 ms,
mma mean 35.445816 ms);
tok/s sitting ran.

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp image | `qw38-llama-authority:cuda-13.0.2` |
| llama.cpp revision | `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` |
| A/B | 4096 rows, scalar vs mma, 3 warm + 30 alternating, include combine |
| P / D128 / D2048 | OPT-021 / OPT-032 protocols, unchanged diagnostics |
| Nsight | not_used |

## Measured sitting

- Device: NVIDIA GeForce RTX 5090 compute 12.0
- measurement_utc: 2026-09-09T19:20:04Z
- P Quartz mean tok/s: 1865.21155 versus OPT-033 1745.10315
- D128 Quartz mean tok/s: 15.0562878 versus OPT-033 15.1528101
- D2048 Quartz mean tok/s: 13.5411425 versus OPT-033 13.5596962
- owns_opt016_parity_gate: false
- substitutes_for_opt016: false

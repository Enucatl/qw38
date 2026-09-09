# OPT-034 — Packed blockwise Q4_K/Q6_K MMV loads

## Claim labels and proof limits

Live exclusive-RTX-5090 A/B of production batch-1 Q4_K/Q6_K decode MMV with
unchanged packed Q8 staging and FP32 lane/reduction order, candidates
`elementwise` versus `packed` blockwise field/scale loads. Keep requires
**byte equality**, **lower weighted MMV time**, **improved D2048**,
and the **cross-workload guard**. This increment does not substitute for the 2K llama.cpp parity gate. The OPT-035 P D128 D2048 are the keep denominators
(P 1865.21155, D128 15.0562878, D2048 13.5411425), not
historical OPT-026 1746.71973, not OPT-032 1637.58594, not OPT-033 1745.10315,
and not OPT-036 1644.04822.

## Decision

**keep** — `reverted`=false;
`keep_sitting_skipped`=false;
selected_mmv_load_path=packed;
A/B winner packed (elementwise weighted mean 0.162414238 ms,
packed weighted mean 0.0734361857 ms);
tok/s sitting ran.

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp image | `qw38-llama-authority:cuda-13.0.2` |
| llama.cpp revision | `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` |
| A/B | production batch-1 Q4_K/Q6_K MMV shapes, elementwise vs packed, 3 warm + 30 alternating, byte-equal, Q8 staging unchanged |
| P / D128 / D2048 | OPT-021 / OPT-032 protocols, unchanged diagnostics |
| Nsight | not_used |

## Measured sitting

- Device: NVIDIA GeForce RTX 5090 compute 12.0
- measurement_utc: 2026-09-09T20:44:11Z
- P Quartz mean tok/s: 1869.84412 versus OPT-035 1865.21155 (retain ≥95%)
- D128 Quartz mean tok/s: 25.3816128 versus OPT-035 15.0562878
- D2048 Quartz mean tok/s: 20.169548 versus OPT-035 13.5411425 (must improve)
- owns_opt016_parity_gate: false
- substitutes_for_opt016: false

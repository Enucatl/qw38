# OPT-033 — Retain attention value sums in registers

## Claim labels and proof limits

Live exclusive-RTX-5090 A/B of 4096-row production stream-K attention including
combine, candidates `global` versus `registers`. Keep requires
**byte-equal output**, **lower component time**, **improved P**, and the
**cross-workload guard**. This increment does not substitute for the 2K llama.cpp parity gate. The OPT-036 P D128 D2048 are the keep denominators
(P 1644.04822, D128 15.200716, D2048 13.5282431), not
historical OPT-026 1746.71973 and not OPT-032 1637.58594.

## Decision

**keep** — `reverted`=false;
`keep_sitting_skipped`=false;
selected_vkq_accum=registers;
A/B winner registers (global mean 53.8520393 ms,
registers mean 44.8948784 ms);
tok/s sitting ran.

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp image | `qw38-llama-authority:cuda-13.0.2` |
| llama.cpp revision | `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` |
| A/B | 4096 rows, global vs registers, 3 warm + 30 alternating, include combine |
| P / D128 / D2048 | OPT-021 / OPT-032 protocols, unchanged diagnostics |
| Nsight | not_used |

## Measured sitting

- Device: NVIDIA GeForce RTX 5090 compute 12.0
- measurement_utc: 2026-09-09T18:11:19Z
- P Quartz mean tok/s: 1745.10315 versus OPT-036 1644.04822
- D128 Quartz mean tok/s: 15.1528101 versus OPT-036 15.200716
- D2048 Quartz mean tok/s: 13.5596962 versus OPT-036 13.5282431
- owns_opt016_parity_gate: false
- substitutes_for_opt016: false

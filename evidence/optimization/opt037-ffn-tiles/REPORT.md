# OPT-037 — Select 4K FFN tiles per projection

## Claim labels and proof limits

Live exclusive-RTX-5090 A/B of 4096-row Q4_K FFN quality MMA I∈{64,128} ×
J∈{32,64,128} independently for gate, up, and down, with shared-Y staging
preserved and 2D scheduling. Keep requires **admitted component wins**,
**improved P**, **frozen MMQ envelopes**,
and the **cross-workload guard**. This increment does not substitute for the 2K llama.cpp parity gate. The OPT-034 P D128 D2048 are the keep denominators
(P 1869.84412, D128 25.3816128, D2048 20.169548), not
historical OPT-026 1746.71973, not OPT-032 1637.58594, not OPT-033 1745.10315,
not OPT-035 1865.21155, and not OPT-036 1644.04822.

## Decision

**reject** — `reverted`=true;
`keep_sitting_skipped`=true;
any_win=false;
gate winner i128_j128 win=false installed i128_j128; up winner i128_j128 win=false installed i128_j128; down winner i128_j128 win=false installed i128_j128;
tok/s sitting skipped.

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp image | `qw38-llama-authority:cuda-13.0.2` |
| llama.cpp revision | `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` |
| A/B | 4096-row gate/up/down independently, I/J sweep, 3 warm + 30 measured, MMA only |
| P / D128 / D2048 | OPT-021 / OPT-032 protocols, unchanged diagnostics |
| Nsight | not_used |

## Measured sitting

- Device: NVIDIA GeForce RTX 5090 compute 12.0
- measurement_utc: 2026-09-09T22:23:11Z
- P Quartz mean tok/s: n/a versus OPT-034 1869.84412 (must improve)
- D128 Quartz mean tok/s: n/a versus OPT-034 25.3816128
- D2048 Quartz mean tok/s: n/a versus OPT-034 20.169548 (guard ≥95%)
- owns_opt016_parity_gate: false
- substitutes_for_opt016: false

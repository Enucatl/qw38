# OPT-036 — Partition KV for vector decode attention

## Claim labels and proof limits

Live exclusive-RTX-5090 A/B of one-token vector attention at 1/4/8/16 KV
partitions with deterministic FP32 partial-statistic merge. Keep requires
**frozen attention envelopes**, **lower component time** than one-partition
at D2048, **improved D2048** Quartz tok/s versus OPT-032, and the
**cross-workload guard**. This increment does not substitute for the 2K llama.cpp parity gate. The decode-oracle P D128 D2048 are the keep denominators
(P 1637.58594, D128 11.8731956, D2048 3.70951414), not
historical OPT-026 1746.71973.

## Decision

**keep** — `reverted`=false;
`keep_sitting_skipped`=false;
selected below-2048=16;
selected at-or-above-2048=16;
D128 A/B winner 16 (1-part mean 0.803483725 ms);
D2048 A/B winner 16 (1-part mean 12.2984858 ms);
tok/s sitting ran.

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp image | `qw38-llama-authority:cuda-13.0.2` |
| llama.cpp revision | `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` |
| A/B | positions 128 and 2048, candidates 1/4/8/16, 3 warm + 30 alternating |
| P / D128 / D2048 | OPT-021 / OPT-032 protocols, unchanged diagnostics |
| Nsight | not_used |

## Measured sitting

- Device: NVIDIA GeForce RTX 5090 compute 12.0
- measurement_utc: 2026-09-09T17:08:11Z
- P Quartz mean tok/s: 1644.04822 versus OPT-032 1637.58594
- D128 Quartz mean tok/s: 15.200716 versus OPT-032 11.8731956
- D2048 Quartz mean tok/s: 13.5282431 versus OPT-032 3.70951414
- owns_opt016_parity_gate: false
- substitutes_for_opt016: false

# OPT-040 — Hoist prompt GDN inverse normalization

## Claim labels and proof limits

Live exclusive-RTX-5090 A/B of shared per-token/key-head Q/K inverse
norms against repeated warp-column L2. Keep requires
**byte-equal quality outputs/state**, **frozen sequential GDN gates**,
**lower complete GDN component time**, **improved P** versus OPT-034,
**95% throughput floors**, and **105% p95 ceilings**. This increment
does not substitute for the 2K llama.cpp parity gate. Copied denominators
are P 1869.84412, D128 26.1887932, D2048 25.3357754.

## Decision

**keep** — `reverted`=false;
`keep_sitting_skipped`=false;
selected_gdn_inverse_path=shared;
4096 A/B winner shared (repeated 11.9163837 ms,
shared 7.66518307 ms);
tok/s sitting ran.

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp image | `qw38-llama-authority:cuda-13.0.2` |
| llama.cpp revision | `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` |
| A/B | token_count 4096, candidates repeated then shared, 3 warm + 30 alternating |
| Correctness | tokens [1, 2, 3, 4, 63, 64, 65, 512, 2048, 4096]; layers [0, 1, 62] |
| P / D128 / D2048 | OPT-021 / OPT-032 protocols, unchanged diagnostics |
| Nsight | not_used |

## Measured sitting

- Device: NVIDIA GeForce RTX 5090 compute 12.0
- measurement_utc: 2026-09-10T09:47:19Z
- P Quartz mean tok/s: 2060.62183 versus OPT-034 1869.84412
- D128 Quartz mean tok/s: 26.1689129 versus OPT-039 26.1887932
- D2048 Quartz mean tok/s: 25.2998886 versus OPT-039 25.3357754
- owns_opt016_parity_gate: false
- substitutes_for_opt016: false

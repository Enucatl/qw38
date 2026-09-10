# OPT-039 — Warp-owned vector decode attention

## Claim labels and proof limits

Live exclusive-RTX-5090 A/B of warp-owned one-token query-head attention
against the accepted 16-partition CTA kernel. Keep requires
**frozen attention envelopes**, **exact candidate KV and state isolation**,
**lower D2048 component time**, **improved D2048** Quartz tok/s versus OPT-034,
**95% throughput floors**, and **105% p95 ceilings**. This increment
does not substitute for the 2K llama.cpp parity gate. Copied OPT-034
denominators are P 1869.84412, D128 25.3816128, D2048 20.169548.

## Decision

**keep** — `reverted`=false;
`keep_sitting_skipped`=false;
selected_decode_attention_vec=warp_query;
D128 A/B winner warp_query (cta 0.0875509307 ms,
warp 0.0246517342 ms);
D2048 A/B winner warp_query (cta 0.802414954 ms,
warp 0.0774026662 ms);
tok/s sitting ran.

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp image | `qw38-llama-authority:cuda-13.0.2` |
| llama.cpp revision | `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` |
| A/B | positions 128 and 2048, candidates cta_group then warp_query, 3 warm + 30 alternating |
| Correctness | [0, 1, 15, 16, 31, 32, 127, 128, 2047, 2048, 2303, 131071] |
| P / D128 / D2048 | OPT-021 / OPT-032 protocols, unchanged diagnostics |
| Nsight | not_used |

## Measured sitting

- Device: NVIDIA GeForce RTX 5090 compute 12.0
- measurement_utc: 2026-09-10T08:57:52Z
- P Quartz mean tok/s: 1868.51721 versus OPT-034 1869.84412
- D128 Quartz mean tok/s: 26.1887932 versus OPT-034 25.3816128
- D2048 Quartz mean tok/s: 25.3357754 versus OPT-034 20.169548
- owns_opt016_parity_gate: false
- substitutes_for_opt016: false

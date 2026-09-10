# OPT-041 — Give prompt QK microtiles warp ownership

## Claim labels and proof limits

Live exclusive-RTX-5090 A/B of warp-owned 16×8 prompt QK microtiles against
shared `cparts` reduction. Keep requires **byte-equal quality attention outputs**,
**frozen attention gates**, **lower complete component time**, **improved P**
versus OPT-040, **95% throughput floors**, and **105% p95 ceilings**. This
increment does not substitute for the 2K llama.cpp parity gate. Copied
denominators are P 2060.62183, D128 26.1689129, D2048 25.2998886.

## Decision

**keep** — `reverted`=false;
`keep_sitting_skipped`=false;
selected_qk_path=warp_microtile;
4096 A/B winner warp_microtile (cparts 35.5102272 ms,
warp_microtile 34.591423 ms);
tok/s sitting ran.

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp image | `qw38-llama-authority:cuda-13.0.2` |
| llama.cpp revision | `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` |
| A/B | token_count 4096, candidates cparts then warp_microtile, 3 warm + 30 alternating |
| Correctness | tokens [16, 17, 31, 32, 33, 63, 64, 65, 512, 2048, 4096]; starts [0, 1, 31, 128, 2048] |
| P / D128 / D2048 | OPT-021 / OPT-032 protocols, unchanged diagnostics |
| Nsight | not_used |

## Measured sitting

- Device: NVIDIA GeForce RTX 5090 compute 12.0
- measurement_utc: 2026-09-10T11:10:31Z
- P Quartz mean tok/s: 2076.98315 versus OPT-040 2060.62183
- D128 Quartz mean tok/s: 26.1599541 versus OPT-040 26.1689129
- D2048 Quartz mean tok/s: 25.2924843 versus OPT-040 25.2998886
- owns_opt016_parity_gate: false
- substitutes_for_opt016: false

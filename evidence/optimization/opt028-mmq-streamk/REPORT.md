# OPT-028 — Q4_K/Q6_K MMQ stream-K

## Claim labels and proof limits

This increment lands **Q4_K/Q6_K MMQ stream-K** for production quality MMA
under unloosened **Q4_K association**, or retains 2D tiling when the A/B
loses or 4K fails.
Live exclusive-RTX-5090 **4K keep/reject** versus the frozen OPT-026 Quartz mean
1746.71973 tok/s. Numeric and exact-state **envelopes unloosened**. This
increment **does not substitute for the 2K llama.cpp parity gate**.
The **OPT-009 byte-exact reference retained**
(`launch_quant_mmq_variant`).

## Decision

**reject** — Quartz mean tok/s 1729.0459 versus OPT-026 baseline 1746.71973.
A/B winner `off` win=false;
`selected_mmq_stream_k_path`=off;
`reverted`=true; `successor_oracle`=false;
`production_mmq_stream_k_installed`=false;
`ladder_exhausted`=false.

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp image | `qw38-llama-authority:cuda-13.0.2` |
| llama.cpp revision | `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` |
| Tokens | exact 4096, not chat-rendered |
| Quartz token IDs | `(42 + index * 997) % kVocabularySize` |
| Quartz path | production `sync_tokens`, attribution null, graphs created, production fused GDN, cache_policy disabled, 0 warm-ups, 3 cold replicates |
| llama.cpp | `llama-bench -p 4096 -n 0 --no-warmup -r 3 -ngl 99` |
| Same sitting | llama.cpp first, then Quartz |
| Nsight | not_used |
| Keep/reject | strictly greater Quartz mean tok/s than 1746.71973 and A/B optimized win |

## Measured 4K sitting

- Device: NVIDIA GeForce RTX 5090 compute 12.0
- measurement_utc: 2026-09-09T08:06:02Z
- Quartz mean tok/s: 1729.0459 (walls 2365.01831, 2369.67603, 2372.12695 ms; tok/s 1731.91052, 1728.50635, 1726.72046)
- Quartz prompt_tokens: 4096; graphs_created: true; prompt_graph_rows: 4096
- llama.cpp live `avg_ts`: 3231.577696 (`avg_ns` 1267543418, `n_prompt` 4096, `n_batch` 2048, `n_ubatch` 512, `flash_attn` -1, `build_commit` cc83d7b, `test_time` 2026-09-09T08:05:53Z)
- `quartz_meets_llama`: false (informational; not this gate)
- `owns_opt016_parity_gate`: false
- `substitutes_for_opt016`: false
- A/B winner: off; ab_win: false
- `ladder_exhausted`: false

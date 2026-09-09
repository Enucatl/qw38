# OPT-026 — Attention fattn stream-K occupancy

## Claim labels and proof limits

This increment lands **fattn occupancy / stream-K** for prompt attention
under unloosened **OPT-005 envelopes unloosened**, or retains occupancy-1
whole-tile fattn when the A/B loses or 4K fails.
Live exclusive-RTX-5090 **4K keep/reject** versus the frozen OPT-025 Quartz mean
1709.21912 tok/s. Numeric and exact-state **envelopes unloosened**. This
increment **does not substitute for the 2K llama.cpp parity gate**.
The **tiled attention remains the reference**
(`launch_attention_prepare_chunk_tiled`).

## Decision

**keep** — Quartz mean tok/s 1746.71973 versus OPT-025 baseline 1709.21912.
A/B winner `stream_k` win=true;
`selected_fattn_path`=stream_k;
`reverted`=false; `successor_oracle`=true;
`production_fattn_optimized`=true;
`ladder_exhausted`=true.

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
| Keep/reject | strictly greater Quartz mean tok/s than 1709.21912 and A/B optimized win |

## Measured 4K sitting

- Device: NVIDIA GeForce RTX 5090 compute 12.0
- measurement_utc: 2026-09-09T05:51:48Z
- Quartz mean tok/s: 1746.71973 (walls 2342.03125, 2344.67603, 2348.2019 ms; tok/s 1748.90918, 1746.9364, 1744.31335)
- Quartz prompt_tokens: 4096; graphs_created: true; prompt_graph_rows: 4096
- llama.cpp live `avg_ts`: 3253.993621 (`avg_ns` 1258815499, `n_prompt` 4096, `n_batch` 2048, `n_ubatch` 512, `flash_attn` -1, `build_commit` cc83d7b, `test_time` 2026-09-09T05:51:40Z)
- `quartz_meets_llama`: false (informational; not this gate)
- `owns_opt016_parity_gate`: false
- `substitutes_for_opt016`: false
- A/B winner: stream_k; ab_win: true
- `ladder_exhausted`: true

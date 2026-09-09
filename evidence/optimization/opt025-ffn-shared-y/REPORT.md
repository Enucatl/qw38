# OPT-025 — Dense FFN shared-Y and SwiGLU-into-down Q8

## Claim labels and proof limits

This increment lands **dense FFN shared-Y and SwiGLU-into-down Q8** for prompt
Q4_K gate/up/down under unloosened **Q4_K association**, or retains per-GEMM
`matrix_prompt` plus BF16 SwiGLU when the A/B loses or 4K fails.
Live exclusive-RTX-5090 **4K keep/reject** versus the frozen OPT-023 Quartz mean
1687.86169 tok/s. Numeric and exact-state **envelopes unloosened**. This
increment **does not substitute for the 2K llama.cpp parity gate**.
The **OPT-009 byte-exact reference retained** (`launch_q8_mmq_bf16_variant`
versus `launch_q8_mmq_bf16_reference`).

## Decision

**keep** — Quartz mean tok/s 1709.21912 versus OPT-023 baseline 1687.86169.
A/B winner `shared_y_swiglu_q8` win=true;
`selected_ffn_path`=shared_y_swiglu_q8;
`reverted`=false; `successor_oracle`=true;
`production_ffn_optimized`=true.

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
| Keep/reject | strictly greater Quartz mean tok/s than 1687.86169 and A/B optimized win |

## Measured 4K sitting

- Device: NVIDIA GeForce RTX 5090 compute 12.0
- measurement_utc: 2026-09-09T05:14:45Z
- Quartz mean tok/s: 1709.21912 (walls 2394.2688, 2396.81519, 2398.16797 ms; tok/s 1710.75195, 1708.93445, 1707.97046)
- Quartz prompt_tokens: 4096; graphs_created: true; prompt_graph_rows: 4096
- llama.cpp live `avg_ts`: 3266.276516 (`avg_ns` 1254072667, `n_prompt` 4096, `n_batch` 2048, `n_ubatch` 512, `flash_attn` -1, `build_commit` cc83d7b, `test_time` 2026-09-09T05:14:37Z)
- `quartz_meets_llama`: false (informational; not this gate)
- `owns_opt016_parity_gate`: false
- `substitutes_for_opt016`: false
- A/B winner: shared_y_swiglu_q8; ab_win: true

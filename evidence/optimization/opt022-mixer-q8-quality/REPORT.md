# OPT-022 — Mixer Q8_0 quality MMQ + shared residual Y

## Claim labels and proof limits

This increment lands **mixer Q8_0 quality MMQ** with **shared residual Y**
under the plan.md **Q8 association** rule. Live exclusive-RTX-5090 **4K keep/reject**
versus the frozen OPT-021 Quartz mean 967.267761 tok/s. Numeric and exact-state
**envelopes unloosened**. This increment **does not substitute for the 2K llama.cpp parity gate**.
The **OPT-009 byte-exact reference retained** (`launch_q8_mmq_bf16_variant` versus
`launch_q8_mmq_bf16_reference`).

## Decision

**keep** — Quartz mean tok/s 1680.80627 versus OPT-021 baseline 967.267761.
`reverted`=false; `successor_oracle`=true;
`production_q8`=quality_mma_shared_y; `shared_residual_y`=true.

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
| Keep/reject | strictly greater Quartz mean tok/s than 967.267761 |

## Measured 4K sitting

- Device: NVIDIA GeForce RTX 5090 compute 12.0
- measurement_utc: 2026-09-09T02:42:58Z
- Quartz mean tok/s: 1680.80627 (walls 2433.14404, 2437.7627, 2439.88013 ms; tok/s 1683.41858, 1680.22913, 1678.771)
- Quartz prompt_tokens: 4096; graphs_created: true; prompt_graph_rows: 4096
- llama.cpp live `avg_ts`: 3227.546524 (`avg_ns` 1269126325, `n_prompt` 4096, `n_batch` 2048, `n_ubatch` 512, `flash_attn` -1, `build_commit` cc83d7b, `test_time` 2026-09-09T02:42:50Z)
- `quartz_meets_llama`: false (informational; not this gate)
- `owns_opt016_parity_gate`: false
- `substitutes_for_opt016`: false
- kernel_nodes: 2; block: [32, 8, 1]

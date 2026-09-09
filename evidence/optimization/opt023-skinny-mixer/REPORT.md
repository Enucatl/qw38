# OPT-023 — Skinny-M mixer dispatch for small output rows

## Claim labels and proof limits

This increment lands **skinny-M mixer dispatch** for mixer Q8_0 GEMMs with
`output_rows < 128` (GDN α/β) under the plan.md **Q8 association** rule, or
retains OPT-022 I=128 quality MMA when the A/B loses or 4K fails.
Live exclusive-RTX-5090 **4K keep/reject** versus the frozen OPT-022 Quartz mean
1680.80627 tok/s. Numeric and exact-state **envelopes unloosened**. This
increment **does not substitute for the 2K llama.cpp parity gate**.
The **OPT-009 byte-exact reference retained** (`launch_q8_mmq_bf16_variant`
versus `launch_q8_mmq_bf16_reference`).

## Decision

**keep** — Quartz mean tok/s 1687.86169 versus OPT-022 baseline 1680.80627.
A/B winner `mma_i32_j128` win=true;
`selected_skinny_mixer_path`=mma_i32_j128;
`reverted`=false; `successor_oracle`=true;
`production_skinny`=true.

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
| Keep/reject | strictly greater Quartz mean tok/s than 1680.80627 and A/B skinny win |

## Measured 4K sitting

- Device: NVIDIA GeForce RTX 5090 compute 12.0
- measurement_utc: 2026-09-09T03:49:04Z
- Quartz mean tok/s: 1687.86169 (walls 2422.67822, 2427.59326, 2429.95752 ms; tok/s 1690.69092, 1687.26782, 1685.62622)
- Quartz prompt_tokens: 4096; graphs_created: true; prompt_graph_rows: 4096
- llama.cpp live `avg_ts`: 3232.279094 (`avg_ns` 1267269105, `n_prompt` 4096, `n_batch` 2048, `n_ubatch` 512, `flash_attn` -1, `build_commit` cc83d7b, `test_time` 2026-09-09T03:48:55Z)
- `quartz_meets_llama`: false (informational; not this gate)
- `owns_opt016_parity_gate`: false
- `substitutes_for_opt016`: false
- A/B winner: mma_i32_j128; ab_win: true

# OPT-024 — Blackwell-aligned Q8_0 D2R for large mixer GEMMs

## Claim labels and proof limits

This increment lands **aligned-SoA D2R for large mixer Q8_0** GEMMs under the
plan.md **Q8 association** rule when it beats quality MMQ, or retains OPT-022
I=128 quality MMA when the A/B loses or 4K fails.
Live exclusive-RTX-5090 **4K keep/reject** versus the frozen OPT-023 Quartz mean
1687.86169 tok/s. Numeric and exact-state **envelopes unloosened**. This
increment **does not substitute for the 2K llama.cpp parity gate**.
The **OPT-009 byte-exact reference retained** (`launch_q8_mmq_bf16_variant`
versus `launch_q8_mmq_bf16_reference`).

## Decision

**reject** — Quartz mean tok/s 1684.9541 versus OPT-023 baseline 1687.86169.
A/B winner `quality_mma` win=false;
`selected_large_mixer_q8_path`=quality_mma;
`reverted`=true; `successor_oracle`=false;
`production_d2r`=false.

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
| Keep/reject | strictly greater Quartz mean tok/s than 1687.86169 and A/B D2R win on every large mixer shape |

## Measured 4K sitting

- Device: NVIDIA GeForce RTX 5090 compute 12.0
- measurement_utc: 2026-09-09T04:20:55Z
- Quartz mean tok/s: 1684.9541 (walls 2427.84814, 2431.31372, 2433.62549 ms; tok/s 1687.0907, 1684.68591, 1683.08557)
- Quartz prompt_tokens: 4096; graphs_created: true; prompt_graph_rows: 4096
- llama.cpp live `avg_ts`: 3207.633195 (`avg_ns` 1277028847, `n_prompt` 4096, `n_batch` 2048, `n_ubatch` 512, `flash_attn` -1, `build_commit` cc83d7b, `test_time` 2026-09-09T04:20:46Z)
- `quartz_meets_llama`: false (informational; not this gate)
- `owns_opt016_parity_gate`: false
- `substitutes_for_opt016`: false
- A/B winner: quality_mma; ab_win: false

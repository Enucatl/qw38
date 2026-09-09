# OPT-021 — Cold 4K prefill keep/reject oracle

## Claim labels and proof limits

This increment pins the **4K keep/reject oracle**. Numeric and exact-state
**envelopes unloosened**. This protocol **does not substitute for the 2K llama.cpp parity gate**.
Timed Quartz walls use **attribution null** and **graphs created**.
The **scout sitting is not the retained fixture**.

Live numbers in `fixtures/opt021_oracle.json`, `llama-bench-4k.json`, and
`quartz-4k.json` are the Measured same-sitting comparison. Scout Quartz
965.204895 tok/s and llama.cpp 3182.476587 tok/s are contract
transparency only.

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
| Keep/reject | strictly greater Quartz mean tok/s than this retained baseline |

## Measured 4K oracle (implementation sitting)

- Device: NVIDIA GeForce RTX 5090 compute 12.0
- measurement_utc: 2026-09-09T01:03:49Z
- Quartz mean tok/s: 967.267761 (walls 4227.67725, 4236.86816, 4239.29785 ms; tok/s 968.853516, 966.751831, 966.197754)
- Quartz prompt_tokens: 4096; graphs_created: true; prompt_graph_rows: 4096
- llama.cpp live `avg_ts`: 3243.626016 (`avg_ns` 1262836272, `n_prompt` 4096, `n_batch` 2048, `n_ubatch` 512, `flash_attn` -1, `build_commit` cc83d7b, `test_time` 2026-09-09T01:03:41Z)
- `quartz_meets_llama`: false (informational; not this gate)
- `owns_opt016_parity_gate`: false
- `substitutes_for_opt016`: false

## Scout versus retained

Scout sitting 2026-09-09 Quartz 965.204895 tok/s and llama.cpp 3182.476587
tok/s are recorded on the contract. The scout sitting is not the retained fixture.
This report's live numbers are the retained OPT-022+ baseline.

## Keep/reject rule for later production changes

A later production change is kept only if its cold exact-4096 mean tok/s is
strictly greater than the retained OPT-021 `quartz.mean_tok_s` (or a later
successor published under the same protocol). Equality is a reject.

Ladder stop (not this gate): later tasks may stop when Quartz cold 4K mean
tok/s is at or above same-protocol llama.cpp `avg_ts`.

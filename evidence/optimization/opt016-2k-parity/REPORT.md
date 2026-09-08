# OPT-016 — 2K prefill parity

## Claim labels and proof limits

This increment uses the **2K yardstick only**. Numeric and exact-state **envelopes unloosened**. There is **no 8K/32K/128K throughput gate**. **Q8_0 remains byte-exact** on the OPT-009 production kernel.

Live numbers in `fixtures/opt016_parity.json`, `llama-bench-2k.json`, and `quartz-2k.json` are the Measured comparison. Historical llama.cpp `avg_ts` 3114.049476 is recorded for transparency and is not a second bar.

## Ranks

Landed: `q4k_q6k_mma_mmq` (J=128), `gdn_fused_token_loop`, `causal_mma_attention`.

Skipped: `optional_2048_prompt_graphs` (Rank 4). Live attribution at 2048 tokens has `graph` 0 ms and `other_idle` 0.467773438 ms (0.005% of 9707.78711 ms wall), below the 1% Rank-4 trigger.

## Protocol

Exact 2048 tokens, unperturbed Quartz `sync_tokens` (attribution null), three cold replicates, graphs created, default production GDN path, cache_policy disabled, exclusive RTX 5090, llama.cpp `llama-bench -p 2048 -n 0 --no-warmup -r 3 -ngl 99` in the same sitting.

## Measured 2K gate (implementation sitting)

- Quartz mean tok/s: 208.758591 (walls 9748.28125, 9825.80566, 9857.68262 ms)
- llama.cpp live `avg_ts`: 3183.528255 (`avg_ns` 643416735, `build_commit` cc83d7b)
- `gate_passed`: false (15.25× behind live llama.cpp)

## Post-rank attribution (not the speed denominator)

`evidence/optimization/opt016-2k-parity/attribution.json` on the same GGUF/GPU after Ranks 1–3:

- ffn_mmq 2551.3667 ms (26.3%)
- gdn 5185.85205 ms (53.4%)
- attention 1966.91724 ms (20.3%)
- graph 0 ms; other_idle 0.467773438 ms

gdn+attention is 73.7% of the remaining 2K wall. Mixer projections stay on `launch_q8_mmq_bf16`. Implementation stopped expanding kernels here.

## Discovered ledger work (no ID assigned)

Admit a production mixer Q8_0 MMA path that keeps the existing OPT-009 byte-exact kernel as the visible reference without loosening that reference, then re-run the frozen 2K gate. 8K/32K/128K throughput and QLT-001 remain later ledger rows.

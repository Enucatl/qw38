# OPT-095 — Share decode KV across six query heads

Status: **performance_rejected**. Authority llama.cpp
`cc83d7b4824f73cfdda4dfbb47ee39804f71b328`, GGUF SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`. Control is in-kernel `warp_query`
with one warp per query head. Candidate `warp_query_gqa6` shares K/V slabs
across six query heads per KV head with block=(32,6) grid=(4,16). n_parts stays
16. No prep launch for either path.

`claims_throughput: true` only when production_kept. Require >=0.10 ms/token
complete 16-layer attention saving and a positive paired interval, or measured
rejection.

## Eligibility

OPT-090 gate: eligible=True; verdict=proceed;
attention_ms=9.0920.

## Numeric policy

Bitwise parity versus warp_query on parity positions. Sampled FP64 uses four
query heads x eight output dimensions. Nonfinite=None.

## Dispatch

Launches: [{"config": "warp_query", "dispatch": {"gqa_block_y": 1, "launch": "warp_query_decode_attention", "n_parts": 16, "path": "warp_query", "prep_block": 32, "prep_grid": 24, "prep_launches": 0, "prepared_q": false, "vec_kv": false}, "mean_ms": 9.6588675}, {"config": "warp_query_gqa6", "dispatch": {"gqa_block_y": 6, "launch": "warp_query_gqa6_decode_attention", "n_parts": 16, "path": "warp_query_gqa6", "prep_block": 32, "prep_grid": 4, "prep_launches": 0, "prepared_q": false, "vec_kv": false}, "mean_ms": 10.104505399999999}]

## Complete 16-layer attention

Control warp_query vs candidate warp_query_gqa6.
Component n=10 means: control 9.659 ms,
candidate 10.105 ms.
Paired CI: -0.5994 .. -0.2919 ms.

## Quality (OPT-073)

quality-v3 engine non-regression=None.

## Decision

Verdict: **performance_rejected** (['negative_complete_saving']).
production_kept=False.
Shipping decode attention GQA stays `warp_query`.

# OPT-103 — Port the one-query vector attention specialization

Status: **performance_rejected**. Authority llama.cpp
`cc83d7b4824f73cfdda4dfbb47ee39804f71b328`, GGUF SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`. Control is in-kernel `warp_query`
(32 threads). Candidate `vec128_online` uses 128 threads, register Q, vectorized
BF16 K/V, online softmax and register V accumulation. Partition count screened
from {4,8,16}; selected_n_parts=16.
`warp_query_gqa6` stays rejected.

`claims_throughput: true` only when production_kept. Require >=0.10 ms/token
complete 16-layer attention saving and a positive paired interval, or measured
rejection.

## Eligibility

OPT-090 gate: eligible=True; verdict=proceed;
attention_ms=9.0920;
llama_dispatch=flash_attn_ext_vec<256,1>.

## Partition screen

{"p128_n16": {"candidate_mean_ms": 1.7836373333333333, "control_mean_ms": 1.6447146666666665, "dispatch": {"gqa_block_y": 4, "launch": "vec128_online_decode_attention", "n_parts": 16, "path": "vec128_online", "prep_block": 32, "prep_grid": 24, "prep_launches": 0, "prepared_q": false, "vec_kv": false}, "n_parts": 16, "position": 128, "saving_ms": -0.13892266666666675}, "p128_n4": {"candidate_mean_ms": 1.7756266666666667, "control_mean_ms": 1.6447146666666665, "dispatch": {"gqa_block_y": 4, "launch": "vec128_online_decode_attention", "n_parts": 4, "path": "vec128_online", "prep_block": 32, "prep_grid": 24, "prep_launches": 0, "prepared_q": false, "vec_kv": false}, "n_parts": 4, "position": 128, "saving_ms": -0.13091200000000014}, "p128_n8": {"candidate_mean_ms": 1.7567146666666666, "control_mean_ms": 1.6447146666666665, "dispatch": {"gqa_block_y": 4, "launch": "vec128_online_decode_attention", "n_parts": 8, "path": "vec128_online", "prep_block": 32, "prep_grid": 24, "prep_launches": 0, "prepared_q": false, "vec_kv": false}, "n_parts": 8, "position": 128, "saving_ms": -0.1120000000000001}, "p2048_n16": {"candidate_mean_ms": 8.622325666666667, "control_mean_ms": 9.244778666666667, "dispatch": {"gqa_block_y": 4, "launch": "vec128_online_decode_attention", "n_parts": 16, "path": "vec128_online", "prep_block": 32, "prep_grid": 24, "prep_launches": 0, "prepared_q": false, "vec_kv": false}, "n_parts": 16, "position": 2048, "saving_ms": 0.6224530000000001}, "p2048_n4": {"candidate_mean_ms": 9.161631999999999, "control_mean_ms": 9.244778666666667, "dispatch": {"gqa_block_y": 4, "launch": "vec128_online_decode_attention", "n_parts": 4, "path": "vec128_online", "prep_block": 32, "prep_grid": 24, "prep_launches": 0, "prepared_q": false, "vec_kv": false}, "n_parts": 4, "position": 2048, "saving_ms": 0.08314666666666781}, "p2048_n8": {"candidate_mean_ms": 8.914890999999999, "control_mean_ms": 9.244778666666667, "dispatch": {"gqa_block_y": 4, "launch": "vec128_online_decode_attention", "n_parts": 8, "path": "vec128_online", "prep_block": 32, "prep_grid": 24, "prep_launches": 0, "prepared_q": false, "vec_kv": false}, "n_parts": 8, "position": 2048, "saving_ms": 0.3298876666666679}}

## Numeric policy

FP64 sampled heads/dims plus per-head envelope versus warp_query.
Nonfinite=0.

## Complete 16-layer attention

Control warp_query vs candidate vec128_online n_parts=16.
Component n=10 means: control 9.263 ms,
candidate 8.873 ms.
Paired CI: 0.0156 .. 0.7645 ms.

## Quality (OPT-073)

quality-v3 engine non-regression=pass.

## P4096

control_tok_s=3063.2619228346375 candidate_tok_s=3059.6966080738252
tok/s delta vs OPT-098 P4096 3046.23: 0.

## Decision

Verdict: **performance_rejected** (['d128_saving_below_0_10_ms']).
production_kept=False.
Shipping decode attention stays `warp_query`.

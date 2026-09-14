# OPT-130 — Implement the evidenced dense decode-attention improvement

Status: **reject**. Shipping decode attention `hybrid_crossover`. Hardware executed.

Parent is authenticated post124 combined plus OPT-127 `decode_segments8`. Primary OPT-129 transfer `occupancy_partition_or_gqa_kv_reuse` is frozen as `hybrid_crossover@1024_vec128_open_nparts8`: occupancy-snapped legal `n_parts=8` (llama D2048 raw ~9) and `vec128_online` past verified_max 4096. GQA6 stays rejected. Rank-2 `llama_mma_decode_consumer` was not launched (unmatched BF16 MMA consumer; shadow F16 cache forbidden). Dense BF16 KV, causal visibility, partial RoPE, GQA mapping and output gates are unchanged.

## Frozen dispatch

second_candidate_launched=`False`. gqa6_admitted=`False`. Graph topology remains two-valued (warp_query vs vec128_online); opening verified_max maps prefixes >4096 onto topology 1. `n_parts` is frozen per topology (not per-token occupancy) so OPT-126/127 captured grids stay representable. Eager fallback is preserved.

## Primitive component

primitive ok=`True`.

| prefix | parent ms | candidate ms | saving ms |
|---:|---:|---:|---:|
| 128 | 0.0350207984 | 0.0354624018 | -0.000441603363 |
| 1023 | 0.0930752009 | 0.0933376029 | -0.000262401998 |
| 1024 | 0.110931203 | 0.0732671991 | 0.0376640037 |
| 2048 | 0.112409599 | 0.111884803 | 0.000524796546 |
| 8192 | 0.569331169 | 0.374316812 | 0.195014358 |
| 32768 | 3.07454085 | 1.75406718 | 1.32047367 |

## Quality

OPT-058 invoked=`True`; candidate NLL measured=`True`; held-out PPL ratio=`1.0`; wikitext PPL ratio=`1.0`.

## Engine tok/s versus parent (decode_segments8)

| workload | parent decode-only | candidate decode-only | delta | parent request | candidate request | delta | geo | CI lower | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| d128 | 59.152732470000004 | 59.15483666 | 0.0021041899999971747 | 57.96904448 | 57.96794167 | -0.0011028099999990104 | 0.9999809666175016 | 0.9997517817002282 | 1.0002689453146059 |
| d2048 | 49.968972789999995 | 51.06167107 | 1.0926982800000076 | 44.1504654 | 45.00633965 | 0.8558742499999994 | 1.0193854020835318 | 1.0192701018051358 | 0.9654837060965445 |
| p4096 | 2895.557764 | 2895.441723 | -0.116041000000223 | 2895.557764 | 2895.441723 | -0.116041000000223 | 0.9999600481713709 | 0.9994616630504249 | 0.0 |

## Verdict

`reject`. reasons=`['d128_throughput_gate', 'measured_loss_or_no_request_win']`. On reject, production pins stay `hybrid_crossover@1024` / n_parts=16 / verified_max=4096 and tok/s speedup versus the sitting parent is 0.

claims_throughput: `False`.


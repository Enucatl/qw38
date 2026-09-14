# OPT-137 — Bounded long-context dense attention MMA candidate

Status: **keep**. Shipping decode attention `dense_bf16_tile_f16_mma_decode_v1`. Hardware executed.

Parent is authenticated post124 combined plus OPT-127 `decode_segments8`. Candidate `dense_bf16_tile_f16_mma_decode_v1` adapts pinned llama fattn-mma-f16 organization for DKQ=DV=256, ncols1=1/ncols2=8 over dense BF16 KV with in-tile F16 conversion. No persistent F16 shadow cache. Production pins stay hybrid_crossover unless keep.

## Quality

OPT-058 invoked=`True`; candidate NLL measured=`True`; held-out PPL ratio=`1.0`; wikitext PPL ratio=`1.0`; long-cache MMA launches=`[48, 48, 48]`.

## Keep policy

`target_guard_v2` verdict `keep`. reasons=`[]`.

claims_throughput: `True`.

tok/s deltas: `{'d8192.decode_only': {'control_rate': 34.90564309, 'candidate_rate': 48.65922203, 'delta': 13.753578940000004, 'g': 1.3940209895173834}, 'd32768.decode_only': {'control_rate': 15.302007380000001, 'candidate_rate': 33.93855744, 'delta': 18.636550059999998, 'g': 2.217915272525001}}`.


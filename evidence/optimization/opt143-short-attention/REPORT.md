# OPT-143 — Production-matched short-decode attention

Status: **no_opportunity**. Shipping decode attention `dense_bf16_tile_f16_mma_decode_v1`. Parent retained on no-keep.

Parent `post124_plus_opt127_decode_segments8_plus_opt137_mma`. Short-decode production path at D2048 is `vec128_online_decode_attention` (hybrid crossover @ 1024, verified_max 4096, n_parts 16). OPT-137 MMA remains at positions `>= 8192`. Rejected OPT-130 occupancy `n_parts=8` path is not reopened.

## Frozen candidate

Candidate: `None`.
Supported mechanism: `None`.
Admission reason: `throughput_alone_cannot_establish_mechanism`.
Reasons: `['throughput_alone_cannot_establish_mechanism', 'throughput_occupancy_dram_alone_cannot_freeze_candidate', 'opt138_expected_benefit_unknown_until_mechanism', 'opt141_llama_mechanism_null', 'warp_query_rejected_as_d2048_evidence']`.

Freeze requires matched positive family excess **and** a source-grounded mechanism for the production dispatch. Occupancy, DRAM throughput, and byte counters alone cannot freeze a kernel. A warp_query replay is not D2048 evidence.

## Family excess (OPT-138)

D2048 `attn_core` middle-window mean `47.51016466666667` ms (n=`3`, one-sided low `47.38279750168242` ms). mean_ms is over twelve middle-window evals, not per eval. Significant positive excess: `True`. D128 `attn_core` mean `4.297360666666666` ms, positive=`True`.

## Production identity (OPT-139)

D2048 path `vec128_online` kernel `vec128_online_decode_attention`. D128 kernel `warp_query_decode_attention`. Warp-query rejected as D2048 evidence: `True`.

## Typed Quartz D2048 counters (OPT-139)

| Slot | Value |
|---|---|
| dram_read_bytes | `8.85` |
| dram_throughput | `2.33` |
| sm_throughput | `5.43` |
| occupancy | `15.62` |
| tensor_activity | `0.0` |
| l2_traffic | `1932797.0` |

Source observations: `False`; SASS observations: `False`.

## Matched llama comparison (OPT-141)

Comparable D2048 attn_core: `True`. Llama kernel `flash_attn_ext_vec`. Quartz kernel `vec128_online_decode_attention`.

| Slot | Quartz | Llama | Unit |
|---|---|---|---|
| `dram_read_bytes` | 8.85 | 9.5 | Mbyte |
| `dram_write_bytes` | 0.0 | 0.0 | byte |
| `l2_traffic` | 1932797.0 | 1803992.0 | sector |
| `dram_throughput` | 2.33 | 29.8 | % |
| `sm_throughput` | 5.43 | 11.51 | % |
| `tensor_activity` | 0.0 | 0.0 | cycle |
| `achieved_occupancy` | 15.62 | 10.8 | % |

Llama `supported_mechanism` remains null. Throughput/occupancy differences are not a named source/SASS-backed change at `vec128_online_decode_attention`.

## Quality

OPT-058 invoked=`False`; candidate NLL measured=`False`; NLL required=`False`; skip_reason=`no_candidate_no_arithmetic_change`. Changed arithmetic requires measured candidate NLL; no_opportunity does not change arithmetic and does not borrow parent NLL.

## Keep policy

`target_guard_v2` target `d2048.decode_only` with complete_request as the target additional guard. Guards: D128, D8192, D32768 decode and P4096 prefill.

Verdict `no_opportunity`. production_kept=`False`.
claims_throughput: `False`.

## Deltas

Candidate measured delta: `None`.
Shipping delta: `0` (zero on no_opportunity/reject; parent retained).
Quality result: `nll_not_required_no_arithmetic_change`.

## Performance evidence checklist

1. **Measurement identity** — Quartz production `decode_segments8` + kept OPT-137 MMA; D2048 component replay `vec128_online_decode_attention`; GGUF `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`; llama revision `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`.
2. **Coverage** — OPT-138 middle-window family excess; OPT-139 target-kernel NCU `--launch-count 1`; OPT-141 matched llama `flash_attn_ext_vec`.
3. **Time accounting** — +47.51 ms is twelve middle-window evals, not per eval; enclosing replay totals are not mixed into typed slots.
4. **Contradiction register** — none between OPT-139 identity, D2048 replay dispatch, and OPT-141 comparable slots. Warp-query as D2048 evidence is rejected (`kernel_selector_mismatch`).
5. **Claim types** — family excess `measured`; D2048 identity `measured`; mechanism `unknown`/`incomplete`; no_opportunity `measured` from absent source/SASS, not from skipping collection.
6. **Target/guard** — OPT-135 `target_guard_v2` opted in; unused for keep because no candidate ran AB/BA.
7. **Independent verification** — verifier PASS (2026-09-15T01:05:00Z): freeze `no_opportunity` with OPT-138/139/141 evidence; quality short-circuit `nll_required=false`, `candidate_nll_not_measured=false`; parent retained; no production change.
8. **Reporting** — candidate measured delta N/A; shipping delta 0; quality N/A (no arithmetic change).

## Raw gates

Sidecars: [`raw/`](raw/) (`preflight.json`, `freeze.json`, phase skips, `report.json`). Structured freeze: [`freeze.json`](freeze.json). Fixture dump: [`answers.json`](answers.json) and `fixtures/opt143_short_attention.json`.

## Status

verdict=`no_opportunity` production_kept=`False` blocked=`False`.
No production kernel or selector change. Independent verification: **PASS** (2026-09-15T01:05:00Z).


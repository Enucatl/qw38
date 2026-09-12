# OPT-099 — Close matched engine attribution blind spots

This increment **claims no performance improvement** and makes **no tok/s
speedup claim**. `claims_throughput` is false. Diagnostic CUDA events are
**not** shipping CUDA-graph timings. Production selectors are unchanged.

Authority llama.cpp `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`.
GGUF SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`.
Token generator `(42 + index * 997) % 248320`.

## Frozen post098_selected

ok=True mismatches={}.
observed={'q4_decode': 'integer_q8_late', 'q4_warps_per_row': 4, 'ffn_decode': 'paired_integer', 'q8_grouping': 'grouped_r1_w4', 'q6_decode': 'integer_q8_1', 'q6_warps_per_row': 2, 'gdn_decode': 'sequential', 'decode_query_prep': 'warp_query', 'prompt_mmq': 'i128_j128_fma_async_x', 'prompt_mmq_wait': 'joined_wait', 'execution_graphs': 'ffn_only', 'mmq_pipeline': 'fma_async', 'mmq_async_x': True, 'ffn_tiles': 'i128_j128'}.

## Conservation

Call counts ok=True.
FFN down first-class=True.
Dropped events rejected=True.

## D128 / D2048

D128 valid=True p95=18.157485966666666.
D2048 valid=True p95=19.7013944.
Uninstrumented wall/ITL/p95 and eager traces are separate records.
Prefix, load, capture, warmup and reset stay outside measured windows.

## Counters

ncu_available=True.
full_ncu_sweep=False.
reason=queried_without_full_set.

## P4096

P4096 valid=True.
Prefill ran in a separate process/sitting.

## Ranking

status=ranked.
gap_attribution_complete=True.
reasons=[].
Matched llama excess is reported only when llama covered the family.
Missing llama coverage is `unknown` / null, never filled with zero.
Families remain non-additive; union and overlap are reported separately.

## Triggers for OPT-100–105

opt100_q8_aligned: go=True reason=matched_excess excess_ms=17.923381116289377.
opt101_gdn_transpose: go=False reason=llama_family_unmapped excess_ms=None.
opt102_q4_repack: go=True reason=matched_excess excess_ms=23.206335663532954.
opt103_attention_vec: go=False reason=llama_family_unmapped excess_ms=None.
opt104_q6_aligned: go=True reason=matched_excess excess_ms=10.569685420933318.
opt105_mmq_x2: go=False reason=excess_below_trigger excess_ms=-81.09209577033357.

## Proof limits

- No throughput or keep claim.
- Eager events are diagnostic.
- FFN down is a first-class family.
- Unexplained wall above 5% stops ranking.
- Do not subtract replay milliseconds from full-engine milliseconds.

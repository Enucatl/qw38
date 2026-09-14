# OPT-136 — Graph accounting and matched production decode

Diagnostics only. `claims_throughput=false`. Shipping delta: N/A. NLL: N/A.

Measured at `2026-09-14T15:52:46Z`. Image `qw38-cuda:13.0.2`.
nsys `NVIDIA Nsight Systems version 2025.3.2.474-253236389321v0`. GPU available `True`.

## Historical OPT-133 sqlite audit

Raw timestamps remain integer nanoseconds. Units come from the export schema.

| Quantity | Expected d128-early | Observed |
|---|---:|---:|
| ordinary_kernel_count | 84 | 84 |
| gpu_graph_count | 96 | 96 |
| kernel_copy_union_ms | 8.24 | 8.24 |
| graph_envelope_union_ms | 186.02 | 186.02 |
| combined_union_ms | 194.26 | 194.26 |
| first_to_last_span_ms | 196.74 | 196.74 |
| outside_combined_envelopes_ms | 2.48 | 2.48 |

Graph envelopes are not leaf kernel time. Internal idle on graph-only
captures is null. `min(old_unobserved, new_idle)` is non-causal.

SQL: `{'kernel': 'SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL', 'memcpy': 'SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY', 'graph': 'SELECT start, end FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE', 'units': 'export_schema / CUPTI documented nanoseconds'}`.

## Coverage

validation ok=`True`
windows=`150`

## Matched decode ratios (identity-matched decode_only, capacity 131072)

- d128: status=`measured` quartz_tok_s=`59.66978170811928` llama_tok_s=`70.69998668222141` ratio=`0.8439857559849323`
- d2048: status=`measured` quartz_tok_s=`50.1397791965909` llama_tok_s=`69.90413144527776` ratio=`0.7172648906429979`
- d8192: status=`measured` quartz_tok_s=`34.939692063358706` llama_tok_s=`67.29221750384848` ratio=`0.5192233717273544`
- d32768: status=`measured` quartz_tok_s=`15.31734670673004` llama_tok_s=`62.79936179795327` ratio=`0.24390927340967433`

## Contradictions

- OPT-108/129 component times come from different sittings than this
  matched decode probe; they are not used as current denominators.
- OPT-115/132 long-context rates used mixed request vs decode-only
  identities in older reports; this harness does not promise to
  reproduce OPT-132 ratios under the new matched output/capacity boundary.

## Status

historical_ok=`True` preflight_ok=`True`
baseline_ok=`True` gpu_blocked=`False`

Unknown causal attribution may remain explicitly unknown. Missing
required captures/coverage are blocked, not done.

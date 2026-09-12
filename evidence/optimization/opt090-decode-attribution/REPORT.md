# OPT-090 — Measure the remaining decode gap

This increment **claims no performance improvement** and makes **no tok/s
speedup claim**. `claims_throughput` is false. Eager diagnostic CUDA
events are **not** shipping CUDA-graph timings. Production selectors
are unchanged.

Authority llama.cpp `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`.
GGUF SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`.
Token generator `(42 + index * 997) % 248320`.

## Idents

Quartz idents: opt088_control, opt089_selected.
Measured: opt088_control, opt089_selected.
measure_once=False.
Pinned llama is a separate matched control; local `../llama.cpp` HEAD
is not the authority engine.

## Conservation

Call counts ok=True.
Dropped events rejected=True.
Unexplained wall limit=0.05.
Replay non-additive audit recorded=True.

## D128 / D2048

D128 valid=True p95=22.184889783333333.
D2048 valid=True p95=23.794095683333335.
Prefix, load, capture, warmup and reset stay outside measured windows.

## Counters

ncu_available=True.
full_ncu_sweep=False.
reason=queried_without_full_set.
Counter absence is explicit; missing counters do not invent bandwidth.

## P4096

P4096 valid=False.
Prefill attribution follows decode and is secondary.

## Ranking

status=ranked.
gap_attribution_complete=True.
reasons=[].
Matched llama excess is reported only when llama covered the family.
Non-additive ceilings are not summed into a token budget.

## Conditional triggers

gdn_reopen_eligible=False reason=no_opt077_timing_capture_defect_repaired.
graph_reopen_eligible=True reason=unhidden_idle_at_both_prefixes.
A newer GPU sitting or broad CI alone cannot reopen GDN.

## Proof limits

- No throughput or keep claim.
- Eager events are diagnostic.
- 64 paired FFN calls, not 128.
- Unexplained wall above 5% stops ranking.
- Do not subtract replay milliseconds from full-engine milliseconds.

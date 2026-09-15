# OPT-138 — Remaining short-decode and prefill excess versus llama

Diagnostics only. `claims_throughput=false`. Shipping delta: N/A. NLL: N/A.

Measured at `2026-09-14T18:51:01Z`. Image `qw38-cuda:13.0.2`.
Parent `post124_plus_opt127_decode_segments8_plus_opt137_mma`. GPU available `True`.

## Identity

pins ok=`True`
opt137 MMA=`True`

## Baseline Quartz vs llama

- d128 metric=`decode_only_ms` quartz_ms=`4286.434619000001` llama_ms=`3617.388042` quartz_tok_s=`59.723295175262294` llama_tok_s=`70.76929459258714`
- d2048 metric=`decode_only_ms` quartz_ms=`5103.649609` llama_ms=`3659.347468` quartz_tok_s=`50.16018332225597` llama_tok_s=`69.95782779270088`
- p4096 metric=`prefill_ms` quartz_ms=`1349.558497` llama_ms=`1262.4316569999999` quartz_tok_s=`3035.0666600263717` llama_tok_s=`3244.5320721230937`

## Family ranking

decode selected=`[{'family': 'attn_core', 'score_ms': 47.51016466666667, 'evidenced': True, 'unstable_edge_only': False, 'd128_middle': {'n': 3, 'mean_ms': 4.297360666666666, 'one_sided_low': 4.291027154819686, 'two_sided_low': 4.288028119882769, 'two_sided_high': 4.306693213450563, 'significant_positive_excess': True, 'df': 2, 'one_sided_critical': 2.919985580355516, 'two_sided_critical': 4.302652729696142, 'evidenced': True}, 'd2048_middle': {'n': 3, 'mean_ms': 47.51016466666667, 'one_sided_low': 47.38279750168242, 'two_sided_low': 47.32248679463196, 'two_sided_high': 47.69784253870137, 'significant_positive_excess': True, 'df': 2, 'one_sided_critical': 2.919985580355516, 'two_sided_critical': 4.302652729696142, 'evidenced': True}, 'source': 'middle_window'}, {'family': 'residual_norm_quant', 'score_ms': 7.209406666666666, 'evidenced': True, 'unstable_edge_only': False, 'd128_middle': {'n': 3, 'mean_ms': 7.209406666666666, 'one_sided_low': 7.167742421139227, 'two_sided_low': 7.148013630702585, 'two_sided_high': 7.270799702630748, 'significant_positive_excess': True, 'df': 2, 'one_sided_critical': 2.919985580355516, 'two_sided_critical': 4.302652729696142, 'evidenced': True}, 'd2048_middle': {'n': 3, 'mean_ms': 6.662013999999999, 'one_sided_low': 6.641534788738445, 'two_sided_low': 6.631837503640103, 'two_sided_high': 6.6921904963598955, 'significant_positive_excess': True, 'df': 2, 'one_sided_critical': 2.919985580355516, 'two_sided_critical': 4.302652729696142, 'evidenced': True}, 'source': 'middle_window'}]`
prefill selected=`[{'family': 'attn_core', 'score_ms': 186.58288100000001, 'evidenced': True, 'stats': {'n': 3, 'mean_ms': 186.58288100000001, 'one_sided_low': 186.10589919623453, 'two_sided_low': 185.88003949496184, 'two_sided_high': 187.2857225050382, 'significant_positive_excess': True, 'df': 2, 'one_sided_critical': 2.919985580355516, 'two_sided_critical': 4.302652729696142, 'evidenced': True}, 'source': 'p4096_complete'}, {'family': 'prompt_mmq', 'score_ms': 27.108745000000038, 'evidenced': True, 'stats': {'n': 3, 'mean_ms': 27.108745000000038, 'one_sided_low': 25.265120375562727, 'two_sided_low': 24.39213003579038, 'two_sided_high': 29.825359964209696, 'significant_positive_excess': True, 'df': 2, 'one_sided_critical': 2.919985580355516, 'two_sided_critical': 4.302652729696142, 'evidenced': True}, 'source': 'p4096_complete'}]`
coverage ok=`True`

## Next experiments

decode=`attn_core`
prefill=`attn_core`

## Status

status=`measured` blocked=`False`

Unsupported counters leave causal explanation explicitly unknown.
Missing trace coverage is blocked. Shipping throughput delta is N/A.

## Whole-wall residual (OPT-142 follow-up)

OPT-138 family-sum gap reconstruction left ~23% decode / ~5.9% prefill
unexplained because overlapping llama families were summed and host-exclusive
CUDA APIs were recorded as 0. OPT-142 reuses these captures and partitions
each Quartz wall with interval unions. Maximum Quartz unresolved share across
required D128/D2048 windows and P4096 is 0.27%. See
[`evidence/optimization/opt142-wall-reconciliation/REPORT.md`](../opt142-wall-reconciliation/REPORT.md).
Family rankings above remain separately qualified.

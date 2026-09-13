# OPT-121 weight traffic requant

Verdict **quality_blocked**. Candidate `q8_to_q4k` vs post-OPT-119 `ffn_only`. Production pin `none`.

## Role map and conversion

Converted tensors `192` src `7130316800` dst `3774873600` saved `3355443200` conversion_ms `15783.9004`. Encoding `q4_k_minmax_ref`. Irreversible. GGUF unmodified. No dense shadow. FP4 study capability_absent=`True`.

## Quality

OPT-058 `--quality` invoked; restored packed/r2=`false`; candidate_nll_measured=`True`; held-out PPL ratio `1.0005700698520772`; wikitext PPL ratio `0.9939496503121583`; recurrence ΔNLL short `3.9215501723764974` long `3.463459781516581` (max 0.02, ok=`False`); opt116=`opt116_generated_v1`.
long-cache GPU `True` ok=`False`.
state/memory=`True` 128k_fit=`True`.

## Full-engine A/B

| Workload | parent tok/s | candidate tok/s | geo ratio | CI lower | p95 ratio | gate |
|---|---:|---:|---:|---:|---:|---|
| D128 (target) | 54.036 | 47.020 | 0.8702 | 0.8697 | 1.1506 | fail |
| D2048 (target) | 41.671 | 37.861 | 0.9086 | 0.9082 | 1.1114 | fail |
| P4096 (guard) | 3015.632 | 3058.693 | 1.0143 | 1.0140 | 0.0000 | pass |

Measured tok/s deltas `{'d128': -7.016271959999997, 'd2048': -3.8101215300000035, 'p4096': 43.0615499999999}`. Sitting deltas `{'d128': 0.0, 'd2048': 0.0, 'p4096': 0.0}`. Keep=False. Reasons: quality_failed, long_cache_failed, d128_throughput_gate, d2048_throughput_gate.

Rejected production dispatch retains baseline Q8/Q6 weights. Memory-only savings cannot admit a keep.

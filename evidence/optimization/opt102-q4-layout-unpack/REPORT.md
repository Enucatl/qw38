# OPT-102 — Q4_K layout and branchless unpack

Status: **reject**. Control `late_w4` (`integer_q8_late` / `raw_gguf` /
`paired_integer`). Candidates `branchless_late` and `aligned_meta`. OPT-093
factored association is not a candidate.

Selected path: `late_w4`.
Shipping Q4 decode: `integer_q8_late`.
Shipping Q4 layout: `raw_gguf`.
Production kept: `False`.
Quality contract: `opt089_strict` reused from authenticated OPT-089.

## Complete FFN

branchless mean_diff_ms=-0.47463029999999995
aligned mean_diff_ms=-0.3297790000000001
control_mean_ms=9.841913700000001

## Decode prefixes

D128 control_mean_ms=574.6887085999999 candidate_mean_ms=583.6533082
D2048 control_mean_ms=628.7437864000001 candidate_mean_ms=636.6536255999999

## P4096

control_tok_s=2990.6265396916924 candidate_tok_s=2994.562748437764
throughput_ratio=1.0013161819751917
tok/s delta vs OPT-098 P4096 3046.23: 0

# OPT-120 — Evaluate physically packed 8-bit and 4-bit KV caches

Status: **quality_blocked**. Production pin `dense_bf16`. Hardware executed on RTX 5090.

Parent is authenticated post113_selected `ffn_only` after OPT-117 rejected graphs, OPT-118 kept lazy/overlap, and OPT-119 kept mixer/FFN Q8 fusion. Primary target is **long-context decode** (D8192 / D32768 / D131040). D128, D2048, and P4096 are non-target guards. Integer Q8 was chosen once; ds4 FP8 writes floats. Group 32, FP16 scales, no committed recent window. Packed payload+scales are authoritative; attention unpacks into registers/shared memory. No dense shadow. At most one full cache design.

## Inventory

Capacity `131072` dense `8589934592` q8q8 `4563402752` q8q4 `3489660928` q4q4 `2415919104` candidate `q8q8` bytes `4563402752` saved `4026531840`. Scales included; no dense shadow.

## Reference, format, quality, long-cache, state

host Q/DQ reference ok=`True`.
packed checkpoint restore / legacy reject ok=`True`.
OPT-058 `--quality` invoked; restored packed/r2=`false`; candidate_nll_measured=`True`; held-out PPL ratio `1.0006340207055555`; wikitext PPL ratio `1.0007110311303178`; opt116=`opt116_generated_v1`.
long-cache GPU `True` free-running `True` ok=`False` prefixes 8192/32768/131040+32.
state/memory=`True` 128k_fit=`True` packed_session_bytes=`445085700` dense_session_bytes=`696743940`.

## Full-engine A/B

| Workload | parent tok/s | candidate tok/s | geo ratio | CI lower | p95 ratio | gate |
|---|---:|---:|---:|---:|---:|---|
| D8192 (target) | 7.849702358 | 4.897418784 | 0.6238988368092301 | 0.6235136668968169 | 1.0249321117122445 | False |
| D32768 (target) | 1.34307015 | 0.4245838762 | 0.3161296390438446 | 0.31573415116282766 | 1.0452881576193231 | False |
| D131040 (target) | 0.12843336900000002 | 0.027626556349999998 | 0.21510418702325107 | 0.21494391963414747 | 1.0572855613714045 | False |
| D128 (guard) | 54.18718109 | 54.084581379999996 | 0.9981064075503897 | 0.9977259582374934 | 1.0020233584038982 | True |
| D2048 (guard) | 41.62626151 | 41.0406414 | 0.985931480431247 | 0.9858019131425172 | 0.999014264637795 | True |
| P4096 (guard) | 2937.071143 | 2316.938355 | 0.7888835394501924 | 0.7846690942579753 | 0.0 | False |

Measured tok/s deltas `{'d8192': -2.952283574, 'd32768': -0.9184862737999999, 'd131040': -0.10080681265000002, 'd128': -0.1025997100000069, 'd2048': -0.5856201100000007, 'p4096': -620.1327880000003}`. Sitting deltas `{'d8192': 0.0, 'd32768': 0.0, 'd131040': 0.0, 'd128': 0.0, 'd2048': 0.0, 'p4096': 0.0}`. Keep=False. Reasons: long_cache_failed, d8192_throughput_gate, d32768_throughput_gate, d131040_throughput_gate, p4096_non_target_guard, measured_quality_fail.

Rejected production dispatch retains dense BF16 KV. Memory-only savings cannot admit a keep.

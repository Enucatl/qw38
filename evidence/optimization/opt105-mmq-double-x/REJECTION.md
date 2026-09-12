# OPT-105 rejection — retain 128x128 joined-wait

Production pin remains `kSelectedMmqDoubleX = false`. FFN I/J stay 128.
`q4_i64_j128_fma_async_x2` is not installed. OPT-065 extra tiles and OPT-097
split wait stay rejected.

Complete 64-layer FFN screen missed the ≥5 ms/P4096 keep gate (saving
**−82.72 ms**). Five-pair P4096 and D128/D2048 guards were not reopened.

Independent verdicts:

{
  "independent_verdicts": {
    "kernel_parity_pass": true,
    "model_quality_pass": true,
    "performance_pass": false,
    "production_kept": true
  },
  "selected_mmq_double_x": false,
  "shipping_unchanged": true,
  "control_kernel": "q4_i128_j128_fma_async_x",
  "candidate_kernel": "q4_i64_j128_fma_async_x2",
  "complete_ffn_saving_ms": -82.7234497,
  "claims_throughput": false,
  "status": "reject_128x128_retained"
}

tok/s delta vs OPT-098 P4096 **3046.23 tok/s**: **0** (rejection).

status=measured_reject.

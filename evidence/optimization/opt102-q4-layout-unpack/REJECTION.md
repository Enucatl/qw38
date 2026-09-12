# OPT-102 rejection — retain integer_q8_late / raw_gguf

Production pins remain `kSelectedQ4DecodePath[] = "integer_q8_late"` and
`kSelectedQ4DeviceLayout[] = "raw_gguf"`. OPT-093 factored association is
unchanged and not reinterpreted.

Complete 64-layer FFN, D128/D2048 five-pair, and P4096 guard did not jointly
win. Keep required ≥0.10 ms/token complete FFN with positive paired CI plus
both decode prefixes.

Independent verdicts:

{
  "late_w4": {
    "kernel_parity_pass": true,
    "model_quality_pass": true,
    "performance_pass": false,
    "production_kept": true,
    "opt074_coverage_unadmitted_blocker": false,
    "incomplete": false
  },
  "branchless_late": {
    "kernel_parity_pass": true,
    "model_quality_pass": true,
    "performance_pass": false,
    "production_kept": false,
    "opt074_coverage_unadmitted_blocker": false,
    "incomplete": false
  },
  "aligned_meta": {
    "kernel_parity_pass": true,
    "model_quality_pass": true,
    "performance_pass": false,
    "production_kept": false,
    "opt074_coverage_unadmitted_blocker": false,
    "incomplete": false
  }
}

tok/s delta vs OPT-098 P4096 **3046.23 tok/s**: **0** (rejection).

status=measured_reject.

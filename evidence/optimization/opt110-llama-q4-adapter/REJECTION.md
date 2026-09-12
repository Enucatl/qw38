# OPT-110 rejection — retain `integer_q8_late` Q4 decode

Production pin remains `integer_q8_late` / `raw_gguf` / 4 warps. Source-faithful
llama Q4_K MMVQ adapter (`llama_q4k_mmvq`) was not admitted.

**Stop reason:** matched primitive **won** (combined staging+dot saving
**+0.050 ms** on identical BF16 and Q4_K at production shapes) and performance
keep bar **met** (complete rotating 64-layer FFN **+0.717 ms/token**, 95% CI
**[0.678, 0.756]**; D128 **55.01 → 58.27 tok/s**; D2048 **53.31 → 56.67
tok/s**; P4096 ratio **0.99985**), but candidate NLL was **not measured**
(`candidate_nll_not_measured`). OPT-073 production quality is not reused.

`quality_unresolved=true`
`reason=["quality_unresolved"]`

`primitive_win=true`
`ffn_keep=true`
`model_quality_pass=false`

Production pins unchanged:
`kSelectedQ4DecodePath[] = "integer_q8_late"`
`kSelectedQ4DeviceLayout[] = "raw_gguf"`
`kSelectedQ4DecodeWarpsPerRow = 4`

Independent verdicts:

```json
{
  "independent_verdicts": {
    "integer_q8_late": {
      "kernel_parity_pass": true,
      "primitive_pass": true,
      "model_quality_pass": true,
      "performance_pass": true,
      "production_kept": true,
      "incomplete": false
    },
    "llama_q4k_mmvq": {
      "kernel_parity_pass": true,
      "primitive_pass": true,
      "model_quality_pass": false,
      "performance_pass": true,
      "production_kept": false,
      "incomplete": true
    }
  },
  "selected_path": "integer_q8_late",
  "shipping_unchanged": true,
  "shipping_q4_decode": "integer_q8_late",
  "status": "quality_blocked",
  "claims_throughput": false,
  "claims_performance_improvement": false
}
```

tok/s delta versus production: **0**.

status=measured_reject.
Full measured context:
[`REPORT.md`](REPORT.md).

# OPT-108 rejection — retain production OPT-107 hybrid

Production pins remain `kSelectedDecodeAttentionVec128Path[] = "warp_query"`
and `kSelectedDecodeAttentionCrossoverThreshold = 1024`. The source-faithful
NVIDIA llama vector stack was not admitted.

**Stop reason:** matched D2048 primitive 95% CI includes 0 (point-estimate
saving ~0.002 ms). The primitive screen requires D2048 faster with
`ci95_low > 0` before engine integration.

`primitive_win=false`
`reason=['matched_primitive_lost']`

Engine integration was not run. Complete-attention 0.50 ms/token was not
attempted. Quality/state/128K and P4096 guard were not run.

Independent verdicts:

```json
{
  "independent_verdicts": {
    "production_opt107": {
      "kernel_parity_pass": true,
      "primitive_pass": true,
      "model_quality_pass": true,
      "performance_pass": true,
      "production_kept": false
    },
    "llama_vec_nvidia": {
      "kernel_parity_pass": true,
      "primitive_pass": false,
      "model_quality_pass": false,
      "performance_pass": false,
      "production_kept": false
    }
  },
  "selected_path": "production_opt107",
  "shipping_unchanged": true,
  "shipping_decode_attention_vec128": "warp_query",
  "crossover_threshold": 1024,
  "status": "primitive_rejected",
  "claims_throughput": false,
  "claims_performance_improvement": false
}
```

tok/s delta versus production: **0**.

status=measured_reject.
Full measured context:
[`REPORT.md`](REPORT.md).

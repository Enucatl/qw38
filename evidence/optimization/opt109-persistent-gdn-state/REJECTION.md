# OPT-109 rejection — retain sequential row-major GDN decode

Production pin remains `sequential` (`prepare_recurrence_window`,
row-major live device state). Persistent col-major session layout
(`persistent_transposed`) was not admitted.

**Stop reason:** pilot recurrence-only **won** on identical buffers (3+10:
sequential **0.02592 ms** vs persistent **0.01102 ms**; mean_diff
**+0.01489 ms**; 95% CI low **+0.01153**; `timed_relayout=0`), but complete
48-layer GDN enclosing **lost** (3+10: sequential **13.190 ms** vs persistent
**13.993 ms**; mean_diff **−0.803 ms**; 95% CI **[−1.053, −0.553]** entirely
negative; required saving ≥ 0.10 ms/token with positive CI).
`saving_ge_0_10_ms=false`. Kernel-only was faster (~0.67 vs ~1.63 ms) but
enclosing wall did not survive.

`complete_gdn_lost=true`
`reason=["complete_gdn_lost"]`

`timed_relayout_launches=0`
`decode_conversions=0`
`capture_key=8b954514af28526225acbdc344b48dcc98485daa454a1835f8f7c43f021433e6`

D128/D2048 engine pairs and P4096 guard were not run (`complete_gdn_lost`).
OPT-077/094/101 historical fixtures are unchanged.

Independent verdicts:

```json
{
  "independent_verdicts": {
    "sequential_row_major": {
      "kernel_parity_pass": true,
      "model_quality_pass": true,
      "performance_pass": false,
      "production_kept": true,
      "incomplete": false
    },
    "persistent_transposed": {
      "kernel_parity_pass": true,
      "model_quality_pass": true,
      "performance_pass": false,
      "production_kept": false,
      "incomplete": false
    }
  },
  "selected_path": "sequential_row_major",
  "shipping_unchanged": true,
  "shipping_gdn_decode": "sequential",
  "status": "rejected",
  "claims_throughput": false,
  "claims_performance_improvement": false
}
```

tok/s delta versus production: **0**.

status=measured_reject.
Full measured context:
[`REPORT.md`](REPORT.md).

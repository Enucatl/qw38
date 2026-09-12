# OPT-101 rejection — retain sequential GDN decode

Production pin remains `sequential`. Transposed warp-column candidate was not
kept. Historical OPT-077/094 evidence is unchanged.

Complete 48-layer GDN n=10: sequential 12.4181503 ms vs
transposed 13.3394116 ms; paired mean_diff_ms
-0.9212612999999997 with 95% CI [-1.2763761588464648,
-0.5661464411535347] (entirely negative; required saving ≥ 0.10 ms
with positive CI). D128 32-token five-pair walls sequential
571.36167 ms vs transposed 584.719043
ms. D2048 sequential 624.1348874 ms vs transposed
636.2994384 ms. P4096 sequential 1322.56177
ms (3097.0198087609924 tok/s) vs transposed
1322.90906 ms (3096.206779323138 tok/s).

## Independent verdicts

{
  "sequential": {
    "kernel_parity_pass": true,
    "model_quality_pass": true,
    "performance_pass": false,
    "production_kept": true,
    "incomplete": false
  },
  "transposed": {
    "kernel_parity_pass": true,
    "model_quality_pass": true,
    "performance_pass": false,
    "production_kept": false,
    "incomplete": false
  }
}

## tok/s

Speedup versus the then-current sequential baseline is **0**
(`claims_throughput=false`). Versus OPT-098 P4096 3046.23 tok/s the production
combination is unchanged, so delta is **0**.

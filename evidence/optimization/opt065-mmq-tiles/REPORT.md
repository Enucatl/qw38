# OPT-065 — Retune MMQ tiles with the active pipeline preserved

Status: **retained 128×128**. Production Q4 prompt FFN tiles stay
`i128_j128` on gate, up, and down. The accepted pipeline remains
`fma_async`. No throughput claim. Complete FFN decides keep versus
control; occupancy is evidence, not the gate.

## Tiles

Four Q4 configurations, all with FMA plus async Y on aligned full tiles
(4096-row P, real layer-0 weights). Isolated GEMM means, 1 warmup + 3 samples:

| Id | I | J | Gate/up ms | Down ms | Occ | Regs | Local | Shared | Extra Y |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **i128_j128** | 128 | 128 | **2.865675** | **2.910101** | 1 | 255 | 80 | 76288 | 18432 |
| i64_j128 | 64 | 128 | 3.011499 | 3.023723 | 1 | 220 | 0 | 56832 | 18432 |
| i128_j64 | 128 | 64 | 3.064309 | 3.082592 | 1 | 246 | 0 | 57600 | 9216 |
| i64_j64 | 64 | 64 | 3.765973 | 3.818656 | 1 | 216 | 0 | 38144 | 9216 |

Complete FFN (staging + gate + up + SwiGLU-Q8 + down + residual) at 4096 rows:
control **9.649205** ms. Selected pair was also `i128_j128`/`i128_j128`
(**9.629227** ms, same tile; keep=false). Occupancy stayed 1 on every
candidate (register-limited). Smaller tiles used less shared memory and
avoided the 80-byte spill on `i128_j128` but lost on complete FFN time.

Synchronous fallback remains for output or prompt tails and misalignment
(`q4_*_sync_fallback`). Q6 stays I=128. Q8 helpers were regression-checked
(one large 128×256×128, one skinny 32×256×128) without a second tuning grid.
Shared-Y and SwiGLU-Q8 staging are unchanged. Stream-K stays off. No
512/1024/2048 prompt-length sweep.

## Resource table

Filled from `mmq_pipeline_kernel_attributes` after opt-in shared bytes on
`fma_async`. Diagnostic and production aligned full tiles report
`pipeline=true fma=true async_y=true` with kernel ids
`q4_i128_j128_fma_async`, `q4_i64_j128_fma_async`, `q4_i128_j64_fma_async`,
and `q4_i64_j64_fma_async`.

## Runs

- Feedback: `build/optimization-runs/OPT-065/feedback/20260911T103300Z-9820fe68`
- Acceptance: `build/optimization-runs/OPT-065/acceptance/20260911T103338Z-ae4e88a7`
- Raw screen: `screen-raw.txt`

tok/s was not remeasured. OPT-069 owns combined E2E / original llama
outcome gates. Speedup is 0 (baseline 128×128 unchanged).

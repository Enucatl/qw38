# OPT-046 rejection

keep sitting failed OPT-044 numerics or the P/D guard versus OPT-045. Production Q4_K decode remains the packed FP32 path
(`selected_q4_decode_path=packed`).

- A/B winner: integer_q8_w4
- keep_sitting_skipped: false
- measurement_utc: 2026-09-10T18:02:50Z
- P Quartz mean tok/s: 2130.48462
- OPT-045 P baseline: 2130.41089
- D128 Quartz mean tok/s: 40.7888718
- D2048 Quartz mean tok/s: 38.3443794

## Why production stays packed

A/B winner `integer_q8_w4` is about 3.50× faster than packed on the weighted
complete MMV (0.0191882669 ms vs 0.067196091 ms) and the integer sitting
measured Quartz P 2130.48462, D128 40.7888718, D2048 38.3443794 tok/s versus
OPT-045 keep denominators. Those throughput numbers do **not** admit the path:

- CUD-001 `q4_k_17x256` host Q8-staged envelope is max_abs **3.0e-4**. Packed
  measures **0.000244140625**. Cooperative integer DP4A measures
  **0.000305175781** and fails `tests/test_cuda_quant_mmv.py`.
- `tests/test_cuda_full_scheduler.py` with the integer production pin reported
  all-NaN logits (248320 nonfinite) and greedy mismatch (cuda=0 vs scalar).
- OPT-044 A/B probe budgets are not a substitute for the frozen CUD-001
  3e-4/2e-4 envelope on `launch_quant_mmv`.

Production `kSelectedQ4DecodePath` remains `packed`. Integer kernels stay
behind `launch_q4k_coop_mmv` / A/B only. Packed FP32 `quant_mmv` is unchanged.

# OPT-082 — CUDA kernel-parity suite

Status: **kernel-parity suite**. `claims_throughput: false`. No production pin
or selector change. Authority remains llama.cpp `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` and GGUF SHA-256
`31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`.

## Policy

Kernel admission proves implementation of the intended quantized operation. It does not prove full-model quality and does not depend on llama GPU versus FP64 consistency.

Reference = independent CPU/dequant of the same quantized weights and
staged/input activations. Classes: `quantized_operation_association`
(`abs > abs_scale*sqrt(K)` **and** `rel > rel_tol` fails an element) and
`same_math_equivalence` (bit identity). Nonfinite count must be 0.

| Family | abs_scale | rel_tol |
|---|---:|---:|
| Q8_0 | 0.05 | 0.05 |
| Q4_K | 0.20 | 0.05 |
| Q6_K | 0.20 | 0.05 |

Canonical checkers: `cuda/kernel_parity.cuh` and `tools/kernel_parity.py`.
Existing `ds4_q4k_association_ok` / `ds4_q8_association_ok` wrappers now call
the canonical helper.

## Host cases

| Case | Family | Class | Verdict | Reason |
|---|---|---|---|---|
| host_q4_or_abs | Q4_K | quantized_operation_association | pass | pass |
| host_q8_both_fail | Q8_0 | quantized_operation_association | fail | association_fail |
| host_q6_reuses_q4 | Q6_K | quantized_operation_association | pass | pass |
| host_same_math_identity | Q8_0 | same_math_equivalence | pass | pass |
| host_nonfinite | Q4_K | quantized_operation_association | fail | nonfinite |

Q8_1 typed `sum_q` versus `sum_x` rejected:
True.

## GPU

available=True ran=True success=True
blocker=none
catalog_count=105 gpu_case_count=105

Aligned MMQ uses M=128 N=128 so `fma_async` / `fma_async_x` actually launch.
Listed N∈{4,8} MMQ shapes are unaligned fallback association cases and are
not counted as the candidate kernel. Down projection is packed Q4_K MMV
(`Q4_K_mmv_packed_down_M3_N1_K256_random_assoc`). Native success fail-closes
on wrong selector, nonfinite, fallback-measured-as-candidate, and CUDA
errors. Association and same-math misses are recorded as documented fails.

## Shipping paths (pass or documented fail)

| Candidate | Case | gpu_pass | launched | reason |
|---|---|---|---|---|
| packed | Q4_K_mmv_packed_M1_N1_K256_random_assoc | True | quant_mmv_packed | pass |
| packed | Q4_K_mmv_packed_M1_N1_K512_random_assoc | True | quant_mmv_packed | pass |
| packed | Q4_K_mmv_packed_M1_N1_K2048_random_assoc | True | quant_mmv_packed | pass |
| packed | Q4_K_mmv_packed_M3_N1_K256_random_assoc | True | quant_mmv_packed | pass |
| packed | Q4_K_mmv_packed_M3_N1_K512_random_assoc | True | quant_mmv_packed | pass |
| packed | Q4_K_mmv_packed_M3_N1_K2048_random_assoc | True | quant_mmv_packed | pass |
| packed | Q4_K_mmv_packed_M17_N1_K256_random_assoc | True | quant_mmv_packed | pass |
| packed | Q4_K_mmv_packed_M17_N1_K512_random_assoc | True | quant_mmv_packed | pass |
| packed | Q4_K_mmv_packed_M17_N1_K2048_random_assoc | True | quant_mmv_packed | pass |
| packed | Q4_K_mmv_packed_M1_N1_K256_zero_assoc | True | quant_mmv_packed | pass |
| packed | Q4_K_mmv_packed_M1_N1_K256_cancel_assoc | True | quant_mmv_packed | pass |
| packed | Q4_K_mmv_packed_M1_N1_K256_minmax_assoc | True | quant_mmv_packed | pass |
| packed | Q4_K_mmv_packed_M1_N1_K256_half_assoc | True | quant_mmv_packed | pass |
| fma_async_x | Q4_K_mmq_fma_async_x_M128_N128_K256_aligned_assoc | True | q4_i128_j128_fma_async_x | pass |
| packed | Q4_K_mmv_packed_M1_N1_K4096_random_assoc | True | quant_mmv_packed | pass |
| packed | Q4_K_mmv_packed_M1_N1_K5120_random_assoc | True | quant_mmv_packed | pass |
| packed | Q4_K_mmv_packed_M1_N1_K6144_random_assoc | True | quant_mmv_packed | pass |
| packed | Q4_K_mmv_packed_down_M3_N1_K256_random_assoc | True | quant_mmv_packed | pass |
| r2_w2 | Q8_0_mmv_r2_w2_M1_N1_K256_random_assoc | True | r2_w2 | pass |
| r2_w2 | Q8_0_mmv_r2_w2_M1_N1_K512_random_assoc | True | r2_w2 | pass |
| r2_w2 | Q8_0_mmv_r2_w2_M1_N1_K2048_random_assoc | True | r2_w2 | pass |
| r2_w2 | Q8_0_mmv_r2_w2_M3_N1_K256_random_assoc | True | r2_w2 | pass |
| r2_w2 | Q8_0_mmv_r2_w2_M3_N1_K512_random_assoc | True | r2_w2 | pass |
| r2_w2 | Q8_0_mmv_r2_w2_M3_N1_K2048_random_assoc | True | r2_w2 | pass |
| r2_w2 | Q8_0_mmv_r2_w2_M17_N1_K256_random_assoc | True | r2_w2 | pass |
| r2_w2 | Q8_0_mmv_r2_w2_M17_N1_K512_random_assoc | True | r2_w2 | pass |
| r2_w2 | Q8_0_mmv_r2_w2_M17_N1_K2048_random_assoc | False | r2_w2 | association_fail |
| integer_q8_1 | Q6_K_mmv_integer_q8_1_M1_N1_K256_random_assoc | True | integer_q8_1 | pass |
| integer_q8_1 | Q6_K_mmv_integer_q8_1_M1_N1_K512_random_assoc | True | integer_q8_1 | pass |
| integer_q8_1 | Q6_K_mmv_integer_q8_1_M1_N1_K2048_random_assoc | True | integer_q8_1 | pass |
| integer_q8_1 | Q6_K_mmv_integer_q8_1_M3_N1_K256_random_assoc | True | integer_q8_1 | pass |
| integer_q8_1 | Q6_K_mmv_integer_q8_1_M3_N1_K512_random_assoc | True | integer_q8_1 | pass |
| integer_q8_1 | Q6_K_mmv_integer_q8_1_M3_N1_K2048_random_assoc | True | integer_q8_1 | pass |
| integer_q8_1 | Q6_K_mmv_integer_q8_1_M17_N1_K256_random_assoc | True | integer_q8_1 | pass |
| integer_q8_1 | Q6_K_mmv_integer_q8_1_M17_N1_K512_random_assoc | True | integer_q8_1 | pass |
| integer_q8_1 | Q6_K_mmv_integer_q8_1_M17_N1_K2048_random_assoc | True | integer_q8_1 | pass |

## Documented GPU fails

| Case | Candidate | Reason | max_abs | abs_tol |
|---|---|---|---:|---:|
| Q4_K_fused_vs_independent_M3_N1_K256_random_same | packed | same_math_fail | 15.9912109 | 0.0 |
| Q8_0_mmv_r1_w4_M17_N1_K2048_random_assoc | r1_w4 | association_fail | 2.30578613 | 2.2627416997969525 |
| Q8_0_mmv_r2_w2_M17_N1_K2048_random_assoc | r2_w2 | association_fail | 2.30580521 | 2.2627416997969525 |

## Proof limit

- no throughput claim
- no production pin change
- kernel admission is not model quality
- no full production M
- generated quantized blocks only

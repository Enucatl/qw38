# OPT-081 — Kernel parity policy reset

Status: **kernel_parity_v1 frozen**. `claims_throughput: false`. No arithmetic
kernel change, no production selector change, and no speedup are claimed.
Authority remains llama.cpp `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` and GGUF SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`.

## Policy

Kernel admission proves implementation of the intended quantized operation. It does not prove full-model quality and does not depend on llama GPU versus FP64 consistency.

Active hierarchy (**Proposed**): structural → kernel parity → same-math →
model quality → performance → release. OPT-059 v2 and OPT-074 production GPU
admission are `historical_diagnostic_only`. CUD-001 `3e-4` is not an active
kernel-parity ceiling.

`opt074_family_admission_required` is **false**. OPT-074 unadmitted rows
(`held_out_exceeds_frozen_calibration_ceiling` /
`opt074_coverage_unadmitted`) are not active keep blockers. Historical
OPT-059/074/070/075/076 fixtures and reports remain as written.

## Reference and classes

Reference = independent CPU/dequant of the **same** quantized weights and
staged/input activations. Classes:
`quantized_operation_association` (element fails only when
`abs > abs_scale*sqrt(K)` **and** `rel > rel_tol`) and
`same_math_equivalence` (bit identity unless a named tiny epsilon is
justified). Nonfinite count must be 0. Hard invariants are never waived by a
tolerance. Q8_1 `sum_q` cannot be equated to `sum_x`.

| Family | Applies | abs_scale | rel_tol |
|---|---|---:|---:|
| Q8_0 | yes | 0.05 | 0.05 |
| Q4_K | yes | 0.20 | 0.05 |
| Q6_K | yes (Q4_K envelope) | 0.20 | 0.05 |
| Q2_K / IQ2 | not_applicable | n/a | n/a |

Canonical checkers: `cuda/kernel_parity.cuh` and `tools/kernel_parity.py`.

## Synthetic cases

| Case | Family | Class | Verdict | Reason |
|---|---|---|---|---|
| or_rule_abs_pass | Q8_0 | quantized_operation_association | pass | pass |
| or_rule_rel_pass | Q8_0 | quantized_operation_association | pass | pass |
| both_fail_rejects | Q8_0 | quantized_operation_association | fail | association_fail |
| nonfinite_rejects | Q8_0 | quantized_operation_association | fail | nonfinite |
| q8_scale_rejects_q4_envelope_gap | Q8_0 | quantized_operation_association | fail | association_fail |
| q4_scale_accepts_same_gap | Q4_K | quantized_operation_association | pass | pass |
| q6_reuses_q4_envelope | Q6_K | quantized_operation_association | pass | pass |
| q2_not_applicable | Q2_K | quantized_operation_association | fail | not_applicable |
| iq2_not_applicable | IQ2 | quantized_operation_association | fail | not_applicable |
| same_math_bit_identity | Q8_0 | same_math_equivalence | pass | pass |
| same_math_rejects_ulp | Q8_0 | same_math_equivalence | fail | association_fail |

Q8_1 typed comparison rejected:
True.
Future keep requiring `opt074_coverage_unadmitted` failed closed:
True.

## Proof limit

- no throughput claim
- no arithmetic kernel change
- no production selector change
- kernel admission is not model quality
- OPT-059/074 are historical_diagnostic_only
- opt074_family_admission_required is false

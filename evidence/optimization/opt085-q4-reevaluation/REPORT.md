# OPT-085 — Re-evaluate Q4 decode candidates

Status: **measured**. Authority llama.cpp
`cc83d7b4824f73cfdda4dfbb47ee39804f71b328`, GGUF SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`. Exactly three configurations:
packed/`paired_staged` control, OPT-075 `integer_q8`/`paired_integer`, and
OPT-076 `integer_q8_late`/`paired_integer` `late_w4`. Four warps/row and
FP32-scale Q8Block. No new fusion, tile, compiler, half-scale Q8_1, or Q4
kernel.

`opt074_coverage_unadmitted` is **not** a blocker.
`opt074_family_admission_required=False`.

## Independent verdicts

| Config | kernel_parity_pass | model_quality_pass | performance_pass | production_kept |
|---|---|---|---|---|
| packed_paired_staged | True | True | False | True |
| integer_q8_paired | True | False | False | False |
| late_w4 | True | False | True | False |

Selected path: `packed_paired_staged`.
shipping_unchanged=True.
production_kept=False.
claims_throughput=False.

## Kernel parity (OPT-082 Q4 idents)

source=this_sitting; host_ok=True;
typed `sum_q` vs `sum_x` rejected=True.

| Ident | Config | pass | gpu/catalog | fallback-as-candidate |
|---|---|---|---|---|
| packed | packed_paired_staged | True | 14/14 | False |
| integer_q8 | integer_q8_paired | True | 6/6 | False |
| integer_q8_late | late_w4 | True | 6/6 | False |

Candidates that fail kernel parity are rejected before quality.

## Quality (OPT-083/084)

Identity-cached OPT-084 suite with `--quality`. Primary gate is regression
versus the shipping Quartz baseline. Pinned llama is inspectable.
A known baseline defect does not reject a non-worsening kernel.
quality-by-config: {'packed_paired_staged': 'shipping_quartz_vs_opt084_freeze', 'integer_q8_paired': 'candidate_specific_quality_not_measured', 'late_w4': 'candidate_specific_quality_not_measured'}.

## Performance

Complete 64-layer rotating FFN on repaired OPT-071 captures. AB/BA pairing.
D2048+32 engine pairs with a separate graph capture per configuration.
this_sitting=True.
Control mean 16.383 ms vs survivor
`late_w4` 9.800 ms.
Paired CI 6.5472 .. 6.6200 ms.
Engine 868.295 vs 639.749 ms.

Historical ranking only (not a keep): packed vs integer_q8_paired
[16.4332861, 11.7726463]; packed vs late_w4
[16.4213763, 9.8184765]. Capture identity shared=
True. A keep requires this sitting's confirmation.

## tok/s

Speedup versus the then-current packed baseline is **0** while packed is
retained. claims_throughput is true only when a candidate is
`production_kept` on this sitting's measured evidence.

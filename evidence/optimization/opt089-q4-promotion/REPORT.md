# OPT-089 — Complete late_w4 admission

Status: **late_w4_kept**. Authority llama.cpp
`cc83d7b4824f73cfdda4dfbb47ee39804f71b328`, GGUF SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`. Exactly two performance
configurations: packed/`paired_staged` control and `late_w4`
(`integer_q8_late`/`paired_integer`, four warps, FP32-scale Q8Block).
Q8 `r1_w4` and remaining OPT-088 selectors stay fixed.
`integer_q8_paired` is parity-only reconstruction, not a third sweep.
No new Q4 kernel.

`opt074_coverage_unadmitted` is **not** a blocker.
`quality_contract_id=opt089_strict`.
`claims_throughput` is true only for this sitting's uninstrumented keep.

## Independent verdicts

| Config | kernel_parity_pass | model_quality_pass | performance_pass | production_kept |
|---|---|---|---|---|
| packed_paired_staged | True | True | False | False |
| late_w4 | True | True | True | True |

Selected path: `late_w4`.
shipping_unchanged=False.
production_kept=True.
claims_throughput=True.
evidence_complete=True.

## Kernel parity (corrected OPT-082 + OPT-089 expansion)

source=this_sitting; host_ok=True;
retained OPT-082 cases=105;
extras=93; catalog=198.
Corrected fused comparator: separate GPU prequant + CUDA SiLU + BF16 RNE.
Q8/Q6 staged Q8_1 references; original-BF16 is approximation-only.
Same-math residual mismatch rejects. Every applicable positive case aggregated.

| Ident | Config | pass | gpu/catalog | parity-only |
|---|---|---|---|---|
| packed | packed_paired_staged | True | 31/31 | False |
| integer_q8 | integer_q8_paired | True | 19/19 | True |
| integer_q8_late | late_w4 | True | 21/21 | False |
| r1_w4 | r1_w4 | True | 39/39 | False |
| r2_w2 | r2_w2 | True | 39/39 | False |
| q6 | q6 | True | 9/9 | False |

Packed fail is a hard stop. integer_q8 reconstruction does not block late.

## Quality (`opt089_strict`)

`--quality` plus quality-config with all effective selectors, applied before
graph creation. Does not restore packed/r2. Authenticate OPT-088 packed/r1
and OPT-084 packed/r2 separately. Candidate NLL is mandatory.
quality-alarm (32 held-out) cannot pass full Q.
quality-by-config: {'packed_paired_staged': 'authenticated_opt088_packed_r1', 'late_w4': 'opt089_strict_pass'}.

## Performance

Screen: 64-layer rotating FFN prefix2048, 1+3 AB/BA, one D2048+32 pair.
Acceptance: 3+10 FFN, five D128 and D2048+32 pairs with p95, P4096 ≥0.95.
this_sitting=True.
Control mean 16.376 ms vs survivor
`late_w4` 9.784 ms.
Paired CI 6.5632 .. 6.6203 ms.
Engine 870.936 vs 639.915 ms.
D128 throughput_ratio=1.394 p95_ratio=0.709.
D2048 throughput_ratio=1.360 p95_ratio=0.732.
P4096 control 1313.887 ms vs candidate 1316.068 ms;
throughput_ratio=0.9983 pass=True.

Historical OPT-082/085/086 reports remain historical and unmodified.
Production pins: Q4 `integer_q8_late` / FFN `paired_integer`. pin_rewrite q4=False ffn=False.

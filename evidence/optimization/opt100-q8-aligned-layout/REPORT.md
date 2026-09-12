# OPT-100 — Aligned Q8 SoA device layout

Status: **measured reject**. Authority llama.cpp
`cc83d7b4824f73cfdda4dfbb47ee39804f71b328`, GGUF SHA-256
`31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`.
Exactly two device layouts: `raw_gguf` control and `aligned_soa` candidate.
Q8 `dp4a_q8_1` / `r1_w4` and grouping `grouped_r1_w4` stay fixed. No Q8 D2R
revival.

`opt074_coverage_unadmitted` is **not** a blocker.
`quality_contract_id=opt089_strict`.
`claims_throughput=false`. Production pin remains `raw_gguf`.

## Independent verdicts

| Config | kernel_parity_pass | model_quality_pass | performance_pass | production_kept |
|---|---|---|---|---|
| raw_gguf | True | True | False | True |
| aligned_soa | True | True | False | False |

Selected path: `raw_gguf`.
shipping_layout=raw_gguf.
production_kept=False.
claims_throughput=False.
evidence_complete=False.

## Kernel parity

12/12 native cases bitwise identical versus raw GGUF Q8_0, including pad
geometry (M=3, K=256), skinny α 48×5120, GDN output 5120×6144, grouped
GDN/attention input, and empty. Host and GPU inverse reconstruction are
byte-exact. `d2r_not_selected=true`. Aligned r1_w4 attributes: 40 registers,
0 local bytes, occupancy 12.

## Quality

Reuse authenticated OPT-089 `late_w4` scores under `opt089_strict`
(same-math lossless layout). `model_quality_pass=true` for both configs.

## Performance

Input projection launches stay 64 (`grouped_r1_w4`). Replacement convert is
in-place and byte-neutral on production mixer tensors.

Feedback screen (1 warmup + 3 rotating mixer rounds): control 6.916 ms vs
candidate 6.906 ms; mean_diff **0.010 ms**; CI95 [−0.160, 0.179];
`positive=false`.

Acceptance (3 warmups + 10 independent rounds, df=9, critical=2.262):

- control mean **6.902 ms**/token
- candidate mean **7.085 ms**/token
- mean_diff (control−candidate) **−0.183 ms**/token
- CI95 **[−0.275, −0.091] ms** (candidate slower; `positive=false`)

Reject: saving below the 0.10 ms/token threshold and the interval is not
positive. D128/P4096 keep gates were not run after the mixer gate failed.
Incidental mixer-phase D2048 five-pair walls were 630.403 vs 628.076 ms
(throughput_ratio 1.004) and do not override the mixer reject.

Pin rewrite layout=False (`kSelectedQ8DeviceLayout[] = "raw_gguf"`).

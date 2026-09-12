# OPT-092 — Grouped Q8 r1_w4 admission

Status: **grouped_r1_w4_kept**. Authority llama.cpp
`cc83d7b4824f73cfdda4dfbb47ee39804f71b328`, GGUF SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`. Exactly two grouping
configurations: `separate` control and `grouped_r1_w4` candidate. Q8
`dp4a_q8_1` / `r1_w4` arithmetic stays fixed.

`opt074_coverage_unadmitted` is **not** a blocker.
`quality_contract_id=opt089_strict`.
`claims_throughput` is true only for this sitting's uninstrumented keep.

## Independent verdicts

| Config | kernel_parity_pass | model_quality_pass | performance_pass | production_kept |
|---|---|---|---|---|
| separate | True | True | False | False |
| grouped_r1_w4 | True | True | True | True |

Selected path: `grouped_r1_w4`.
shipping_grouping=grouped_r1_w4.
production_kept=True.
claims_throughput=True.
evidence_complete=True.

## Kernel parity

source=None; host_ok=None;
catalog=None.
Grouped vs separate exact staged bytes and bitwise FP32 outputs.

## Quality

Reuse authenticated OPT-089 `late_w4` scores under `opt089_strict`.

## Performance

Launch counts: control 240, candidate 64
(−176). Screen ≥0.1 ms/token complete mixer.
this_sitting=True.
Control mean 7.573 ms vs candidate
7.054 ms.
Paired CI 0.4862 .. 0.5529 ms.
Engine 642.751 vs 627.414 ms.
D128 throughput_ratio=1.024 p95_ratio=0.976.
D2048 throughput_ratio=1.025 p95_ratio=0.975.
P4096 throughput_ratio=0.9986 pass=True.
Pin rewrite grouping=False.

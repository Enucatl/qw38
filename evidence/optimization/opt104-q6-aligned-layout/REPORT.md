# OPT-104 — Aligned Q6_K SoA device layout

Status: **measured_reject**. Authority llama.cpp
`cc83d7b4824f73cfdda4dfbb47ee39804f71b328`, GGUF SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`. Exactly two device layouts:
`raw_gguf` control and `aligned_soa` candidate. Production Q6 decode stays
`integer_q8_1` / 2 warps. Combined attention-output plus logits.

`opt074_coverage_unadmitted` is **not** a blocker.
`quality_contract_id=opt089_strict`.
`claims_throughput` is true only for this sitting's uninstrumented keep.

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

source=this_sitting; host_ok=True;
catalog=12.
Byte-exact inverse reconstruction and bitwise same-association MMV versus
raw GGUF Q6_K. Replacement is in-place; production Q6 shapes are
byte-neutral.

## Quality

Reuse authenticated OPT-089 `late_w4` scores under `opt089_strict`
(same-math lossless layout).

## Performance

Combined Q6 launches stay 17 (16 attention outputs + logits).
Screen ≥0.1 ms/token complete Q6 decode.
this_sitting=True.
Control mean 1.559 ms vs candidate
1.566 ms.
Paired CI -0.0156 .. 0.0012 ms.
Engine 638.221 vs 638.573 ms.
D128 throughput_ratio=n/a p95_ratio=n/a.
D2048 throughput_ratio=n/a p95_ratio=n/a.
P4096 throughput_ratio=n/a pass=None.
Pin rewrite layout=False.

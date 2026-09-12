# OPT-101 — Transpose and register-shard decode GDN state

Status: **rejected**. Authority llama.cpp
`cc83d7b4824f73cfdda4dfbb47ee39804f71b328`, GGUF SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`. Control is sequential
`prepare_recurrence_window`. Candidate is warp-column `transposed` with
register-sharded col-major device state, four warps/CTA, `grid=(48,32)`.
Convolution stays a separate launch. Canonical checkpoint payload remains
row-major. OPT-077/094 historical fixtures are not rewritten.

OPT-099 matched excess is the expectation only; OPT-077 replay values are not
added to wall time. `{"go": false, "reason": "llama_family_unmapped", "family": "gdn_core", "excess_ms": null, "quartz_gdn_core_ms": null, "opt077_replay_not_added_to_wall": true, "used_as_expectation_only": true}`.

`claims_throughput: False`.
Production pin `sequential`.
`evidence_complete=True`.

## Layout

logical `row_major_fp32` sha256 `3ac4ad9f78bcbdd1f4aa80da8617b32d6e84384783be3d314fec91c1661a4721`;
device `col_major_fp32` sha256 `e709048e6ff6ce6223cb14aae0a6c686ae18980d82213f8bd34c266200b26741`;
checkpoint `row_major_fp32`; element count `786432`.

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

## Complete 48-layer GDN

control_mean_ms=12.4181503
candidate_mean_ms=13.3394116
mean_diff_ms=-0.9212612999999997
ci95_low=-1.2763761588464648
ci95_high=-0.5661464411535347
saving_ge_0_10_ms=False
positive=False
capture_key=8b954514af28526225acbdc344b48dcc98485daa454a1835f8f7c43f021433e6

## D128 / D2048

D128 control_mean_ms=571.36167 candidate_mean_ms=584.719043
D128 throughput_ratio=0.9771298359021243 guards_pass=False
D2048 control_mean_ms=624.1348874 candidate_mean_ms=636.2994384
D2048 throughput_ratio=0.9808818117312575 guards_pass=False

## P4096 guard

control_ms=1322.56177 candidate_ms=1322.90906
control_tok_s=3097.0198087609924 candidate_tok_s=3096.206779323138
throughput_ratio=0.9997374800653342

## Decision

`production_kept=False`; shipping GDN decode
`sequential`.

## tok/s

Speedup versus the then-current sequential baseline is **0.0**.
Versus OPT-098 P4096 **3046.23 tok/s** the production combination is unchanged, so
delta is **0** (speedup 0).

# OPT-094 — Conditional OPT-077 tile32 GDN timing replication

Status: **no_reopen**. Authority llama.cpp
`cc83d7b4824f73cfdda4dfbb47ee39804f71b328`, GGUF SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`.

Conditional replication of OPT-077 `tile32` versus `sequential` requires
OPT-090 `gdn_reopen_eligible=true` with a demonstrated before/after timing or
capture repair. Prior OPT-077 control 12.51649 ms vs tile32 11.89390 ms had CI
−0.1643 to 1.4094 ms (inconclusive, not numerical rejection).

## Eligibility

`gdn_reopen_eligible=False`; verdict `no_reopen`;
reason `no_opt077_timing_capture_defect_repaired`.

## Independent verdicts

{
  "sequential": {
    "kernel_parity_pass": false,
    "model_quality_pass": false,
    "performance_pass": false,
    "production_kept": true,
    "incomplete": false,
    "not_applicable_reason": "no_reopen_eligibility"
  },
  "tile32": {
    "kernel_parity_pass": false,
    "model_quality_pass": false,
    "performance_pass": false,
    "production_kept": false,
    "incomplete": true,
    "not_applicable_reason": "no_reopen_eligibility"
  }
}

## Decision

`production_kept=True`; shipping GDN decode
`sequential`; `claims_throughput=
False`.

## tok/s

Speedup versus the then-current sequential baseline is **0.0**
while sequential is retained unless a full replication promote occurs.

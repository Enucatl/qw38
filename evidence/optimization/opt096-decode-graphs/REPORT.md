# OPT-096 — Conditional eight-layer decode graphs

Status: **no_reopen_overhead_below_trigger**. Authority llama.cpp
`cc83d7b4824f73cfdda4dfbb47ee39804f71b328`, GGUF SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`.

`claims_throughput: true` only when production_kept selects decode_segments8.
Otherwise finish as no-reopen and retain ffn_only.

## Eligibility

graph_reopen_eligible=True;
verdict=no_reopen_overhead_below_trigger; reason=idle_below_trigger.
D128 idle=0.0695 ms/token.
D2048 idle=0.0766 ms/token.
Threshold 0.5 ms/token removable unhidden overhead at both prefixes.

## Complete 64-layer decode body

Component n=None means: control n/a ms,
candidate n/a ms.
Paired CI: n/a .. n/a ms.

## Quality (OPT-073)

quality-v3 engine non-regression=None.

## Decision

production_kept=True.
Shipping execution graphs `ffn_only`.

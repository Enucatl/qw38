# OPT-109 — Keep transposed GDN state for the session lifetime

Status: **rejected**. Authority llama.cpp
`cc83d7b4824f73cfdda4dfbb47ee39804f71b328`, GGUF SHA-256
`31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`.
Control is sequential row-major `prepare_recurrence_window`. Candidate
`persistent_transposed` keeps col-major FP32 device state for the session
lifetime with zero timed per-layer relayout; canonical checkpoint bytes remain
row-major. Convolution stays a separate launch. OPT-077/094/101 historical
fixtures are not rewritten.

`claims_throughput: false`. `claims_performance_improvement: false`.
Production pin `sequential`. `evidence_complete=true`.

## Claim labels and proof limits

Persistent session layout only in the candidate path; production pin stays
`sequential`. Pilot recurrence-only (3 warmups + 10 paired CUDA events on
identical buffers) must beat the current recurrence before session ownership
changes. Complete 48-layer GDN enclosing (3+10, same capture key as OPT-101)
requires saving ≥ 0.10 ms/token with a positive 95% paired interval. D128/D2048
engine pairs and P4096 guard run only for a surviving session design. Component
(recurrence-only, conversion-only) and enclosing wall time are separate claims;
do not sum overlapping intervals. OPT-101 measured rejection (per-token relayout)
is retained; OPT-109 tests layout lifetime, not the recurrence kernel alone.

## Sitting identity

- device: NVIDIA GeForce RTX 5090
- llama_revision: `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`
- gguf_sha256: `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`
- execution_graphs: `ffn_only` (unchanged; OPT-114)
- shipping GDN decode pin: `sequential` (unchanged)
- capture_key: `8b954514af28526225acbdc344b48dcc98485daa454a1835f8f7c43f021433e6`
- candidate dispatch: `prepare_recurrence_decode_transposed`, `grid=(48,32)`,
  38 registers, occupancy 12, 4 warps/CTA

## Eligibility

OPT-101 warp-column recurrence admitted. Pilot must win on identical buffers
before persistent session layout is exercised in complete GDN.

## Correctness and parity

Native correctness tier passed: layout round-trip (logical row-major ↔ device
col-major ↔ checkpoint row-major), seq_abs ~1e-9 on isolated recurrence,
FP64 sample oracle inside tolerance, CUDA graph AB/BA address variants stable.
Quality (OPT-073 reuse), state/lifetime/checkpoint/graph, and 128K memory
ledger passed. `kernel_parity_pass=true`, `model_quality_pass=true`.

## Layout

logical `row_major_fp32` sha256 `3ac4ad9f78bcbdd1f4aa80da8617b32d6e84384783be3d314fec91c1661a4721`;
device `col_major_fp32` sha256 `e709048e6ff6ce6223cb14aae0a6c686ae18980d82213f8bd34c266200b26741`;
checkpoint `row_major_fp32`; element count `786432`.

## Pilot recurrence-only (3+10 on identical buffers)

`pilot_lost=false`. Control sequential recurrence on row-major buffers;
candidate persistent recurrence on pre-existing col-major buffers.
`timed_relayout=0`.

| path | mean ms | saving ms | 95% CI low | positive |
|---|---:|---:|---:|---|
| sequential_row_major | 0.02592 | — | — | — |
| persistent_transposed | 0.01102 | +0.01489 | +0.01153 | **true** |

Pilot stop rule did not fire; session-design evaluation proceeded to complete
48-layer GDN.

## Complete 48-layer GDN (3+10 enclosing)

`positive=false`. `saving_ge_0_10_ms=false`. `timed_relayout_launches=0`;
`decode_conversions=0`.

| path | mean ms | saving ms | 95% CI | positive |
|---|---:|---:|---|---|
| sequential_row_major | 13.190 | — | — | — |
| persistent_transposed | 13.993 | **−0.803** | **[−1.053, −0.553]** | **false** |

Kernel-only CUDA events were faster for the candidate (~0.67 ms vs ~1.63 ms
mean) but enclosing wall time lost. Session design therefore did not survive
complete GDN acceptance.

## Skipped after complete GDN loss

- Five uninstrumented D128/D2048 engine pairs (`reason=complete_gdn_lost`)
- P4096 prefill throughput guard (`reason=complete_gdn_lost`)

## Independent verdicts

```json
{
  "sequential_row_major": {
    "kernel_parity_pass": true,
    "model_quality_pass": true,
    "performance_pass": false,
    "production_kept": true,
    "incomplete": false
  },
  "persistent_transposed": {
    "kernel_parity_pass": true,
    "model_quality_pass": true,
    "performance_pass": false,
    "production_kept": false,
    "incomplete": false
  }
}
```

## Decision

Verdict: **rejected** (`complete_gdn_lost`). Production sequential row-major
retained; shipping GDN decode stays `sequential`. Persistent col-major session
layout is diagnostic-only and was not installed. Tok/s delta versus the sitting
sequential baseline: **0**.

Evidence also in
[`fixtures/opt109_persistent_gdn_state.json`](../../../fixtures/opt109_persistent_gdn_state.json);
rejection record in
[`REJECTION.md`](REJECTION.md).

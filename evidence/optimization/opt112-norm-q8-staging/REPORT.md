# OPT-112 — Fuse decode normalization into typed Q8 staging

Status: **deferred_below_trigger**. Authority llama.cpp
`cc83d7b4824f73cfdda4dfbb47ee39804f71b328`, GGUF SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`.

`claims_throughput: false`. `claims_performance_improvement: false`.
This sitting finished as `deferred_below_trigger` and retained
`current_bf16_q8_staging`. Fusion candidates were not implemented.

## Claim labels and proof limits

Screen-first host admission from OPT-099 D128/D2048 family aggregates (OPT-090
as peer). OPT-099 unmatched `ffn_norm` 26.288 ms / 32 tokens = 0.8215 ms/token
is an upper-bound clue, not a removable budget. Isolated decode
`activation_quant` is 0 (restage sits inside projection intervals);
`residual_mixer` is the foldable add launch for
`residual_norm_to_typed_q8_rounded`. Min potentially removable
**0.2496 ms/token** is below the **0.50 ms/token** trigger at both prefixes.
Fusion candidates `norm_to_typed_q8_rounded` and
`residual_norm_to_typed_q8_rounded` were not implemented. Quality/OPT-058 not
required on defer. Native fusion test, complete decode replay, and engine
throughput pairs were not run.

## Screen

Source `fixtures/opt099_matched_attribution.json`. OPT-099 unmatched
`ffn_norm` D2048 window 26.288 ms is an upper-bound clue
(0.8215 ms/token), not a removable budget.

| Prefix | ffn_norm ms/token (clue) | activation_quant ms/token | residual_mixer ms/token | potentially removable ms/token |
| --- | ---: | ---: | ---: | ---: |
| D128 | 0.8209 | 0.0000 | 0.2496 | 0.2496 |
| D2048 | 0.8215 | 0.0000 | 0.2525 | 0.2525 |

Trigger 0.50 ms/token at both prefixes.
min removable=0.2496.
verdict=deferred_below_trigger; reason=upper_bound_clue_is_not_removable_budget.

Decode attribution has no first-class `activation_quant` leaf (Q8 restage sits
inside FFN/mixer projection intervals). `residual_mixer` is the foldable add
launch for `residual_norm_to_typed_q8_rounded`; the FP32 residual store still
has to exist. RMSNorm arithmetic in `ffn_norm` / `input_norm` is retained.

## Consumers

GDN (48) and attention (16) mixer inputs reuse Q8_1 among same-encoding
groups. FFN (64) currently restages through sitting `llama_q4k_mmvq`
`block_q8_1`; a `paired_integer` pin would need FP32-scale `Q8Block` instead.
`logits_norm` stays on BF16. Encodings are not cross-reused.

## Decision

production_kept=True.
Shipping path `current_bf16_q8_staging`.
fusion_implemented=False.
**tok/s delta vs production: 0** (defer; production unchanged).

## Verification (2026-09-13T06:52:00Z)

**Verdict: pass** — measured defer is honest; production pins unchanged.

**Screen numbers traced from `fixtures/opt112_norm_q8_staging.json`**
- `min_potentially_removable_ms_per_token`: **0.24962933131322917** (D128
  `residual_mixer_ms_per_token`).
- D128: `ffn_norm_ms_per_token` 0.8209 (clue), `activation_quant_ms_per_token`
  0.0, `residual_mixer_ms_per_token` 0.2496, `potentially_removable_ms_per_token`
  0.2496, `threshold_ms_per_token` 0.5, `above_trigger` false.
- D2048: `ffn_norm_ms_per_token` 0.8215 (clue), `activation_quant_ms_per_token`
  0.0, `residual_mixer_ms_per_token` 0.2525, `potentially_removable_ms_per_token`
  0.2525, `above_trigger` false.
- `verdict`: `deferred_below_trigger`; `reason`:
  `upper_bound_clue_is_not_removable_budget`; `fusion_implemented`: false;
  `shipping_unchanged`: true; `production_kept`: true.

**Production unchanged**
- No OPT-112 references under `cuda/`; `cuda/full_scheduler.cu` unchanged.
- Fixture: `selected_path=current_bf16_q8_staging`,
  `claims_throughput=false`, `claims_performance_improvement=false`.

**Host validation**
```
uv run ruff format .
# 252 files left unchanged
uv run pytest -q tests/test_opt112_norm_q8_staging.py
# 12 passed in 0.03s
```

## Documentation (2026-09-13T06:52:00Z)

Added defer paragraph to `docs/06-system-optimization.md` and audit row to
`docs/65-documentation-audit.md`. Cross-links verified against
`pins/opt112_norm_q8_staging_contract.json`,
`pins/opt112_iteration_contract.json`, and
`fixtures/opt112_norm_q8_staging.json`.

## Delivery (2026-09-13T06:52:00Z)

Delivery authorized after independent verification pass. Marked OPT-112 `done`
with verdict `deferred_below_trigger`. Coupled IDs: none. First eligible pending
task: **OPT-113**.

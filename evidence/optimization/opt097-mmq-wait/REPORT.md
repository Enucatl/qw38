# OPT-097 MMQ split X/Y wait report

Updated: 2026-09-12T10:05:00Z

## Decision

- control: `joined_wait` (`q4_i128_j128_fma_async_x`)
- candidate: `split_xy_wait` (`q4_i128_j128_fma_async_x_split_wait`)
- production_kept: `true` (joined_wait retained; split_xy_wait rejected)
- selected_mmq_split_xy_wait: `false`

## Independent verdicts

- `kernel_parity_pass`: `true` (bitwise exact on aligned K256/512/5120/17408 and fallbacks)
- `model_quality_pass`: `true` (parity-only; full Q not required for reject)
- `performance_pass`: `false` (complete FFN 8.474 ms control vs 8.540 ms candidate)
- `production_kept`: `true`

## Complete FFN screen (N4096, 3 samples)

- control joined_wait: **8.474** ms
- candidate split_xy_wait: **8.540** ms
- saving: **-0.066** ms (regression; keep=false)

## Eligibility

- OPT-090 P4096 gate/up mean **9.517** ms/pair; estimated removable wait **5.177** ms → proceed

## Notes

Split wait overlaps X copy with half1 MMA but regressed on this GPU. Production pin remains `kSelectedMmqSplitXYWait = false`. P4096/decode-guard acceptance sittings deferred (`incomplete: true`).

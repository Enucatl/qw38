# OPT-105 MMQ 64x128 double raw-X report

Updated: 2026-09-12T17:33:55Z

## Decision

- control: `i128_j128` (`q4_i128_j128_fma_async_x`)
- candidate: `i64_j128_x2` (`q4_i64_j128_fma_async_x2`)
- production_kept: `True`
- selected_mmq_double_x: `False`

## Independent verdicts

- `kernel_parity_pass`: `True`
- `model_quality_pass`: `True`
- `performance_pass`: `False`
- `production_kept`: `True`

## Complete 64-layer FFN screen

- control: 639.904419
- candidate: 722.627869
- saving_ms: -82.7234497
- keep: False

## GEMM screen (P4096, 1 warmup + 3 samples)

- gate/up control **2.571** ms vs candidate **2.911** ms
- down control **2.557** ms vs candidate **2.900** ms

## Resources

- control i128 shared **94736** B, occupancy 1, regs 255
- candidate i64×2 shared **75280** B, occupancy 1, regs 244
- device opt-in **101376** B

## tok/s delta vs OPT-098 P4096

- baseline: 3046.23 tok/s
- post: 3046.23 tok/s (unchanged)
- delta: **0** (rejection)

## Notes

Two raw-X slots on 64x128 vs one-slot 128x128 joined-wait control. Same
direction as OPT-065 (64×128 slower than 128×128); the second X slot did not
recover it. OPT-097 split wait stays rejected. Tails use the synchronous
reference. P4096 five-pair and D128/D2048 guards deferred after the failed
≥5 ms FFN screen (`incomplete: true`).


# OPT-070 — Revalidate installed Q8/MMQ keeps

Status: **measured this sitting**. Authority llama.cpp `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`, GGUF
SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`. No new kernel/layout sweep. Screen-only `keep=true` is
rejected by the runner. Shipping selectors remain `r2_w2` and `fma_async_x`.

## OPT-069 reporting correction

REPORT.md was unmeasured after the sitting; rewritten from retained sidecars without altering numerical samples.
Measurement UTC of the retained sitting: 2026-09-11T12:18:36Z.
Three outcomes remain unpassed versus llama / OPT-056; the previous
unmeasured labels were a report bug, not a missing sitting.

## Arithmetic and quality

Q8 OPT-074: q8_mixer_k5120 v2_admitted=False coverage=unadmitted; q8_mixer_k6144 v2_admitted=False coverage=unadmitted.
MMQ OPT-074: q4_down_k17408 v2_admitted=False coverage=unadmitted; q4_gate_up_k5120 v2_admitted=False coverage=unadmitted.
Missing coverage is unresolved admission, not a numerically validated keep.
OPT-073 quality-v3 absolute=fail;
engine non-regression=pass;
quality-v2 all=False.

## Q8 r1_w4 vs r2_w2 (D2048 complete rotating mixer)

Control `r1_w4` vs installed `r2_w2`.
48 GDN input/output groups and 16 attention input/output groups in layer
order, including RMS/Q8_1 staging and OPT-058 invalidation on both sides.
X-prefetch held at the installed `fma_async_x` pin.

Component n=10 means: control 7.455 ms,
candidate 7.506 ms.
Paired CI (control-candidate): -0.0726 ..
-0.0303 ms; mean diff -0.0514 ms.
Engine D2048+32 five pairs: 874.076 vs
876.572 ms; regression UB
3.8115 ms vs 2% limit
17.4815 ms;
e2e non-regression=True.
Verdict: **inconclusive**
(['opt074_coverage_unadmitted']).
Q8 cannot revert: OPT-074 mixer families are unadmitted, so control
admission does not pass. Shipping `r2_w2` stays unresolved.
Proof boundary: complete rotating mixer + five target engine pairs.

## MMQ fma_async vs fma_async_x (P4096, 64 FFNs)

Tile 128x128. Complete FFN includes staging, SwiGLU, down, residual.
Q8 decision held at the resulting Q8 verdict / installed r2_w2.

Component n=10 means: control 887.292 ms,
candidate 838.523 ms.
Paired CI: 47.5883 ..
49.9496 ms; mean diff 48.7690 ms.
Engine P4096 five pairs: 1437.039 vs
1388.133 ms; regression UB
-38.2990 ms vs 2% limit
28.7408 ms;
e2e non-regression=True.
Verdict: **inconclusive**
(['opt074_coverage_unadmitted']).
MMQ cannot be a numerically validated keep: OPT-074 Q4 families are
unadmitted. Shipping `fma_async_x` stays unresolved.
Proof boundary: 64-layer complete prompt FFN + five target engine pairs.

## Production dispatch

Scoped Q8 layout and MMQ async-X overrides are applied in the production
translation units before separate graph capture.
Q8 launches: [{'config': 'r1_w4', 'last_q8_layout': 'r1_w4', 'last_mmq_kernel': '', 'last_mmq_async_x': False, 'effective_q8_rows_skinny': 1}, {'config': 'r2_w2', 'last_q8_layout': 'r2_w2', 'last_mmq_kernel': '', 'last_mmq_async_x': False, 'effective_q8_rows_skinny': 2}].
MMQ launches: [{'config': 'fma_async', 'last_q8_layout': '', 'last_mmq_kernel': 'q4_i128_j128_fma_async', 'last_mmq_async_x': False, 'effective_q8_rows_skinny': 2}, {'config': 'fma_async_x', 'last_q8_layout': '', 'last_mmq_kernel': 'q4_i128_j128_fma_async_x', 'last_mmq_async_x': True, 'effective_q8_rows_skinny': 2}].
Engine graph proof: Q8 override_before_capture=True
graph_capture_separate=True pairs=5;
MMQ override_before_capture=True
graph_capture_separate=True pairs=5.

## tok/s

This increment does not rerun OPT-069 P/D oracles. Official tok/s delta
versus the then-current OPT-069 baseline is 0 (shipping selectors unchanged).
OPT-069 retained P4096 Quartz 2914.65698 tok/s vs llama
3142.517034.

# OPT-064 — Improve Q8 decode row grouping

Status: **kept**. Production Q8 path stays `dp4a_q8_1`. OPT-047 historical
warps remain 4/4/4. OPT-064 layout pin is **r2_w2** (2 rows/CTA, 2 warps/row)
for skinny, medium, and wide. One rule; skinny was not split. No throughput
claim. No D2048 sitting in this increment.

## Layouts

| Id | RowsPerCta | WarpsPerRow | Rotating weighted ms | Hot weighted ms | Regs | Local | Occ |
|---|---:|---:|---:|---:|---:|---:|---:|
| r1_w4 | 1 | 4 | 3.34318942 | 2.47074127 | 40 | 0 | 12 |
| **r2_w2** | 2 | 2 | **2.67639479** | 2.85320511 | 40 | 0 | 12 |
| r4_w1 | 4 | 1 | 2.84620795 | 3.01243728 | 40 | 0 | 12 |
| r8_w1 | 8 | 1 | 3.88437331 | 2.99929586 | 40 | 0 | 6 |

Weighted complete mixer uses OPT-060 counts: `48 * (gdn_input + gdn_output) +
16 * attn`. Screen: four layouts × three groups × (1 warmup + 3 paired
samples), hot and rotating. Keep uses **rotating**. Saving versus control:
**0.667 ms/token** (above the 0.10 ms effort threshold). r8_w1 lost on
rotating. r4_w1 was second. No larger tuning grid. No end-to-end for the
three losing variants.

Q8_0 bytes and Q8_1 scale/quant semantics are unchanged. 34-byte blocks keep
2-byte-aligned `get_int_b2` loads. Single-warp rows shuffle-reduce with no
shared array or CTA barrier. No persistent SoA copy and no OPT-024 repack.

## Staging

`matrix_vector` reuses Q8_1 staging for identical activation pointer and
column count. Pointer identity is not data identity: overwriting the buffer
at the same address without `invalidate_q8_decode_staging()` restaged 1 time
(stale); after invalidate, stage count became 2 and the output changed.
RMS writes of `normalized_` now clear the cookie.

## Correctness

Smoke M17/K32: four layouts match each other; independent decode of GPU Q8_1
and Q8_0. Correctness: M1/K32, M7/K64, M17/K512, M48/K5120, signed extremes,
zero, cancellation, tail guards, and held-out layers 62/63 sampled 16 rows ×
4 captures on real K5120/K6144.

## Runs

- Feedback: `build/optimization-runs/OPT-064/feedback/20260911T094708Z-8764c23e`
- Acceptance: `build/optimization-runs/OPT-064/acceptance/20260911T094719Z-88153dad`
- Raw screen: `screen-raw.txt`

tok/s was not remeasured. Mixer rotating delta is control 3.343 ms → 2.676 ms
(-0.667 ms/token). OPT-069 owns combined E2E / original llama outcome gates.

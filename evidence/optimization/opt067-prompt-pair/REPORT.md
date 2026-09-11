# OPT-067 — Pair prompt gate/up tiles with a BF16 SwiGLU epilogue

Status: **measured rejection**. Production prompt FFN stays separate
gate/up on OPT-065/066 `i128_j128` `fma_async` / `fma_async_x` with fused
SwiGLU+Q8. Paired `i64_j64` remains a non-production symbol.
`kSelectedFfnPromptPairPath` is `off`. No throughput claim.

## OPT-061 5 ms gate

Rotating prompt-FFN enclosing time from OPT-061: **699.91 ms** / 4096-token
prompt (64 layers). Avoided traffic if pairing dropped both FP32 gate/up
stores:

| Quantity | Bytes |
|---|---:|
| FP32 gate+up stores / layer | 570,425,344 |
| BF16 activated store / layer | 142,606,336 |
| Staged-Y Q8 / layer (reload) | 23,592,960 |
| FP32 stores / 64-layer prompt | 36,507,222,016 |

At the OPT-061 useful-byte estimate of **1015 GB/s** (not a DRAM-counter
claim), avoided FP32 stores are about **35.97 ms/prompt**, above the
**5 ms** go threshold. The kernel was therefore written. Complete-FFN
measurement, not the traffic estimate, decides keep versus reject.

## Candidates

Exactly two screen configurations:

| Id | Path | Notes |
|---|---|---|
| **control** | separate `i128_j128` `fma_async_x` + fused SwiGLU+Q8 | Best existing separate tile from OPT-065/066 |
| **candidate** | paired `i64_j64` FMA, Y loaded once per K256, BF16 SwiGLU epilogue, standalone BF16→Q8 | No OPT-066 X pipeline on this kernel |

Aligned paired dumps match separate `i64_j64` GEMMs bit-exactly. Activated
BF16 matches the separate i128 control (exact on production 16×4 K5120 and
aligned 64×64; DS4 envelope on unique M65 tails). Graph and eager outputs
match for the paired path. Workspace buffers were not removed
(high-water **1,819,620,864** bytes). Launch count 6 → 5.

## Resources

Queried after opt-in shared bytes:

| Variant | Occ | Regs | Local | Shared |
|---|---:|---:|---:|---:|
| control `i128_j128` `fma_async` | 1 | 255 | 64 | 76288 |
| paired aligned | 1 | 247 | 0 | 38144 |
| paired fallback | 1 | 255 | 24 | 38144 |

Device opt-in 101376 B. Extra X bytes: 0 (X pipeline not combined).

## Complete FFN screen (4096 rows, 1 warmup + 3 samples)

Control **8.51732349** ms. Candidate **11.2768526** ms. Occupancy 1, no
aligned spills. **keep=false** (latency lost). No further paired tile.

## Runs

- Feedback: `build/optimization-runs/OPT-067/feedback/20260911T112035Z-78a3da3d`
- Acceptance: `build/optimization-runs/OPT-067/acceptance/20260911T112046Z-0b3830bb`

tok/s was not remeasured. OPT-069 owns combined E2E / original llama
outcome gates. Speedup is 0 (separate gate/up unchanged).

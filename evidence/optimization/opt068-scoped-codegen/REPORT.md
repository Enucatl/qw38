# OPT-068 — Scoped optimized CUDA compilation

This increment **claims no throughput improvement**. Production CUDA flags
stay `-O2 --fmad=false`. Host `-O2 -ffp-contract=off` is unchanged.

## Family choice (before compiling variants)

OPT-061 rotating complete-family ranking:

| Rank | Family | Rotating enclosing |
|---|---|---|
| 1 | prompt-ffn | 699.91 ms |
| 2 | decode-ffn | 17.20 ms |
| 3 | decode-mixer | 5.44 ms |

Selected **Q4 prompt MMQ** (`q4_prompt_mmq`), tile `i128_j128`, production
kernel `q4_i128_j128_fma_async_x`. Cooperative Q4 decode and Q8 decode were
not compiled. Launch wrappers and optimized instantiations live in
`cuda/q4_prompt_mmq.cu`. `full_scheduler.cu` does not include the MMA device
header.

## Builds

Three isolated object directories, each with an OPT-057-style
`nvccflags.stamp`. Linked engine objects (scheduler, host, strict CUDA)
used default `NVCCFLAGS`. No `--use_fast_math`, FTZ, approximate math, or
`-maxrregcount`. Explicit `__fmaf_rn` / `__fmul_rn` / `__fadd_rn` remain
noncontracting.

| Id | Flags | Object SHA-256 | Bytes |
|---|---|---|---|
| `o2_fmad_false` | `-O2 --fmad=false -Xptxas=-v` | `0d88926180d9757c33613fc56045e67c5136c8bfd96b1b9aef9ca45690e2c945` | 7377760 |
| `o3_fmad_false` | `-O3 --fmad=false -Xptxas=-v` | `369bff7dece415fc240c15a1cc1672cef9e4c2d1c3b40ef57b98c32ad75c395f` | 7385848 |
| `o3_fmad_true` | `-O3 --fmad=true -Xptxas=-v` | `bed220df6ca53c77c5c44e2d73f524c3c9eb7eac2c05cf448d34be526df4d303` | 6900280 |

First-time compilation setup was an explicit `make -j3` of the three
variant objects (MMA header). Warm feedback did not recompile.

## ptxas / SASS (hypothesis, not acceptance)

Queried production pipeline kernel: occupancy 1, 255 registers, 64 local
bytes, 76288 shared. Async-X attributes: 229 registers, 0 local, 94736
shared. Same for all three builds.

| Variant | FFMA | FMUL | FADD |
|---|---|---|---|
| O2 | 12987 | 61317 | 36148 |
| O3 | 12987 | 61317 | 36148 |
| O3+FMA | 39212 | 35092 | 9923 |

O3 does not change those opcode counts versus O2. `--fmad=true` contracts
FMUL/FADD into FFMA; that is not a keep condition. Instruction-count
decrease is a hypothesis, not acceptance.

## Numerics

Smoke tiny 17×256×17 and correctness random/cancel 128×256×128 plus held-out
layers 62/63, 16 rows × 4 prompts at production K (5120/17408), shared FP64
decode references: all three builds passed. FMA max-abs on held-out was
within the same ds4 envelope as O2 (example L62 gate 0.038814 vs 0.038814).
Contraction did not fail v2-style primitive checks on these inputs, but
FMA still needs a complete-cost win for production admission.

## Screen complete FFN (layer 0, 4096 rows, 1 warmup + 3 samples)

| Variant | Mean ms | 64-layer prompt delta vs O2 |
|---|---|---|
| O2 control | **8.33649063** | — |
| O3 | 8.34392548 | −0.476 ms (slower) |
| O3+FMA | 8.34093857 | −0.285 ms (slower) |

Component gate is ≥5 ms/4096-token prompt. Neither candidate is faster.
O3 is cosmetic (same SASS opcode counts, no complete-cost win) so O2 is
retained. Strict/host flags were inspected unchanged.

## Keep

**Retain `-O2 --fmad=false`.** Rejected flags: `-O3 --fmad=false`,
`-O3 --fmad=true`. Production Make rule for the family stays the default
`NVCCFLAGS`. Diagnostic variant directories and stamps remain for later
regression. No tok/s sitting; OPT-069 owns combined E2E.

Runs:
- feedback `build/optimization-runs/OPT-068/feedback/20260911T114617Z-0641bcd2` (16.35 s / 300 s, compile cache hit)
- acceptance `build/optimization-runs/OPT-068/acceptance/20260911T114635Z-c7cb6219` (25.03 s / 7200 s)

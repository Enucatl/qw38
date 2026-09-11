# OPT-066 — Overlap packed Q4 weight fetch with prompt MMA

Status: **keep**. Production Q4 prompt MMQ on the accepted OPT-065
`i128_j128` tile now prefetches packed 144-byte Q4 blocks with `cp.async`
into a single 16-byte-aligned shared slot, overlapping the existing FMA
plus packed-Y pipeline. Arithmetic is byte-identical to the Y-only
control. No throughput (tok/s) claim. Complete FFN at P4096 decides keep.

## Variants

Control is `fma_async` (Y ring only). Candidate is the same tile with
raw-X prefetch (`q4_i128_j128_fma_async_x`). No other tiles.

Two-stage extra shared is `2*I*144+16 = 36880` bytes. Added to the
existing Y-pipelined footprint (76288) that is **113168** bytes, which
exceeds this device's `sharedMemPerBlockOptin` of **101376**. The kernel
therefore uses one prefetched raw slot (`I*144+16 = 18448` bytes).
Candidate shared is **94736**. Slot lifetime: unpack into the decoded X
tile, then reuse the raw slot for the next K256 block while MMA consumes
decoded X; wait_all covers the combined X+Y commit group; a CTA barrier
follows every consumer.

4096-row P, real layer-0 rotating weights, 1 warmup + 3 paired samples:

| Variant | Gate/up ms | Down ms | Occ | Regs | Local | Shared |
|---|---:|---:|---:|---:|---:|---:|
| control `fma_async` | 2.769579 | 2.791424 | 1 | 255 | 64 | 76288 |
| **candidate `fma_async_x`** | **2.535787** | **2.536107** | 1 | 229 | 0 | 94736 |

Complete FFN (staging + gate + up + SwiGLU-Q8 + down + residual):
control **9.360267** ms, candidate **8.614912** ms (`keep=true`).

Tails and misalignment stay on the synchronous fallback
(`q4_i128_j128_sync_fallback`). Exact GPU output matches the control on
M128/N128 K256/512/768, tail M129/N129/K768, and a stale-shared relaunch
with a different activation. 16×8 FP64 samples per synthetic shape and
64 cached host samples per production FFN shape all passed. Q6/Q8,
stream-K, and MMQ quantization are unchanged.

## Stage lifetime

| Stage | Behavior |
|---|---|
| warmup | `cp.async` raw X[k0] (I×9 16-byte chunks) + Y[k0,h0]; one commit group; wait_all; unpack into decoded X; CTA barrier |
| iter k load | half 0 issues next Y and raw X[k+1] into the free raw slot in the same commit group |
| iter k compute | MMA both K128 halves on decoded X and the active Y stage |
| next-stage ready | wait_all after half 0 (combined X+Y group); unpack after half 1 MMA |
| drain | last kb0 issues no next X; Y drain matches the existing pipeline |

## SASS and sanitizer

SM120 encodes `cp.async` as `LDGSTS.E.BYPASS.128` (47 hits in
`build/quant_mmv.cuda.o`). Candidate launch attributes: occupancy 1,
229 registers, 0 local bytes, 94736 dynamic shared. Control still spills
64 local bytes. `compute-sanitizer --tool memcheck` on correctness
(K256/512/768, tail, stale shared, production samples): 0 errors.

## Runs

- Feedback: `build/optimization-runs/OPT-066/feedback/20260911T105659Z-f7f8ceb9`
- Acceptance: `build/optimization-runs/OPT-066/acceptance/20260911T105704Z-056f463e`
- Raw screen: `screen-raw.txt`
- SASS excerpt: `sass-excerpt.txt`

tok/s was not remeasured. OPT-069 owns combined E2E / original llama
outcome gates. Component complete-FFN delta is −0.745 ms (1.086× vs
Y-only control). Official tok/s speedup for this increment is 0 until
that sitting.

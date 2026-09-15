# OPT-151 — Long-decode QK/PV output-producing MMA

Status: **screened_out** (draft documentation; verifier pass pending). Shipping decode
attention `dense_bf16_tile_f16_mma_decode_v1`. Hardware executed on
`NVIDIA GeForce RTX 5090` with image `qw38-cuda:13.0.2`.
Source `b4582b94168a94342027c004ca4289c54bffffff` (dirty). Measured at
`2026-09-15T12:58:20Z`.

Parent is shipping OPT-137 `dense_bf16_tile_f16_mma_decode_v1` (scalar QK/PV
plus zero-weight MMA helper) on `decode_segments8` with OPT-148 flash-vec
below 8192. Candidate `decode_attention_qk_pv_mma_v2` uses output-producing
MMA for both QK and PV with llama Ampere D256 ncols1=1/ncols2=8 64-thread
geometry, dense BF16 tiles converted once to F16, GQA6 padded to eight
columns. Position>=8192 boundary unchanged. OPT-137 historical keep is not
rewritten (`kSelectedOpt137DenseMma = true`). Production pin
`kSelectedOpt151QkPvMma` remains `false`.

Structured JSON: [`preflight.json`](preflight.json),
[`correctness.json`](correctness.json), [`mma-dep.json`](mma-dep.json),
[`screen.json`](screen.json), [`quality.json`](quality.json),
[`mechanism.json`](mechanism.json), [`compiled.json`](compiled.json),
[`compiled-sass.txt`](compiled-sass.txt), [`report.json`](report.json),
[`fixtures/opt151_qk_pv_mma.json`](../../../fixtures/opt151_qk_pv_mma.json).

## Identity

- Parent stack: `post124_plus_opt127_decode_segments8_plus_opt137_mma_plus_opt148_flash_vec`.
- Candidate ID: `decode_attention_qk_pv_mma_v2`.
- GGUF `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`.
- Llama revision `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`.
- Capacity `131072`; frozen partitions `131 / 170 / 170` (8448 / 33024 / 131072).
- Screen: 1 warmup + 3 alternating parent/candidate pairs per shape, 16-layer
  complete attention family (query prep, BF16→F16, QK, softmax, PV, partials,
  merge, output gate).

## Candidate

Both QK and PV consume `mma.m16n8k16.row.col.f32.f16.f16.f32` accumulators
(PTX fragment packing: A = K or V^T; B = Q or softmax weights; C = scores/VKQ).
No `0.0F * mma_scores` zero-weight helper. Dense BF16 tiles convert once via
OPT-111 rounding. Llama geometry: 64 threads (2 warps), occupancy pin 4,
nbatch_fa=64, nstages=2. Heads 6–7 masked in the eight-column tile.

Implementation: `cuda/opt151_qk_pv_mma_decode.cuh` (OPT-137 kernel untouched).
B-fragment packing was initially swapped on K vs N (`mma-dep` failed with
`qk_max_abs=0.0448`); after fixing Q/weight packing as
`B[k=tid*2+{0,1}][n=groupID]`, `qk_max_abs=3.35e-8`.

## Screen

Complete 16-layer attention family. Candidate is slower at both targets;
parent retained. Acceptance quality, state-memory, and performance
(`target_guard_v2` 3+10 AB/BA) skipped.

| prefix | parent ms | candidate ms | saving ms | candidate faster | vs parent |
| --- | ---: | ---: | ---: | --- | ---: |
| D8192 | 3.25240541 | 6.30413866 | −3.05173326 | false | ~1.94× slower |
| D32768 | 13.8037758 | 38.4661217 | −24.6623459 | false | ~2.79× slower |

advance=`False`; screened_out=`True`; reason=`screen_candidate_not_faster_at_both_targets`.

### Llama complete-attention comparison (OPT-150 identities)

Llama native 16-layer complete-attention family times from OPT-136 eligibility
(preflight / mechanism sidecars). Same GGUF and llama revision as OPT-150.
These are native fattn-mma-f16 family costs, not whole-engine decode-only
ratios.

| prefix | llama ms | parent ms | parent/llama | candidate ms | candidate/llama |
| --- | ---: | ---: | ---: | ---: | ---: |
| D8192 | 0.482374 | 3.252405 | 6.74 | 6.304139 | 13.07 |
| D32768 | 1.434741 | 13.803776 | 9.62 | 38.466122 | 26.81 |

OPT-136 quartz complete-attention excess (parent baseline, not this screen):
D8192 `12.374 ms`, D32768 `48.101 ms`. Closer llama MMA organization did not
close the gap; the v2 candidate regressed versus the scalar parent at both
targets despite real output-producing HMMA.

**Independent screen reconstruction:** native screen emits row means only; per-pair
round arrays are not captured in `screen.json` (unlike OPT-149
`d2048-pair-reconstruction.json`). Verifier cannot recompute means from raw
pairs without rerunning screen or extending the native workload.

## Geometry versus llama and OPT-137

| item | llama Ampere D256 ncols=8 | OPT-137 parent | OPT-151 v2 |
| --- | --- | --- | --- |
| nthreads | 64 (2 warps) | 256 (8 warps) | 64 (2 warps) |
| occupancy pin | 4 | 4 | 4 |
| nbatch_fa | 64 | 64 | 64 |
| nstages target | 2 | 2 (load convert) | 2 (load convert) |
| ncols1/ncols2 | 1 / 8 | 1 / 8 | 1 / 8 |
| QK | m16n8k16 MMA | scalar FP32 + warp_sum | m16n8k16 MMA |
| PV | m16n8k16 MMA | scalar FP32 | m16n8k16 MMA |
| dummy MMA | none | `0.0F * mma_scores[0]` | none |
| KV | F16 cache | dense BF16, convert once | dense BF16, convert once |
| Q scale | 1/sqrt(256) F16 RN | FP32 prepared_q | 1/sqrt(256) then F16 RN |
| partitions | n/a | frozen 131 / 170 / 170 | same frozen buckets |

v2 follows llama's 64-thread MMA geometry rather than OPT-137's eight-warp
scalar layout. Residual costs inside family time: dense BF16 tile load,
OPT-111 FP32 round-trip to F16, two-warp softmax combine, partition merge,
query prep. Unknown attribution: how much of the ~1.9–2.8× regression is MMA
instruction count (1112 HMMA vs parent 2), occupancy/thread geometry, or
softmax/merge overhead.

## Correctness

Identity vs parent at positions 128, 1023, 1024, 4096, 4097, 8191, 8192,
8192 layers 3/7/63, 8193, 32767, 32768, 32769, 65535, 65536, 131071: all
`ok`. Below 8192, parent and candidate share the same kernel (max_abs 0).
At >=8192, candidate path is `decode_attention_qk_pv_mma_v2` with max_abs
about 0.005–0.008 versus scalar parent (F16 MMA vs FP32 dots). Zero
nonfinites. Graph/eager, cancellation, handoff, and same-math workloads
passed.

Isolated MMA products (`mma-dep`): `qk_max_abs=3.35e-8`, `pv_max_abs=0`,
both accumulators data-dependent, no zero-weight helper.

## Mechanism

Compiled `decode_attention_qk_pv_mma_v2` contains `HMMA` count `1112`
([`compiled.json`](compiled.json), [`compiled-sass.txt`](compiled-sass.txt)).
Graph mechanism at D8192 and D32768: 48 MMA launches each, topology
recapture 0, graph bytes 23068672. `kSelectedOpt151QkPvMma` stays false.
HMMA count and mma-dep prove real tensor-core QK/PV products; they do not
imply a keep.

## Quality

OPT-058 invoked=`False` (skipped after screen fail). Candidate NLL was not
measured (`candidate_nll_not_measured=true`). Long-cache 8K/32K/131040+32
quality was not run. Screened-out retains parent; quality skip is not a
stubbed NLL pass.

Held-out NLL `None`. ppl_ratio=`None`.

## Keep policy

`target_guard_v2` verdict `screened_out`.
reasons=`['screen_candidate_not_faster_at_both_targets']`.
claims_throughput: `False` (screened_out; no pin flip).
tok/s deltas: `{}`.

## Deltas

Candidate measured delta (complete 16-layer attention family):

- D8192: parent `3.25240541` ms → candidate `6.30413866` ms; saving
  `−3.05173326` ms (~1.94× slower).
- D32768: parent `13.8037758` ms → candidate `38.4661217` ms; saving
  `−24.6623459` ms (~2.79× slower).

Shipping delta: `0` (parent retained; `kSelectedOpt151QkPvMma = false`).

## Coverage

- Preflight: OPT-136 coverage valid; positive excess at D8192/D32768; no
  OPT-129 0.113 ms misuse.
- Correctness: 28 identity cases + mma-dep + same-math + cancellation +
  handoff; all `ok`.
- Screen: D8192 and D32768 complete-family means recorded.
- Mechanism: compiled SASS, graph launch counts, mma-dep at acceptance.
- Quality/state/performance: **not run** (screened_out gate).
- Compiled: candidate kernel symbol present; 1112 HMMA; data-dependent outputs.

## Contradictions

- **Closer llama geometry hypothesis falsified** — 64-thread MMA QK/PV is
  slower than OPT-137 scalar at both screen targets; organization match alone
  is not sufficient.
- **OPT-137 keep not rewritten** — parent scalar path remains shipping; v2
  mechanism success does not retroactively prove OPT-137's zero-weight MMA was
  output-producing.
- **Screen vs OPT-150 replay parent times** — OPT-150 D8192 quartz family
  `3.15950942` ms vs this screen parent `3.25240541` ms (different run/seed);
  not a path disagreement.
- **Per-pair reconstruction gap** — screen JSON lacks round arrays; independent
  mean verification requires rerun or native extension.

## Performance evidence checklist (draft, unverified)

1. **Measurement identity** — parent stack post-148; shipping decode attention
   `dense_bf16_tile_f16_mma_decode_v1`; candidate `decode_attention_qk_pv_mma_v2`;
   GGUF and llama pin recorded; screen 1 warmup + 3 pairs × 2 shapes × 16 layers.
2. **Coverage** — preflight, correctness, screen, mechanism, compiled artifacts
   present; quality/state/performance correctly absent after screen fail.
3. **Time accounting** — screen means are complete-family enclosing times
   (query prep through merge), not leaf HMMA or byte-size-only estimates.
4. **Contradictions** — llama-geometry hypothesis rejected by measured screen;
   OPT-137 historical keep preserved; see Contradictions section.
5. **Claim types** — screen and mechanism `measured`; llama comparison rows
   `derived` from OPT-136 sidecars; quality NLL `not measured`.
6. **Target/guard** — `target_guard_v2` opted in for survivors; skipped on
   screened_out; frozen targets D8192/D32768 decode_only (256 evals) not reached.
7. **Independent verification** — pending verifier pass; screen per-pair
   reconstruction **not available** in current artifacts.
8. **Reporting** — candidate measured delta and shipping delta reported
   separately; shipping delta `0`.

## Commands

```
uv run python tools/opt151_qk_pv_mma.py --phase preflight --mode feedback --run-dir build/optimization-runs/opt151
# ok=true

uv run python tools/opt151_qk_pv_mma.py --phase correctness --mode feedback --run-dir build/optimization-runs/opt151
# first mma-dep fail qk_max_abs=0.0448 (B-fragment K/N swapped); after PTX
# m16n8k16 B packing fix, ok=true

uv run python tools/opt151_qk_pv_mma.py --phase screen --mode feedback --run-dir build/optimization-runs/opt151
# ok=true, advance=false, screened_out=true

flock build/optimization-runs/qw38-gpu.lock docker run --rm --gpus all ... qw38-cuda:13.0.2 make cuda-opt151-diagnostics
# ok, includes qw38-cuda-opt058-quality-baseline-test (not invoked for NLL)

uv run python tools/opt151_qk_pv_mma.py --phase mechanism --mode acceptance --run-dir build/optimization-runs/opt151
# ok=true (HMMA 1112 + mma-dep). quality/state-memory/performance not run.

uv run python tools/opt151_qk_pv_mma.py --phase report --mode acceptance --run-dir build/optimization-runs/opt151
# verdict=screened_out

uv run ruff check tools/opt151_qk_pv_mma.py tests/test_opt151_qk_pv_mma.py
uv run pytest -q tests/test_opt151_qk_pv_mma.py
# 7 passed
```

## Status

verdict=`screened_out` production_kept=`False` blocked=`False`.
Shipping decode attention `dense_bf16_tile_f16_mma_decode_v1`.
OPT-148 flash-vec below 8192 unchanged. Prompt pin `prefill_attention_8x8_v1`
unchanged. Draft documentation **unverified**; verifier pass pending. No commit.

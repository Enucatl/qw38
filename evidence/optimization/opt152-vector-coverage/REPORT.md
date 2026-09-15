# OPT-152 — Vector attention coverage throughout the sub-8K region

Status: **keep** (draft documentation; verifier pass pending). Survivor
`flash_vec_all_short_v1`. Shipping decode attention `decode_attention_flash_vec_v1`
unchanged. Production coverage pin flipped to `all_short`. Hardware executed on
`NVIDIA GeForce RTX 5090` with image `qw38-cuda:13.0.2`. Source
`553c10444d44856620cc38773346eee8cda3a642` (dirty). Measured at
`2026-09-15T14:00:04Z`.

Parent `post148_flash_vec_parent_plus_opt137_mma` with kept OPT-148 flash-vec in
`[1024,4096]`, `warp_query` below 1024 and on `[4097,8191]`, OPT-137 MMA at
`>=8192`. OPT-151 remains unselected. This task is dispatch-only; OPT-148
arithmetic and `n_parts=16` are unchanged. `verified_max` stays 4096.

Structured JSON: [`raw/correctness.json`](raw/correctness.json),
[`raw/screen.json`](raw/screen.json), [`raw/quality.json`](raw/quality.json),
[`raw/state-memory.json`](raw/state-memory.json),
[`raw/performance.json`](raw/performance.json), [`raw/report.json`](raw/report.json),
[`dispatch-records.json`](dispatch-records.json),
[`family-time-reconstruction.json`](family-time-reconstruction.json),
[`answers.json`](answers.json),
[`fixtures/opt152_vector_coverage.json`](../../../fixtures/opt152_vector_coverage.json).

## Identity

- Parent stack: OPT-148 `decode_attention_flash_vec_v1` + OPT-137 MMA +
  OPT-147 `prefill_attention_8x8_v1` on `decode_segments8`.
- Candidate ID: `flash_vec_all_short_v1` (dispatch overlay; same launch
  `flash_vec_decode_attention`).
- Rejected alternate: `flash_vec_gap_only_v1` (screened but not advanced).
- GGUF `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`.
- Llama revision `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`.
- Capacity `131072`; crossover `1024`; MMA threshold `8192`; `verified_max=4096`.
- Mechanism: `unknown`; causal claim withheld.

## Candidates

Both candidates reuse the kept OPT-148 consumer without raising
`kSelectedDecodeAttentionVerifiedMax` or reviving OPT-130 `n_parts=8`.

| candidate | flash-vec range | below 1024 | `[4097,8191]` | targets |
| --- | --- | --- | --- | --- |
| `flash_vec_gap_only_v1` | `[1024,8191]` | `warp_query` | flash-vec (closes 4K gap) | D6144 |
| `flash_vec_all_short_v1` | `[0,8191]` | flash-vec | flash-vec | D512, D6144 |

Frozen choice rule: prefer `all_short` only if D512 and D6144 complete-family
means improve and D128 loses at most 2%; else `gap_only` if D6144 improves;
else `screened_out`.

Survivor: `flash_vec_all_short_v1` (`all_short_both_targets_improved_d128_within_2pct`).

## Dispatch (parent vs keep)

Quartz position is zero-based; llama padded `K.ne[1]` is not Quartz position.
Llama switches to `fattn-mma-f16` at padded `>=8192` (Quartz position 7936+).
Quartz OPT-137 MMA remains at position `>=8192`.

| position | parent path | keep (`all_short`) path | change |
| --- | --- | --- | --- |
| 0–1023 | `warp_query` | `decode_attention_flash_vec_v1` | newly admitted |
| 1024–4096 | `decode_attention_flash_vec_v1` | same | unchanged |
| 4097–8191 | `warp_query` | `decode_attention_flash_vec_v1` | gap closed |
| >=8192 | `dense_bf16_tile_f16_mma_decode_v1` | same | unchanged |

Final dispatch ranges: flash-vec `[0,8191]`, MMA `>=8192`, `verified_max=4096`.
Production pin: `kSelectedDecodeAttentionFlashVecCoverage = "all_short"`.
Shipping decode kernel pin unchanged: `kSelectedDecodeAttentionFlashVec = true`.

## Correctness

Positions 0,1,127,128,511,1023,1024,4095,4096,4097,6144,8190,8191,8192 with
layers 3/7/63, sampled FP64, candidate-row visibility, empty/tail partitions,
and actual post-launch dispatch. Graph replay used live `DecodeLaunchState.position`
across 1024/4096/8192 for both candidates; host-only selector equality is not
the proof.

Correctness sidecar `ok=true`. Graph/eager repeat, prompt/decode handoff,
cancellation, and checkpoint round-trip passed. See
[`dispatch-records.json`](dispatch-records.json) for per-position path labels
and llama kernel identities.

## Screen

Warmups `1`, pairs `3`, shapes D128/D512/D6144, 16-layer complete family
(prep, conversion, combine included). Both candidates screened in; only
`all_short` advanced under the frozen choice rule.

| candidate | shape | parent ms | candidate ms | saving ms | parent path → candidate path |
| --- | ---: | ---: | ---: | ---: | --- |
| gap_only | D128 | 0.509 | 0.503 | 0.006 | warp_query → warp_query |
| gap_only | D512 | 0.926 | 0.914 | 0.012 | warp_query → warp_query |
| gap_only | D6144 | 6.970 | 0.912 | 6.058 | warp_query → flash_vec |
| all_short | D128 | 0.507 | 0.426 | 0.081 | warp_query → flash_vec |
| all_short | D512 | 0.929 | 0.431 | 0.498 | warp_query → flash_vec |
| all_short | D6144 | 6.955 | 0.903 | 6.052 | warp_query → flash_vec |

Independent reconstruction from per-pair rounds:
[`family-time-reconstruction.json`](family-time-reconstruction.json).

## Quality

OPT-058 invoked=`True`; candidate NLL measured=`True`; skip_reason=`None`.
Native quality-region cases exercised newly admitted positions under
`flash_vec_all_short_v1`.

| case | control NLL | candidate NLL |
| --- | ---: | ---: |
| held_out_wikitext_1024 | 1.7875990840085783 | 1.785908735866253 |
| wikitext_nll | 1.5249350773895778 | 1.5254441451559835 |

Held-out ppl_ratio=`0.9983`; wikitext ppl_ratio=`1.0005` (max `1.01`).

## State and memory

Graph/eager live-position replay, checkpoint round-trip, and 128k memory_fit
`passed=true` (free `3510632448`, reserve `1610612736`). No graph workspace
regression versus parent.

## Keep policy

`target_guard_v2` with frozen survivor targets D512 and D6144 `decode_only`
(3 warmups + 10 AB/BA pairs, 256 decode evals). Guards D128/D2048/D8192/D32768
`decode_only` and `complete_request` plus P4096 prefill. Target L>1, guard
L>=0.98, decode p95<=1.05. 2% materiality reported separately and is not a
keep threshold.

| role | workload | metric | control tok/s | candidate tok/s | g | note |
| --- | --- | --- | ---: | ---: | ---: | --- |
| target | D512 | decode_only | 57.721 | 60.381 | 1.046 | pass |
| target | D6144 | decode_only | 39.203 | 58.194 | 1.484 | pass |
| guard | D128 | decode_only | 59.806 | 60.751 | 1.016 | within 2% |
| guard | D2048 | decode_only | 59.706 | 59.715 | 1.000 | held |
| guard | D8192 | decode_only | 48.843 | 48.842 | 1.000 | MMA unchanged |
| guard | D32768 | decode_only | 34.045 | 34.045 | 1.000 | held |

Verdict `keep`. `production_kept=true`. `claims_throughput=true`.

## Deltas

### Candidate measured delta

Acceptance `target_guard_v2` whole-engine decode_only (parent coverage vs
`flash_vec_all_short_v1` overlay, 10 alternating pairs):

- **D512**: 57.721 → 60.381 tok/s (**+2.66**, **g=1.046**).
- **D6144**: 39.203 → 58.194 tok/s (**+18.99**, **g=1.484**).

Screen complete-family attention (not whole-engine): D512 parent 0.929 ms →
candidate 0.431 ms; D6144 parent 6.955 ms → candidate 0.903 ms.

### Shipping delta

Production change on keep: coverage pin `parent` → `all_short`; decode
attention kernel pin stays `decode_attention_flash_vec_v1`. Because the shipped
change is exactly the admitted coverage overlay, shipping throughput delta
matches candidate measured delta on the frozen targets:

- **D512 decode_only**: **+2.66** tok/s (**g=1.046**).
- **D6144 decode_only**: **+18.99** tok/s (**g=1.484**).

Prefixes outside the newly admitted regions (D2048, D8192, D32768) show
neutral shipping impact (g≈1.000) as expected.

Quality result: `measured`.

## Coverage

- Correctness: 14 positions × 3 layers, graph replay at 1024/4096/8192, both
  coverage overlays.
- Screen: D128/D512/D6144 complete-family means for both candidates.
- Quality: OPT-058 candidate NLL + native quality-region at admitted boundaries.
- State/memory: graph replay, checkpoint, 128k reserve gate.
- Performance: `target_guard_v2` 3+10 AB/BA across 7 workloads.
- Independent verification: family-time reconstruction from screen rounds.

## Contradictions

- **Llama padded boundary vs Quartz position** — llama `fattn-mma-f16` at
  padded `K.ne[1]>=8192` is not the same as Quartz MMA at position `>=8192`;
  exact threshold parity remains untested (documented in OPT-150).
- **Complete-family screen vs whole-engine guards** — D128 complete-family
  screen shows a large flash-vec win (~16%), but whole-engine D128 decode_only
  guard is only g=1.016 because D128 was a non-regression guard, not a target;
  prefixes above 128 tokens dominate whole-engine decode_only.
- **Numerically identical candidate and shipping deltas** — expected because
  only the coverage pin flips; decode kernel identity is unchanged.

## Performance evidence checklist (draft, unverified)

1. **Measurement identity** — parent post-148 flash-vec + OPT-137 MMA; survivor
   `flash_vec_all_short_v1`; GGUF and llama pin recorded; screen 1+3 pairs,
   acceptance 3+10 AB/BA, 256 decode evals.
2. **Coverage** — 14 identity positions, graph live-position replay,
   D128/D512/D6144 screen, llama padded kernel labels in dispatch records.
3. **Time accounting** — screen means are complete-family enclosing times
   (prep through merge), not leaf-only.
4. **Contradictions** — llama padded K.ne[1] 8192 is not Quartz position;
   Quartz MMA remains position>=8192; see Contradictions section.
5. **Claim types** — dispatch experiment; mechanism `unknown`; screen and
   acceptance `measured`; quality NLL `measured`.
6. **Target/guard** — OPT-135 `target_guard_v2` for screened-in survivor.
7. **Independent verification** — family-time reconstruction from screen rounds;
   verifier pending on acceptance pair recomputation.
8. **Reporting** — candidate measured delta and shipping delta reported
   separately; numerically equal on keep because only coverage pin flips.

## Commands

```
uv run python tools/opt152_vector_coverage.py --phase correctness --mode feedback --run-dir build/optimization-runs/opt152
# ok=true

uv run python tools/opt152_vector_coverage.py --phase screen --mode feedback --run-dir build/optimization-runs/opt152
# screened_in=true, survivor=flash_vec_all_short_v1

uv run python tools/opt152_vector_coverage.py --phase quality --mode acceptance --run-dir build/optimization-runs/opt152
# candidate_nll_measured=true, ppl_ratio=0.9983

uv run python tools/opt152_vector_coverage.py --phase state-memory --mode acceptance --run-dir build/optimization-runs/opt152
# ok=true, 128k memory_fit passed

uv run python tools/opt152_vector_coverage.py --phase performance --mode acceptance --run-dir build/optimization-runs/opt152
# target_guard_v2 pass; D512 g=1.046, D6144 g=1.484

uv run python tools/opt152_vector_coverage.py --phase report --mode acceptance --run-dir build/optimization-runs/opt152
# verdict=keep, production_coverage=all_short

uv run pytest -q tests/test_opt152_vector_coverage.py
# 9 passed
```

## Status

verdict=`keep` production_kept=`True` coverage_pin=`all_short`.
Shipping decode `decode_attention_flash_vec_v1`. Prompt pin
`prefill_attention_8x8_v1`. Draft documentation **unverified**; verifier pass
pending. No commit.

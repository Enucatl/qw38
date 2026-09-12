# OPT-107 — Dispatch decode attention by measured prefix crossover

Status: **production_kept**. Authority llama.cpp
`cc83d7b4824f73cfdda4dfbb47ee39804f71b328`, GGUF SHA-256
`31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`. Control is
global in-kernel `warp_query`. Candidate is a prefix-aware hybrid that keeps
OPT-103 kernels unchanged and selects `vec128_online` only on a measured range.

`uses_opt098_arrays_for_keep_reject: false`. Fresh OPT-114 sitting is the
control. OPT-103 rejection evidence is retained; this is a new hybrid, not a
reinterpretation of the global OPT-103 candidate.

## Claim labels and proof limits

OPT-103 kernels, arithmetic, `n_parts=16`, KV layout, query preparation, and
merge unchanged; selector is a context-position branch only; screen thresholds
512, 1024, 1536, 2048; measure complete 16-layer attention at 128, 512, 1024,
1536, 2048, 4096; lowest threshold with positive paired saving at the threshold
and every longer measured position through 4096; below threshold and outside
verified range dispatch byte-for-byte to `warp_query`; 128K representative
verifies fallback and is not an extrapolated threshold; fresh OPT-114 sitting is
the keep/reject control; OPT-098 arrays are historical calibration only; D128
remains `warp_query` with output identity and 2% non-regression; D2048 requires
>=0.10 ms/token complete attention saving with a positive 95% paired interval;
no new vector kernel, GQA6 retry, KV type change, approximate math, partition
sweep, or prompt-attention change; component and engine measurements are separate
claims; the D2048 win is opportunistic and is not evidence that the underlying
vector stack is solved.

## Sitting identity

- device: NVIDIA GeForce RTX 5090
- llama_revision: `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`
- gguf_sha256: `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`
- execution_graphs: `ffn_only` (unchanged; OPT-114)
- shipping vec128 pin: `warp_query` (unchanged)
- crossover threshold: **1024** (`verified_max=4096`)
- OPT-114 control baselines (reporting deltas only): D128 53.4602 tok/s; D2048
  48.3830 tok/s; P4096 2981.0894 tok/s

## Eligibility

OPT-103 split retained (`warp_query` at D128, positive paired saving at D2048).
OPT-114 `aa_verdict=repeatable`. Eligible to proceed.

```json
{"eligible": true, "measurement_utc": "2026-09-12T22:26:45Z", "opt103_d128_mean_diff_ms": -0.10387520000000003, "opt103_d2048_mean_diff_ms": 0.3900411999999999, "opt103_d2048_positive": true, "opt103_rejection_retained": true, "opt103_shipping": "warp_query", "opt114_aa_verdict": "repeatable", "phase": "eligibility", "reason": "opt103_split_and_opt114_repeatable", "schema_version": 1, "task": "OPT-107", "uses_opt098_arrays_for_keep_reject": false, "verdict": "proceed"}
```

## Threshold screen (feedback; n=3 paired; CI rule, not keep-quality)

Measured positions [128, 512, 1024, 1536, 2048, 4096]. Screen thresholds
[512, 1024, 1536, 2048]. Lowest admitted threshold: **1024** (512 rejected:
`ci95_low < 0` at position 512). `verified_max=4096`. 128K fallback checks
selector only and is not an extrapolated threshold.

| pos | mean_diff_ms (warp − vec128) | ci95_low | positive |
|---:|---:|---:|---|
| 128 | 0.168 | 0.073 | true |
| 512 | 0.215 | −0.216 | false |
| 1024 | 0.465 | 0.190 | true |
| 1536 | 0.969 | 0.102 | true |
| 2048 | 1.025 | 0.417 | true |
| 4096 | 1.884 | 0.527 | true |

```json
{"admitted": true, "measurement_utc": "2026-09-12T22:21:51Z", "phase": "select", "reason": "lowest_positive_paired_saving", "rejected": [{"admitted": false, "failed_positions": [512], "missing": [], "threshold": 512}], "required_positions": [1024, 1536, 2048, 4096], "schema_version": 1, "task": "OPT-107", "threshold": 1024, "uses_opt098_arrays": false}
```

## Acceptance (3 warmup + 10 paired component; five 32-token AB/BA engine pairs)

Control is concurrent global `warp_query`. Hybrid uses threshold 1024. OPT-114
baselines are reporting deltas only, not keep/reject gates.

| Workload | path | control ms | hybrid ms | saving ms | 95% CI | engine control tok/s | hybrid tok/s | vs OPT-114 tok/s |
|---|---|---:|---:|---:|---|---:|---:|---:|
| D128 | warp_query | 1.830 | 1.762 | 0.068 | 0.025 .. 0.111 | 54.965 | 54.798 | +1.338 |
| D2048 | vec128_online | 11.845 | 10.354 | 1.491 | 1.092 .. 1.889 | 50.226 | 53.167 | +4.784 |

D128 engine 2% guard: regression_upper 3.56 ms vs 11.64 ms bound, pass.
D2048 component saving >= 0.10 ms with `ci95_low > 0`; engine candidate_mean <
control.

## Guards and parity

- **Quality (OPT-073):** `quality_v3_engine_non_regression=pass`;
  `quality_v3_quartz_engine_non_regression=pass`; inherited absolute fail
  remains visible.
- **State:** checkpoint/restore, cancellation, graph/eager, candidate-KV pass.
- **P4096 prefill:** hybrid 3060.7 vs control 3063.6 tok/s (ratio 0.999);
  prompt path unchanged.
- **128K fallback:** position 131072 dispatches `warp_query`; `extrapolated=false`.
- **Native correctness:** `qw38-cuda-opt107-attention-crossover-test --workload
  correctness` passed; `threshold_pin=1024`; D128-range `seq_abs=0`; no extra
  kernel/sync/alloc/copy from selector branch.

## Decision

Verdict: **keep**
(`d128_warp_query`, `d2048_component_ci_positive`, `saving_ge_0_10_ms`,
`d2048_engine_improved`, `quality`). `production_kept=true`.

Shipping vec128 pin stays `warp_query`; production crossover threshold=
**1024**. Positions `< 1024` and `> 4096` remain `warp_query`. Tok/s delta vs
OPT-114 D2048: **+4.784** (reporting only).

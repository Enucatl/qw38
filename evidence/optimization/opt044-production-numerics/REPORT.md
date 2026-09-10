# OPT-044 production numerics

Status: **Measured** independent host FP64 / llama Q8_1 freeze; production dispatch remains strict. `claims_performance_improvement: false`.

## Roles

- **Strict reference:** retained kernels, `--fmad=false`, original CUD-001/003 and scalar-oracle contracts.
- **Optimized production:** permitted Q8_1 staging, reordered FP32 reductions, FMA, F16 MMA with FP32 accumulation, and individually validated approximate transcendentals. Not installed. Unrepresented shapes stay on the strict path.

## OPT-042 large-shape reproduction

Host fill checksums match the frozen OPT-042 synthetic identities for `q4_k_17x256` and `q4k_gate_up`. OPT-042 `numeric_reject` is unchanged.

| Shape | Packed vs host max_abs (OPT-042) | FP32 host vs FP64 staged max_abs | Llama Q8_1 vs FP64 original max_abs |
|---|---:|---:|---:|
| q4k_gate_up | 0.000335693359 | 0.000335188401 | 3.11893277 |
| q4k_down | 0.00231933594 | 0.00263586978 | 2.98918227 |

The CUD-001 miss on production FFN shapes is already present in serial FP32 versus FP64 staged accumulation. Quantization versus original BF16 is ~3 abs for both Quartz Q8 and llama Q8_1. Pathological random Q6_K `q6_k_257x512` emits nonfinites; that case stays on the strict path and does not inflate ceilings.

## Frozen production ceilings (vs original BF16)

- q4k_gate_up abs `3.27488041` (llama `3.11893277`).
- q4k_down abs `3.13864238` (llama `2.98918227`).
- vs-staged llama error is zero in this FP64 GEMM replica, so those ceilings remain the strict 3e-4/2e-4 envelopes. Current production large-shape staged misses are existing failures, not a reason to loosen CUD-001.

## Quality suite

- Legacy QLT-001 PPL ratio stays **1.05**.
- Production-optimization PPL ratio is **1.01** on calibration and held-out 1024-target spans.
- Recurrence incremental NLL stays **0.02**.
- Continuation runner-up margin is **0** (no stored near-tie evidence).
- Current Quartz vs pinned llama wikitext PPL ratio: **1.0002643259599846** (legacy pass `True`, production 1.01 pass `True`).
- Held-out 1024-target span is `stream[16385:17409]` from frozen `recurrence_long.context`, disjoint from `wikitext_nll` `stream[1:1025]`. Scores for that span were not re-run in this sitting.

Baseline production-1.01 failures are reported, not tuned away.

## Proof limit

- No speedup claim.
- No unvalidated fast kernel in production.
- Structural/session exactness is not an accuracy compromise.


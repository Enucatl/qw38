# OPT-108 — Reproduce the actual NVIDIA pinned-llama vector-attention stack

Status: **primitive_rejected**. Authority llama.cpp
`cc83d7b4824f73cfdda4dfbb47ee39804f71b328`, GGUF SHA-256
`31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`.
Control is production OPT-107 hybrid (`production_opt107`: `warp_query` below
1024, `vec128_online` on `[1024, 4096]`). Candidate `llama_vec_nvidia` is the
source-faithful NVIDIA `flash_attn_ext_vec<256,1>` adapter: 128 threads,
16-byte copies, prepared Q once per head, occupancy/KV-length partitions (not
hardcoded 16), NVIDIA float2 V accumulation. BF16 KV path; AMD half2 and F16
cache migration are not used.

`claims_throughput: false`. `claims_performance_improvement: false`.
Matched primitive screen must win D2048 with a positive 95% CI before engine
integration; complete 16-layer attention, quality/state/128K, and P4096 guard
were not run after the primitive stop rule fired.

## Claim labels and proof limits

Source-faithful pinned organization only in the candidate TU; production pins
unchanged. Primitive positions 128, 512, 2048, 4096 on identical query/KV/output
buffers; 3 warmups + 10 paired CUDA events per position. Stop if matched
primitive loses; do not integrate a losing adapter. D2048 requires
`ci95_low > 0` on the matched primitive before complete-attention 0.50 ms/token
is attempted. OPT-103 and OPT-107 rejection evidence is retained; OPT-107 hybrid
remains shipping. Fresh OPT-114 sitting is historical calibration only for this
task; control is production OPT-107 on matched buffers, not OPT-098 arrays.
Component and engine measurements are separate claims; a point-estimate D2048
win whose CI includes 0 is not admission.

## Sitting identity

- device: NVIDIA GeForce RTX 5090
- llama_revision: `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`
- gguf_sha256: `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`
- execution_graphs: `ffn_only` (unchanged; OPT-114)
- shipping vec128 pin: `warp_query` (unchanged)
- crossover threshold: **1024** (`verified_max=4096`; OPT-107)
- candidate attrs (live): occupancy 2, 170 SMs, 217 regs, 0 local; D128
  `n_parts=1`, D2048 `n_parts=9`

## Eligibility

OPT-107 hybrid admitted. Eligible to proceed to matched primitive screen.

```json
{"eligible": true, "measurement_utc": "2026-09-12T22:50:02Z", "opt107_threshold": 1024, "opt107_production_kept": true, "phase": "eligibility", "reason": "opt107_hybrid_admitted", "schema_version": 1, "task": "OPT-108", "verdict": "proceed", "control": "production_opt107"}
```

## Correctness and parity

Native correctness tier passed: production vs candidate seq_abs ~1e-8–1e-9 at
short/crossover/D2048/4096 shapes; FP64 sample oracle inside 5e-3; zero and
graph/eager cases finite after OOB softmax weights were zeroed on short prefixes.
Parity phase `pass=true`; `production_pin_unchanged=true`. Engine integration,
quality suite, state/cache/128K, and P4096 prefill guard were **not** executed.

## Primitive screen (must win before engine integration)

3 warmups + 10 paired CUDA events on identical buffers. Control is production
OPT-107 hybrid path at each position. `primitive_win=false`.

| pos | control ms | candidate ms | saving ms | 95% CI | n_parts | positive |
|---:|---:|---:|---:|---|---:|---|
| 128 | 0.02168 | 0.02449 | −0.00281 | −0.00673 .. 0.00111 | 1 | false |
| 512 | 0.03222 | 0.02481 | +0.00741 | 0.00557 .. 0.00926 | 3 | true |
| 2048 | 0.03317 | 0.03101 | +0.00216 | **−0.00099 .. 0.00531** | 9 | **false** |
| 4096 | 0.04648 | 0.03726 | +0.00922 | 0.00738 .. 0.01106 | 14 | true |

D2048 is a point-estimate win whose **95% CI includes 0**; it fails the
positive-CI primitive gate. D128 is slower with CI including 0 (not a significant
regression, but not a primitive win). D512 and D4096 show positive CIs but do
not satisfy the D2048 stop rule.

```json
{"choose": {"d2048_faster": false, "d2048_positive": false, "d128_regressed": false, "missing": [], "primitive_win": false, "reason": "matched_primitive_lost"}, "primitive_win": false}
```

## Skipped after primitive stop rule

- Engine integration (complete 16-layer attention at D128/D2048)
- Complete-attention 0.50 ms/token acceptance gate
- Quality (OPT-073 successor policy), state/cache round-trip, 128K memory
- Five uninstrumented D128/D2048 engine pairs and P4096 prefill guard

## Decision

Verdict: **primitive_rejected** (`matched_primitive_lost`).
Production OPT-107 hybrid unchanged: shipping vec128 pin stays `warp_query`;
crossover threshold **1024**. Candidate adapter is diagnostic-only and was not
integrated. Engine integration is skipped when the matched primitive loses.
Tok/s delta versus production: **0**.

Evidence also in
[`fixtures/opt108_llama_vector_stack.json`](../../../fixtures/opt108_llama_vector_stack.json);
rejection record in
[`REJECTION.md`](REJECTION.md).

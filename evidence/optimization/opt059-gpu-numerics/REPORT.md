# OPT-059 GPU numerics v2

Status: **Measured policy freeze**. Llama GPU coverage for production K is
**unadmitted** until the authority exporter returns actual projection outputs
for those rows. `claims_performance_improvement: false`. No P/D oracles. OPT-046
is **not** installed.

## Why this sitting exists

OPT-044 classified CUD-001 as retained-reference arithmetic, but production still
reported a global `strict` selector and later tasks compared optimized kernels to
the 3e-4 scalar envelope. The OPT-044 llama number is a **host replica**
(`half(sum(q)*d)`), not GPU reduction. OPT-059 freezes a v2 policy from **actual
llama GPU error** when that export exists, and an explicit per-family admission
table in the meantime. It does **not** raise the old envelope just enough to pass
OPT-046.

## Arithmetic roles

| Role | Meaning |
|---|---|
| Strict reference | Retained kernels, `--fmad=false`, original CUD-001/003. Unrepresented shapes. |
| Optimized production | Selected Q8 `dp4a_q8_1` (OPT-047) and Q6 vocab `integer_q8_1` (OPT-048). |
| Engine summary | **mixed** — do not label the whole engine strict while those pins are selected. |
| Unrepresented fallback | `kSelectedProductionNumericsPath[] = "strict"` |

v1 `pins/production_numerics_contract.json` remains `selected=strict`,
`optimized_admitted=false`. Tests of v1 history stay on that contract. Current
dispatch tests read the admission table.

## Q8_1 staging audit

Do not reinterpret existing staged buffers. Consumer change is a later candidate.

| Path | Stored sum | Authority? |
|---|---|---|
| Quartz `quantize_bf16_q8_1` | `half(sum(integer quants))` in `Q8_1Block::q8_sum` | existing production format `quartz_q8_1_sum_q` |
| Distinct `quantize_bf16_q8_1_sum_x` | `half(sum(original x))` | llama GPU `quantize.cu::quantize_q8_1` |
| llama CPU `quantize_row_q8_1_ref` | `half(sum(q)*d)` | not GPU |
| OPT-044 host replica | `half(sum(q)*d)` | v1 diagnostic only |
| Pinned Q4 MMV | ignores stored sum; `dp4a(0x01010101, …)` recomputes integer q8 sums | |

IEEE half is exact for integers through 2048. **2047** survives; **2049** loses a
unit. **4064** is even and may round exactly (ULP 2 above 2048). Negative
counterparts match.

## Frozen case IDs (before candidate tuning)

Calibration layers **0, 3, 31, 32**; tokens **128, 512**. Held-out layers **62,
63**; tokens **2048, 4095**. Families: Q4 gate/up K5120, Q4 down K17408, Q8 mixer
K5120 and K6144, Q6 vocab K5120. At most 16 output rows and 4 captured
activations per family/K. At most 64 production FP64 dots per phase.

OPT-043 preprojection captures alone are insufficient. Down/SwiGLU inputs and
final-norm vocab inputs are required (`maybe_capture_down_input` after SwiGLU).

## v2 ceilings

Reference-only, declared before candidates:

- abs/RMS: `max(strict_ceiling, 1.25 * llama_gpu_error + 1e-6)`
- cosine loss: `max(1e-7, 1.25*(1-cos_llama)+1e-7)`
- old 1.05 headroom stored as diagnostics only
- missing llama GPU error → **unadmitted**, strict fallback
- nonfinite / pathological llama → strict, no budget from NaN/Inf
- zero vectors: abs/norm, not cosine
- FP32 rounding guard counts floating scale/group reductions, **not** integer
  multiplies; Q4_K group int product max = 15×127×32 = 60960; int32 overflow max
  documented. A bad total-error result cannot be waived by that bound.

Production K families copied from v1 currently have `llama_gpu_error = null`
and stay **unadmitted**. Filling those numbers requires
`qw38-llama-projection-export` on the pinned llama revision
`cc83d7b4824f73cfdda4dfbb47ee39804f71b328` and GGUF SHA
`31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`. Bind the
CUDA backend, `ggml_backend_synchronize` plus `cudaDeviceSynchronize`, then read
FP32 outputs. Do not substitute a CPU reimplementation.

## Llama GPU authority

`qw38-llama-projection-export` runs pinned llama.cpp CUDA `ggml_mul_mat`, then
`ggml_backend_synchronize` and `cudaDeviceSynchronize` before reading FP32
outputs. This sitting exported a **synthetic** Q4 17×256 probe (16 rows), not
GGUF production-K rows and not the OPT-044 checksummed fill:

- vs FP64 original max_abs **19.0954106**, RMS **13.1575881**, nonfinite 0
- sidecar: `evidence/optimization/opt059-gpu-numerics/llama-gpu-export/`
- CUDA `ggml_cpy` F32→Q8_1 aborts in this revision; staging bytes are not
  exported that way. Q8_1 sum semantics are audited from kernel source and the
  Quartz `sum_q` / `sum_x` variants.

Production K families remain **unadmitted**. Do not copy this synthetic probe
error onto the OPT-044 checksummed `q4_k_17x256` family.

## Admission mapping

`kProductionAdmission[]` is the explicit family / columns / staging / variant
table. `production_numerics_optimized_admitted()` is true because selected Q8/Q6
pins already dispatch optimized kernels. That is not v2 GPU admission
(`v2_admitted` remains false until llama GPU error is measured). An admission
record allows testing a candidate; production keep still requires task
acceptance.

## OPT-046 independent verdict

Winner `integer_q8_w4` is **not** installed. Production Q4 pin stays `packed`.

- vs FP64 original (17×256): max_abs **0.85785675** (quantization vs original BF16)
- vs FP64 staged: max_abs **3.05175781e-5** (meets the strict staged envelope)
- Historical CUD-001 vs scalar host MMV: **0.000305175781** > **3e-4** — fail
- v2 original: **unadmitted** (no llama GPU error for this sitting)
- Later Q4 work (OPT-062) owns any keep. Do not retune v2 budgets to pass it.

## Quality retained

PPL ratio **≤ 1.01** on both spans, recurrence incremental NLL **≤ 0.02**,
OPT-058 functional checks. Teacher-forced top-1 swaps: margin = 2× admitted max
logit error; only top-1/runner-up swaps within that bound. Logit error is
**unadmitted until reference logits**. No free-running token-identity
requirement. No speedup claim.

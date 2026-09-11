# OPT-063 — Fuse cooperative integer gate/up and SwiGLU

Status: **wired, not installed**. Production `kSelectedQ4DecodePath` remains
`packed`. `kSelectedFfnDecodePath` remains `paired_staged`. No throughput claim.
OPT-062's unfused integer Q8Block path is the complete-FFN control.

## Why this sitting exists

OPT-062 wired admitted cooperative Q8Block dots onto gate, up, and down as
separate launches. Gate and up share one activation vector. This task fuses
those two dots and the admitted SwiGLU epilogue without concatenating GGUF
weights or changing Q4 scale/min arithmetic.

Technique from pinned llama.cpp `mmvq.cu::mul_mat_vec_q` `has_fusion`
(cc83d7b, MIT): two weight pointers, one staged activation, separate
accumulators, SwiGLU after row reduction. Quartz-owned rewrite keeps the
OPT-062 group-shuffle reduction.

## Configurations

| Config | Staging | Gate/up | Down | Status |
|---|---|---|---|---|
| Unfused integer control | Q8Block FP32 scale, one shared stage | `q4k_coop_mmv_prequant_q8` ×2 + BF16 SwiGLU | cooperative `q4k_coop_mmv_q8` | OPT-062 admitted control |
| paired_integer 4-warp | same Q8Block stage | `q4k_coop_gate_up_swiglu_prequant_q8` | same down | testable; screen-faster |
| packed paired_staged | Q8Block | packed fused gate/up | packed `quant_mmv` | production control; logged only |

Warps per row: **4**. 4-warp kernel: 40 registers, 0 local bytes (no spills),
1024 B shared, 12 active blocks/SM. 2-warp was queried for attributes only and
not launched.

## Dispatch contract

- Selector `paired_integer` plus `integer_q8` launches the fused kernel.
- One gate/up stage, a separate down stage. No fused 32-value Q8 quant after
  SwiGLU.
- Trace/diagnostic separate gate/up taps fall back to the unfused integer
  path and label staging `q8_fp32_unfused_trace`.
- Q8 decode staging caches are invalidated before the fused launch.
- Dispatch records cover eager work and CUDA graph capture.

## Live sitting (2026-09-11)

Verification feedback:
`build/optimization-runs/OPT-063/feedback/20260911T091010Z-4024fe4e`
(`result_class=ok`, 1.93 s). Verification acceptance:
`build/optimization-runs/OPT-063/acceptance/20260911T091014Z-b6ab1891`
(`result_class=ok`, 4.88 s, screen 2.98 s).

Smoke/correctness: fused vs unfused vs staged FP64 max_abs **0** on M17/K256
(zero, alternating, large_neg, tail CTA), M33/K512, and 16 sampled rows of
M17408/K5120 × 4 vectors. All outputs finite. Output guards intact. Graph/eager
fused variants equal. Trace unfused fallback labeled.

Complete FFN (real layers 0/31/63, rotating 64 layers, 1 warmup + 3 samples;
gate/up charged 64 times, down 64 times, staging and SwiGLU included):

| Path | Gate/up | Down | Mean ms |
|---|---|---|---|
| integer_q8 unfused | `q4k_coop_mmv_prequant_q8` ×2 + SwiGLU | `q4k_coop_mmv_q8` | 12.3049278 |
| paired_integer | `q4k_coop_gate_up_swiglu_prequant_q8` | `q4k_coop_mmv_q8` | 11.2725649 |
| packed paired_staged (optional) | packed fused | packed | 15.8205442 |

Residual/down/activated prefix/trace/graph-eager max_abs **0**, nonfinite **0**.
Q8 scratch **19584** bytes. Screen winner is paired_integer (~1.09× vs unfused
integer, ~1.40× vs packed). Production pin stays packed: OPT-062 did not
install integer Q8Block (`v2_admitted=false`), and this sitting did not run
combined v2 NLL / functional answers. `claims_throughput` remains false.

## Commands

```sh
uv run pytest -q tests/test_opt063_integer_ffn.py
uv run python tools/run_optimization_task.py --task OPT-063 --mode feedback
uv run python tools/run_optimization_task.py --task OPT-063 --mode acceptance
```

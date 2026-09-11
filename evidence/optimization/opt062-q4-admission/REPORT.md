# OPT-062 Q4 admission — cooperative dots on every decode FFN leg

Status: **wired, not installed**. Production `kSelectedQ4DecodePath` remains
`packed`. `kSelectedFfnDecodePath` remains `paired_staged`. No throughput claim.
OPT-046's 3.5× complete-MMV figure is not this sitting's win.

## Why this sitting exists

OPT-046 already has cooperative integer Q4_K dots, but production decode FFN
takes `paired_staged` packed fusion for gate/up. `launch_quant_mmv_prequant`
also stayed on packed FP32. Changing the Q4 selector alone therefore moved
**down** and left gate/up on packed. This task wires the admitted Q8Block
cooperative path onto all three legs and records the launches that actually
ran.

## Configurations

| Config | Staging | Gate/up | Down | Status |
|---|---|---|---|---|
| Control | Q8Block FP32 scale | packed `q4k_gate_up_swiglu_prequant` | packed `quant_mmv` | production |
| Q8Block candidate | Q8Block FP32 scale, one shared stage | `launch_q4k_coop_mmv_prequant_q8` ×2, existing BF16 SwiGLU | cooperative `launch_q4k_coop_mmv` via `matrix_vector` | testable; screen-faster |
| Half-scale Q8_1 | quartz_q8_1_sum_q / llama_q8_1_sum_x | skipped | skipped | OPT-059 `v2_admitted=false` |

Warps per row: **4** (OPT-046 winner). No warps×staging×fusion sweep.
Integer paired fusion is OPT-063.

## Dispatch contract

- A `Q8Block*` function dispatches only Q8Block kernels. Selecting
  `integer_q8_1` must **not** reinterpret that pointer as Q8_1.
- Integer Q8Block FFN: one gate/up stage, a separate down stage.
- Dispatch records cover eager work and CUDA graph capture.
- Q8/Q6 reuse of the same workspace still invalidates decode staging.

## Historical pins

OPT-046 rejection and packed production pin are unchanged. Strict reference
tests still call the packed launcher explicitly. CUD-001 envelopes are not
the production gate; staged FP64 is.

## Live sitting (2026-09-11)

Implementation feedback:
`build/optimization-runs/OPT-062/feedback/20260911T084506Z-b0ae8a9d`
(`result_class=ok`, 3.35 s). Implementation acceptance:
`build/optimization-runs/OPT-062/acceptance/20260911T084528Z-e8dda304`
(`result_class=ok`, 4.38 s, screen 2.70 s).

Delivery verification (2026-09-11T08:49:10Z): feedback
`build/optimization-runs/OPT-062/feedback/20260911T084905Z-7c69b80a`
(`result_class=ok`, 1.65 s); acceptance
`build/optimization-runs/OPT-062/acceptance/20260911T084909Z-e97840bd`
(`result_class=ok`, 4.34 s, screen 2.69 s). Ruff clean; host pytest 19 passed.

Smoke/correctness: Q4 scale/min layout exact on all eight groups. Packed and
integer_q8 MMV vs original and staged FP64 max_abs **0** on M17/K256,
M19/K512, sampled M16/K5120 and M16/K17408 (zero, alternating, ±group,
independent group0–7 ±sums). Integer vs packed max_abs **0**. Integer launch
variant `q4k_coop_mmv_q8`. Typed `Q8Block*` prequant stays packed under
`integer_q8_1` and uses `q4k_coop_mmv_prequant_q8` under `integer_q8`. Buffer
restage delta_abs **214096**. Graph/eager prequant variants equal. Production
FP64 dots **128**. Half-scale skipped.

Complete FFN (real layer-0 weights, synthetic residual; rotating 64 layers,
1 warmup + 3 samples):

| Path | Gate | Up | Down | Stages | Mean ms |
|---|---|---|---|---|---|
| packed paired_staged | `q4k_gate_up_swiglu_prequant` | same | `quant_mmv_packed` | 1 + 1 | 15.7643099 |
| integer_q8 | `q4k_coop_mmv_prequant_q8` | `q4k_coop_mmv_prequant_q8` | `q4k_coop_mmv_q8` | 1 + 1 | 12.4760742 |

Integer vs packed: residual max_abs **7.62939453e-06**, down **7.62939453e-06**,
activated prefix **0**, graph/eager residual **0**, nonfinite **0**. Graph
capture recorded the same integer variants. Q8 scratch **19584** bytes
(`q8_workspace_bytes(17408)`). Selected/effective Q4 remain `packed`.

Component screen winner is integer_q8 (~1.26× complete FFN). Production pin
stays packed: OPT-059 `integer_dp4a_q8` is testing-admitted only
(`v2_admitted=false`), and this sitting did not run combined v2 NLL / functional
answer quality. D2048 probe not run (`historical_oracles=false`).
`claims_throughput` remains false.

## Commands

```sh
uv run pytest -q tests/test_opt062_q4_admission.py tests/test_opt059_numerics.py
uv run python tools/run_optimization_task.py --task OPT-062 --mode feedback
uv run python tools/run_optimization_task.py --task OPT-062 --mode acceptance
```

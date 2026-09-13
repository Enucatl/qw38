# OPT-122 — Lower-precision persistent GDN state

Status: **no_material_opportunity**. Parent authenticated post113_selected `ffn_only` after OPT-118 kept lazy/overlap. OPT-117 graphs remain rejected. Production pin stays `fp32`.

`claims_throughput: false`. `claims_performance_improvement: false`.

## Sitting identity

- llama_revision: `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`
- gguf_sha256: `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`
- parent: `post113_selected`
- execution_graphs: `ffn_only`
- measured_at: `2026-09-13T19:48:20Z`

## Traffic inventory (complete decode token after OPT-118)

Sequential GDN reads committed recurrent twice in registers and writes the candidate once. The 3.1 MiB per-layer working set fits in RTX 5090 L2, so DRAM is charged 1 read + 1 write of recurrent plus FP32 convolution rings. Decode commits by pointer swap. Prompt carry D2D production traffic is `False`.

| Item | Value |
|---|---:|
| gdn_core_ms | 2.00809646 |
| wall_ms | 18.1397324 |
| decode_d2d_bytes | 1013760 |
| dram_fp32_bytes | 317718528 |
| dram_bf16_bytes | 166723584 |
| dram_q8_bytes | 95944704 |
| peak_fp32_ms | 0.177298278 |
| peak_bf16_ms | 0.093037717 |
| fp32_recurrence_ms | 0.887004733 |
| bf16_recurrence_ms | 0.590308785 |
| memcpy_rec_fp32_ms | 0.204576001 |
| memcpy_rec_bf16_ms | 0.10368 |

## Storage candidates (at most two)

BF16 recurrent matrices with FP32 update/accumulation. Block-scaled integer 8-bit (group 32, FP16 scales) only if calibrated error and net byte savings justify the more aggressive stage. Convolution stays FP32.

- BF16 host RMS `0.00753242459`; Q8 RMS `0.017616956`.
- BF16 critical-path / roofline upper `0.0` ms; leaf save `0.2966959480000001` ms (never admits).
- Q8 critical-path upper `0.0` ms; justified `False`.
- Admitted engine candidates: `[]`.

## Materiality versus A/A

OPT-115 A/A geo P4096/D128/D2048 = 1.0001 / 1.0000 / 0.9999. Relative resolution `9.999999999998899e-05` → `0.0017423150837441827` ms on D128 wall `17.423150837443746` ms. Enclosing gdn_core is the complete-request sink. Recurrence-only leaf timing never admits. Decode commits by pointer swap; prompt carry D2D is unused at the 4096 chunk pin. DRAM is 1 read + 1 write of recurrent (L2 covers the second register pass) plus FP32 conv rings. Roofline upper bound is 0 when compute already exceeds FP32 DRAM time, so shrinking packed traffic cannot shorten gdn_core.

**Verdict: `no_material_opportunity`.** Production pin remains FP32. Complete-request A/B was not required after the quantified bound. Tok/s delta versus the sitting baseline is **0** (baseline unchanged).

## Quality / state / 128K

No packed encoding was admitted into the session, so OPT-116/OPT-058 candidate NLL is not in scope. 128K GDN capacity remains 158859264 B FP32. Checkpoint format is unchanged.

## Independent verdicts

- kernel_correctness: `True`
- model_quality: `True` (FP32 retained)
- state_memory: `True`
- performance: `True`
- production_kept: `True`


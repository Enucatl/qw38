# OPT-018 — Q4_K/Q6_K MMA MMQ quality

## Claim labels and proof limits

This increment admits production Q4_K/Q6_K prompt MMQ quality MMA
(`MMQ_ITER_K=256`, packed load-tiles, Q8_1 MMQ Y in the existing workspace).
Q4_K/Q6_K MMQ vs CPU dequant GEMM uses ds4 Q4_K parity association gate
(`abs > 0.20*sqrt(K)` and `rel > 0.05`).
fixed abs/rms retired for Q4_K/Q6_K MMQ admission.
variant retained with exact Q8 staging.
The parity gate owner remains blocked. This remasurement is
not an end-to-end 2K tok/s gate. There is no 8K/32K/128K throughput gate.

Live MMA envelope, J pin, FFN before/after, attribution, and remasurement are
**Measured**. llama.cpp Q4_K/Q6_K load-tiles / vec-dot / quantize_mmq_q8_1
provenance at `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` and ds4 `test_mmq_parity.cu` association gate are
**External**.

## Production path

`launch_quant_mmq` uses quality MMA when `prompt_rows >= 8` (J=128) and
`launch_quant_mmq_variant` otherwise. Mixer Q8_0 stays on `launch_q8_mmq_bf16`.
Decode `launch_quant_mmv` is unchanged.

Inner-loop repair on the quality kernel: float accumulators, Q4_K DS4 Y
(`half2` scale + prequant sum), half2 X scales at SRAM stride 76, and
`__launch_bounds__(256, 1)`.

## Component MMA association (**Measured**)

MMA and the retained variant are admitted against host CPU dequant-weight ×
BF16→float activation GEMM under the ds4 Q4_K parity rule (plan.md option C):
an element fails only when both `abs_error > 0.20 * sqrt(K)` and
`rel_error > 0.05`, with zero non-finites. Cases include prompt rows 8/9/64/65 on 17×256 and 1024×5120,
plus FFN 17408×5120 and 5120×17408 at 2048 rows, plus Q6_K 5120×6144.

## J pin (**Measured**)

FFN-shape sweep at 2048 prompt rows on 17408×5120 and 5120×17408, 3 CUDA-event
replicates, 0 warm-ups:
`mma_prompt_tile_2048=128 mean_ms=3.75303459 pinned=128 occupancy=1`
`mma_speed_2048_17408x5120 mma_ms=1.64595187 variant_ms=155.836014 occupancy=1`

Raw samples: `q4k-q6k-mma-j-sweep-raw.txt`.

## llama-competitive FFN bar (**Measured**, not the parity gate)

- FFN before (OPT-017 attribution): 2524.67725 ms
- FFN after: 399.287018 ms
- llama.cpp 2K wall: 636.184782 ms (`avg_ts` 3219.6604)
- `llama_competitive`: True
- Quartz mean tok/s: 625.792114 (walls [3272, 3275.05591, 3270.90405])
- `would_pass_opt016`: False (informational)
- `owns_opt016_parity_gate`: false; `opt016_status`: blocked

## Post-remasurement attribution (**Measured**)

OPT-014 categories on the same GGUF/GPU after the unperturbed samples
(`measurement_utc` 2026-09-08T18:33:22Z):

- ffn_mmq 399.287018 ms
- gdn 1643.46082 ms
- attention 1148.47461 ms
- wall 3194.70337 ms
- tok_s 641.061096

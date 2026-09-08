# OPT-017 — mixer Q8_0 MMA MMQ

## Claim labels and proof limits

This increment admits production mixer Q8_0 prompt MMQ MMA. Numeric
**envelopes unloosened**. **OPT-009 Q8_0 reference remains byte-exact** on
`launch_q8_mmq_bf16_variant` versus `launch_q8_mmq_bf16_reference`.
**OPT-016 remains the parity gate owner**. This remasurement is
**not an end-to-end 2K tok/s gate**. There is **no 8K/32K/128K throughput gate**.

Live MMA envelope, J pin, attribution, and remasurement are **Measured**.
llama.cpp Q8_0 MMQ MMA provenance at `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` is **External**.

## Production path

`launch_q8_mmq_bf16` uses MMA when `prompt_rows >= 8` (J=128) and the
OPT-009 tiled variant otherwise. Q8_0 is not routed through
`launch_quant_mmq`. Decode `q8_mmv_bf16` is unchanged.

## Component MMA envelope (**Measured**)

CUD-002 versus `launch_quant_mmq_variant` `kQ8_0` (`gpu_staged_*`): max abs
`5e-4`, RMS `2.5e-4`, zero non-finites. Cases: prompt rows 8/9/64/65 on
17×256 and 1024×5120, plus 48×256 tail. Versus unstaged BF16
`launch_q8_mmq_bf16_variant` is informational only and is not a pass/fail
gate.

## J pin (**Measured**)

Mixer-shape sweep at 2048 prompt rows on 12288×5120, 10240×5120, and
5120×6144, 3 CUDA-event replicates, 0 warm-ups:
`q8_mma_prompt_tile_2048=128 mean_ms=21.7802238 pinned=128 occupancy=2`
`q8_mma_speed_2048_12288x5120 mma_ms=8.85369587 variant_ms=42.9801064 occupancy=2`

Raw samples: `q8-mma-j-sweep-raw.txt`.

## Live 2K remasurement (**Measured**, not the OPT-016 gate)

- Quartz mean tok/s: 375.57019 (walls [5451.81201, 5454.271, 5453.04346])
- llama.cpp live avg_ts: 3208.29056 (`build_commit` cc83d7b)
- `would_pass_opt016`: False (informational)
- `owns_opt016_parity_gate`: false; `opt016_status`: blocked

## Post-remasurement attribution (**Measured**)

OPT-014 categories on the same GGUF/GPU after the unperturbed samples:

- ffn_mmq 2524.67725 ms
- gdn 1646.56665 ms
- attention 1208.98462 ms
- wall 5383.7417 ms

Mixer Q8_0 time remains inside `gdn` and `attention` until OPT-020.

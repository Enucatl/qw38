# OPT-042 — Study integer decode MMV admissibility

## Claim labels and proof limits

Live exclusive-RTX-5090 diagnostic of packed FP32 Q4_K decode MMV versus a Q4_K integer-dot (`__dp4a`) candidate. Proof boundary: claims no performance improvement; fixed CUD-001 envelope; unchanged FP32-scale Q8 staging; production dispatch unchanged; accepted keep denominators remain unchanged; does not substitute for the 2K llama.cpp parity gate; integer-dot is diagnostic only.

## Classification

- admissibility: `numeric_reject`
- promote_to_production_ab: false
- claims_performance_improvement: false
- publishes_successor_oracle: false
- selected_mmv_load_path: `packed`
- reverted: false

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp revision | `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` |
| Candidates | packed then integer, 3 warm + 30 alternating pairs |
| Views | complete (stage+MMV) and prequant (MMV only) |
| Envelope | CUD-001 max abs 3e-4, RMS 2e-4, zero nonfinites |
| Keep denominators | copied from OPT-041; unchanged |
| Nsight | not_used |

## Measured sitting

- Device: NVIDIA GeForce RTX 5090 compute 12.0
- measurement_utc: 2026-09-10T12:09:07Z
- synthetic weighted complete ms packed=0.06731271 integer=0.0604874678
- real weighted complete ms packed=0.0676808506 integer=0.0615350045
- staging_equal: true
- occupancy integer: {"4": 12, "8": 6, "16": 3}
- q8_0_mixer_sibling: not_run
- q6k_logits_sibling: not_run
- owns_opt016_parity_gate: false
- substitutes_for_opt016: false

# OPT-038 — Refresh post-ladder attribution and source gap map

## Claim labels and proof limits

This increment **claims no performance improvement**. It retains **D128 and D2048 oracles**
with **exclusive subsystem breakdowns**, **independent raw host-wall accounting**,
**matched pinned llama.cpp** public-API measurements, **matched-component experiment specifications**,
and a **recorded next-task order**. **accepted keep denominators remain unchanged**.
This protocol **does not substitute for the 2K llama.cpp parity gate**.
**llama-bench random decode is informational**.

Live numbers in `fixtures/opt038_post_ladder_gap.json` and this directory are the
Measured same-sitting exclusive RTX 5090 record. Accepted keep denominators remain
the OPT-034 copies (P 1869.84412, D128 25.3816128,
D2048 20.169548 tok/s). This increment does not publish a successor
oracle.

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp image | `qw38-llama-authority:cuda-13.0.2` |
| llama.cpp revision | `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` |
| P | exact 4096, attribution null, graphs created, 0 warm-ups, 3 cold replicates |
| D128/D2048 | prefix 128 or 2048 then 256 predetermined tokens, 3 warm + 30 measured |
| llama.cpp P | `llama-bench -p 4096 -n 0 --no-warmup -r 3 -ngl 99` |
| llama.cpp D | `qw38-llama-decode-oracle` via pinned `llama.h` and `llama_time_us` |
| Nsight | not_used |

## Measured sitting

- Device: NVIDIA GeForce RTX 5090 compute 12.0
- measurement_utc: 2026-09-10T07:55:02Z
- P Quartz mean tok/s: 1872.63806 ; llama.cpp avg_ts: 3241.632261
- D128 Quartz mean tok/s: 25.3562603 ; llama.cpp mean tok/s: 68.8080723
- D2048 Quartz mean tok/s: 20.1668205 ; llama.cpp mean tok/s: 67.0727771
- p_gap: 1.7310511466374876 ; d2048_gap: 3.325897461129284 ; decode_deficit_larger: true
- mmv_ms: 31.804891599999998 ; attention_core_ms_d2048: 13.2069445
- prefill_attn_ms: 579.519836 ; prefill_gdn_ms: 576.816162
- next_task: OPT-042
- next_task_order: OPT-042, OPT-041, OPT-040, OPT-039
- scout_recommended_order: OPT-039, OPT-040, OPT-042, OPT-041 (informational; not required to equal next_task_order)
- claims_performance_improvement: false
- publishes_successor_oracle: false
- owns_opt016_parity_gate: false
- substitutes_for_opt016: false
- opt031_not_in_order: true

## Independent raw host-wall accounting

Prefill 4K: raw_host_wall_ms 2218.05591; gpu_event_sum_ms 2217.47239493; graph_host_interval_ms 0.177492008; adjusted_reconstruction_ms 2218.05371; wall_raised false; prompt_graph_launches 64

Decode D128: raw_host_wall_ms 37.4397621; gpu_event_sum_ms 37.0530860501; graph_host_interval_ms 0.183418036; adjusted_reconstruction_ms 37.3150978; wall_raised false

Decode D2048: raw_host_wall_ms 49.1544418; gpu_event_sum_ms 48.8410518079; graph_host_interval_ms 0.210362986; adjusted_reconstruction_ms 49.0563354; wall_raised false

Matched-component experiment specifications live in `COMPONENT-PROTOCOL.md`. Those A/B experiments were not run in this sitting.

# OPT-032 — Freeze decode oracle and refresh sink attribution

## Claim labels and proof limits

This increment **claims no performance improvement**. It freezes **D128 and D2048 oracles**
with **exclusive decode categories** and **matched pinned llama.cpp** public-API
measurements, plus a **recorded next-task order**. This protocol
**does not substitute for the 2K llama.cpp parity gate**.
**llama-bench random decode is informational**.

Live numbers in `fixtures/opt032_decode_oracle.json` and this directory are the
Measured same-sitting exclusive RTX 5090 record. Historical OPT-026 Quartz mean
1746.71973 tok/s is contract transparency, not a keep.

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
- measurement_utc: 2026-09-09T15:07:45Z
- P Quartz mean tok/s: 1637.58594 ; llama.cpp avg_ts: 3139.909678
- D128 Quartz mean tok/s: 11.8731956 ; llama.cpp mean tok/s: 68.506172
- D2048 Quartz mean tok/s: 3.70951414 ; llama.cpp mean tok/s: 66.9333082
- p_gap: 1.9174014635225802 ; d2048_gap: 18.04368595829102 ; decode_deficit_larger: true
- mmv_ms: 53.609988200000004 ; attention_core_ms_d2048: 201.241058
- next_task: OPT-036
- next_task_order: OPT-036, OPT-033, OPT-035, OPT-034, OPT-037
- claims_performance_improvement: false
- owns_opt016_parity_gate: false
- substitutes_for_opt016: false

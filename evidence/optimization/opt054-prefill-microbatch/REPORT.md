# OPT-054 — Tune internal prefill batches without early commit

## Claim labels and proof limits

Live exclusive-RTX-5090 A/B of internal 512/1024/2048/4096 prefill microbatches
inside one atomic 4096-token transaction. Keep requires
**atomic 4096-token transaction**, **internal 512/1024/2048/4096 microbatches**,
**no early commit**, **graphs-off isolation then graphs-on shipping**,
**keep 4096 unless a smaller size wins complete P**,
**D128/D2048 95% floors and p95 inside 105% versus OPT-053**,
**OPT-044 admits cross-size arithmetic drift**, and
**does not substitute for the 2K llama.cpp parity gate**. Copied denominators
are P 2895.42773, D128 37.5605927, D2048 35.7286987.

## Decision

**no-change** — `reverted`=false;
`keep_sitting_skipped`=true;
selected_prompt_microbatch_rows=4096;
graphs-off A/B winner 4096 (512 1654.02405 ms / 2476.38477 tok/s, 1024 1518.56348 ms / 2697.28589 tok/s, 2048 1451.12952 ms / 2822.62891 tok/s, 4096 1441.02893 ms / 2842.41333 tok/s);
graphs-on 4096 1441.89697 ms /
2840.70215 tok/s;
tok/s sitting skipped.

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp revision | `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` |
| A/B | complete 4096-token execute_prompt_chunk, graphs-off then graphs-on 4096 |
| Cancellation | no publish after internal batches 1, 2, or the final batch |
| P / D128 / D2048 | OPT-021 / OPT-032 protocols, unchanged diagnostics |
| Nsight | not_used |

## Measured sitting

- Device: NVIDIA GeForce RTX 5090 compute 12.0
- measurement_utc: 2026-09-11T00:33:24Z
- extra_workspace_bytes: 0
- graphs-off 512: 1654.02405 ms / **2476.38477** tok/s
- graphs-off 1024: 1518.56348 ms / **2697.28589** tok/s
- graphs-off 2048: 1451.12952 ms / **2822.62891** tok/s
- graphs-off 4096: 1441.02893 ms / **2842.41333** tok/s
- graphs-on 4096: 1441.89697 ms / **2840.70215** tok/s
- diagnostic 2K from empty: 680.203308 ms / **3010.8645** tok/s (not historical P)
- diagnostic 4K append at prefix 2048: 1640.02295 ms / **2497.52612** tok/s
- diagnostic 4K append at prefix 4096: 1835.90747 ms / **2231.04932** tok/s
- P Quartz mean tok/s: 2895.42773 versus OPT-053 2895.42773 (copied; keep sitting skipped)
- D128 Quartz mean tok/s: 37.5605927 versus OPT-053 37.5605927
- D2048 Quartz mean tok/s: 35.7286987 versus OPT-053 35.7286987
- owns_opt016_parity_gate: false
- substitutes_for_opt016: false

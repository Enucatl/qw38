# OPT-052 — Remove redundant GDN arithmetic and state traffic

## Claim labels and proof limits

Live exclusive-RTX-5090 A/B of hoisted scaled Q/K and decay against the
OPT-040 shared-inverse warp-column loop. Keep requires
**OPT-044 production-numerics budgets**, **like-arithmetic shared control remains OPT-040 hoisted inverses**,
**conv+preprocessing+recurrence+gated complete cost**, **hoisted scaled Q/K and decay**,
**admitted FMA and approximate exp as separate variants**, **optional conversion-inclusive state transpose**,
**nonzero incoming state and outer-chunk restore**, **same-path transaction/restore exactness**,
**95% throughput floors versus OPT-051 keep**, **105% p95 ceilings versus OPT-051**, and
**does not substitute for the 2K llama.cpp parity gate**. Copied denominators
are P 2489.33008, D128 37.6655884, D2048 35.7582932.

## Decision

**keep** — `reverted`=false;
`keep_sitting_skipped`=false;
selected_gdn_preproc_path=transpose;
4096 A/B winner transpose (shared 26.516851046666666 ms, preproc 25.61543477 ms, preproc_fma 25.03394993 ms, approx_exp 25.352060756666667 ms, transpose 23.217959473333334 ms);
tok/s sitting ran.

Like-arithmetic `preproc` is byte-equal to shared and faster on the complete
conv+preprocessing+recurrence+gated component. Conversion-inclusive
`transpose` is also byte-equal and is the A/B winner. `preproc_fma` and
`approx_exp` stay eligible under production-numerics abs 3e-4 / rms 2e-4
but lose the like-arithmetic 1% preference to `transpose`.

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp revision | `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` |
| A/B | complete conv+preproc+recurrence+gated, candidates shared / preproc / preproc_fma / approx_exp / transpose |
| Correctness | tokens [1, 2, 3, 4, 63, 64, 65, 512, 2048, 4096]; layers [0, 32, 62]; outer chunks 64+1 and 256+256 |
| P / D128 / D2048 | OPT-021 / OPT-032 protocols, unchanged diagnostics |
| Nsight | not_used |

## Measured sitting

- Device: NVIDIA GeForce RTX 5090 compute 12.0
- measurement_utc: 2026-09-10T23:07:40Z
- P Quartz mean tok/s: 2692.64575 versus OPT-051 2489.33008
- D128 Quartz mean tok/s: 37.5680695 versus OPT-051 37.6655884
- D2048 Quartz mean tok/s: 35.6793633 versus OPT-051 35.7582932
- owns_opt016_parity_gate: false
- substitutes_for_opt016: false

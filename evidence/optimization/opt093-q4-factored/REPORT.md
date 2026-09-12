# OPT-093 — Factor paired-group Q4 integer dots

Status: **measured reject**. Control `previous_selected`
(`integer_q8_late` / `paired_integer`). Candidate `factored_pair_w4`
(`integer_q8_factored` / `paired_integer`, 4 warps).

Provenance: pinned llama `vecdotq.cuh::vec_dot_q4_K_q8_1_impl_vmmq`
(cc83d7b, MIT). Quartz rewrite in `cuda/q4k_decode_dots.cuh`.

## Mechanism

Paired-group Q4 integer scale/min factoring: `pair=(lane%16)/4`, `pack=lane%4`,
groups `(2*pair, 2*pair+1)`, K blocks `2*warp+lane/16` stride 8. Exact int32
dot/min sums; FP32 `d*sum_d - dmin*sum_m` via two independent single-group
projections (same-math bitwise). Launch variants:
`q4k_coop_mmv_factored_{prequant_,}q8`,
`q4k_coop_gate_up_swiglu_factored_prequant_q8`.

## Eligibility

OPT-090 gate passed: `ffn_gate_up_max_removable_ms` ≈ 411.6,
`material_q4_cost=true`, `no_go_bandwidth_bound=false`.

## Kernel parity

13/13 native cases bitwise identical (`QW38_HOST_NATIVE=1`).

## Complete rotating FFN screening (acceptance)

10 paired samples, df=9, critical=2.262:

- control mean 9.977 ms/token
- candidate mean 10.482 ms/token
- mean diff (control−candidate) **−0.505 ms/token**
- CI95 **[−0.543, −0.466] ms** (candidate slower; `positive=false`)

Reject: saving below 0.10 ms threshold and interval not positive.

## Independent verdicts (`factored_pair_w4`)

| Verdict | Value |
|---------|-------|
| `kernel_parity_pass` | true |
| `model_quality_pass` | false (candidate Q not measured; performance reject first) |
| `performance_pass` | false |
| `production_kept` | false |

Control retained: `integer_q8_late` / `paired_integer`.
`claims_throughput`: false.

## D2048+32 engine pairs (informational)

Uninstrumented AB pairs show lower ITL for candidate (~1.35× throughput ratio),
but complete rotating FFN component replay is the authoritative keep gate and
rejects the candidate.

## Tok/s delta

Production baseline unchanged (reject). Speedup **0** versus sitting control.

# OPT-078 — Prepare decode queries once and vector-load KV

Status: **measured**. Authority llama.cpp
`cc83d7b4824f73cfdda4dfbb47ee39804f71b328`, GGUF SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`. Control is in-kernel `warp_query`
RMS+RoPE per partition. Candidates hoist decode Q prep (`prepared_q`) and add
legal 16-byte KV loads (`prepared_q_veckv`). n_parts stays 16. Prompt fattn
hoist is unchanged. Candidate K/V staging stays separate. Capture/setup is
cached under repaired OPT-071 decode-attention captures.

`claims_throughput: false` unless production_kept. Require >=0.10 ms/token
complete 16-layer attention saving and a positive paired interval, or measured
rejection. If the extra prep launch offsets the kernel saving, reject.

## Numeric policy

Same-path control vs candidate GPU outputs must match for unchanged arithmetic.
Sampled independent FP64 uses four query heads x eight output dimensions at
two positions. Nonfinite=0.

## Mechanism

The retained `warp_query_decode_attention` path repeats lane-0 RMS over 256
elements and RoPE in each of 16 partitions per query head; each lane loads eight
adjacent BF16 dimensions through scalar loop accesses. This differs from
OPT-050's already-hoisted prompt preparation.

Candidates add a decode-only preparation launcher with one warp per query head,
retaining the exact ordered FP32 norm and existing RoPE operations first.
Prepared FP32 Q is written to the existing normalized-query workspace once; all
16 attention partitions consume it. `prepared_q_veckv` adds legal 16-byte KV
loads per lane into eight BF16 values with the existing dim=lane*8+i mapping,
reduction/softmax order and ascending-part merge unchanged. Unaligned pointers
fall back to the alignment-safe scalar path. Candidate K/V preparation stays
separate from committed cache visibility. Prepared Q is rewritten for every
new input/position and graph capture; pointer identity is never a skip.

## Configurations

| id | role | prepared_q | vec_kv | prep_launches | launch |
|---|---|---|---|---|---|
| `warp_query` | control | false | false | 0 | `warp_query_decode_attention` |
| `prepared_q` | candidate | true | false | 1 | `prepare_decode_query+warp_query_prepared_q` |
| `prepared_q_veckv` | candidate | true | true | 1 | `prepare_decode_query+warp_query_prepared_q_veckv` |

Production shape: 24 query heads, 4 KV heads, width 256, rotary 64.
Tiny positions 0/1/31/32; production positions 127/128/2047/2048.

## Dispatch

Control `warp_query`: `prep_launches=0`, `prep_grid=0`, `prep_block=0`,
`n_parts=16`, `prepared_q=false`, `vec_kv=false`.

Candidates `prepared_q` / `prepared_q_veckv`: `prep_launches=1`,
`prep_grid=24` (one CTA per query head), `prep_block=32` (one warp per head),
`n_parts=16`, `prepared_q=true`; `vec_kv` distinguishes the two survivors.
The extra prep launch is inside the timed complete-attention boundary (query/key
prep, core, merge, gating, FP32-to-BF16 cast).

Launches: [{"config": "warp_query", "dispatch": {"launch": "warp_query_decode_attention", "n_parts": 16, "path": "warp_query", "prep_block": 0, "prep_grid": 0, "prep_launches": 0, "prepared_q": false, "vec_kv": false}, "mean_ms": 9.0920387}, {"config": "prepared_q", "dispatch": {"launch": "prepare_decode_query+warp_query_prepared_q", "n_parts": 16, "path": "prepared_q", "prep_block": 32, "prep_grid": 24, "prep_launches": 1, "prepared_q": true, "vec_kv": false}, "mean_ms": 10.3147649}]

## Native proofs

Native target `build/qw38-cuda-opt078-decode-attention-test`; pytest host
contract and native screen passed before feedback/acceptance.

| proof | result |
|---|---|
| seq_abs vs FP64 sampled reference | 0 (all tiny and prod 127/128 cases) |
| prepared-Q refresh on new input/position/graph capture | pass |
| candidate vs committed KV visibility | pass |
| alignment-safe scalar fallback | pass |
| same-path control/candidate exact outputs (unchanged arithmetic) | pass |
| nonfinite | 0 |
| n_parts | 16 |
| occupancy (query-head prep grid) | 24 |

## Feedback d128 (position 128, n=3)

Three configurations x (1 warmup + 3 samples) complete 16-layer rounds.
Survivor for paired screen: `prepared_q_veckv`.

| config | mean_ms |
|---|---|
| `warp_query` (control) | 1.693 |
| `prepared_q` | 1.851 |
| `prepared_q_veckv` (survivor) | 1.745 |

Control vs survivor: mean_diff **-0.052 ms**; CI95 -0.121 .. +0.017 ms;
positive=false; saving_ge_0_10_ms=false. Verdict **inconclusive** (feedback).

Engine D128 pair: control 866.382 ms vs candidate 869.041 ms
(regression_upper 2.659 ms). dispatch_ok=true.

## Feedback d2048 (position 2048, n=3)

Survivor for paired screen: `prepared_q`.

| config | mean_ms |
|---|---|
| `warp_query` (control) | 9.570 |
| `prepared_q` (survivor) | 9.108 |
| `prepared_q_veckv` | 9.507 |

Control vs survivor: mean_diff **+0.462 ms**; CI95 -0.164 .. +1.088 ms;
positive=false (CI not positive at n=3). Verdict **inconclusive** (feedback).

Engine D2048 pair: control 867.836 ms vs candidate 869.184 ms
(regression_upper 1.349 ms). dispatch_ok=true.

## Complete 16-layer attention acceptance (position 2048, n=10)

Control `warp_query` vs survivor `prepared_q` on repaired OPT-071 rotating
decode-attention captures. Warmups=3, samples=10, engine_pairs=5.

| metric | control | candidate |
|---|---|---|
| component mean_ms | 9.092 | 10.315 |
| paired mean_diff_ms | — | -1.223 |
| paired CI95 (control−candidate) | — | -1.970 .. -0.476 |
| t_crit (df=9) | — | 2.262 |
| positive | — | false |
| saving_ge_0_10_ms | — | false |

Engine D2048+32 n=5: control **875.121 ms** vs candidate **876.809 ms**;
mean_regression 1.688 ms; regression_upper **4.118 ms**; t_crit df4=2.132;
e2e_non_regression=true (NR holds at 2% gate).

## Quality (OPT-073)

Identity-cached OPT-073 quality preflight ran once per survivor; no long P/D
protocol. quality-v3 absolute=fail; engine non-regression=pass;
quality-v3_quartz_engine_non_regression=pass; quality-v2 all=false;
opt056_visible=true; identity_cached=true. A functional baseline defect cannot
excuse new output/state errors.

## Decision

Verdict: **performance_rejected** (['negative_complete_saving']).
production_kept=False; retain_reason=performance; shipping_unchanged=True.
Shipping decode query-prep stays `warp_query`. Extra prep launch offsets kernel
saving at acceptance n=10 despite d2048 feedback hint.

## tok/s

Speedup versus the then-current warp_query baseline is **0.0** while warp_query
is retained unless production_kept.

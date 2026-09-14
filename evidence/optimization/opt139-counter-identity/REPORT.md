# OPT-139 — Normalize NCU evidence and authenticate production kernel identity

Diagnostics only. `claims_throughput=false`. `claims_performance_improvement=false`.
Shipping delta: N/A. NLL: N/A. Candidate: null. No production kernel or selector change.
No throughput keep claim.

Measured at `2026-09-14T23:34:29Z`. Image `qw38-cuda:13.0.2`.
Parent `post124_plus_opt127_decode_segments8_plus_opt137_mma`.
Selector `decode_segments8`. Replay binary hash
`006f6053049ec98cc75ab7a08a1fe07cabfe66e3e06e4741fd949dee4122ea57`.
GPU available `True`. NCU `True`. Pins authenticated `True`.

Structured JSON: [`identity.json`](identity.json),
[`counter-evidence.json`](counter-evidence.json),
[`fixtures/opt139_counter_identity.json`](../../../fixtures/opt139_counter_identity.json).

## Raw artifacts

| Shape | NCU report (`.ncu-rep`) | Exported CSV | Capture stdout |
|---|---|---|---|
| D128 `attn_core` | [`raw/d128-attn_core.ncu-rep`](raw/d128-attn_core.ncu-rep) | [`raw/d128-attn_core.ncu.csv`](raw/d128-attn_core.ncu.csv) | [`raw/d128-attn_core.ncu.txt`](raw/d128-attn_core.ncu.txt) |
| D2048 `attn_core` | [`raw/d2048-attn_core.ncu-rep`](raw/d2048-attn_core.ncu-rep) | [`raw/d2048-attn_core.ncu.csv`](raw/d2048-attn_core.ncu.csv) | [`raw/d2048-attn_core.ncu.txt`](raw/d2048-attn_core.ncu.txt) |

First pass wrote `.ncu-rep` only (56.3 s fresh NCU). Second pass reused reports and
exported long-format CSV via `ncu --import … --csv` (1.4 s). Enclosing replay
`enclosing_ms` totals remain on sibling objects and are not mixed into target-kernel
typed slots.

Production trace sources for identity (largest-duration `attn_core` kernel):

- D128: `build/optimization-runs/opt138/traces/quartz-d128-node-r0.2.sqlite`
- D2048: `build/optimization-runs/opt138/traces/quartz-d2048-node-r0.2.sqlite`

## Identity

| Shape | Hybrid path | Expected kernel | Trace stem | Identity | Replay dispatch |
|---|---|---|---|---|---|
| D128 | `warp_query` | `warp_query_decode_attention` | `warp_query_decode_attention` | ok | `path_for_position=warp_query` |
| D2048 | `vec128_online` | `vec128_online_decode_attention` | `vec128_online_decode_attention` | ok | `path_for_position=vec128_online` |

Hybrid crossover threshold 1024; verified max 4096. A `warp_query` replay is not
automatically D2048 evidence — rejected with `kernel_selector_mismatch` when
observed kernel is `warp_query_decode_attention` but expected is
`vec128_online_decode_attention`.

Prefill `attn_core` eligible=`False` reason=`prefill_decode_attention_ineligible_until_opt140`.

## Typed counter records

Selected NCU metrics (no full sweep): `dram__bytes_op_read`, `dram__bytes_op_write`,
`dram__bytes`, `dram__throughput.avg.pct_of_peak_sustained_elapsed`, `lts__t_sectors`,
`lts__t_sector_hit_rate`, `sm__pipe_tensor_cycles_active`,
`sm__throughput.avg.pct_of_peak_sustained_elapsed`,
`sm__warps_active.avg.pct_of_peak_sustained_active`,
`smsp__warps_issue_stalled_long_scoreboard`, `smsp__warps_issue_stalled_barrier`.

Per-launch slots come from the identified target kernel only (`--launch-count 1`,
`--replay-mode application`). Measured zero stays zero; absent metrics stay null.

### D128 — `warp_query_decode_attention`

| Slot | Value | Unit | Notes |
|---|---|---|---|
| `dram_read_bytes` | 586.75 | Kbyte | from `dram__bytes_op_read.sum` |
| `dram_write_bytes` | 0.0 | byte | measured zero |
| `dram_throughput` | 1.14 | % | |
| `l2_traffic` | 146745.0 | sector | from `lts__t_sectors.sum` |
| `l2_hit_rate` | 1.0 | — | `lts__t_sector_hit_rate.max_rate`; pct 78.63% also captured |
| `sm_throughput` | 3.92 | % | |
| `tensor_activity` | 0.0 | cycle | measured zero |
| `achieved_occupancy` | 4.62 | % | |
| `stalls.long_scoreboard` | 19247.9 | warp | do not sum stall percentages |
| `stalls.barrier` | 0.0 | warp | measured zero |
| `registers` | null | — | `metric_absent_from_capture` |
| `local_memory_spills` | null | — | `metric_absent_from_capture` |

Launch: grid `(24, 16, 1)`, block `(32, 1, 1)`, launch_id `0`.

### D2048 — `vec128_online_decode_attention`

| Slot | Value | Unit | Notes |
|---|---|---|---|
| `dram_read_bytes` | 8.85 | Mbyte | from `dram__bytes_op_read.sum` |
| `dram_write_bytes` | 0.0 | byte | measured zero |
| `dram_throughput` | 2.33 | % | |
| `l2_traffic` | 1932797.0 | sector | from `lts__t_sectors.sum` |
| `l2_hit_rate` | 1.0 | — | `lts__t_sector_hit_rate.max_rate`; pct 84.21% also captured |
| `sm_throughput` | 5.43 | % | |
| `tensor_activity` | 0.0 | cycle | measured zero |
| `achieved_occupancy` | 15.62 | % | |
| `stalls.long_scoreboard` | 184201.53 | warp | do not sum stall percentages |
| `stalls.barrier` | 31915.07 | warp | |
| `registers` | null | — | `metric_absent_from_capture` |
| `local_memory_spills` | null | — | `metric_absent_from_capture` |

Launch: grid `(24, 16, 1)`, block `(32, 4, 1)`, launch_id `0`.

## Mechanism admission

| Shape | Identity | Admission reason | `supported_mechanism` | `candidate` |
|---|---|---|---|---|
| D128 | ok | `throughput_alone_cannot_establish_mechanism` | null | null |
| D2048 | ok | `throughput_alone_cannot_establish_mechanism` | null | null |
| P4096 prefill `attn_core` | ineligible | `prefill_decode_attention_ineligible_until_opt140` | null | null |

Causal mechanism remains **explicitly unknown**. Identity and typed counters are
admitted; throughput, occupancy, and byte counters alone do not establish a named
source/SASS-backed mechanism. OPT-138's historical `ERR_NVGPUCTRPERM` is superseded
for this run, but that repair does not imply a supported optimization hypothesis.

## Performance evidence checklist (diagnostics)

1. **Measurement identity** — engine `quartz`, selector `decode_segments8`, replay
   hash above, GGUF `models/Qwen3.8-27B-Q4_K_M.gguf`, decode component replay at
   positions 128 and 2048, `cache_mode=rotating`, warmups 0, samples 1.
2. **Coverage** — bounded `attn_core` only; no full `ncu --set full` sweep; enclosing
   replay totals separated; prefill decode-attention deferred to OPT-140; llama counters
   out of scope (OPT-141).
3. **Time accounting** — enclosing `enclosing_ms` kept separate from target-kernel
   counters; stall warp counts not summed across metrics or shapes.
4. **Contradictions** — none identified between identity traces and replay dispatch.
5. **Claim types** — counter values `measured`; mechanism `unknown`/`incomplete`
   pending named source/SASS.
6. **Target/guard** — instrumentation-only; OPT-135 not applicable; shipping delta N/A.
7. **Independent verification** — pending verifier pass on this draft.
8. **Reporting** — candidate measured delta N/A; shipping delta N/A; quality N/A.

## Status

status=`measured` blocked=`False`.

Unsupported or identity-rejected mechanisms remain explicitly unknown.
Throughput alone cannot establish a mechanism. Candidate stays null.

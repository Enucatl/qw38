# OPT-140 — Production-boundary P4096 prefill attention replay

Diagnostics only. `claims_throughput=false`. `claims_performance_improvement=false`.
Shipping delta: N/A. NLL: N/A. Candidate: null. No production kernel or selector change.
No attention optimization. No new quality tolerance. Profiled NCU runs are not throughput samples.

Measured at `2026-09-15T00:05:49Z`. Image `qw38-cuda:13.0.2`.
Parent `post124_plus_opt127_decode_segments8_plus_opt137_mma`.
Selector `decode_segments8`. Replay binary hash
`130469b13ef1cb8dbe100bc8d5a05fb27653520f2401704ac8606c53c9273561`.
GPU available `True`. NCU `True`. Pins authenticated `True` (unverified by documentation agent).

Structured JSON: [`boundary-manifest.json`](boundary-manifest.json),
[`replay-evidence.json`](replay-evidence.json),
[`counter-evidence.json`](counter-evidence.json),
[`fixtures/opt140_prefill_attention_replay.json`](../../../fixtures/opt140_prefill_attention_replay.json).

## Raw artifacts

| Phase | Artifact | Path |
|---|---|---|
| Replay stdout | prompt-attention capture | [`replay/prompt-attention.stdout.txt`](replay/prompt-attention.stdout.txt) |
| Replay rounds | rotating sample log | [`replay/rounds.jsonl`](replay/rounds.jsonl) |
| Replay capture | capture bundle | [`replay/capture-bundle.json`](replay/capture-bundle.json) |
| NCU report | `.ncu-rep` | [`raw/p4096-attn_core.ncu-rep`](raw/p4096-attn_core.ncu-rep) |
| NCU export | `.ncu.csv` | [`raw/p4096-attn_core.ncu.csv`](raw/p4096-attn_core.ncu.csv) |
| NCU capture | `.ncu.txt` | [`raw/p4096-attn_core.ncu.txt`](raw/p4096-attn_core.ncu.txt) |

First NCU pass used `regex:fattn_mma_pipeline_opt111_base` and profiled nothing (CUDA
symbol is `fattn_mma_pipeline_kernel`). Second pass used
`regex:fattn_mma_pipeline_kernel --launch-count 1 --replay-mode application` and
wrote the `.ncu-rep`. CSV export reused the report. Enclosing replay `enclosing_ms`
totals remain on replay rounds and are not mixed into target-kernel typed slots.

## Boundary manifest

Source: [`boundary-manifest.json`](boundary-manifest.json).

| Field | Value |
|---|---|
| Phase | `prefill` |
| Replay family | `prompt-attention` |
| Workload | P4096 (`prefill_tokens=4096`) |
| Capacity | `131072` |
| Empty initial state | `true` |
| Prefix reuse | `false` |
| Final-token logits only | `true` |
| Chunk count | `1` |
| Query start | `0` |
| Attention layers | `16` |
| Cache mode | `rotating` |
| Production weights | `true` |
| Synthetic weights | `false` |
| Complete family replay | `true` |
| Counter kernel | `fattn_mma_pipeline_opt111_base` |
| Counter kernel is complete family | `false` |
| Decode-attention substitution | `false` (rejected) |
| Boundary ok | `true` |
| Identity ok | `true` |

**Production leaves included** (attn_core accounting):

- `split_attention_rows`
- `stage_chunk`
- `bf16_f16_convert`
- `fattn_mma_pipeline_opt111_base`
- `merge`
- `epilogue`
- `fp32_to_bf16`

**Explicitly excluded:**

- `mixer_mmq`, `ffn_mmq`, `logits`, `commit_sync`, `embedding`

Complete family replay runs split/stage/convert/core/merge/epilogue. NCU
`--kernel-name` targets `fattn_mma_pipeline_opt111_base` only
(`counter_kernel_is_complete_family=false`). A single repeatedly cached tile is not
equated with complete prompt cost.

## Replay parity

Capture key `698a857387d5b5779f4921723f6592d72b248b3b4fd74f55d0c3116909cfcd11`.
Protocol: 1 warmup + 3 alternating rotating rounds (`cache_mode=rotating`).
Existing abs tolerance `2e-3` reused; no new quality tolerance.

| Check | Value | Status |
|---|---|---|
| Layers compared | 16 | ok (unverified) |
| `max_abs` | 0.0 | equal |
| `kv_max_abs` | 0.0 | equal |
| Nonfinite | 0 | equal |
| Parity equal | `true` | under abs tol 2e-3 |
| Dispatch path | `opt111_base` | |
| Launch | `fattn_mma_pipeline_opt111_base` | |
| `decode_attention_substitution` | `false` | |

### Timing samples (not throughput claims)

| Round | Kind | enclosing_ms | kernel_only_ms | cache |
|---|---|---|---|---|
| 0 | warmup | 418.777 | 227.981 | rotating |
| 0 | sample | 420.241 | 228.074 | rotating |
| 1 | sample | 419.552 | 228.104 | rotating |
| 2 | sample | 420.376 | 228.141 | rotating |

### Causal tails and chunk boundaries

All 16 attention layers (slots 0–15, layers 3/7/11/…/63): `causal_tail_abs=0.0`,
`chunk_start_abs=0.0`, `chunk_mid_abs=0.0`, `equal=true` under existing abs tol.

## Decode-attention substitution rejection

OPT-138 correctly refused decode-attention replay for prefill `attn_core` production
comparison. OPT-140 enforces the same policy at the production boundary.

| Replay family | Boundary ok | Identity ok | Reason |
|---|---|---|---|
| `prompt-attention` | `true` | `true` | valid prefill replay |
| `decode-attention` | `false` | `false` | `prefill_decode_attention_ineligible_until_opt140`, `replay/production boundary mismatch` |

Preflight `decode_attention_rejected` records mismatches
`prefill_decode_attention_ineligible_until_opt140` and
`replay/production boundary mismatch`. Fixture `decode_attention_substitution=false`.
Dispatch stdout confirms `decode_attention_substitution=false` and
`complete_family_replay=true`.

## Typed counter records

Selected NCU metrics (no full sweep): `dram__bytes_op_read`, `dram__bytes_op_write`,
`dram__bytes`, `dram__throughput.avg.pct_of_peak_sustained_elapsed`, `lts__t_sectors`,
`lts__t_sector_hit_rate`, `sm__pipe_tensor_cycles_active`,
`sm__throughput.avg.pct_of_peak_sustained_elapsed`,
`sm__warps_active.avg.pct_of_peak_sustained_active`,
`smsp__warps_issue_stalled_long_scoreboard`, `smsp__warps_issue_stalled_barrier`.

Per-launch slots come from the identified target kernel only (`--launch-count 1`,
`--replay-mode application`). Measured zero stays zero; absent metrics stay null.

### P4096 prefill — `fattn_mma_pipeline_opt111_base`

| Slot | Value | Unit | Notes |
|---|---|---|---|
| `dram_read_bytes` | 252.12 | Mbyte | from `dram__bytes_op_read.sum` |
| `dram_write_bytes` | 168.42 | Mbyte | from `dram__bytes_op_write.sum` |
| `dram_throughput` | 1.22 | % | |
| `l2_traffic` | 216065551.0 | sector | from `lts__t_sectors.sum` |
| `l2_hit_rate` | 1.0 | — | `lts__t_sector_hit_rate.max_rate`; pct 93.46% also captured |
| `sm_throughput` | 22.4 | % | |
| `tensor_activity` | 810024960.0 | cycle | from `sm__pipe_tensor_cycles_active.sum` |
| `achieved_occupancy` | 8.33 | % | |
| `stalls.long_scoreboard` | 1039196.58 | warp | do not sum stall percentages |
| `stalls.barrier` | 326412.84 | warp | |
| `registers` | null | — | `metric_absent_from_capture` |
| `local_memory_spills` | null | — | `metric_absent_from_capture` |

Target kernel:
`void fattn_mma_pipeline_kernel<16, 0, 1, 32, 2, 2, 1, 2, 1, 1, 0>(...)`.
Launch: grid `(4, 256, 2)`, block `(128, 1, 1)`, launch_id `0`.
Identity mismatches: `[]`.

## Mechanism admission

| Shape | Identity | Admission reason | `supported_mechanism` | `candidate` |
|---|---|---|---|---|
| P4096 prefill `attn_core` | ok | `throughput_alone_cannot_establish_mechanism` | null | null |

Causal mechanism remains **explicitly unknown** (unverified). Identity and typed
counters are admitted; throughput, occupancy, and byte counters alone do not
establish a named source/SASS-backed mechanism. OPT-139 prefill ineligibility is
resolved by this matched prompt-attention replay; that resolution does not imply a
supported optimization hypothesis.

## Performance evidence checklist (diagnostics)

1. **Measurement identity** — engine `quartz`, selector `decode_segments8`, replay
   hash above, GGUF `models/Qwen3.8-27B-Q4_K_M.gguf`, P4096 prompt-attention at
   capacity 131072, `cache_mode=rotating`, 1 warmup + 3 samples.
2. **Coverage** — bounded `attn_core` only; no full `ncu --set full` sweep; enclosing
   replay totals separated; complete family replay vs counter kernel distinguished.
3. **Time accounting** — enclosing `enclosing_ms` kept separate from target-kernel
   counters; stall warp counts not summed across metrics.
4. **Contradictions** — none identified between boundary manifest, replay dispatch,
   and counter identity (unverified).
5. **Claim types** — counter values `measured`; mechanism `unknown`/`incomplete`
   pending named source/SASS.
6. **Target/guard** — instrumentation-only; OPT-135 not applicable; shipping delta N/A.
7. **Independent verification** — pending verifier pass on this draft.
8. **Reporting** — candidate measured delta N/A; shipping delta N/A; quality N/A.

## Status

status=`measured` blocked=`False`.

Matched P4096 replay: `true`. Decode-attention substitution rejected: `true`.
Throughput alone cannot establish a mechanism. Candidate stays null.
No attention optimization shipped. Production selector unchanged.

Preflight make_rc=`0` gpu=`True` ncu=`True`.

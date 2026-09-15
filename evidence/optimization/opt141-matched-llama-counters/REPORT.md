# OPT-141 — Matched llama counters for the selected gap families

Diagnostics only. `claims_throughput=false`. `claims_performance_improvement=false`.
Shipping delta: N/A. NLL: N/A. Candidate: null. No production kernel or selector change.
NCU duration is not a benchmark. Occupancy is not a bandwidth conclusion.

Measured at `2026-09-15T00:29:07Z`. Image `qw38-cuda:13.0.2`.
Parent `post124_plus_opt127_decode_segments8_plus_opt137_mma`.
Selector `decode_segments8`. Llama binary hash
`4a02206a38b8c01a7f9b313ef2bd5ff42844790dfc3772aba43dca5c6ea9f794`.
Pinned llama `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`.
GPU available `True`. NCU `True`. Pins authenticated `True` (unverified by documentation agent).

Structured JSON: [`identity.json`](identity.json),
[`counter-evidence.json`](counter-evidence.json),
[`comparisons.json`](comparisons.json),
[`fixtures/opt141_matched_llama_counters.json`](../../../fixtures/opt141_matched_llama_counters.json).

Resolves `llama_ncu_kernel_identity_unresolved` for the four OPT-138
phase/family selections. Private llama diagnostic reused; production
authority checkout is untouched.

## Raw artifacts

| Phase / family | Workload | Kernel | `.ncu-rep` | `.ncu.csv` | `.ncu.txt` |
|---|---|---|---|---|---|
| decode `attn_core` | D128 | `flash_attn_ext_vec` | [`raw/decode-attn_core-d128.ncu-rep`](raw/decode-attn_core-d128.ncu-rep) | [`raw/decode-attn_core-d128.ncu.csv`](raw/decode-attn_core-d128.ncu.csv) | [`raw/decode-attn_core-d128.ncu.txt`](raw/decode-attn_core-d128.ncu.txt) |
| decode `attn_core` | D2048 | `flash_attn_ext_vec` | [`raw/decode-attn_core-d2048.ncu-rep`](raw/decode-attn_core-d2048.ncu-rep) | [`raw/decode-attn_core-d2048.ncu.csv`](raw/decode-attn_core-d2048.ncu.csv) | [`raw/decode-attn_core-d2048.ncu.txt`](raw/decode-attn_core-d2048.ncu.txt) |
| decode `residual_norm_quant` | D128/D2048 | `quantize_q8_1` | [`raw/decode-residual_norm_quant-d128.ncu-rep`](raw/decode-residual_norm_quant-d128.ncu-rep) | [`raw/decode-residual_norm_quant-d128.ncu.csv`](raw/decode-residual_norm_quant-d128.ncu.csv) | [`raw/decode-residual_norm_quant-d128.ncu.txt`](raw/decode-residual_norm_quant-d128.ncu.txt) |
| prefill `attn_core` | P4096 | `flash_attn_ext_f16` | [`raw/prefill-attn_core-p4096.ncu-rep`](raw/prefill-attn_core-p4096.ncu-rep) | [`raw/prefill-attn_core-p4096.ncu.csv`](raw/prefill-attn_core-p4096.ncu.csv) | [`raw/prefill-attn_core-p4096.ncu.txt`](raw/prefill-attn_core-p4096.ncu.txt) |
| prefill `prompt_mmq` | P4096 | `mul_mat_q` | [`raw/prefill-prompt_mmq-p4096.ncu-rep`](raw/prefill-prompt_mmq-p4096.ncu-rep) | [`raw/prefill-prompt_mmq-p4096.ncu.csv`](raw/prefill-prompt_mmq-p4096.ncu.csv) | [`raw/prefill-prompt_mmq-p4096.ncu.txt`](raw/prefill-prompt_mmq-p4096.ncu.txt) |

First counters pass used `--nvtx-include` and profiled nothing (llama uses
`nvtxRangePushA`, not Start/End); `mul_mat_q<` missed NCU short name `mul_mat_q`.
Diagnosed retry: drop NVTX filter; regex `flash_attn_ext_vec`, `flash_attn_ext_f16`,
`^quantize_q8_1$`, `^mul_mat_q$`; `--kernel-name regex:… --launch-count 1
--replay-mode application`. Second pass: six typed records (five collections +
residual D2048 reuse of D128). Prefill attn CSV reused from the same sitting. No
full metric sweep.

## Selections

| Selection | Status | Reason |
|---|---|---|
| `decode.attn_core` | `matched` | — |
| `decode.residual_norm_quant` | `matched` | — |
| `prefill.attn_core` | `matched` | — |
| `prefill.prompt_mmq` | `matched` | — |

## Identity

Source: [`identity.json`](identity.json). Trace sqlite paths under
`build/optimization-runs/opt138/traces/`.

| Key | Kernel | Function | NCU regex | Replay family | Identity ok | Fusion |
|---|---|---|---|---|---|---|
| `decode.attn_core.d128` | `flash_attn_ext_vec` | `flash_attn_ext_vec<256,1>` | `flash_attn_ext_vec` | `decode-attention` | `true` | — |
| `decode.attn_core.d2048` | `flash_attn_ext_vec` | `flash_attn_ext_vec<256,1>` | `flash_attn_ext_vec` | `decode-attention` | `true` | — |
| `decode.residual_norm_quant.d128` | `quantize_q8_1` | `quantize_q8_1` | `^quantize_q8_1$` | `decode-mixer` | `true` | `fused_boundary_mismatch` |
| `decode.residual_norm_quant.d2048` | `quantize_q8_1` | `quantize_q8_1` | `^quantize_q8_1$` | `decode-mixer` | `true` | `fused_boundary_mismatch` |
| `prefill.attn_core.p4096` | `flash_attn_ext_f16` | `flash_attn_ext_f16<256,256,8,8,false,false>` | `flash_attn_ext_f16` | `prompt-attention` | `true` | `fused_boundary_mismatch` |
| `prefill.prompt_mmq.p4096` | `mul_mat_q` | `mul_mat_q<GGML_TYPE_Q4_K,128,false>` | `^mul_mat_q$` | `prompt-ffn` | `true` | — |

Decode attn: largest leaf `flash_attn_combine_results` is enclosing fused work;
matched compute kernel is `flash_attn_ext_vec`. D128 and D2048 authenticated
separately (`n_kv` 128 vs 2048). Prefill attn mapped via OPT-140
prompt-attention identity (`flash_attn_ext_f16`, not `flash_attn_ext_vec`).
P4096 kernel-leaf source/launch mapping only; no fabricated CUPTI graph-node IDs.

## Typed llama counters

Selected NCU metrics (no full sweep): `dram__bytes_op_read`, `dram__bytes_op_write`,
`dram__bytes`, `dram__throughput.avg.pct_of_peak_sustained_elapsed`, `lts__t_sectors`,
`lts__t_sector_hit_rate`, `sm__pipe_tensor_cycles_active`,
`sm__throughput.avg.pct_of_peak_sustained_elapsed`,
`sm__warps_active.avg.pct_of_peak_sustained_active`,
`smsp__warps_issue_stalled_long_scoreboard`, `smsp__warps_issue_stalled_barrier`.

Per-launch slots come from the identified target kernel only (`--launch-count 1`,
`--replay-mode application`). Measured zero stays zero; absent metrics stay null.

### Decode attn — `flash_attn_ext_vec`

| Workload | dram_read | dram_write | dram_throughput | l2_traffic | sm_throughput | occupancy | tensor | Reused |
|---|---|---|---|---|---|---|---|---|
| D128 | 1.11 Mbyte | 0.0 byte | 4.96 % | 212987 sector | 1.72 % | 8.13 % | 0.0 cycle | no |
| D2048 | 9.5 Mbyte | 0.0 byte | 29.8 % | 1803992 sector | 11.51 % | 10.8 % | 0.0 cycle | no |

### Decode residual — `quantize_q8_1`

| Workload | dram_read | dram_write | dram_throughput | l2_traffic | sm_throughput | occupancy | tensor | Reused |
|---|---|---|---|---|---|---|---|---|
| D128 | 26.11 Kbyte | 0.0 byte | 0.35 % | 5078 sector | 0.38 % | 17.13 % | 0.0 cycle | no |
| D2048 | 26.11 Kbyte | 0.0 byte | 0.35 % | 5078 sector | 0.38 % | 17.13 % | 0.0 cycle | no (shared dispatch) |

### Prefill attn — `flash_attn_ext_f16`

| Workload | dram_read | dram_write | dram_throughput | l2_traffic | sm_throughput | occupancy | tensor | Reused |
|---|---|---|---|---|---|---|---|---|
| P4096 | 151.94 Mbyte | 92.57 Mbyte | 4.87 % | 226109556 sector | 30.19 % | 14.73 % | 1142161408 cycle | yes |

### Prefill MMQ — `mul_mat_q`

| Workload | dram_read | dram_write | dram_throughput | l2_traffic | sm_throughput | occupancy | tensor | Reused |
|---|---|---|---|---|---|---|---|---|
| P4096 | 79.37 Mbyte | 134.64 Mbyte | 6.92 % | 129106525 sector | 52.75 % | 16.67 % | 838860800 cycle | no |

All six records: `identity.ok=true`, `error=null`, `supported_mechanism=null`,
`candidate=null`, `throughput_alone_cannot_establish_mechanism`.

## Quartz vs llama

Source: [`comparisons.json`](comparisons.json). Compare only where work, units and
normalization match. Fused-boundary mismatch rejects leaf comparison.

| Key | Comparable | Reason | Quartz kernel | Llama kernel |
|---|---|---|---|---|
| `decode.attn_core.d128` | `true` | — | `warp_query_decode_attention` | `flash_attn_ext_vec` |
| `decode.attn_core.d2048` | `true` | — | `vec128_online_decode_attention` | `flash_attn_ext_vec` |
| `decode.residual_norm_quant.d128` | `false` | `quartz_counters_unusable` | — | `quantize_q8_1` |
| `decode.residual_norm_quant.d2048` | `false` | `quartz_counters_unusable` | — | `quantize_q8_1` |
| `prefill.attn_core.p4096` | `false` | `fused_boundary_mismatch` | `fattn_mma_pipeline_opt111_base` (16×2) | `flash_attn_ext_f16` (8×8) |
| `prefill.prompt_mmq.p4096` | `false` | `quartz_counters_unusable` | — | `mul_mat_q` |

**Decode attn D128 comparable slots** (units match): dram_write 0/0 byte; l2
212987 vs 146745 sector; dram_throughput 4.96 vs 1.14 %; sm_throughput 1.72 vs
3.92 %; tensor 0/0 cycle; occupancy 8.13 vs 4.62 %. dram_read skipped (unit
mismatch: Quartz Kbyte vs llama Mbyte).

**Decode attn D2048 comparable slots**: dram_read 9.5 vs 8.85 Mbyte; dram_write
0/0 byte; l2 1803992 vs 1932797 sector; dram_throughput 29.8 vs 2.33 %;
sm_throughput 11.51 vs 5.43 %; tensor 0/0 cycle; occupancy 10.8 vs 15.62 %.

**Residual** (`fused_boundary_mismatch`): llama `quantize_q8_1` vs Quartz
`rms_norm_fp32_to_bf16_parallel` fusion; enclosing `rms_norm_f32`, `l2_norm_f32`,
`quantize_q8_1`; all counter slots incomparable.

**Prefill attn** (`fused_boundary_mismatch`): tile shape Quartz 16×2 vs llama 8×8;
enclosing `flash_attn_stream_k_fixup_general`, `flash_attn_mask_to_KV_max`; all
counter slots incomparable.

## Mechanism admission

| Shape | Identity | Admission reason | `supported_mechanism` | `candidate` |
|---|---|---|---|---|
| All four selections | ok | `throughput_alone_cannot_establish_mechanism` | null | null |

Causal mechanism remains **explicitly unknown** (unverified). Identity and typed
llama counters are admitted; throughput, occupancy, and byte counters alone do not
establish a named source/SASS-backed mechanism.

## Next experiment

No supported mechanism: throughput/occupancy alone cannot admit a candidate.
Required source/SASS observations are still absent.

Resolving measurement: named SASS/source observations on `flash_attn_ext_vec` vs
Quartz D2048 `vec128_online_decode_attention` and on
`flash_attn_ext_f16<256,256,8,8>` vs OPT-140 `fattn_mma_pipeline_opt111_base`,
with a disconfirming family-excess repeat after any later candidate.

## Independent raw-report inspection

Decode [`raw/decode-attn_core-d128.ncu.csv`](raw/decode-attn_core-d128.ncu.csv):
one launch, stem `flash_attn_ext_vec`, typed `dram_read_bytes=1.11 Mbyte`,
`dram_throughput=4.96 %`, `l2_traffic=212987 sector` match the structured D128
record.

Prefill [`raw/prefill-attn_core-p4096.ncu.csv`](raw/prefill-attn_core-p4096.ncu.csv):
one launch, stem `flash_attn_ext_f16`, typed `dram_read_bytes=151.94 Mbyte`,
`dram_throughput=4.87 %`, `l2_traffic=226109556 sector` match the structured
P4096 record.

NCU duration is not used as a tok/s or keep metric.

## Performance evidence checklist (diagnostics)

1. **Measurement identity** — engine `llama`, selector `decode_segments8`, llama
   hash above, GGUF `models/Qwen3.8-27B-Q4_K_M.gguf`, OPT-136 private diagnostic,
   pinned revision `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`.
2. **Coverage** — four OPT-138 selections; five identified kernels (residual
   D128/D2048 share dispatch); bounded per-kernel `--launch-count 1`; no full
   `ncu --set full` sweep; one diagnosed retry after first-pass filter miss.
3. **Time accounting** — NCU duration excluded from keep metrics; enclosing fused
   siblings (`flash_attn_combine_results`, `rms_norm_f32`) not equated with matched
   compute kernels.
4. **Contradictions** — none identified between trace identity, typed counters,
   and comparisons (unverified).
5. **Claim types** — counter values `measured`; mechanism `unknown`/`incomplete`
   pending named source/SASS.
6. **Target/guard** — instrumentation-only; shipping delta N/A; NLL N/A.
7. **Independent verification** — pending verifier pass on this draft.
8. **Reporting** — candidate measured delta N/A; shipping delta N/A; quality N/A.

## Status

status=`measured` blocked=`False`.
`llama_ncu_kernel_identity_unresolved=false`.
An unresolved required identity is blocked, not a successful skip.
Throughput alone cannot establish a mechanism. Candidate stays null.
No production changes. Production selector unchanged.

Preflight make_rc=`0` gpu=`True` ncu=`True`.

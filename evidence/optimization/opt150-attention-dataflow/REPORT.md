# OPT-150 — Current decode-attention arithmetic and matched scaling

Diagnostics only. `claims_throughput=false`. Shipping delta: N/A. NLL: N/A.
No production scheduling, selector, or kernel changes.

Measured at `2026-09-15T12:16:37Z`. Image `qw38-cuda:13.0.2`.
Parent `post124_plus_opt127_decode_segments8_plus_opt137_mma_plus_opt148_flash_vec`.
Execution graph `decode_segments8`. Shipping decode attention
`decode_attention_flash_vec_v1`. Allocated capacity `131072`. Pinned llama
`cc83d7b4824f73cfdda4dfbb47ee39804f71b328`.

Structured JSON: [`identity.json`](identity.json), [`dispatch-table.json`](dispatch-table.json),
[`dataflow.json`](dataflow.json), [`compiled.json`](compiled.json),
[`historical-corrections.json`](historical-corrections.json),
[`correctness.json`](correctness.json), [`replay.json`](replay.json),
[`matched.json`](matched.json), [`attention-ceiling.json`](attention-ceiling.json),
[`time-reconstruction.json`](time-reconstruction.json), [`answers.json`](answers.json),
[`fixtures/opt150_attention_dataflow.json`](../../../fixtures/opt150_attention_dataflow.json).

## Identity

- Source revision: `ddc93e75d364e5f557d662cb662d10d6843d4b87` (dirty).
- GPU: `NVIDIA GeForce RTX 5090`.
- NVCC flags: `-O2 --fmad=false`.
- Crossover `1024`, verified_max `4096`, MMA threshold `8192`.
- GGUF `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`.
- Production pins unchanged: `kSelectedDecodeAttentionFlashVec`, `kSelectedOpt137DenseMma`.

## Historical corrections

- **OPT-129** `approximately 0.050 ms at prefix 8192`: That number is the BF16 llama-vector adapter enclosing path (matched_bf16_enclosing mean 0.05033 ms/layer), explicitly not the native selected fattn-mma-f16 ncols1=1 ncols2=8 kernel. The sitting was non-exclusive and allocated prefix+1, not production capacity 131072. ([`evidence/optimization/opt129-matched-attention/REPORT.md`](../opt129-matched-attention/REPORT.md)).
- **OPT-136** `D32768 ratio 0.243909`: That ratio is pre-OPT-137 whole-engine decode-only throughput (quartz 15.317 tok/s / llama 62.799 tok/s) at capacity 131072, not a component attention ratio and not post-148 flash-vec. ([`evidence/optimization/opt136-graph-accounting/REPORT.md`](../opt136-graph-accounting/REPORT.md)).

Current matched whole-engine ratios below (1.16–1.55 quartz/llama ms/token) use the
post-148 parent and OPT-136/142 decode-only accounting; they do not reproduce OPT-136's
pre-137 tok/s ratios and are not contradictions once parent identity differs.

## Dispatch

Quartz uses a zero-based position and attends over `position+1` tokens. Llama
flash-attn uses padded used `K.ne[1]` = `max(256, pad(used_max_p1, 256))`, not
allocated capacity. Do not copy the number 8192 across those identities.

| position | visible | quartz path | topology | n_parts | graph xing | llama padded | llama kernel |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | `warp_query` | 0 | 16 |  | 256 | `flash_attn_ext_vec<256,1>` |
| 128 | 129 | `warp_query` | 0 | 16 |  | 256 | `flash_attn_ext_vec<256,1>` |
| 1023 | 1024 | `warp_query` | 0 | 16 |  | 1024 | `flash_attn_ext_vec<256,1>` |
| 1024 | 1025 | `decode_attention_flash_vec_v1` | 1 | 16 | topology_0_to_1 | 1280 | `flash_attn_ext_vec<256,1>` |
| 4096 | 4097 | `decode_attention_flash_vec_v1` | 1 | 16 |  | 4352 | `flash_attn_ext_vec<256,1>` |
| 4097 | 4098 | `warp_query` | 0 | 16 | topology_1_to_0 | 4352 | `flash_attn_ext_vec<256,1>` |
| 6144 | 6145 | `warp_query` | 0 | 16 |  | 6400 | `flash_attn_ext_vec<256,1>` |
| 7935 | 7936 | `warp_query` | 0 | 16 |  | 7936 | `flash_attn_ext_vec<256,1>` |
| 7936 | 7937 | `warp_query` | 0 | 16 |  | 8192 | `fattn-mma-f16_ncols1=1_ncols2=8` |
| 8191 | 8192 | `warp_query` | 0 | 16 |  | 8192 | `fattn-mma-f16_ncols1=1_ncols2=8` |
| 8192 | 8193 | `dense_bf16_tile_f16_mma_decode_v1` | 2 | 132 | topology_0_to_2 | 8448 | `fattn-mma-f16_ncols1=1_ncols2=8` |
| 8193 | 8194 | `dense_bf16_tile_f16_mma_decode_v1` | 2 | 132 |  | 8448 | `fattn-mma-f16_ncols1=1_ncols2=8` |
| 32768 | 32769 | `dense_bf16_tile_f16_mma_decode_v1` | 3 | 148 | topology_2_to_3 | 33024 | `fattn-mma-f16_ncols1=1_ncols2=8` |

`n_parts` in the table is the host dispatch plan. Post-launch
`last_decode_attention_n_parts` at positions 8192/32768 reads 131/170 in the
correctness capture while paths and topologies match; treat as a bookkeeping
delta, not a path disagreement.

Llama switches to fattn-mma-f16 at padded `K.ne[1]>=8192` (Quartz position
7936+). Quartz OPT-137 MMA starts at position 8192.

## Source-to-output dataflow

### Quartz OPT-137 MMA (`position >= 8192`)

- QK: `fp32_dot_plus_warp_sum`; MMA helper `mma_qk_eight_zero_weight`.
- Softmax: `per_kv_row_online`.
- PV: `scalar_fp32_weight_times_v`; PV MMA present=`False`.
- GQA: `ncols2=8_live_heads_0_5_mask_6_7`.
- Loads/conversion: `dense_bf16_tile_to_f16_opt111_rounding`.
- Partitions: `frozen_graph_buckets_8448_33024_131072`.
- Merge: `merge_decode_kv_parts`.

### Pinned llama fattn-mma-f16 D256 ncols1=1 ncols2=8

- QK: `mma_kq`. Softmax: `tiled_rescale_nbatch_fa_64_ampere`.
- PV: `mma_vkq`. GQA: `ncols2=8_shared_kv_when_gqa_ratio>4`.
- Threads/occupancy/stages: `64` / `4` / `2`.

## Compiled MMA survival

- Source zero-weight add present: `True`.
- SASS tool: `cuobjdump`; kernel symbol `True`; HMMA count `2`; survives=`True`.
- Blocked: `False` reason `None`.
- Raw: [`compiled-sass.txt`](compiled-sass.txt) (171850 bytes).

Production kernel SASS contains HMMA.16816 ops. That shows the compiler kept
tensor-core instructions in `opt137::dense_bf16_tile_f16_mma_decode`. It does
not prove the zero-weight QK add is on the output-producing dataflow; PV stays
scalar.

## Correctness

Status `measured`. Dispatch ok `True`. Finite `True`. Sampled FP64 ok `True`.
13 dispatch launches; 9 sampled FP64 checks (layers 3/7/63 × positions
128/1024/8192). Max sampled abs error ~7.5e-9.

## Family timing (1 warmup + 3 alternating rounds, 16 layers)

- `D2048` position `2048`: quartz `0.630378664` ms; llama enclosing `0.0365333334` ms; adapter `0.0135253333` ms; selected `flash_attn_ext_vec<256,1>`. Adapter is not native F16 cost.
- `D8192` position `8192`: quartz `3.15950942` ms; llama enclosing `None` ms; adapter `None` ms; selected `fattn-mma-f16_ncols1=1_ncols2=8`. OPT-129 llama-vector adapter replay illegal memory access (prefix+1 sitting); family times left null rather than substituting vec for MMA.
- `D32768` position `32768`: quartz `13.6802988` ms; llama enclosing `0.331808001` ms; adapter `0.19691734` ms; selected `fattn-mma-f16_ncols1=1_ncols2=8`. Adapter is not native F16 cost; D32768 enclosing uses matched BF16 path, not native selected-kernel F16 alone.

## Matched decode probes (32 evals, capacity 131072)

Unprofiled decode-only; denominator `decode_only_ms / eval_count`. Instrumented
attribution kept separate per OPT-142.

- D2048: quartz `16.629501979166665` ms/token vs llama `14.3661436875` ms/token; ratio `1.1575480755935927`. Instrumented `16.6455008125` ms/token.
- D8192: quartz `20.410231281250002` ms/token vs llama `17.072758635416665` ms/token; ratio `1.1954852591256049`. Instrumented `20.4114125625` ms/token.
- D32768: quartz `29.36345990625` ms/token vs llama `18.9222396875` ms/token; ratio `1.5517962139358934`. Instrumented `29.387756979166667` ms/token.

Raw: [`raw-matched-decode.json`](raw-matched-decode.json), [`raw-matched-records.json`](raw-matched-records.json).

## Independent time reconstruction

D2048 family times recomputed from `replay.json` round arrays
(`independent_recompute.quartz_mean_ms` matches `quartz_mean_ms`; n_quartz=3).

| Quantity | ms |
| --- | ---: |
| Quartz complete attention family | 0.6303786633333334 |
| Llama enclosing (adapter included) | 0.0365333334 |
| Adapter conversion (charged separately) | 0.0135253333 |
| Llama native kernel (enclosing − adapter) | 0.023008000100000002 |
| Family ratio quartz/llama enclosing | 17.254890388221003 |

Artifact: [`time-reconstruction.json`](time-reconstruction.json).

## Attention-only speedup ceiling

Family-time ratio is not a whole-engine multiplier.

- `D2048` family ratio `50.554318636562904` (denom `same_sitting_complete_attention_family_ms`, llama arm uses selected native `0.0124693336` ms); engine ratio `1.1575480755935927` (denom `decode_only_ms / eval_count`).
- `D8192` family ratio `None` (llama family IMA; denom `same_sitting_complete_attention_family_ms`); engine ratio `1.1954852591256049` (denom `decode_only_ms / eval_count`).
- `D32768` family ratio `41.22956275548039` (denom `same_sitting_complete_attention_family_ms`); engine ratio `1.5517962139358934` (denom `decode_only_ms / eval_count`).

## Contradictions

- **OPT-129 ~0.050 ms @8192** — BF16 adapter enclosing path, not native MMA; corrected above and in [`historical-corrections.json`](historical-corrections.json).
- **OPT-136 D32768 0.243909** — pre-OPT-137 whole-engine tok/s ratio, not attention component; corrected above.
- **OPT-136 vs OPT-150 matched ratios** — different parent (`+opt148_flash_vec`), not a same-identity disagreement; current ratios are the authenticated baseline for this parent.
- **Host `n_parts` 132/148 vs post-launch 131/170** — documented; paths/topologies agree.
- **Llama MMA @7936 vs Quartz MMA @8192** — padding/position identity difference, not a silent kernel swap.
- **D8192 replay llama family null** — documented IMA on OPT-129 adapter sitting; selected kernel is fattn-mma-f16; vec/adapter times are not used as native MMA cost.

## Coverage

- Dispatch: 13/13 required positions launched and finite (`correctness.json`).
- Numerical: 9/9 sampled FP64 checks ok; max abs ~7.5e-9.
- Replay: 3/3 shapes quartz measured; llama family 2/3 (D8192 blocked by IMA).
- Matched: 3/3 prefixes × 32 evals, capacity 131072, unprofiled decode-only.
- Compiled: production `dense_bf16_tile_f16_mma_decode` SASS captured.
- Reconstruction: D2048 independent recompute from round arrays.

## Performance evidence checklist (diagnostics)

1. **Measurement identity** — parent `post124_plus_opt127_decode_segments8_plus_opt137_mma_plus_opt148_flash_vec`; source `ddc93e75` (dirty); selector `decode_segments8`; capacity 131072; GGUF and llama pin recorded in [`identity.json`](identity.json); warmups/samples explicit per phase.
2. **Coverage** — required dispatch, correctness, replay, matched, and compiled artifacts present; D8192 llama family replay incomplete (IMA), not zero-filled.
3. **Time accounting** — matched uses OPT-136 unprofiled decode-only / 32 evals; instrumented attribution separate (OPT-142); adapter conversion charged separately in family replay; family ratios not used as whole-engine multipliers.
4. **Contradictions** — OPT-129/136 historical claims reconciled; see Contradictions section.
5. **Claim types** — arithmetic/dataflow `measured`/`derived`; HMMA survival `measured` with data-dependence limit explicit; historical OPT-129/136 numbers `historical` only.
6. **Target/guard** — diagnostics only; shipping throughput delta N/A; NLL N/A; `claims_throughput=false`.
7. **Independent verification** — pending verifier pass; draft conclusions below are **unverified**.
8. **Reporting** — candidate measured delta N/A; shipping delta N/A; quality N/A.

## Inputs for OPT-151/152

- OPT-151: long-decode QK/PV still scalar plus a zero-weight MMA helper; llama selected MMA at padded n_kv>=8192 (Quartz position 7936+) while Quartz MMA starts at position 8192.
- OPT-152: sub-8K vector coverage; Quartz returns to warp_query on [4097,8191] while llama stays on vec until padded 8192.

## Status

status=`measured` blocked=`False` claims_throughput=false production_kept=true.
NLL N/A. Shipping throughput delta N/A.

Zero-weight MMA data dependence into QK output remains unproven beyond source
`0.0F * mma_scores[0]` plus two surviving HMMA ops. D8192 llama family replay
remains blocked. Verifier pass pending.

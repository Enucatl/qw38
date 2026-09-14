# OPT-129 — Compare matched Quartz and llama decode-attention components

Status: **diagnostics complete**. Authority llama.cpp `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`. GGUF SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`.

`claims_throughput: false`. `claims_performance_improvement: false`. Production `hybrid_crossover@1024` is unchanged.

## Sitting identity

- measurement_utc: `2026-09-14T00:35:17Z`
- llama_revision: `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`
- crossover: **1024** (verified_max=4096)
- device: NVIDIA GeForce RTX 5090
- image: `qw38-cuda:13.0.2`
- hardware_executed: `True`
- sitting: **not exclusive** (unrelated GPU residents were left running)
- capacity rule: allocated == visible `prefix+1` (not production 131072)
- full_scheduler dispatch: `launch_attention_prepare_partitioned`

## Kernel identities

Flash-attn auto configuration is not treated as the launched kernel. Quartz launches were recorded from `launch_attention_prepare_partitioned`. Llama selected kernels are from pinned `fattn.cu::ggml_cuda_get_best_fattn_kernel` with `get_n_kv` padded used length (`max(n_pad, 256)`), **not** full `n_ctx`. Ada+ decode with `Q.ne[1]==1` and unquantized KV selects VEC unless `gqa_ratio>4` and `K.ne[1]>=8192`, in which case MMA_F16 ncols1=1 ncols2=8.

- live f16 occupancy: 3 / matched occupancy: 2 on 170 SMs; f16 regs 167, local bytes 0

```json
{
  "128": {
    "quartz": "warp_query_decode_attention",
    "llama_padded_nkv": 256,
    "llama_selected": "flash_attn_ext_vec<256,1>",
    "llama_vec_is_selected": true
  },
  "1023": {
    "quartz": "warp_query_decode_attention",
    "llama_padded_nkv": 1024,
    "llama_selected": "flash_attn_ext_vec<256,1>",
    "llama_vec_is_selected": true
  },
  "1024": {
    "quartz": "vec128_online_decode_attention",
    "llama_padded_nkv": 1280,
    "llama_selected": "flash_attn_ext_vec<256,1>",
    "llama_vec_is_selected": true
  },
  "2048": {
    "quartz": "vec128_online_decode_attention",
    "llama_padded_nkv": 2304,
    "llama_selected": "flash_attn_ext_vec<256,1>",
    "llama_vec_is_selected": true
  },
  "8192": {
    "quartz": "warp_query_decode_attention",
    "llama_padded_nkv": 8448,
    "llama_selected": "fattn-mma-f16_ncols1=1_ncols2=8",
    "llama_vec_is_selected": false
  },
  "32768": {
    "quartz": "warp_query_decode_attention",
    "llama_padded_nkv": 33024,
    "llama_selected": "fattn-mma-f16_ncols1=1_ncols2=8",
    "llama_vec_is_selected": false
  }
}
```

## Replay

Identical Q/K/V per (layer, prefix). Native Quartz is BF16 physical. Matched layout is BF16 physical consumed by both shipping Quartz and OPT-108 llama-vec. Native llama dtype is F16/F16 token-major via a diagnostic adapter whose conversion cost is reported separately and in the enclosing adapter path. F16 vec at prefixes whose source-selected kernel is MMA is **not** a matched llama result. Native BF16 vs llama F16 is **not** identical arithmetic.

Quartz always launches `n_parts=16`. Matched llama-vec occupancy n_parts were 1/4/5/9/21/21 at prefixes 128/1023/1024/2048/8192/32768.

| layer | prefix | quartz kernel | llama selected | f16 vec matched llama? | quartz enclosing ms | matched BF16 ms | adapter convert ms | f16 kernel ms | adapter enclosing ms |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| 3 | 128 | warp_query_decode_attention | flash_attn_ext_vec<256,1> | True | 0.03631 | 0.02716 | 0.01300 | 0.00951 | 0.01864 |
| 3 | 1023 | warp_query_decode_attention | flash_attn_ext_vec<256,1> | True | 0.09740 | 0.02551 | 0.01041 | 0.01038 | 0.02563 |
| 3 | 1024 | vec128_online_decode_attention | flash_attn_ext_vec<256,1> | True | 0.11004 | 0.02451 | 0.00952 | 0.01055 | 0.02582 |
| 3 | 2048 | vec128_online_decode_attention | flash_attn_ext_vec<256,1> | True | 0.11372 | 0.02840 | 0.01252 | 0.01237 | 0.03613 |
| 3 | 8192 | warp_query_decode_attention | fattn-mma-f16_ncols1=1_ncols2=8 | False | 0.57053 | 0.05098 | 0.03799 | 0.03475 | 0.08548 |
| 3 | 32768 | warp_query_decode_attention | fattn-mma-f16_ncols1=1_ncols2=8 | False | 3.07606 | 0.16665 | 0.19624 | 0.13019 | 0.33204 |
| 7 | 128 | warp_query_decode_attention | flash_attn_ext_vec<256,1> | True | 0.03731 | 0.02316 | 0.00531 | 0.00818 | 0.01900 |
| 7 | 1023 | warp_query_decode_attention | flash_attn_ext_vec<256,1> | True | 0.09636 | 0.02481 | 0.00836 | 0.01155 | 0.02559 |
| 7 | 1024 | vec128_online_decode_attention | flash_attn_ext_vec<256,1> | True | 0.11129 | 0.02497 | 0.00856 | 0.01040 | 0.02474 |
| 7 | 2048 | vec128_online_decode_attention | flash_attn_ext_vec<256,1> | True | 0.11201 | 0.02982 | 0.01325 | 0.01296 | 0.03481 |
| 7 | 8192 | warp_query_decode_attention | fattn-mma-f16_ncols1=1_ncols2=8 | False | 0.56993 | 0.05096 | 0.03716 | 0.03531 | 0.08310 |
| 7 | 32768 | warp_query_decode_attention | fattn-mma-f16_ncols1=1_ncols2=8 | False | 3.08089 | 0.16740 | 0.19874 | 0.13085 | 0.33239 |
| 63 | 128 | warp_query_decode_attention | flash_attn_ext_vec<256,1> | True | 0.03548 | 0.02792 | 0.01037 | 0.01006 | 0.01877 |
| 63 | 1023 | warp_query_decode_attention | flash_attn_ext_vec<256,1> | True | 0.09619 | 0.02636 | 0.01014 | 0.01012 | 0.02538 |
| 63 | 1024 | vec128_online_decode_attention | flash_attn_ext_vec<256,1> | True | 0.10809 | 0.02589 | 0.00895 | 0.01034 | 0.02506 |
| 63 | 2048 | vec128_online_decode_attention | flash_attn_ext_vec<256,1> | True | 0.11324 | 0.02959 | 0.01249 | 0.01256 | 0.03471 |
| 63 | 8192 | warp_query_decode_attention | fattn-mma-f16_ncols1=1_ncols2=8 | False | 0.57210 | 0.04905 | 0.03813 | 0.03517 | 0.08135 |
| 63 | 32768 | warp_query_decode_attention | fattn-mma-f16_ncols1=1_ncols2=8 | False | 3.07585 | 0.16698 | 0.19534 | 0.12916 | 0.33145 |

### Prefix means (3 attention layers) and 16-layer branch

| prefix | quartz kernel | llama selected | quartz ms | matched BF16 ms | adapter ms | f16 kernel ms | 16-layer quartz ms |
|---:|---|---|---:|---:|---:|---:|---:|
| 128 | warp_query_decode_attention | flash_attn_ext_vec<256,1> | 0.03636 | 0.02608 | 0.00956 | 0.00925 | 0.582 |
| 1023 | warp_query_decode_attention | flash_attn_ext_vec<256,1> | 0.09665 | 0.02556 | 0.00964 | 0.01069 | 1.546 |
| 1024 | vec128_online_decode_attention | flash_attn_ext_vec<256,1> | 0.10980 | 0.02512 | 0.00901 | 0.01043 | 1.757 |
| 2048 | vec128_online_decode_attention | flash_attn_ext_vec<256,1> | 0.11299 | 0.02927 | 0.01275 | 0.01263 | 1.808 |
| 8192 | warp_query_decode_attention | fattn-mma-f16_ncols1=1_ncols2=8 | 0.57085 | 0.05033 | 0.03776 | 0.03508 | 9.134 |
| 32768 | warp_query_decode_attention | fattn-mma-f16_ncols1=1_ncols2=8 | 3.07760 | 0.16701 | 0.19677 | 0.13007 | 49.242 |

### Materiality versus OPT-125 request

- D128 16-layer enclosing: **0.5818026858666667** ms (share of OPT-125 request **0.03151523134535869**)
- D2048 16-layer enclosing: **1.8078037546666668** ms (share of OPT-125 request **0.07541943073286053**)
- D32768 16-layer enclosing: **49.24160128** ms/token

OPT-108 primitive screen (exclusive paired events) measured shipping hybrid 0.033 ms vs llama-vec 0.031 ms at D2048, CI including 0. This sitting's matched llama-vec (~0.029 ms) agrees with OPT-108; shipping Quartz (~0.113 ms) does **not**. Absolute Quartz milliseconds are sitting-sensitive. Historical OPT-125 D32768 decode-only was ~15 ms/token; a 49 ms attention-only branch cannot be the production engine cost at capacity 131072. Treat long-context Quartz ms as an upper bound from this non-exclusive component replay.

The 1024 crossover is vs warp_query, not vs llama-vec: at prefix 1023 warp_query ~0.097 ms, at 1024 vec128 ~0.110 ms, while matched llama-vec stays ~0.025 ms on both sides. Shipping `hybrid_crossover@1024` is unchanged.

## Numerical

Quartz BF16 vs OPT-108 BF16 llama-vec is a same-representation comparison (max_abs ~1e-8 to 1e-9, nonfinite 0). Quartz BF16 vs llama F16 is **not** identical arithmetic; the same abs scale is reported only as a diagnostic.

| layer | prefix | max_abs quartz vs matched | max_abs quartz vs f16 | nonfinite | q_hash |
|---:|---:|---:|---:|---:|---|
| 3 | 128 | 1.11758709e-08 | 1.11758709e-08 | 0 | `dfb76fd49d54ec9f` |
| 3 | 1023 | 5.58793545e-09 | 5.58793545e-09 | 0 | `dfb76fd49d54ec9f` |
| 3 | 1024 | 4.65661287e-09 | 4.65661287e-09 | 0 | `dfb76fd49d54ec9f` |
| 3 | 2048 | 2.79396772e-09 | 2.79396772e-09 | 0 | `dfb76fd49d54ec9f` |
| 3 | 8192 | 2.79396772e-09 | 11.3616056 | 0 | `dfb76fd49d54ec9f` |
| 3 | 32768 | 2.79396772e-09 | 0.129812568 | 0 | `dfb76fd49d54ec9f` |
| 7 | 128 | 1.11758709e-08 | 1.11758709e-08 | 0 | `bb44e1c6f9d9ec10` |
| 7 | 1023 | 3.7252903e-09 | 3.7252903e-09 | 0 | `bb44e1c6f9d9ec10` |
| 7 | 1024 | 4.65661287e-09 | 4.65661287e-09 | 0 | `bb44e1c6f9d9ec10` |
| 7 | 2048 | 2.79396772e-09 | 2.79396772e-09 | 0 | `bb44e1c6f9d9ec10` |
| 7 | 8192 | 2.32830644e-09 | 2.38502932 | 0 | `bb44e1c6f9d9ec10` |
| 7 | 32768 | 2.09547579e-09 | 0.124400578 | 0 | `bb44e1c6f9d9ec10` |
| 63 | 128 | 1.11758709e-08 | 1.11758709e-08 | 0 | `81e956507ba009d8` |
| 63 | 1023 | 3.7252903e-09 | 3.7252903e-09 | 0 | `81e956507ba009d8` |
| 63 | 1024 | 4.65661287e-09 | 4.65661287e-09 | 0 | `81e956507ba009d8` |
| 63 | 2048 | 2.79396772e-09 | 2.79396772e-09 | 0 | `81e956507ba009d8` |
| 63 | 8192 | 2.79396772e-09 | 16.8496933 | 0 | `81e956507ba009d8` |
| 63 | 32768 | 2.79396772e-09 | 0.132315442 | 0 | `81e956507ba009d8` |

## Adapter cost

Convert BF16 physical → llama F16 token-major is timed separately from the F16 kernel. At prefix 32768 convert ~0.197 ms/layer exceeds the F16 kernel ~0.130 ms/layer; paying that every token requires a persistent F16 cache, which OPT-130 forbids.

```json
{
  "128": {
    "adapter_mean_ms": 0.00955733333,
    "llama_f16_kernel_mean_ms": 0.009250133283333333,
    "adapter_enclosing_mean_ms": 0.01880319973333333
  },
  "1023": {
    "adapter_mean_ms": 0.009636266773333334,
    "llama_f16_kernel_mean_ms": 0.0106858667,
    "adapter_enclosing_mean_ms": 0.025533866133333333
  },
  "1024": {
    "adapter_mean_ms": 0.009011200006666666,
    "llama_f16_kernel_mean_ms": 0.010429866599999999,
    "adapter_enclosing_mean_ms": 0.025207466733333333
  },
  "2048": {
    "adapter_mean_ms": 0.012753066933333333,
    "llama_f16_kernel_mean_ms": 0.012629333266666667,
    "adapter_enclosing_mean_ms": 0.0352170666
  },
  "8192": {
    "adapter_mean_ms": 0.03775999943333333,
    "llama_f16_kernel_mean_ms": 0.03507626676666667,
    "adapter_enclosing_mean_ms": 0.08331093436666666
  },
  "32768": {
    "adapter_mean_ms": 0.19677439833333335,
    "llama_f16_kernel_mean_ms": 0.13006933533333334,
    "adapter_enclosing_mean_ms": 0.33195946633333334
  }
}
```

## Ranking for OPT-130

Attention material: **True**. OPT-130 disposition: `proceed`.

At most two causal transfers. Rank 1 is the matched-layout BF16 occupancy/partition / no-warp_query-fallback candidate. Rank 2 is source-grounded MMA and is **not** a matched llama result.

```json
[
  {
    "id": "occupancy_partition_or_gqa_kv_reuse",
    "rank": 1,
    "source": "llama vec occupancy n_parts vs Quartz hardcoded 16; GQA=6 duplicated KV rereads in warp_query; shipping warp_query fallback past verified_max 4096",
    "estimated_removable_branch_ms": 1.3394602880000002,
    "enclosing_request_benefit": 1.3394602880000002,
    "long_context_matched_layout_branch_ms": 46.56943922666667,
    "d8192_matched_layout_branch_ms": 8.3283627392,
    "required_layout_workspace": "BF16 physical, no new cache",
    "numerical_risk": "low_for_partition_count, medium_for_gqa6",
    "provenance": "OPT-108 llama_vec_nvidia / OPT-095 gqa6 (both rejected)",
    "matched_result": true,
    "reason": "Matched BF16-physical llama-vec vs shipping hybrid on identical buffers where llama source-selects VEC (prefixes <=2048). OPT-108 D2048 control was 0.033 ms with a 95% CI that included 0 versus llama-vec 0.031 ms; this sitting's larger Quartz number is non-exclusive and must be re-screened before OPT-130 treats D2048 as a proven win. Long-context BF16 llama-vec vs warp_query is the same layout but is not a matched llama result: llama source-selects MMA at n_kv>=8192."
  },
  {
    "id": "llama_mma_decode_consumer",
    "rank": 2,
    "source": "pinned fattn.cu BEST_FATTN_KERNEL_MMA_F16 for Ada+ decode when gqa_ratio>4 and K.ne[1]>=8192; ncols1=1 ncols2=8",
    "estimated_removable_branch_ms": null,
    "enclosing_request_benefit": "unknown_until_mma_launched",
    "required_layout_workspace": "Port MMA onto dense BF16 physical or keep the existing BF16 cache and a BF16 MMA consumer. Native llama MMA reads F16/F16 token-major and needs stream-K/fixup workspace. Diagnostic BF16->F16 convert is not a production path; OPT-130 forbids a shadow F16 cache.",
    "numerical_risk": "medium",
    "provenance": "MIT ggml fattn-mma-f16 at cc83d7b4824f73cfdda4dfbb47ee39804f71b328",
    "matched_result": false,
    "reason": "Source-selected llama kernel at prefixes 8192/32768 is MMA, not the diagnostic F16 vec actually launched. Quartz shipping is warp_query past verified_max 4096. Adapter convert cost grows with prefix and cannot be amortized per token without a forbidden F16 cache."
  }
]
```

## Production

Shipping decode attention remains `hybrid_crossover@1024`. No selector, kernel, or throughput claim.


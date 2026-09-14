# OPT-131 — Fuse one measured residual decode launch chain

Status: **no_material_opportunity**. Parent is OPT-127 kept `decode_segments8` (`opt127_kept_decode_segments8`) with OPT-128 no-opportunity and OPT-130 reject (`hybrid_crossover` unchanged). No production fusion pin is added.

`claims_throughput: False`.

## Sitting identity

- llama_revision: `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`
- gguf_sha256: `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`
- parent: `opt127_kept_decode_segments8` / `decode_segments8`
- measured_at: `2026-09-14T01:25:49Z`

## Causal upper bound

decode_segments8 already captures in-graph producer/consumer launches. Remaining opportunity is intermediate traffic plus the single out-of-graph embedding BF16→FP32 cast. OPT-119 mixer/FFN norm→Q8 is retained and not recreated. Last residual→logits_norm is after the last eight-layer cancellation poll and is not fused.

Parent graph wall `16.370451` ms/token. Eager wall `19.5440044` ms/token. Launch gap already removed `3.17355347` ms/token.

Freeze threshold `0.2` ms/token. 2% request materiality `0.32740902` ms/token.

## Ranked residual chains

| Rank | Name | avoided launches on parent | traffic upper ms | causal bound ms | freeze legal |
|---:|---|---:|---:|---:|---|
| 1 | mixer_residual_add_to_ffn_norm | 0 | 0.026214400000000002 | 0.026214400000000002 | True |
| 2 | gdn_recurrence_to_gated_output | 0 | 0.02359296 | 0.02359296 | False |
| 3 | attn_query_split | 0 | 0.01572864 | 0.01572864 | True |
| 4 | attn_output_cast | 0 | 0.00786432 | 0.00786432 | True |
| 5 | last_residual_to_logits_norm | 0 | 0.00040960000000000004 | 0.00232760005 | False |
| 6 | embed_bf16_to_fp32 | 0 | 0.00020480000000000002 | 0.00221579995 | True |
| 7 | prepare_gdn_gates_to_tiled | 0 | 0.00036864 | 0.00036864 | False |

### mixer_residual_add_to_ffn_norm

residual_add_fp32 → ffn_norm rms_norm_fp32_to_bf16 in `cuda/full_scheduler.cu`. in_graph=`True` gdn=`False` cancel_boundary=`False` opt119_reuse=`True`.

### gdn_recurrence_to_gated_output

launch_gdn_prepare_tiled → launch_gdn_gated_output in `cuda/gdn_step.cu,cuda/scheduler_primitives.cu`. in_graph=`True` gdn=`True` cancel_boundary=`False` opt119_reuse=`False`.

### attn_query_split

launch_q8_mixer_input_group → launch_split_attention_query_gate in `cuda/scheduler_primitives.cu,cuda/full_scheduler.cu`. in_graph=`True` gdn=`False` cancel_boundary=`False` opt119_reuse=`False`.

### attn_output_cast

launch_attention_prepare → launch_fp32_to_bf16 in `cuda/full_scheduler.cu,cuda/scheduler_primitives.cu`. in_graph=`True` gdn=`False` cancel_boundary=`False` opt119_reuse=`False`.

### last_residual_to_logits_norm

execute_ffn residual_add last layer → launch_rms_norm_fp32_dispatch output_norm in `cuda/full_scheduler.cu`. in_graph=`False` gdn=`False` cancel_boundary=`True` opt119_reuse=`True`.

### embed_bf16_to_fp32

launch_quant_row_decode → bf16_to_fp32 in `cuda/full_scheduler.cu`. in_graph=`False` gdn=`False` cancel_boundary=`False` opt119_reuse=`False`.

### prepare_gdn_gates_to_tiled

launch_prepare_gdn_gates → launch_gdn_prepare_tiled in `cuda/scheduler_primitives.cu,cuda/gdn_step.cu`. in_graph=`True` gdn=`True` cancel_boundary=`False` opt119_reuse=`False`.

## OPT-109 GDN prior

isolated_win=`True` enclosing_loss=`True` enclosing_saving_ms=`-0.803`. OPT-109 isolated recurrence won but complete 48-layer GDN enclosing lost. A GDN chain is not frozen without a new enclosing win.

## Lifetime / cancellation / equivalence

lifetime=`True` cancel=`True` equivalence=`True`. Graph vs eager greedy tokens match on the unchanged fallback. Candidate/committed GDN and KV buffers stay separate. No fusion across the eight-layer poll.

## Frozen candidates (at most two)

Admitted: `[]`. Frozen: `none`. decode_segments8 already captures in-graph short kernels, so avoided launches on the parent are zero for those chains. Causal upper bound is remaining intermediate traffic at 100 GB/s plus the single out-of-graph embed cast. OPT-119 norm→Q8 is reused, not recreated. GDN chains stay unfrozen because OPT-109 isolated win / enclosing loss. Last residual→logits_norm crosses the last segment cancellation poll. Candidates freeze only when the bound clears 0.2000 ms/token (max of 0.2 ms and 1% of parent graph wall). 2% request materiality is 0.3274 ms/token. A kernel-only speedup is not a keep.

**Verdict: `no_material_opportunity`.** shipping fusion `none`. D128 tok/s delta `0.0`; speedup `0.0`; baseline_unchanged `True`.


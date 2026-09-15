# OPT-149 — Decode mixer RMSNorm fused into Q8_1 staging screen

Status: **screened_out**. Shipping decode attention `decode_attention_flash_vec_v1`. Shipping decode-norm-q8 `current_bf16_q8_staging`. Parent retained; shipping decode-norm-q8 stays `current_bf16_q8_staging`.

Parent `post124_plus_opt127_decode_segments8_plus_opt137_mma_plus_opt147_8x8_plus_opt148_flash_vec` with kept OPT-147 prompt `prefill_attention_8x8_v1` and kept OPT-148 decode `decode_attention_flash_vec_v1`. Control is `current_bf16_q8_staging` (BF16 RMSNorm plus `launch_quantize_bf16_q8_1`). Candidate `norm_to_q8_1_screen_v1` fuses eligible mixer input_norm into `quartz_q8_1_sum_q` with in-register BF16 rounding. FFN `llama_q4k_mmvq`, Q8Block, and logits BF16 stay unfused. OPT-110 is not revived. Q8 byte size alone is not a win.

## Candidate

Candidate ID: `norm_to_q8_1_screen_v1`. Launch `rms_norm_fp32_to_q8_1` / `residual_add_norm_fp32_to_q8_1`. Selected group `mixer_input_norm_q8_1`.
Supported mechanism: `unknown`. Screen admission does not require SASS.
Missing required work is incomplete/blocked, never `no_opportunity`.

## Correctness

Compared fused Q8_1 versus control GPU staging for GDN and attention layers, with and without residual add, mixed-consumer retain, stale-buffer rejection, graph/eager equality, and sampled FP64 refs. Identity gate is GPU control versus candidate Q8 bytes.

Correctness sidecar ok=`True`.

## Screen (D2048 complete decode-mixer family)

Warmups `1`, alternating pairs `3`, 64 mixer layers, residual+norm+Q8+projections included, plus one short engine pair (prefix 2048, 32 output tokens).
Control mean `2.101024` ms. Candidate mean `2.255574` ms. Saving `-0.15455` ms. Screened in: `False`. Reason: `complete_family_time_did_not_improve`.

Independent reconstruction of one D2048 pair set: [`d2048-pair-reconstruction.json`](d2048-pair-reconstruction.json).

## Quality

OPT-058 invoked=`False`; candidate NLL measured=`False`; NLL required=`False`; skip_reason=`screened_out_retain_parent`. Screened-in survivors must measure candidate NLL because this is an arithmetic/output-encoding change. Screened-out retains parent and does not stub `candidate_nll_not_measured`.

Held-out NLL `None`. ppl_ratio=`None`.

## Keep policy

`target_guard_v2` target `d2048.decode_only` with D2048 `complete_request` as the target additional guard. Guards: D128/D8192/D32768 `decode_only` and `complete_request` plus P4096 prefill. Mechanism remains unknown and is not a substitute for those gates.

Verdict `screened_out`. production_kept=`False`.
claims_throughput: `False`.

## Deltas

Candidate measured delta: `{'rounds': [{'warmup': False, 'sample_index': 0, 'control_ms': 2.101248, 'candidate_ms': 2.24912}, {'warmup': False, 'sample_index': 1, 'control_ms': 2.102592, 'candidate_ms': 2.258944}, {'warmup': False, 'sample_index': 2, 'control_ms': 2.099232, 'candidate_ms': 2.258656}], 'reconstructed_control_mean_ms': 2.1010240000000002, 'reconstructed_candidate_mean_ms': 2.2555733333333334, 'reconstructed_saving_ms': -0.1545493333333332, 'reconstructed_faster': False, 'layers': 64, 'control_mean_ms': 2.101024, 'candidate_mean_ms': 2.255574, 'saving_ms': -0.15455, 'faster': False, 'screened_in': False, 'pairs': 3, 'warmups': 1, 'ok': True}`.
Shipping delta: `0` (zero on screened_out/reject; parent retained).
Quality result: `nll_not_required_screened_out`.

## Performance evidence checklist

1. **Measurement identity** — Quartz production `decode_segments8` + kept OPT-137 MMA + kept OPT-147 `prefill_attention_8x8_v1` + kept OPT-148 `decode_attention_flash_vec_v1`; parent decode-norm-q8 `current_bf16_q8_staging`; candidate launch `rms_norm_fp32_to_q8_1`; shipping after this task `current_bf16_q8_staging`; GGUF `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`; llama revision `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`.
2. **Coverage** — native complete decode-mixer family screen at D2048 with rotating layers plus one short engine pair; correctness over GDN/attention, residual, mixed consumers, stale buffers, graph/eager.
3. **Time accounting** — screen means are complete-family enclosing times including residual, RMSNorm, Q8 staging and four mixer projections, not Q8 byte-size-only or leaf-only.
4. **Contradiction register** — fused Q8_1 vs BF16+restage is the OPT-112 hypothesis under test, not a freeze blocker. OPT-110 is not revived. Mechanism stays unknown without source/SASS evidence.
5. **Claim types** — mechanism `unknown`; screen `measured`; keep only after quality/state/target_guard_v2.
6. **Target/guard** — OPT-135 `target_guard_v2` opted in for screened-in survivors.
7. **Independent verification** — D2048 pair reconstruction recomputes means from raw alternating rounds.
8. **Reporting** — candidate measured delta and shipping delta are separate; shipping delta is 0 unless `kSelectedDecodeNormQ81Fusion` flips on keep.

## Raw gates

Sidecars: [`raw/`](raw/). Reconstruction: [`d2048-pair-reconstruction.json`](d2048-pair-reconstruction.json). Fixture: `fixtures/opt149_norm_q8.json`.

## Status

verdict=`screened_out` production_kept=`False` blocked=`False`.
Production fusion pin `current_bf16_q8_staging`. Decode attention `decode_attention_flash_vec_v1`. Prompt pin `prefill_attention_8x8_v1`.


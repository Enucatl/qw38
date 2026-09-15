# OPT-148 — Short-decode flash-style vector attention screen

Status: **keep**. Shipping decode attention `decode_attention_flash_vec_v1`. Production pin flipped to `decode_attention_flash_vec_v1`.

Parent `post124_plus_opt127_decode_segments8_plus_opt137_mma` with kept OPT-147 prompt `prefill_attention_8x8_v1`. Parent D2048 path is `vec128_online` (`hybrid_crossover` window [1024,4096], n_parts=16). Candidate `decode_attention_flash_vec_v1` is a Quartz-owned flash-style vector path using llama `flash_attn_ext_vec` only as the implementation boundary. OPT-137 MMA remains at positions `>= 8192`. D128 stays `warp_query`. OPT-130 is not revived.

## Candidate

Candidate ID: `decode_attention_flash_vec_v1`. Launch `flash_vec_decode_attention`.
Supported mechanism: `unknown`. Screen admission does not require SASS.
Missing required work is incomplete/blocked, never `no_opportunity`.

## Correctness

Compared flash-vec versus `vec128_online` at positions 0, 1023, 1024, 2047, 2048, 4096, 8192 with rotating layers and sampled FP64 refs. D128 must stay `warp_query`; positions `>=8192` must stay MMA.

Correctness sidecar ok=`True`.

## Screen (D2048 complete family)

Warmups `1`, alternating pairs `3`, 16 rotating attention layers, prep+combine+conversion included, plus one short engine pair (prefix 2048, 32 output tokens).
Control mean `1.775925` ms. Candidate mean `0.601931` ms. Saving `1.173995` ms. Screened in: `True`. Reason: `None`.

Independent reconstruction of one D2048 pair set: [`d2048-pair-reconstruction.json`](d2048-pair-reconstruction.json).

## Quality

OPT-058 invoked=`True`; candidate NLL measured=`True`; NLL required=`True`; skip_reason=`None`. Screened-in survivors must measure candidate NLL because this is an arithmetic change. Screened-out retains parent and does not stub `candidate_nll_not_measured`.

Held-out NLL `1.7875990840085783`. ppl_ratio=`1.0`.

## Keep policy

`target_guard_v2` target `d2048.decode_only` with D2048 `complete_request` as the target additional guard. Guards: D128/D8192/D32768 `decode_only` and `complete_request` plus P4096 prefill. Mechanism remains unknown and is not a substitute for those gates.

Verdict `keep`. production_kept=`True`.
claims_throughput: `True`.

## Deltas

Candidate measured delta: `{'d2048.decode_only': {'control_rate': 49.44886580977065, 'candidate_rate': 59.765056814895765, 'delta': 10.316191005125113, 'g': 1.2086232764293652}}`.
Shipping delta: `{'d2048.decode_only': {'control_rate': 49.44886580977065, 'candidate_rate': 59.765056814895765, 'delta': 10.316191005125113, 'g': 1.2086232764293652}}` (zero on screened_out/reject; parent retained).
Quality result: `measured`.

## Performance evidence checklist

1. **Measurement identity** — Quartz production `decode_segments8` + kept OPT-137 MMA + kept OPT-147 `prefill_attention_8x8_v1`; parent decode `hybrid_crossover`/`vec128_online`; candidate launch `flash_vec_decode_attention`; shipping after this task `decode_attention_flash_vec_v1`; GGUF `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`; llama revision `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`.
2. **Coverage** — native complete-family screen at D2048 with rotating layers plus one short engine pair; correctness over required positions.
3. **Time accounting** — screen means are complete-family enclosing times including prep, conversion, combine and 16-layer rotation, not leaf-only.
4. **Contradiction register** — flash-style tile organization vs `vec128_online` is the hypothesis under test, not a freeze blocker. Mechanism stays unknown without source/SASS evidence.
5. **Claim types** — mechanism `unknown`; screen `measured`; keep only after quality/state/target_guard_v2.
6. **Target/guard** — OPT-135 `target_guard_v2` opted in for screened-in survivors.
7. **Independent verification** — D2048 pair reconstruction recomputes means from raw alternating rounds.
8. **Reporting** — candidate measured delta and shipping delta are separate; shipping delta is 0 unless production pin flips on keep.

## Raw gates

Sidecars: [`raw/`](raw/). Reconstruction: [`d2048-pair-reconstruction.json`](d2048-pair-reconstruction.json). Fixture: `fixtures/opt148_short_decode_flash.json`.

## Status

verdict=`keep` production_kept=`True` blocked=`False`.
Production pin `decode_attention_flash_vec_v1`. Prompt pin `prefill_attention_8x8_v1`.


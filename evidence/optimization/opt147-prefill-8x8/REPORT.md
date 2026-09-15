# OPT-147 — Prefill 8x8 flash-attention tile screen

Status: **keep**. Shipping prompt attention `prefill_attention_8x8_v1`. Production pin flipped to `prefill_attention_8x8_v1`.

Parent `post124_plus_opt127_decode_segments8_plus_opt137_mma`. Parent P4096 path is `fattn_mma_pipeline_opt111_base` (`opt111_base`, 16×2). Candidate `prefill_attention_8x8_v1` is llama-shaped 8×8 with GQA ratio six padded to ncols2=8 (heads 6 and 7 masked). OPT-111 arithmetic, BF16 state, causal masks, chunking and final-token output policy are unchanged. OPT-137 MMA remains at positions `>= 8192`.

## Candidate

Candidate ID: `prefill_attention_8x8_v1`. Launch `fattn_mma_pipeline_prefill_attention_8x8_v1`.
Supported mechanism: `unknown`. Screen admission does not require SASS.
Missing required work is incomplete/blocked, never `no_opportunity`.

## Correctness

Compared 8×8 versus `opt111_base` at prompt lengths 1, 32, 128, 2048, 4096, plus causal boundary/tail, chunk-mid, GQA head sampling, finite outputs, graph/eager capture, and prompt-to-decode KV handoff.

Correctness sidecar ok=`True`.

## Screen (P4096 complete family)

Warmups `1`, alternating pairs `3`, 16 rotating attention layers, prep+combine+conversion included.
Control mean `222.146301` ms. Candidate mean `201.478683` ms. Saving `20.667618` ms. Screened in: `True`. Reason: `None`.

Independent reconstruction of one P4096 pair set: [`p4096-pair-reconstruction.json`](p4096-pair-reconstruction.json).

## Quality

OPT-058 invoked=`True`; candidate NLL measured=`True`; NLL required=`True`; skip_reason=`None`. Screened-in survivors must measure candidate NLL. Screened-out retains parent and does not stub `candidate_nll_not_measured`.

Held-out NLL `1.7875990840085783`. ppl_ratio=`1.0`.

## Keep policy

`target_guard_v2` target `p4096.prefill`. Guards: D128/D2048/D8192/D32768 `decode_only` and `complete_request`. Mechanism remains unknown and is not a substitute for those gates.

Verdict `keep`. production_kept=`True`.
claims_throughput: `True`.

## Deltas

Candidate measured delta: `{'p4096.prefill': {'control_rate': 2997.7860688227747, 'candidate_rate': 3032.417338625696, 'delta': 34.63126980292145, 'g': 1.01158065552246}}`.
Shipping delta: `{'p4096.prefill': {'control_rate': 2997.7860688227747, 'candidate_rate': 3032.417338625696, 'delta': 34.63126980292145, 'g': 1.01158065552246}}` (zero on screened_out/reject; parent retained).
Quality result: `measured`.

## Performance evidence checklist

1. **Measurement identity** — Quartz production `decode_segments8` + kept OPT-137 MMA; parent prompt `opt111_base`; candidate launch `fattn_mma_pipeline_prefill_attention_8x8_v1`; shipping after this task `prefill_attention_8x8_v1`; GGUF `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`; llama revision `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`.
2. **Coverage** — native complete-family screen at P4096 with rotating layers; correctness over required lengths, causal tails, GQA pad masks.
3. **Time accounting** — screen means are complete-family enclosing times including prep, conversion, combine and 16-layer rotation, not leaf-only.
4. **Contradiction register** — 16×2 vs 8×8 tile mismatch is the hypothesis under test, not a freeze blocker. Decode NCU is not prefill evidence.
5. **Claim types** — mechanism `unknown`; screen `measured`; keep only after quality/state/target_guard_v2.
6. **Target/guard** — OPT-135 `target_guard_v2` opted in for screened-in survivors.
7. **Independent verification** — P4096 pair reconstruction recomputes means from raw alternating rounds.
8. **Reporting** — candidate measured delta and shipping delta are separate; shipping delta is 0 unless production pin flips on keep.

## Raw gates

Sidecars: [`raw/`](raw/). Reconstruction: [`p4096-pair-reconstruction.json`](p4096-pair-reconstruction.json). Fixture: `fixtures/opt147_prefill_8x8.json`.

## Status

verdict=`keep` production_kept=`True` blocked=`False`.
Production pin `prefill_attention_8x8_v1`.


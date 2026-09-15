# OPT-144 — Complete-boundary P4096 prefill attention

Status: **no_opportunity**. Shipping prompt attention `opt111_base`. Parent retained on no-keep.

Parent `post124_plus_opt127_decode_segments8_plus_opt137_mma`. OPT-143 did not keep a short-decode candidate. P4096 production path is `fattn_mma_pipeline_opt111_base` (`opt111_base`). OPT-137 MMA remains at positions `>= 8192`. Rejected OPT-130 occupancy path is not reopened. OPT-111 is not re-ported.

## Frozen candidate

Candidate: `None`.
Supported mechanism: `None`.
Admission reason: `throughput_alone_cannot_establish_mechanism`.
Reasons: `['throughput_alone_cannot_establish_mechanism', 'throughput_occupancy_dram_alone_cannot_freeze_candidate', 'opt138_expected_benefit_unknown_until_mechanism', 'opt141_fused_boundary_mismatch_16x2_vs_8x8', 'tile_shape_mismatch_cannot_freeze_without_comparable_counters', 'opt141_llama_mechanism_null', 'decode_attention_substitution_rejected', 'tensor_activity_nonzero_not_assumed_zero']`.

Freeze requires matched positive family excess **and** a source-grounded mechanism for the production prompt-attention dispatch. Occupancy, DRAM throughput, and byte counters alone cannot freeze a kernel. Decode NCU is not P4096 prefill evidence. OPT-111 `opt111_base` is already shipping.

## Family excess (OPT-138)

P4096 `attn_core` complete-family mean `186.58288100000001` ms (n=`3`, one-sided low `186.10589919623453` ms). mean_ms is matched P4096 complete-family attn_core excess, not a promised end-to-end saving; family excess exceeds the ~87.2 ms net whole-engine gap. Significant positive excess: `True`. Approximate net whole-engine gap `87.2` ms. Offsetting families and residual mean this is not a promised end-to-end saving. OPT-142 complete gap closure is not claimed (`False`).

## Production identity (OPT-140)

P4096 path `opt111_base` kernel `fattn_mma_pipeline_opt111_base`. Replay family `prompt-attention`. Decode-attention substitution `False`. Decode NCU rejected as prefill evidence: `True`.

Complete family includes split/stage/convert/core/merge/epilogue. Counter kernel is `fattn_mma_pipeline_opt111_base` only (`counter_kernel_is_complete_family=false`).

## Typed Quartz P4096 counters (OPT-140)

| Slot | Value |
|---|---|
| dram_read_bytes | `252.12` |
| dram_throughput | `1.22` |
| sm_throughput | `22.4` |
| occupancy | `8.33` |
| tensor_activity | `810024960.0` |
| l2_traffic | `216065551.0` |

Source observations: `False`; SASS observations: `False`. Tensor activity nonzero: `True` (do not assume zero from aggregate replay text).

## Matched llama comparison (OPT-141)

Comparable P4096 attn_core: `False`. Fusion reason: `fused_boundary_mismatch`. Llama kernel `flash_attn_ext_f16<256,256,8,8>`. Quartz kernel `fattn_mma_pipeline_opt111_base` (16×2). Leaf counters are incomparable; tile-shape mismatch is not a freeze.

| Slot | Quartz | Llama | Unit |
|---|---|---|---|
| _(none — fused_boundary_mismatch)_ | | | |

Llama `supported_mechanism` remains null. Throughput/occupancy differences are not a named source/SASS-backed change at `fattn_mma_pipeline_opt111_base`.

## Quality

OPT-058 invoked=`False`; candidate NLL measured=`False`; NLL required=`False`; skip_reason=`no_candidate_no_arithmetic_change`. Changed arithmetic requires measured candidate NLL; no_opportunity does not change arithmetic and does not borrow parent NLL.

## Keep policy

`target_guard_v2` target `p4096.prefill`. Guards: D128/D2048/D8192/D32768 `decode_only` and `complete_request`. Prompt and whole-request effects are reported separately when a candidate times; unused here because no candidate ran AB/BA.

Verdict `no_opportunity`. production_kept=`False`.
claims_throughput: `False`.

## Deltas

Candidate measured delta: `None`.
Shipping delta: `0` (zero on no_opportunity/reject; parent retained).
Quality result: `nll_not_required_no_arithmetic_change`.
Prompt effect: N/A. Whole-request effect: N/A.

## Performance evidence checklist

1. **Measurement identity** — Quartz production `decode_segments8` + kept OPT-137 MMA + shipping `opt111_base`; P4096 prompt-attention replay `fattn_mma_pipeline_opt111_base`; GGUF `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`; llama revision `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`.
2. **Coverage** — OPT-138 complete-family excess; OPT-140 rotating prompt-attention replay and target-kernel NCU `--launch-count 1`; OPT-141 matched llama `flash_attn_ext_f16` with fused-boundary rejection.
3. **Time accounting** — +186.58 ms is matched family excess, not a promised whole-engine saving versus ~87.2 ms net gap; enclosing replay totals are not mixed into typed slots.
4. **Contradiction register** — none between OPT-140 identity, prompt-attention dispatch, and OPT-141 fused-boundary mismatch. Decode NCU as prefill evidence is rejected. Tensor activity is measured nonzero.
5. **Claim types** — family excess `measured`; P4096 identity `measured`; mechanism `unknown`/`incomplete`; no_opportunity `measured` from absent source/SASS, not from skipping collection.
6. **Target/guard** — OPT-135 `target_guard_v2` opted in; unused for keep because no candidate ran AB/BA.
7. **Independent verification** — verifier PASS 2026-09-15T01:13:26Z; pytest 38 passed; ruff clean; quality skip `nll_required=false`, `candidate_nll_not_measured=false`.
8. **Reporting** — candidate measured delta N/A; shipping delta 0; quality N/A (no arithmetic change). Prompt vs whole-request split unused.

## Raw gates

Sidecars: [`raw/`](raw/) (`preflight.json`, `freeze.json`, phase skips, `report.json`). Structured freeze: [`freeze.json`](freeze.json). Fixture dump: [`answers.json`](answers.json) and `fixtures/opt144_prefill_attention.json`.

## Status

verdict=`no_opportunity` production_kept=`False` blocked=`False`.
No production kernel or selector change.


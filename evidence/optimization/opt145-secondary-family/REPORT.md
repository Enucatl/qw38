# OPT-145 — Remaining residual_norm_quant or prompt_mmq family

Status: **no_opportunity**. Parent retained on no-keep. Shipping Q4 decode `llama_q4k_mmvq`; prompt MMQ `fma_async_x`; prompt attention `opt111_base`.

Parent `post124_plus_opt127_decode_segments8_plus_opt137_mma`. OPT-143/144 attention work did not keep a candidate, so residual_norm_quant decode and prompt_mmq prefill evidence was refreshed against the same production objects.

## Family selection

Rank by **excess / matched whole-workload interval**, not raw milliseconds. Twelve decode evals are not compared with one P4096 prompt by ms.

Selected family: `residual_norm_quant`. Supported fraction: `0.03576104492604553`. Excess ms: `7.209406666666666`. Winning workload: `d128` (`decode_only`).

| Workload | Kind | Excess ms | Window ms | Fraction | Positive |
|---|---|---|---|---|---|
| `d128` | `decode` | `7.209406666666666` | `201.599441` | `0.03576104492604553` | `True` |
| `d2048` | `decode` | `6.662013999999999` | `246.21243` | `0.027057992157422755` | `True` |

Unselected family: `prompt_mmq`. Unselected reason: `smaller_supported_excess_fraction`. Recorded for future prioritization; no automatic second experiment.

| Workload | Kind | Excess ms | Window ms | Fraction | Positive |
|---|---|---|---|---|---|
| `p4096` | `prefill` | `27.108745000000038` | `1347.756601` | `0.02011397679661599` | `True` |

## Frozen candidate

Candidate: `None`.
Supported mechanism: `None`.
Admission reason: `quartz_leaf_record_missing_or_unusable`.
Reasons: `['quartz_leaf_record_missing_or_unusable', 'throughput_occupancy_dram_alone_cannot_freeze_candidate', 'fused_boundary_mismatch_quartz_rms_plus_bf16_quant', 'decode_mixer_stalls_cannot_identify_rmsnorm', 'opt112_deferred_fusion_not_revived', 'opt131_chain_fusion_not_revived', 'opt138_expected_benefit_unknown_until_mechanism', 'quartz_counters_unusable', 'opt110_not_re_ported']`.

Freeze requires matched positive family excess **and** a source-grounded mechanism at the fused comparison boundary. Occupancy, DRAM, and byte counters alone cannot freeze a kernel. Decode-mixer replay stalls cannot identify an RMSNorm bottleneck. OPT-110/112/131/122/130 are not revived.

## Resolving measurement

Kind `fused_boundary_identity`. Quartz kernel `rms_norm_fp32_to_bf16_parallel`. Llama kernel `quantize_q8_1`. Fusion ok `False` reason `fused_boundary_mismatch`. Named source/SASS `False`.

fused residual+rms+bf16+quant vs llama rms_norm_f32+quantize_q8_1; leaf counters incomparable; decode-mixer replay stalls are not an RMSNorm bottleneck

## Quality

OPT-058 invoked=`False`; candidate NLL measured=`False`; NLL required=`False`; skip_reason=`no_candidate_no_arithmetic_change`. Changed arithmetic requires measured candidate NLL; no_opportunity does not change arithmetic and does not borrow parent NLL.

## Keep policy

`target_guard_v2` target `d128.decode_only`. Roles `{'target_workloads': ['d128'], 'targets': [{'workload': 'd128', 'kind': 'decode', 'primary_metric': 'decode_only'}], 'guards': [{'workload': 'd2048', 'kind': 'decode'}, {'workload': 'd8192', 'kind': 'decode'}, {'workload': 'd32768', 'kind': 'decode'}, {'workload': 'p4096', 'kind': 'prefill'}], 'target_metric': 'decode_only', 'complete_request_target_guard': True}`.

Verdict `no_opportunity`. production_kept=`False`.
claims_throughput: `False`.

## Deltas

Candidate measured delta: `None`.
Shipping delta: `0` (zero on no_opportunity/reject; parent retained).
Quality result: `nll_not_required_no_arithmetic_change`.

## Performance evidence checklist

1. **Measurement identity** — Quartz production `decode_segments8` + kept OPT-137 MMA; shipping Q4 `llama_q4k_mmvq`; RMS `rms_norm_fp32_to_bf16_parallel` / `parallel_fma`; prompt MMQ `quant_mmq_mma_quality_kernel` / `fma_async_x`; GGUF `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`; llama revision `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`. OPT-143/144 did not change production, so OPT-138 family windows remain the current sitting.
2. **Coverage** — OPT-138 matched family excess on decode middle windows and P4096 complete; OPT-141 fused-boundary rejection for residual_norm_quant; prompt_mmq fusion ok with quartz leaf counters unusable.
3. **Time accounting** — family excess is a fraction of the matched whole-workload interval (`quartz_window_ms`); overlapping graph parents are not summed with children.
4. **Contradiction register** — raw ms would prefer prompt_mmq (+27.11 vs +7.21); fraction ranking prefers residual_norm_quant. Decode-mixer stalls are not treated as an RMSNorm diagnosis. OPT-112 deferred fusion is not a freeze.
5. **Claim types** — family excess `measured`; fraction ranking `derived`; mechanism `unknown`/`incomplete`; no_opportunity `measured` from absent source/SASS after one fused-boundary resolving check.
6. **Target/guard** — OPT-135 `target_guard_v2` opted in; unused for keep because no candidate ran AB/BA.
7. **Independent verification** — verifier PASS (2026-09-15T01:26:00Z): 59 pytest passed; ruff clean; freeze/report verdict `no_opportunity`; parent retained; shipping delta 0.
8. **Reporting** — candidate measured delta N/A; shipping delta 0; quality N/A (no arithmetic change). Unselected family evidence retained.

## Raw gates

Sidecars: [`raw/`](raw/) (`preflight.json`, `freeze.json`, phase skips, `report.json`). Structured freeze: [`freeze.json`](freeze.json). Selection: [`selection.json`](selection.json). Fixture dump: [`answers.json`](answers.json) and `fixtures/opt145_secondary_family.json`.

## Status

verdict=`no_opportunity` production_kept=`False` blocked=`False`.
No production kernel or selector change.


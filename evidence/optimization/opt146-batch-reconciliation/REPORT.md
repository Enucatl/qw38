# OPT-146 — Measure the combined stack and publish the next remaining-gap decision

Diagnostics only. `claims_throughput=false`. Shipping delta: **0**.
NLL: N/A. No production selector, kernel, or scheduling change.
Do not read this report as parity, final optimization, or release readiness.
OPT-016 / OPT-056 / CMP-003 remain unchanged.

Measured at `2026-09-15T02:10:31Z`.
Parent `post124_plus_opt127_decode_segments8_plus_opt137_mma`. Selector `decode_segments8`. Capacity `131072`.

## Frozen identity

- pins ok=`True`
- execution graphs `decode_segments8`
- OPT-137 dense MMA `True` threshold `8192`
- Q4 decode `llama_q4k_mmvq`
- Quartz binary `7fd0efb4abb5b0943d2462812a4b88e0c99d7c372b61a2cfff7807b3099377e5`
- llama profile `4a02206a38b8c01a7f9b313ef2bd5ff42844790dfc3772aba43dca5c6ea9f794`
- GPU lock `build/optimization-runs/qw38-gpu.lock`
- decode prefixes `[128, 2048, 8192, 32768]`
- P4096 prefill; eval tokens `256`; warmups `3`; pairs `10`

## OPT-137 control

- reconstructable=`True`
- method=`current_production_is_opt137_kept_stack`
- same sitting as final=`True`
- historical rates used as paired gain=`False`

OPT-143/144/145 did not keep. Production pins remain the OPT-137 kept stack. A distinct pre-keep control cannot be reconstructed without changing production pins.

## Net batch gains

- shipping delta **0**
- candidate measured delta **0**
- batch keeps: `0`
- OPT-143/144/145 were measured `no_opportunity`; parent retained.

## OPT-143 / OPT-144 / OPT-145

| Task | Verdict | Family | Keep | Shipping | Mechanism |
|---|---|---|---|---:|---|
| OPT-143 | no_opportunity | attn_core | false | 0 | None |
| OPT-144 | no_opportunity | attn_core | false | 0 | None |
| OPT-145 | no_opportunity | residual_norm_quant | false | 0 | None |

## Matched Quartz vs llama throughput

Same-sitting unprofiled runs. OPT-137 control is this production binary.

| Workload | Metric | Quartz mean | llama mean | Gap | Quartz tok/s | llama tok/s | n |
|---|---|---:|---:|---:|---:|---:|---:|
| d128 | decode_only | 4305.0961 | 3635.4261 | 669.6700 | 59.4645 | 70.4183 | 10 |
| d128 | complete_request | 4387.0979 | 3704.0018 | 683.0961 | 58.3530 | 69.1146 | 10 |
| d2048 | decode_only | 5113.7515 | 3680.2777 | 1433.4738 | 50.0611 | 69.5600 | 10 |
| d2048 | complete_request | 5753.0358 | 4320.9318 | 1432.1040 | 44.4982 | 59.2465 | 10 |
| d8192 | decode_only | 5271.6531 | 3829.5311 | 1442.1220 | 48.5616 | 66.8490 | 10 |
| d8192 | complete_request | 8412.3806 | 6454.1093 | 1958.2713 | 30.4313 | 39.6647 | 10 |
| d32768 | decode_only | 7532.9496 | 4106.3390 | 3426.6106 | 33.9840 | 62.3427 | 10 |
| d32768 | complete_request | 29468.0103 | 16415.9841 | 13052.0262 | 8.6874 | 15.5946 | 10 |
| p4096 | complete_request | 1367.0656 | 1270.5693 | 96.4962 | 2996.2214 | 3223.7768 | 10 |
| p4096 | prefill | 1367.0656 | 1270.5693 | 96.4962 | 2996.2214 | 3223.7768 | 10 |

Rates use matching metric identities (decode_only vs decode_only, complete_request vs complete_request, prefill vs prefill). Setup, graph creation and warmup are outside the denominators.

## Wall reconciliation (OPT-142 equation, <=5% unresolved)

- Quartz windows within limit: `21` / `21`
- max unresolved share: `0.0026` (limit `0.05`)
- coverage ok: `True`
- no proportional allocation: `True`

| Window | Unresolved share | Within 5% | Window perturbed |
|---|---:|---|---|
| decode-d128-early-r0 | 0.0025 | True | False |
| decode-d128-early-r1 | 0.0026 | True | False |
| decode-d128-early-r2 | 0.0025 | True | False |
| decode-d128-middle-r0 | 0.0018 | True | False |
| decode-d128-middle-r1 | 0.0017 | True | False |
| decode-d128-middle-r2 | 0.0017 | True | False |
| decode-d128-late-r0 | 0.0019 | True | False |
| decode-d128-late-r1 | 0.0020 | True | False |
| decode-d128-late-r2 | 0.0020 | True | False |
| decode-d2048-early-r0 | 0.0021 | True | False |
| decode-d2048-early-r1 | 0.0017 | True | False |
| decode-d2048-early-r2 | 0.0019 | True | False |
| decode-d2048-middle-r0 | 0.0013 | True | False |
| decode-d2048-middle-r1 | 0.0014 | True | False |
| decode-d2048-middle-r2 | 0.0013 | True | False |
| decode-d2048-late-r0 | 0.0024 | True | False |
| decode-d2048-late-r1 | 0.0015 | True | False |
| decode-d2048-late-r2 | 0.0016 | True | False |
| prefill-d4096-prefill-r0 | 0.0001 | True | False |
| prefill-d4096-prefill-r1 | 0.0001 | True | False |
| prefill-d4096-prefill-r2 | 0.0001 | True | False |

Decode whole-request nsys vs unprofiled decode_only_ms is not the
window perturbation check; OPT-142 residual uses the 12-token
partition. Prefill native chrono matches the capture window.

## Top-two positive-excess families (fresh captures)

### decode

- `attn_core` score_ms=`47.5277` evidenced=`True`
- `residual_norm_quant` score_ms=`7.1542` evidenced=`True`

### prefill

- `attn_core` score_ms=`186.1318` evidenced=`True`
- `prompt_mmq` score_ms=`28.5925` evidenced=`True`


## Next-experiment specifications

At most one spec per phase. `candidate=null` when no source-grounded mechanism exists.

### decode

- selected family `attn_core`
- measured excess ms `47.5277`
- candidate `None`
- supported mechanism `None`
- prior experiment `OPT-143` (`no_opportunity`)
- resolving measurement: OPT-143 already measured no_opportunity for attn_core: throughput_alone_cannot_establish_mechanism. A later experiment requires a new named source/SASS or fused-boundary identity, not a rerun of occupancy/DRAM ranking.

### prefill

- selected family `attn_core`
- measured excess ms `186.1318`
- candidate `None`
- supported mechanism `None`
- prior experiment `OPT-144` (`no_opportunity`)
- resolving measurement: OPT-144 already measured no_opportunity for attn_core: throughput_alone_cannot_establish_mechanism. A later experiment requires a new named source/SASS or fused-boundary identity, not a rerun of occupancy/DRAM ranking.

## Independent reconstruction

- paired throughput `d2048.decode_only` ok=`True`
- family/window `d2048.middle.r0` ok=`True`

## Performance evidence checklist

1. Measurement identity: frozen binary hashes, selectors, capacity 131072, GGUF, llama revision, warmups/samples, decode_only / complete_request / prefill.
2. Coverage: OPT-136 `audit_window_from_tables` plus `performance_evidence.py --validate` on OPT-146 `coverage.json`.
3. Time accounting: OPT-142 disjoint wall partition; CPU overlapping GPU is not additive; no proportional allocation.
4. Contradiction register: sitting hash may differ from OPT-137/138 historical captures; historical rates are not paired gains.
5. Claim types: throughput gaps `measured`; shipping delta `derived` from no-keep; historical OPT-137 tok/s `historical`.
6. Target/guard: OPT-135 `target_guard_v2` statistics used for paired log-ratio CIs; this task does not keep a candidate.
7. Independent verification: one paired D2048 decode_only mean and one D2048 middle wall reconstructed from raw intervals.
8. Reporting: candidate measured delta, shipping delta, quality, and completeness are separate. Diagnostics shipping tok/s change is 0, not N/A, because the batch produced no keep versus the OPT-137 starting stack.

## Non-claims

- Not parity.
- Not release readiness.
- Not a production keep.
- OPT-016, OPT-056, and CMP-003 are unchanged.


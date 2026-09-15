# OPT-142 — Reconcile remaining whole-wall profiling residual

Diagnostics only. `claims_throughput=false`. Shipping delta: N/A. NLL: N/A.
No production scheduling, selector, or kernel changes.

Measured at `2026-09-15T00:49:54Z`.
Parent `post124_plus_opt127_decode_segments8_plus_opt137_mma`. Selector `decode_segments8`.

## Identity

pins ok=`True`
sitting binary matches OPT-138 capture=`False`
reuse OPT-138 traces=`True`

Current sitting Quartz binary hash may differ from the OPT-138 capture
hash. This task reuses the authenticated OPT-138 traces because the
question is the residual inside those captures, not a new sitting.

Structured JSON: [`coverage.json`](coverage.json),
[`per-window.json`](per-window.json), [`reproduce.json`](reproduce.json),
[`answers.json`](answers.json),
[`fixtures/opt142_wall_reconciliation.json`](../../../fixtures/opt142_wall_reconciliation.json).

## Accounting fault diagnosed

OPT-138 reported approximately **23.6%** decode and **5.9%** prefill
family-sum residual shares. Reproducing those windows from retained sqlite
**before** changing the acceptance equation confirmed the same numbers.
The fault is bookkeeping, not missing device work:

1. **Family-sum overlap** — llama decode family unions double-count
   concurrent kernels (~21.5 ms on D2048 middle). GPU wall uses a disjoint
   union; overlapping family totals are labeled bookkeeping only.
2. **Window endpoints** — `opt136.window` / `opt138.prefill` NVTX envelopes
   sit outside `cudaProfilerApi`. Acceptance uses in-capture eval/prefill
   NVTX (`eval_or_prefill_nvtx`); Quartz P4096 uses `qw38.prefill_chunk`.
3. **Host APIs unmeasured** — runtime `nameId` was not resolved against
   `StringIds`, so host-exclusive time was recorded as zero.

The corrected partition assigns GPU union (graph parents not added on top of
children), host APIs exclusive of GPU overlap, instrumentation exclusive of
GPU+host, and labels the remainder `unresolved_process_gap` (not proven
device-wide idle). No proportional allocation. No widened 5% tolerance.

## Raw artifacts

| Artifact | Path |
|---|---|
| OPT-138 trace root (42 sqlite) | [`build/optimization-runs/opt138/traces/`](../../../build/optimization-runs/opt138/traces/) |
| D2048 middle reproduce (Quartz) | `quartz-d2048-node-r0.2.sqlite` |
| D2048 middle reproduce (llama) | `llama-d2048-node-r0.2.sqlite` |
| P4096 prefill reproduce (Quartz) | `quartz-p4096-node-r0.sqlite` |
| P4096 prefill reproduce (llama) | `llama-p4096-node-r0.sqlite` |
| Per-window reconciliation | [`per-window.json`](per-window.json) (21 Quartz windows) |
| Coverage validation | [`coverage.json`](coverage.json) |

Recapture not required: retained traces answer the residual question inside
the OPT-138 capture identity.

## OPT-138 residual reproduced (family-sum method, host unmeasured)

### decode prefix=2048 window=middle r0

- family-sum residual share of Quartz wall: `0.2361` (ranking_complete=`False`)
- disjoint wall-gap residual share: `0.0000` (ranking_complete=`True`)
- Quartz unresolved share: `0.0013`
- Llama family-overlap bookkeeping ms: `21.4769`
- Quartz window source: `eval_or_prefill_nvtx`

### prefill prefix=4096 window=prefill r0

- family-sum residual share of Quartz wall: `0.0587` (ranking_complete=`False`)
- disjoint wall-gap residual share: `0.0000` (ranking_complete=`True`)
- Quartz unresolved share: `0.0001`
- Llama family-overlap bookkeeping ms: `0.7310`
- Quartz window source: `eval_or_prefill_nvtx`

## Per-window Quartz wall partition

| Window | Wall ms | GPU union ms | Host exclusive ms | Unresolved ms | Unresolved share | Source | Within 5% |
|---|---:|---:|---:|---:|---:|---|---|
| decode-d128-early-r0 | 198.5538 | 191.6627 | 6.4397 | 0.4515 | 0.0023 | eval_or_prefill_nvtx | True |
| decode-d128-early-r1 | 198.6102 | 191.6967 | 6.3974 | 0.4819 | 0.0024 | eval_or_prefill_nvtx | True |
| decode-d128-early-r2 | 198.7821 | 191.6521 | 6.6016 | 0.5284 | 0.0027 | eval_or_prefill_nvtx | True |
| decode-d128-middle-r0 | 201.6441 | 193.9519 | 7.3120 | 0.3801 | 0.0019 | eval_or_prefill_nvtx | True |
| decode-d128-middle-r1 | 201.6135 | 193.9604 | 7.3081 | 0.3451 | 0.0017 | eval_or_prefill_nvtx | True |
| decode-d128-middle-r2 | 201.5898 | 193.9766 | 7.2852 | 0.3280 | 0.0016 | eval_or_prefill_nvtx | True |
| decode-d128-late-r0 | 203.7712 | 196.1977 | 7.2493 | 0.3242 | 0.0016 | eval_or_prefill_nvtx | True |
| decode-d128-late-r1 | 204.0795 | 196.2520 | 7.4674 | 0.3600 | 0.0018 | eval_or_prefill_nvtx | True |
| decode-d128-late-r2 | 203.9296 | 196.1821 | 7.3798 | 0.3677 | 0.0018 | eval_or_prefill_nvtx | True |
| decode-d2048-early-r0 | 228.5903 | 220.7904 | 7.3776 | 0.3988 | 0.0017 | eval_or_prefill_nvtx | True |
| decode-d2048-early-r1 | 228.7508 | 220.8562 | 7.4063 | 0.4883 | 0.0021 | eval_or_prefill_nvtx | True |
| decode-d2048-early-r2 | 228.7573 | 220.8195 | 7.4338 | 0.5040 | 0.0022 | eval_or_prefill_nvtx | True |
| decode-d2048-middle-r0 | 246.2556 | 238.5766 | 7.3525 | 0.3264 | 0.0013 | eval_or_prefill_nvtx | True |
| decode-d2048-middle-r1 | 246.3413 | 238.7558 | 7.2382 | 0.3472 | 0.0014 | eval_or_prefill_nvtx | True |
| decode-d2048-middle-r2 | 246.3879 | 238.6169 | 7.4116 | 0.3594 | 0.0015 | eval_or_prefill_nvtx | True |
| decode-d2048-late-r0 | 247.2399 | 239.2973 | 7.5265 | 0.4161 | 0.0017 | eval_or_prefill_nvtx | True |
| decode-d2048-late-r1 | 246.9915 | 239.3702 | 7.2750 | 0.3464 | 0.0014 | eval_or_prefill_nvtx | True |
| decode-d2048-late-r2 | 246.9234 | 239.3195 | 7.2567 | 0.3472 | 0.0014 | eval_or_prefill_nvtx | True |
| prefill-d4096-prefill-r0 | 1347.8423 | 1344.6364 | 3.1291 | 0.0768 | 0.0001 | eval_or_prefill_nvtx | True |
| prefill-d4096-prefill-r1 | 1346.7835 | 1343.5577 | 3.1408 | 0.0850 | 0.0001 | eval_or_prefill_nvtx | True |
| prefill-d4096-prefill-r2 | 1345.1447 | 1341.9303 | 3.1279 | 0.0866 | 0.0001 | eval_or_prefill_nvtx | True |

Host exclusive is CUDA runtime/driver/sync APIs that do not overlap
GPU union. Overlapping `cudaEventSynchronize` is wait duration, not
added GPU wall. Remaining process-scoped gaps are **unresolved**,
not proven device-wide idle. Family-sum overlap is bookkeeping.

Llama P4096 node traces have no `NVTX_EVENTS` table, so those llama
windows use GPU-span fallback. Host time before the first and after
the last llama GPU interval is unobservable there. Quartz P4096 uses
`qw38.prefill_chunk`. The acceptance gate is Quartz unresolved share.

## Independent raw-interval reconstruction

Decode D2048 middle r0 and prefill P4096 r0 were independently
reconstructed from raw sqlite intervals (see host test
`test_independent_raw_interval_reconstruction`):

- Conservation holds on both windows (`conservation_ok=true`).
- D2048 middle Quartz unresolved share `0.0013`; disjoint Q-L wall-gap
  residual share `0.0000`.
- P4096 prefill Quartz unresolved share `0.0001`; disjoint Q-L wall-gap
  residual share `0.0000`.
- Legacy family-sum residual shares remain `0.2361` / `0.0587` as
  bookkeeping-only artifacts.

## Performance evidence checklist (diagnostics)

1. **Measurement identity** — reuse authenticated OPT-138 traces; selector
   `decode_segments8`; parent `post124_plus_opt127_decode_segments8_plus_opt137_mma`;
   sitting Quartz hash differs from capture hash (documented).
2. **Coverage** — 21 required Quartz windows (D128/D2048 early/middle/late ×3,
   P4096 prefill ×3); `coverage.json` status `valid`, `window_count=21`.
3. **Time accounting** — GPU union disjoint; CPU overlapping GPU not additive;
   graph-parent exclusion; family overlap labeled bookkeeping; no proportional
   allocation; process gaps not proven device-wide idle.
4. **Contradictions** — family-sum residual vs disjoint partition explained;
   llama P4096 uses GPU-span fallback (Quartz gate only).
5. **Claim types** — partition shares `measured`; family rankings separately
   qualified; no optimization inferred from unexplained time.
6. **Target/guard** — instrumentation-only; shipping delta N/A; NLL N/A.
7. **Independent verification** — verifier pass 2026-09-15T00:51:00Z (see dossier).
8. **Reporting** — candidate measured delta N/A; shipping delta N/A; quality N/A.

## Status

status=`measured`
blocked=`False`
missing observation=`None`
unresolved limit=`0.05` of Quartz wall
max Quartz unresolved share=`0.0027` (21/21 windows within limit)
disjoint Q-L wall-gap residual=`0`

Family rankings from OPT-138 remain separately qualified. This task
does not redistribute unexplained time across families and does not
widen the 5% tolerance. No production scheduling, selector, or kernel
changes.


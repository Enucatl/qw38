# OPT-125 — Decode timing, GPU activity, and traffic evidence

Status: `analysis`. measurement_utc=`2026-09-14T10:44:19Z`.
Diagnostics only. claims_throughput=`false`. claims_performance_improvement=`false`.
production_kept=`true`.

## Identity

device=`NVIDIA GeForce RTX 5090` source=`81be729883b9fb946da46fa2774c45e67bdced33`
state=`dirty` parent=`combined_opt118_opt119`
graphs=`ffn_only` llama pin=`cc83d7b4824f73cfdda4dfbb47ee39804f71b328`.
post113 ok=`True`.
pinned llama ok=`True`.
hardware_executed=`True`.
gpu_blocker=`None`.

## Frozen metric names

Correction date `2026-09-13`. First sampled token needs
eval: `False`.
Prefill (or the previous decode eval) already produced logits for the last prompt token. Sampling that token does not require a new eval. The first decode eval consumes the first emitted token to produce logits for the next sample. Planned-token engine probes skip the sampler and count eval_calls = emitted decode tokens.

| Name | Clock / boundary |
|---|---|
| setup | session/workspace/graph create; excluded from request and decode |
| prefill | request start through `sync_tokens` |
| time_to_first_emitted_token | prefill plus first decode eval on planned-token probes |
| decode_only | after prefill through last decode eval; denominator = emitted tokens |
| sample_eval_output | host sample + eval + output inside decode; count evals explicitly |
| complete_request | before prefill through last decode eval; OPT-123 `run_session_arm` tok/s |
| fixed_token_engine_probe | planned tokens, no sampler |

OPT-123 `run_session_arm` starts the clock before prefill and divides outputs
by that entire wall. OPT-115's request probe subtracts prefill. Pinned llama
decode oracle is decode-only. Historical fields are retained; corrections are
dated `2026-09-13`.

## Historical corrections (stored fields)

| Workload | historical combined tok/s | metric then | decode-only tok/s | request tok/s | llama tok/s | llama metric | published ratio | matched decode-only ratio |
|---|---:|---|---:|---:|---:|---|---:|---:|
| D128 | 54.0159 | complete_request | 54.9718 | 54.0159 | 68.6696 | decode_only | 0.7866 | 0.8005 |
| D2048 | 41.6331 | complete_request | 46.5290 | 41.6331 | 67.1855 | decode_only | 0.6197 | 0.6925 |

Long-context OPT-123 request tok/s versus decode-only from wall−ttft:

| Probe | historical request tok/s | decode-only tok/s | prefill ms | decode ms |
|---|---:|---:|---:|---:|
| D8192 | 8.0657 | 33.4012 | 3009.6069 | 958.0502 |
| D32768 | 1.3519 | 15.0229 | 21628.5762 | 2130.0774 |
| D131040 | n/a (OPT-123 OOM) | 4.7025 | 242361.1410 | 6804.9351 |

Fresh exclusive sitting (2026-09-14, post124 `ffn_only`):

| Probe | request tok/s | decode-only tok/s | TTFT ms | decode p50 ms |
|---|---:|---:|---:|---:|
| D8192 | 8.0652 | 33.4012 | 3039.5828 | 29.9374 |
| D32768 | 1.3469 | 15.0229 | 21695.2188 | 66.5586 |
| D131040 | 0.1284 | 4.7025 | 242573.7810 | 212.6837 |

41.420→8.066 (D8192) and 22.118→1.352 (D32768) mix OPT-115 decode-only with
OPT-123 complete-request. They are not evidence of a decode regression until
both sides use the same boundary.

## Fresh matched baseline

hardware=`True` reason=`None`.
A/A `{'p4096': {'decode_only': {'n': 5, 'df': 4, 'geometric_ratio': 1.0001252633794766, 'ci95_low': 0.9990043088540476, 'ci95_high': 1.0012474756963252, 'se': 0.0004957748498430327, 'critical': 2.262, 'ratios': [0.9987464517005102, 1.0017056920083245, 0.9999325549612427, 1.0006131376901581, 0.9996309389889908], 'equality_excluded': False, 'incomplete': False, 'guard': 0.02, 'point_outside_guard': False, 'status': 'repeatable', 'a_mean_tok_s': 2995.248826, 'b_mean_tok_s': 2995.622362, 'a_p50_tok_s': 2993.54858, 'a_p95_tok_s': 3007.054202, 'b_p50_tok_s': 2993.34668, 'b_p95_tok_s': 3005.0583479999996}, 'request': {'n': 5, 'df': 4, 'geometric_ratio': 1.0001252633794766, 'ci95_low': 0.9990043088540476, 'ci95_high': 1.0012474756963252, 'se': 0.0004957748498430327, 'critical': 2.262, 'ratios': [0.9987464517005102, 1.0017056920083245, 0.9999325549612427, 1.0006131376901581, 0.9996309389889908], 'equality_excluded': False, 'incomplete': False, 'guard': 0.02, 'point_outside_guard': False, 'status': 'repeatable', 'a_mean_tok_s': 2995.248826, 'b_mean_tok_s': 2995.622362, 'a_p50_tok_s': 2993.54858, 'a_p95_tok_s': 3007.054202, 'b_p50_tok_s': 2993.34668, 'b_p95_tok_s': 3005.0583479999996}}, 'd128': {'decode_only': {'n': 5, 'df': 4, 'geometric_ratio': 0.9999604288825266, 'ci95_low': 0.9996124624373044, 'ci95_high': 1.0003085164553371, 'se': 0.00015386417964301618, 'critical': 2.262, 'ratios': [0.9996727537522762, 1.0005303636404148, 0.99976476153131, 1.000026795057591, 0.999807707188602], 'equality_excluded': False, 'incomplete': False, 'guard': 0.02, 'point_outside_guard': False, 'status': 'repeatable', 'a_mean_tok_s': 55.1259323, 'b_mean_tok_s': 55.12375028, 'a_p50_tok_s': 55.1240654, 'a_p95_tok_s': 55.174413279999996, 'b_p50_tok_s': 55.12257, 'b_p95_tok_s': 55.16351848}, 'request': {'n': 5, 'df': 4, 'geometric_ratio': 0.9999657002030068, 'ci95_low': 0.999616513413697, 'ci95_high': 1.0003150089705075, 'se': 0.00015440307287509824, 'critical': 2.262, 'ratios': [0.9996740782365553, 1.0005390979209496, 0.9997814454875178, 1.0000275509207455, 0.9998065668697674], 'equality_excluded': False, 'incomplete': False, 'guard': 0.02, 'point_outside_guard': False, 'status': 'repeatable', 'a_mean_tok_s': 54.1690071, 'b_mean_tok_s': 54.1671486, 'a_p50_tok_s': 54.1673393, 'a_p95_tok_s': 54.21652068, 'b_p50_tok_s': 54.165947, 'b_p95_tok_s': 54.20613248}}, 'd2048': {'decode_only': {'n': 5, 'df': 4, 'geometric_ratio': 1.000022129457035, 'ci95_low': 0.9997062265309231, 'ci95_high': 1.0003381322071312, 'se': 0.0001396754381687485, 'critical': 2.262, 'ratios': [0.9996421894361814, 1.0004352933919805, 1.0002327866749432, 0.9998768766889298, 0.9999236961932287], 'equality_excluded': False, 'incomplete': False, 'guard': 0.02, 'point_outside_guard': False, 'status': 'repeatable', 'a_mean_tok_s': 46.60169984, 'b_mean_tok_s': 46.60272982, 'a_p50_tok_s': 46.5956993, 'a_p95_tok_s': 46.62537384, 'b_p50_tok_s': 46.5994225, 'b_p95_tok_s': 46.61589205999999}, 'request': {'n': 5, 'df': 4, 'geometric_ratio': 1.0000445384812533, 'ci95_low': 0.9996763250443811, 'ci95_high': 1.0004128875431593, 'se': 0.0001628049686596568, 'critical': 2.262, 'ratios': [0.9996586934653082, 1.0005462659200686, 1.000258740653582, 1.0000007312930999, 0.9997585261497719], 'equality_excluded': False, 'incomplete': False, 'guard': 0.02, 'point_outside_guard': False, 'status': 'repeatable', 'a_mean_tok_s': 41.71910476, 'b_mean_tok_s': 41.72096252, 'a_p50_tok_s': 41.7089462, 'a_p95_tok_s': 41.74860002, 'b_p50_tok_s': 41.719738, 'b_p95_tok_s': 41.74175646}}}`.

Combined / post113 / llama raw records are retained under
`build/optimization-runs/opt125/` when GPU phases run. Fixed-token engine
probes are labeled separately from free-running sample/eval/output loops.

## Activity trace

method=`cuda_event_engine_attribution`.
nsys=`True` ncu=`True`
cupti_linked=`False`.
Nsight Systems is installed in the pinned CUDA image; this phase still used CUDA-event attribution only (no nsys capture run).
Windows: early / middle / late on the same 256-output trajectory.

## Disjoint intervals

Leaf gaps from `classify_timeline` are **unobserved**, not GPU idle.
CPU waits that overlap device-active intervals are not counted twice.

| Workload | device_active_ms | proven_inactive_ms | unobserved_ms | leaf_gap_ms (opt115 label) |
|---|---:|---:|---:|---:|
| D128 | 102.2116 | 0.0000 | 126.9594 | 4564.8082 |
| D2048 | 140.9033 | 0.0000 | 126.6968 | 5365.0995 |

## Weight reads and busy-time contradiction

D128 busy vs compulsory: bound_supported=
`False` busy_below_compulsory=
`True`.
D2048 timeline vs request mismatch=
`True`.
Observed idle is not an unavoidable lower bound. DRAM/L2 was not
counter-sampled; 1792 GB/s remains a listed peak.

Embedding consumers read one row, not the full table. Combined-stack decode
D2H is the 4-byte lazy index, not full logits.

## Long-context / OPT-016 2K

Fresh probes: hardware=`True`
reason=`None`. Fresh exclusive long-cache probes measured at capacity 131072 after stopping zanzara-archive GPU services. Historical OPT-123 request tok/s values are retained; fresh decode-only and request rates confirm the boundary correction (request includes prefill).
OPT-016 2K Quartz blocked: `decode_segments8_requires_matching_session_capacity` (not OOM on exclusive sitting).
Historical OPT-123 D131040 OOM on a non-exclusive sitting is retained in
fixtures; the exclusive sitting measured D131040+32 successfully.

## Ranking for OPT-126–131

| Task | disposition | causal ms | confidence | reason |
|---|---|---:|---|---|
| OPT-126 | proceed | n/a | source_grounded | OPT-117 recapture on frontier/GDN pointer changes is a source-grounded graph-input instability. Missing profiling does not block this correctness prerequisite. |
| OPT-127 | proceed | n/a | conditional_on_opt126 | Recapture-free replay is the mechanism that can convert OPT-126 into request time. Quantify after stable inputs; do not treat ~10 ms leaf gaps as the recapture savings. |
| OPT-128 | insufficient_evidence | n/a | low_until_api_correlated_inactive | Host stalls overlapping GPU execution must not be counted twice. Leaf-gap ~10 ms/token is unobserved, not proven CPU-submission starvation. Proceed only after OPT-125 activity traces attribute true inactive spans. |
| OPT-129 | proceed | 6.6078 | source_grounded_component_unknown | Matched decode-only D2048 still trails pinned llama after boundary correction, but 0.620× was a mixed-boundary whole-engine ratio and is not an attention component ratio. Component replay is required; shipping hybrid_crossover@1024 stays. |
| OPT-130 | insufficient_evidence | n/a | blocked_on_opt129 | No matched attention-component candidate is frozen. OPT-129 must name at most two causal transfers before implementation. |
| OPT-131 | insufficient_evidence | n/a | blocked_on_graph_host_attention | Residual launch-chain fusion is only justified after graph, host, and attention verdicts. OPT-119 already kept norm→Q8. |

Parent for later tasks: `combined_opt118_opt119`. Keep protocol: 3+5 screen,
3+10 acceptance, both decode-only and complete-request metrics, 2% request
materiality reported separately.

## OPT-133 Nsight reconciliation (2026-09-14T12:50:00Z)

Follow-on [`OPT-133`](../../tasks/OPT-133.md) bounded Nsight capture on the
admitted `decode_segments8` stack (nsys **2025.3.2**, measurement
`2026-09-14T12:01:33Z`) reconciles this report's CUDA-event **unobserved**
intervals against hardware `gputrace`:

| Window set | OPT-125 unobserved (12-token scale) | Nsight gpu_idle_ms | Unresolved |
| --- | ---: | ---: | ---: |
| mean over D128/D2048 × early/middle/late | **126.4 ms** | **188–237 ms** | **0 ms** |

Fraction of OPT-125 unobserved explained by Nsight hardware idle: **1.0**. Dominant
CUDA API time in the bounded window is **`cudaEventSynchronize`** (~193–244 ms/window)
from this task's event-attribution probe — not a separate removable budget. Actual
GPU kernel union ~**8 ms**/window. See
[`evidence/optimization/opt133-decode-nsys-trace/REPORT.md`](../opt133-decode-nsys-trace/REPORT.md).

This section amends interpretation only; OPT-125 sidecars and CUDA-event tables above
are unchanged.

## Answers

1. **Is 10 ms/token true inactive?** `False` as **removable** headroom.
   At CUDA-event attribution (this task): ~10 ms/token is the OPT-115/124 leaf-gap
   remainder after event-union busy time; proven inactive ms/token=**0.0**; relabeled
   **unobserved**. **OPT-133 amendment:** Nsight hardware idle fully accounts for that
   unobserved remainder (fraction **1.0**) — the GPU was inactive during event-leaf gaps,
   not running missing kernels. That idle is dominated by `cudaEventSynchronize` waits
   from event instrumentation, not proven removable decode time.
2. **How much is causally removable?** `n/a` ms/token proven.
   Only API-correlated proven device-inactive spans are removable host/submission time.
   OPT-133 does not establish removable ms/token from the leaf-gap remainder. Source-grounded
   graph (OPT-126/127) and attention (OPT-129) work may still proceed without that number.
3. **Do llama and long-context ratios survive matched boundaries?**
   llama `False`
   (D128 published `0.7866` vs matched
   `0.8005`; D2048 published
   `0.6197` vs matched
   `0.6925`).
   Fresh sitting decode-only combined/llama: D128
   `0.8012`
   (`55.1248` /
   `68.8069`), D2048
   `0.6918`
   (`46.6022` /
   `67.3645`).
   long-context `False`
   (D8192 request `8.0657` vs
   decode-only `33.4566`).
   Fresh exclusive long-cache probes measured at 131072 capacity (see table above).
   OPT-016 2K Quartz remains blocked by decode_segments8 session-capacity requirements (not OOM).

Unknowns are valid findings. Throughput improvement is not a requirement.

## OPT-136 dated correction (2026-09-14)

Derived GPU idle from OPT-133 `cuda_gpu_trace` is **invalid**. The capture used
`--cuda-graph-trace=graph`, but the GPU sum omitted
`CUPTI_ACTIVITY_KIND_GRAPH_TRACE`. Subtracting ordinary kernel/copy union from
the window does not prove hardware idle; graph-internal activity is unknown
until node tracing covers those envelopes. Cross-run
`min(old_unobserved, new_idle)` is **not causal**.

Raw OPT-133 sample arrays and historical admission are preserved. Current
derived idle fields are null. Evidence:
[`evidence/optimization/opt136-graph-accounting/historical-reconciliation.json`](../../evidence/optimization/opt136-graph-accounting/historical-reconciliation.json).

Do not treat graph envelopes as continuous busy time, and do not replace the
old ~188 ms idle claim with a claim of ~194 ms continuous hardware busy time.

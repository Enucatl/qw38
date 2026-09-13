# OPT-127 — Replay decode segments without per-token recapture

Status: **keep**. Production selector `decode_segments8`. Hardware executed.

Parent is authenticated post124 combined OPT-118+119 `ffn_only`. Candidate is recapture-free `decode_segments8` replay on OPT-126 stable inputs (`DecodeLaunchState`, 16 segment graphs, `stable_pingpong_committed_slot`). Keep requires OPT-116 quality, state/memory/cancellation/checkpoint/128K, and a measured request gain on D128/D2048 with P4096 as a non-target guard. Capture, instantiate, and upload are instrumented separately from launch-state uploads and graph launches. No weight-byte reduction is claimed.

## Recapture elimination

recapture_proof ok=`True`; invalidation policy `recapture_on_pingpong_or_kv_identity_or_missing_topologies;no_recapture_on_frontier_slot_token;crossover_1024_selects_prebound_topology;past_verified_max_4096_uses_topology_0;session_or_kv_replacement_invalidates`. Steady-state tokens inside one topology, including alternating GDN commits, must show zero capture/instantiate/destroy/recapture deltas. Crossover 1024 selects the prebound topology-1 graphs; the transition cost is inside the crossing request, not a recapture.

Coarser candidate admitted=`False`. Existing eight-layer segmentation was screened first. Remaining graph submission gaps were not proven material under OPT-125 (proven device inactive 0 ms), and cancellation still polls at eight-layer boundaries, so no coarser persistent kernel was admitted.

## Same-math, quality, state

same_math=`True` exact=`True`; quality=`True` candidate_nll_measured=`True` opt058_invoked=`True` held-out ppl ratio=`1.0`; state/memory=`True`; cancellation=`True`.

## Full-engine A/B (3 warmups + 10 AB/BA; decode-only and request)

| Workload | metric | parent tok/s | candidate tok/s | geo ratio | CI lower | p95 ratio | gate |
|---|---|---:|---:|---:|---:|---:|---|
| D128 | request | 54.208061220000005 | 58.05710373000001 | 1.0710033971824118 | 1.0695656067760302 | 0.9321973087645687 | True |
| D128 | decode_only | 55.21354715 | 59.25954018 | 1.0732775184836347 | 1.0719055661413812 | 0.9321973087645687 | reported |
| D2048 | request | 41.655003750000006 | 44.27746507 | 1.0629573143930846 | 1.0619425982860256 | 0.9333057190228863 | True |
| D2048 | decode_only | 46.68674316 | 50.00288773 | 1.071030008151505 | 1.070235939545154 | 0.9333057190228863 | reported |
| P4096 | request | 2899.645532 | 2895.647804 | 0.9986212207014872 | 0.9981643782607919 | 0.0 | True |
| P4096 | decode_only | 2899.645532 | 2895.647804 | 0.9986212207014872 | 0.9981643782607919 | 0.0 | reported |

## Historical engine probes (OPT-117 recapturing path)

OPT-117 verdict `retain_ffn_only` with per-token recapture. Those numbers are not a recapture-free replay measurement and are not reused as this sitting's parent.

D128 request tok/s delta vs combined parent `3.849042510000004`. Reasons: `[]`.

Launch preparation stays inside the measured request (`setup_ms` + prefill + decode). Short-output and 256-output amortization are in the handoff and A/B sidecars. User-visible output remains per token; cancellation still polls after each eight-layer segment and commits atomically only on success.

Keep=True. Shipping `decode_segments8`. Weight-byte reduction claimed=false.

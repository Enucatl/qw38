# OPT-080 — Combined batch validation and llama outcome gates

## Claim labels and proof limits

combined freeze of OPT-070-079 keep/reject/no-go selectors; OPT-069 and OPT-056 retained as history; same-sitting P/D128/D2048 versus pinned llama.cpp; parity gap is Tq-Tl; +5% throughput gap is Tq-Tl/1.05; do not label the parity gap as the +5% bar; decode p95 no worse than llama for the OPT-056 outcome; quality-v2 remains the original absolute gate; quality-v3 does not replace OPT-056 or OPT-016; full quality v2 on the frozen combination; original OPT-016 2K parity evidence; OPT-056 and OPT-016 stay blocked unless their gates pass; rejected candidates must not leak into production; instrumented timings are not release throughput; preflight is not release evidence; failed required quality stops release before long timing; diagnostic performance is not a release or keep.

Preflight quality-v2 remains blocked. This is not release evidence.
`gate.passed` is False. OPT-056 remains blocked. OPT-016 stays blocked.
Quality-v3 does not replace quality-v2, OPT-056, or OPT-016.

## Frozen combination

Keeps:

- OPT-064 r2_w2 (OPT-070 inconclusive retain)
- OPT-066 fma_async_x (OPT-070 inconclusive retain)
- OPT-079 kv_once

Rejected or retained:

- OPT-070 Q8/MMQ inconclusive; shipping unchanged
- OPT-071 instrumentation repair; no kernel keep
- OPT-072 rejection review; no kernel keep
- OPT-073 quality-v3 policy; quality-v2 still fail
- OPT-074 GPU admission; missing coverage is not a keep
- OPT-075 packed Q4 retained; integer not installed
- OPT-076 packed retained; late-reduction not installed
- OPT-077 sequential GDN retained
- OPT-078 warp_query retained; prepared_q rejected

## Three outcomes

| Outcome | Verdict |
|---|---|
| Internal improvement with quality | unpassed |
| Llama parity (Quartz >= llama, Tq-Tl) | unpassed |
| OPT-056 +5% (Tq-Tl/1.05, p95, quality, OPT-016) | unpassed |

## Remaining latency budget (OPT-069 sitting, unchanged)

| Workload | Quartz tok/s | llama tok/s | parity ms (Tq-Tl) | +5% ms (Tq-Tl/1.05) |
|---|---:|---:|---:|---:|
| P 4096 | 2914.65698 | 3142.517034 | 101.89738823402377 | 163.9647110002427 |
| D128 | 37.1789093 | 68.8708796 | 3168.5234884718725 | 3345.528287325584 |
| D2048 | 35.4498482 | 67.3394867 | 3419.836789346795 | 3600.866921708588 |

tok/s delta vs OPT-069 baseline: 0 (speedup 1.00×). Diagnostic performance
plan: `evidence/optimization/opt080-batch-gate/diagnostic-performance-plan.json`.
No automatic next optimization sweep.

# OPT-069 — Combined batch validation and llama outcome gates

## Reporting correction (OPT-070)

This file was rewritten from retained OPT-069 measured sidecars. Historical numerical samples were not altered. The previous text labeled the three outcomes unmeasured after the sitting had already produced quartz-p.json, decode oracles, 2K parity, and quality-v2 artifacts.

## Claim labels and proof limits

combined freeze of OPT-062-068 keep/reject selectors; same-sitting P/D128/D2048 versus pinned llama.cpp; parity gap is Tq-Tl; +5% throughput gap is Tq-Tl/1.05; do not label the parity gap as the +5% bar; decode p95 no worse than llama for the OPT-056 outcome; full quality v2 on the frozen combination; original OPT-016 2K parity evidence; OPT-056 and OPT-016 stay blocked unless their gates pass; rejected candidates must not leak into production; instrumented timings are not release throughput; preflight is not release evidence.

This sitting reports three distinct outcomes. Failed historical gates stay
failed. `gate.passed` is False. OPT-056 remains
blocked because the original +5% / p95 / quality conditions did not pass.

## Frozen combination

Keeps:

- OPT-064 r2_w2
- OPT-066 fma_async_x

Rejected or retained:

- OPT-062 integer Q4 not installed
- OPT-063 paired integer not installed
- OPT-065 i128_j128 retained
- OPT-067 prompt pair off
- OPT-068 O2 --fmad=false retained

Q4 path is packed on gate/up/down. Q8 production layout is r2_w2. MMQ is
`fma_async` with `async_x`. Prompt pair is off. Tiles stay i128_j128.
NVCCFLAGS stay `-O2 --fmad=false`. Graphs are `ffn_only`. Batch size is
4096. Workspace extra X bytes:
18448; shared 94736;
X ring stages 1.

## Three outcomes

| Outcome | Verdict |
|---|---|
| Internal improvement with quality | unpassed |
| Llama parity (Quartz >= llama, Tq-Tl) | unpassed |
| OPT-056 +5% (Tq-Tl/1.05, p95, quality, OPT-016) | unpassed |

## Measured sitting

- Device: NVIDIA GeForce RTX 5090 compute 12.0
- power_limit_w: 400.0
- measurement_utc: 2026-09-11T12:18:36Z
- source_revision: retained (reporting_correction)

| Workload | Quartz tok/s | llama tok/s | OPT-056 baseline | parity ms (Tq-Tl) | +5% ms (Tq-Tl/1.05) |
|---|---:|---:|---:|---:|---:|
| P 4096 | 2914.65698 | 3142.517034 | 2808.49609 | 101.89738823402377 | 163.9647110002427 |
| D128 | 37.1789093 | 68.8708796 | 37.4816246 | 3168.5234884718725 | 3345.528287325584 |
| D2048 | 35.4498482 | 67.3394867 | 35.7208481 | 3419.836789346795 | 3600.866921708588 |

Decode p95 ms: D128 Quartz 27.0246696 vs llama 14.526; D2048 Quartz 28.3053131 vs llama 14.627.

Quality v2 all=False status=fail.
Legacy QLT-001 and OPT-056 functional verdicts remain historical and are not
retroactive passes of the corrected prompt suite.

OPT-016 2K: Quartz 3132.88379 vs llama 3132.053862; point comparison gate_passed=True. This increment does not own the OPT-016 ledger row.

State/memory: memory_fit=True; checkpoint=True; cancellation frontier 0.

Matched-token OPT-060 family attribution is a separate record and is not mixed
into these uninstrumented throughput numbers.

Budget: this is a long sitting, not a five-minute check. Preflight is not
release evidence.

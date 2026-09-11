# OPT-069 — Combined batch validation and llama outcome gates

## Claim labels and proof limits

combined freeze of OPT-062-068 keep/reject selectors; same-sitting P/D128/D2048 versus pinned llama.cpp; parity gap is Tq-Tl; +5% throughput gap is Tq-Tl/1.05; do not label the parity gap as the +5% bar; decode p95 no worse than llama for the OPT-056 outcome; full quality v2 on the frozen combination; original OPT-016 2K parity evidence; OPT-056 and OPT-016 stay blocked unless their gates pass; rejected candidates must not leak into production; instrumented timings are not release throughput; preflight is not release evidence.

This report is the frozen configuration record. Preflight is not release
evidence. Throughput and quality verdicts stay unmeasured until the release
sitting overwrites this file.

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

Q4 path is packed. Q8 production layout is r2_w2. MMQ is `fma_async` with
`async_x`. Prompt pair is off. Tiles stay i128_j128. NVCCFLAGS stay
`-O2 --fmad=false`. Graphs are `ffn_only`. Batch size is
4096. Workspace extra X bytes:
{'mmq_x_shared_bytes': 94736, 'mmq_x_extra_bytes': 18448, 'x_ring_stages': 1}.

## Three outcomes

| Outcome | Verdict |
|---|---|
| Internal improvement with quality | unmeasured |
| Llama parity (Quartz >= llama, Tq-Tl) | unmeasured |
| OPT-056 +5% (Tq-Tl/1.05, p95, quality, OPT-016) | unmeasured |

OPT-056 and OPT-016 stay blocked unless their actual respective gates pass.

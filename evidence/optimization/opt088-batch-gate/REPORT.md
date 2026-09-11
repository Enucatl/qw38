# OPT-088 — Combined post-reset production gate

## Claim labels and proof limits

combined freeze of OPT-085 packed Q4, OPT-086 r1_w4 Q8 and fma_async_x MMQ, OPT-079 kv_once, OPT-087 no_additional_reopen; independent kernel_parity_pass per family; quality is not one boolean; quartz_vs_baseline_quality_delta and quartz_vs_llama_quality_delta stay distinct; opt074_coverage_unadmitted is not a blocker; absolute task-accuracy fail inherited from OPT-084 does not by itself block; new regression versus OPT-084 does block; OPT-080 remains historical and is not reinterpreted; OPT-056 and OPT-016 stay blocked unless their owning conditions pass; parity gap is Tq-Tl; +5% throughput gap is Tq-Tl/1.05; do not label the parity gap as the +5% bar; decode p95 no worse than llama for the OPT-056 outcome; preflight is not release evidence; failed required quality stops release before long timing; diagnostic performance is not a release or keep; rejected candidates must not leak into production.

This sitting reports independent kernel-parity, quality-delta, recurrence,
performance, keep, and release-eligible fields. Failed historical gates stay
failed. `gate.passed` is False. OPT-056 remains
blocked unless the original +5% / p95 / quality conditions pass. OPT-080
remains historical and is not reinterpreted. Quality is not one boolean.

## Frozen combination

Keeps:

- OPT-085 packed Q4
- OPT-086 r1_w4 Q8 revert
- OPT-086 fma_async_x MMQ keep
- OPT-079 kv_once
- OPT-087 no_additional_reopen

Rejected or retained:

- OPT-085 integer_q8_paired and late_w4 not installed
- OPT-086 r2_w2 Q8 reverted
- OPT-086 fma_async MMQ not selected
- OPT-087 no additional reopen

Q4 path is packed. Q8 production layout is r1_w4. MMQ is `fma_async` with
`async_x`. Attention pipeline stays `kv_once`. OPT-087 is
`no_additional_reopen`. NVCCFLAGS stay `-O2 --fmad=false`. Graphs are
`ffn_only`. Batch size is 4096.

## Independent fields

| Field | Status |
|---|---|
| kernel_parity_pass (q4/q8/mmq/kv_once) | True/True/True/True |
| model_quality_pass | True |
| quartz_vs_baseline_quality_delta | pass delta_nll=0.0 ppl_ratio=1.0 |
| quartz_vs_llama_quality_delta | delta=-0.00021362203932273616 |
| recurrence_state_status | pass incremental_nll=0.0 |
| performance_pass | False |
| production_kept | True |
| release_eligible | True |

## Three historical outcomes (not relabeled)

| Outcome | Verdict |
|---|---|
| Internal improvement with quality | unpassed |
| Llama parity (Quartz >= llama, Tq-Tl) | unpassed |
| OPT-056 +5% (Tq-Tl/1.05, p95, quality, OPT-016) | unpassed |

## Measured sitting

- Device: NVIDIA GeForce RTX 5090 compute 12.0
- power_limit_w: 400.0
- measurement_utc: 2026-09-11T23:12:15Z
- source_revision: 27179f41593ae8252438d6a46f1edce60f3fa642 (dirty)

| Workload | Quartz tok/s | llama tok/s | OPT-069 baseline | parity ms (Tq-Tl) | +5% ms (Tq-Tl/1.05) |
|---|---:|---:|---:|---:|---:|
| P 4096 | 2956.4502 | 3170.927662 | 2914.65698 | 93.70973445984464 | 155.22095131447213 |
| D128 | 37.482605 | 69.132553 | 37.1789093 | 3126.8036094170743 | 3303.1384279862104 |
| D2048 | 35.732132 | 67.719321 | 35.4498482 | 3384.1103573938576 | 3564.1251007560395 |

Decode p95 ms: D128 Quartz 26.8055744 vs llama 14.461; D2048 Quartz 28.0852242 vs llama 14.552.

OPT-016 2K: Quartz 3181.19946 vs llama 3173.955939; point comparison gate_passed=True. This increment does not own the OPT-016 ledger row.

State/memory: memory_fit=True; checkpoint=True; cancellation frontier 0.

Budget: this is a long sitting, not a five-minute check. Preflight is not
release evidence.

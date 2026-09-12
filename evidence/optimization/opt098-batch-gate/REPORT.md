# OPT-098 — Combined post-088 outcome gate

## Claim labels and proof limits

combined freeze of post088_control from authenticated OPT-088 and post098_selected from OPT-089-097 keeps; independent kernel_parity_pass per family; quality is not one boolean; quartz_vs_baseline_quality_delta and quartz_vs_llama_quality_delta stay distinct; opt074_coverage_unadmitted is not a blocker; absolute task-accuracy fail inherited from OPT-084 does not by itself block; new regression versus OPT-084 does block; OPT-088 remains historical and is not reinterpreted; OPT-056 and OPT-016 stay blocked unless their owning conditions pass; parity gap is Tq-Tl; +5% throughput gap is Tq-Tl/1.05; do not label the parity gap as the +5% bar; decode p95 no worse than llama for the OPT-056 outcome; preflight is not release evidence; failed required quality stops release before long timing; diagnostic performance is not a release or keep; rejected candidates must not leak into production; internal improvement is selected versus fresh OPT-088 control not llama; strict_ppl_ratio_max=1.01 applies because concession is inactive; release_eligible is never a synonym for opt056_pass.

`gate.passed` is False. OPT-088 remains historical and is
not reinterpreted. Quality is not one boolean.

## Three independent outcomes

| Outcome | Verdict |
|---|---|
| Internal improvement vs OPT-088 control | unpassed |
| Llama parity (Quartz >= llama, Tq-Tl) | unpassed |
| OPT-056 +5% (Tq-Tl/1.05, p95, quality, OPT-016) | unpassed |

## Measured sitting

| Workload | Selected tok/s | llama tok/s | OPT-088 control | parity ms | +5% ms |
|---|---:|---:|---:|---:|---:|
| P 4096 | 3046.23218 | 3170.927662 | 2956.4502 | 52.87633312178468 | 114.38754997641217 |
| D128 | 37.2903061 | 69.132553 | 37.482605 | 3162.0237472078 | 3338.3585657769363 |
| D2048 | 35.4885025 | 67.719321 | 35.732132 | 3433.2942882907378 | 3613.3090316529197 |

Decode p95 ms: D128 Quartz 27.0138912 vs llama 14.461 vs OPT-088 control 26.8055744; D2048 Quartz 28.2724323 vs llama 14.552 vs OPT-088 control 28.0852242.

OPT-016 2K: Quartz 3165.44849 vs llama 3173.955939; gate_passed=False.

State/memory: memory_fit=True; checkpoint=True; cancellation frontier 0.

## Independent fields

| Field | Status |
|---|---|
| kernel_parity_pass | True |
| model_quality_pass | True |
| performance_pass | False |
| production_kept | True |
| release_eligible | True |

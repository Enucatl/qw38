# OPT-106 — Combined post-098 outcome gate

## Claim labels and proof limits

combined freeze of post098_control from authenticated OPT-098 and post106_selected equal to that combination; OPT-100 through OPT-105 rejected with no production selector change; independent kernel_parity_pass per family; quality is not one boolean; quartz_vs_baseline_quality_delta and quartz_vs_llama_quality_delta stay distinct; opt074_coverage_unadmitted is not a blocker; absolute task-accuracy fail inherited from OPT-084 does not by itself block; new regression versus OPT-084 does block; OPT-098 remains historical and is not reinterpreted; OPT-056 and OPT-016 stay blocked unless their owning conditions pass; parity gap is Tq-Tl; +5% throughput gap is Tq-Tl/1.05; do not label the parity gap as the +5% bar; decode p95 no worse than llama for the OPT-056 outcome; preflight is not release evidence; failed required quality stops release before long timing; diagnostic performance is not a release or keep; rejected candidates must not leak into production; internal improvement is selected versus authenticated OPT-098 control not llama; llama numbers are measured in this sitting and not reused from OPT-098; strict_ppl_ratio_max=1.01 applies because concession is inactive; release_eligible is never a synonym for opt056_pass; OPT-099 matched family attribution is diagnostic and non-additive; selected equals control so Quartz is measured once.

`gate.passed` is False. OPT-098 remains historical and is
not reinterpreted. Quality is not one boolean. post106_selected equals
post098_selected. Llama numbers were measured in this sitting.

## Three independent outcomes

| Outcome | Verdict |
|---|---|
| Internal improvement vs OPT-098 control | unpassed |
| Llama parity (Quartz >= llama, Tq-Tl) | unpassed |
| OPT-056 +5% (Tq-Tl/1.05, p95, quality, OPT-016) | unpassed |

## Measured sitting

| Workload | Selected tok/s | llama tok/s | OPT-098 control | parity ms | +5% ms |
|---|---:|---:|---:|---:|---:|
| P 4096 | 3036.84448 | 3252.583232 | 3046.23218 | 89.46169853594392 | 149.42868635309674 |
| D128 | 53.5021248 | 69.195195 | 37.2903061 | 1085.1778363812805 | 1261.3530200840823 |
| D2048 | 49.400589 | 67.3503264 | 35.4885025 | 1381.1035222364067 | 1562.10451869893 |

Decode p95 ms: D128 Quartz 18.8904171 vs llama 14.486 vs OPT-098 control 27.0138912; D2048 Quartz 20.3424911 vs llama 14.611 vs OPT-098 control 28.2724323.

OPT-016 2K: Quartz 3109.80151 vs llama 3141.838681; gate_passed=False.

State/memory: memory_fit=False; checkpoint=True; cancellation frontier 0.

Matched family attribution is diagnostic and non-additive (OPT-099).

## Independent fields

| Field | Status |
|---|---|
| kernel_parity_pass | True |
| model_quality_pass | True |
| performance_pass | False |
| production_kept | True |
| release_eligible | True |
| internal_improvement_with_quality | False |
| llama_parity | False |
| opt056_plus5 | False |

# OPT-135 — Separate target improvement from guard non-regression

Status: **target_guard_v2 frozen (unverified draft)**. Host policy only.
`claims_throughput: false`. No CUDA change, no production pin change, and no
shipping throughput delta. Historical OPT-130 admission remains `reject`.

## Policy scope

`target_guard_v2` is a prospective internal admission rule. Tasks opt in
explicitly (`keep_policy_id` / candidate `policy_id`). Unknown IDs fail closed.
New OPT-137 uses this evaluator; OPT-136/138 diagnostics do not perform keep
admission. Completed historical tasks keep their frozen policies. OPT-016,
OPT-056, CMP-003 and `plan.md` release gates are unchanged.

A candidate contract freezes disjoint nonempty `targets` and `guards`, dispatch
region, unchanged branches, quality/state gates and thresholds, then hashes that
freeze before the first screen sample. Acceptance must present the same hash.

Acceptance uses ten paired AB/BA rounds. For each workload,
`x_i = log(candidate_rate_i / control_rate_i)`, `g = exp(mean(x))`,
`se = sample_sd(x)/sqrt(10)`, and one-sided 95% bounds
`L,U = exp(mean(x) ± 1.8331129326536335·se)` (df=9). `[L,U]` is not a two-sided
95% CI. Zero variance sets `L=U=g`. Any acceptance count other than ten is
invalid for this version.

Every target must satisfy `L > 1.0`. Every throughput guard must satisfy
`L >= 0.98`; neutrality that still lies above that floor is a pass, including
when the interval contains 1. Decode targets use `decode_only` as primary and
`complete_request` as an additional 0.98 guard. Decode workloads require each
paired p95 ITL ratio `<= 1.05`. `material_2pct = (g >= 1.02)` is reported per
target and is not a hidden keep threshold.

Verdicts: `keep` only with complete passing evidence; `reject` for measured
quality/state failure, target `U<=1`, throughput-guard `U<0.98`, or a p95
violation; `inconclusive` when valid intervals establish neither; `incomplete`
for absent or invalid required evidence. Precedence is incomplete, reject,
inconclusive, keep.

## Historical OPT-130 replay

Raw OPT-130 pairs were replayed under counterfactual roles: D2048 decode-only
target, D2048 complete-request plus D128 decode/request and P4096 prefill as
guards. Available per-metric intervals support a D2048 target pass with D128
and P4096 guards above the 0.98 floor. That replay is labeled counterfactual
and **does not alter historical admission**. Preserved historical verdict:
`reject`. Shipping impact remains 0.

## Validator

```sh
uv run python tools/performance_keep_policy.py --contract pins/performance_keep_policy_v2.json --self-check fixtures/opt135_target_guard_policy.json
```

Self-check fixtures carry independently calculated expected decisions. The CLI
writes no historical fixture.

## Proof limit

- policy only; no shipping throughput change
- historical OPT-130 admission remains reject
- unknown policy IDs fail closed
- release gates OPT-016/OPT-056/CMP-003 unchanged

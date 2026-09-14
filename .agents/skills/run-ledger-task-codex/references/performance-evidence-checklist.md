# Performance evidence checklist

Use this checklist for any ledger increment that ranks a bottleneck, claims
idle or excess time, compares engines, or reports a keep/reject/throughput
delta. Prompts can enforce how evidence is requested and reviewed; they cannot
themselves prove profiler completeness. OPT-136 supplies the executable
interval checks. Until those checks exist, reconstruct coverage from raw
artifacts independently and label unavailable checks `unavailable`. Do not
manufacture an executable-check pass.

Read the eight adversarial cases in
[performance-evidence-review-cases.md](performance-evidence-review-cases.md)
before ranking or verifying a performance claim. Admission, planning,
documentation, verification, and completion each apply this checklist.

## 1. Measurement identity

For each rate record all of: engine/commit, actual loaded binary hash, build
flags/image/tool versions, GGUF, selectors, input token hash, prefix,
output/eval counts, first-token convention, allocated capacity, populated
length, sampling/output policy, graph mode, clocks/residents, warmups/samples,
and exact numerator/start/end events.

Distinguish prefill, decode-only, complete-request, component, and public
sample/eval/output metrics. Setup, graph creation, warmup, and checkpoint
restore stay outside rate denominators. A ratio requires a matching metric
identity on both arms. Refuse a Quartz complete-request versus llama
decode-only comparison.

## 2. Coverage before ranking

Name captured tables, units, clock domain, capture bounds, streams, expected
and observed graph/node counts, family mapping, and unclassified work before
ranking any sink or idle remainder.

Graph envelopes are not leaf kernel time. Graph-level data cannot prove
continuous activity or internal idle. Missing node tracing forbids
graph-internal idle and component-completeness claims. Do not subtract
ordinary kernel/copy union from a window that contains graph envelopes and
call the remainder proven idle; internal graph activity is unknown until node
tracing covers it.

A rebuilt diagnostic binary is necessary but not sufficient. Stale binaries
are not evidence. OPT-020-style CUDA-event categories cannot rank production
graph execution until coverage of that graph is established.
`hardware_gpu_trace=true` does not imply graphs were captured. Graph-level
capture that omits `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` from the GPU sum is
incomplete coverage.

Missing coverage must not become zero excess. Unmapped llama families make
excess `null`/`unknown`; never treat missing corresponding llama work as
Quartz minus zero.

Use OPT-136's checked `coverage.json` when available; do not substitute its
existence or a success flag for inspecting the assertions and source rows.
Validate required coverage with the shared CLI (nonzero exit is a fail;
`null`/`unavailable` is not a pass):

```sh
uv run python tools/performance_evidence.py --validate <coverage.json>
```

## 3. Time accounting

Union overlapping intervals. Do not sum graph parents and children, nested
NVTX and kernels, or overlapping CPU waits and GPU work. CPU
`cudaEventSynchronize` duration is wait duration, not removable GPU idle; do
not count an API synchronize span that is concurrent with GPU work as idle or
add it to GPU wall time.

A cross-run or cross-stack aggregate cannot identify overlap of intervals.
Never use `min(old_unobserved, new_idle)` as proof of causal explanation. If
a same-run leaf sum exceeds its enclosing duration because nesting or overlap
was summed, reconstruct a disjoint union before ranking.

## 4. Contradiction register

List conflicting sources, their identities, quantitative disagreement, current
disposition, and resolving check. A component exceeding its enclosing duration
on the same run is an error; differing sittings are a comparability problem
until resolved. Do not choose whichever number supports the planned candidate.
Unresolved material contradictions stop affected rankings, not unrelated
verified findings. Do not claim a production multiple (for example 4x) from
unreconciled component times taken in different sittings.

## 5. Claim types

Each conclusion is exactly one of `measured`, `derived`, `hypothesis`,
`incomplete`, or `historical`. A derivation includes formula, units, and input
links. Bytes divided by assumed bandwidth is a conditional estimate. A
removable-time upper bound requires a demonstrated bound on all relevant
costs. Graphs do not make kernel-node/fusion overhead zero by definition.
Historical numbers are references, not current denominators. Only verified
findings enter the final narrative.

## 6. Target and guard roles

Freeze region, primary metric, targets, guards, thresholds, and policy ID
before candidate timing. Apply OPT-135 when the task opts into
[`target_guard_v2`](../../../../pins/performance_keep_policy_v2.json); do not
invent a second policy in a stage prompt. Validate the implemented policy with:

```sh
uv run python tools/performance_keep_policy.py --contract pins/performance_keep_policy_v2.json --self-check fixtures/opt135_target_guard_policy.json
```

A guard needs non-regression, not demonstrated improvement. If a target
improves and a guard CI contains 1 but lies above its non-regression floor,
apply the opted-in target/guard policy, not a guard-improvement test.
Preserve historical OPT-130 rejection and release gates.

## 7. Independent verification

Pass task paths and required phases to the verifier without pre-deciding a
verdict. The verifier independently queries raw records for the largest
claimed gap and every claimed eliminated gap; checks coverage, units, and
identities; and records calculations. A second agent restating generated JSON
is not independent evidence. Failed evidence checks return to implementation.
Missing hardware is `incomplete`/`blocked`, not a measured rejection or no
opportunity. Required capture or candidate NLL not run is
`incomplete`/`blocked`; do not fabricate `no_material_opportunity`, a quality
pass, or a measured reject.

## 8. Reporting and delivery

Record candidate measured delta, shipping delta, quality result, and evidence
completeness separately. Every reported ratio must use a matching metric
identity and window. Diagnostics use `N/A` for shipping throughput changes.
Rejected candidates may have measured gains while shipping delta is zero; do
not conflate a rejected candidate's measured result with zero shipping impact.

Documentation precedes verification as draft preparation. Label draft
conclusions `unverified`. Verification assesses that draft and all artifacts.
Do not ask the documentation stage to supply a future verifier verdict.
Delivery may publish the verified verdict but must not invent or revise
scientific conclusions; a semantic report change returns to verification.

# Performance evidence review cases

Independent review uses these eight adversarial cases against both ledger
runner skills. Expected decisions are enforced by the cited passages in
[performance-evidence-checklist.md](performance-evidence-checklist.md). Do not
invoke delivery to apply a case.

## Case 1 — Graph envelopes are not proven idle

**Scenario.** Graph capture reports a 196.740 ms window, 8.243 ms ordinary GPU
activity, and 186.016 ms graph envelopes. A ranking subtracts ordinary
activity from the window and treats 188.497 ms as proven idle.

**Expected decision.** Reject 188.497 ms proven idle; internal graph activity
is unknown.

**Policy.** Checklist §2:

> Graph envelopes are not leaf kernel time. Graph-level data cannot prove
> continuous activity or internal idle. Missing node tracing forbids
> graph-internal idle and component-completeness claims. Do not subtract
> ordinary kernel/copy union from a window that contains graph envelopes and
> call the remainder proven idle; internal graph activity is unknown until node
> tracing covers it.

## Case 2 — Mismatched request versus decode-only ratio

**Scenario.** A comparison divides Quartz complete-request time by llama
decode-only time, or otherwise mixes those metric classes in one ratio.

**Expected decision.** Refuse the comparison.

**Policy.** Checklist §1:

> Distinguish prefill, decode-only, complete-request, component, and public
> sample/eval/output metrics. Setup, graph creation, warmup, and checkpoint
> restore stay outside rate denominators. A ratio requires a matching metric
> identity on both arms. Refuse a Quartz complete-request versus llama
> decode-only comparison.

## Case 3 — Synchronize wait is not GPU idle

**Scenario.** An API `cudaEventSynchronize` (or equivalent) span overlaps
concurrent GPU work. A ranking counts that CPU wait as removable GPU idle or
adds it to GPU wall time.

**Expected decision.** Refuse to count the synchronize span as idle or add it
to GPU wall time.

**Policy.** Checklist §3:

> CPU `cudaEventSynchronize` duration is wait duration, not removable GPU
> idle; do not count an API synchronize span that is concurrent with GPU work
> as idle or add it to GPU wall time.

## Case 4 — Nested or overlapping leaf sum

**Scenario.** On the same run, the sum of leaf intervals exceeds the enclosing
duration because graph parents and children, nested NVTX and kernels, or
overlapping streams were added rather than unioned.

**Expected decision.** Reconstruct a disjoint union before ranking. Treat the
exceeding leaf sum as an accounting error, not a ranking input.

**Policy.** Checklist §3:

> Union overlapping intervals. Do not sum graph parents and children, nested
> NVTX and kernels, or overlapping CPU waits and GPU work.

> If a same-run leaf sum exceeds its enclosing duration because nesting or
> overlap was summed, reconstruct a disjoint union before ranking.

Checklist §4:

> A component exceeding its enclosing duration on the same run is an error;

## Case 5 — Cross-sitting component contradiction

**Scenario.** An OPT-108-like 0.033 ms component and an OPT-129-like 0.113 ms
component for the same named work come from different sittings. A ranking
uses the pair to claim a production 4x gap.

**Expected decision.** Record the contradiction. Do not claim a production 4x
result without reconciliation.

**Policy.** Checklist §4:

> A component exceeding its enclosing duration on the same run is an error;
> differing sittings are a comparability problem until resolved. Do not choose
> whichever number supports the planned candidate. Unresolved material
> contradictions stop affected rankings, not unrelated verified findings. Do
> not claim a production multiple (for example 4x) from unreconciled component
> times taken in different sittings.

Checklist §5:

> Historical numbers are references, not current denominators.

## Case 6 — Missing llama GDN mapping

**Scenario.** Quartz reports GDN time but the llama capture has no mapped GDN
family. A ranking computes excess as Quartz minus zero.

**Expected decision.** Excess is `null`/`unknown`, never Quartz minus zero.

**Policy.** Checklist §2:

> Missing coverage must not become zero excess. Unmapped llama families make
> excess `null`/`unknown`; never treat missing corresponding llama work as
> Quartz minus zero.

## Case 7 — Guard non-regression, not guard improvement

**Scenario.** The task opts into the OPT-135 target/guard policy. The target
improves. A guard CI contains 1 but lies entirely above its non-regression
floor.

**Expected decision.** Apply the opted-in target/guard policy. The guard
passes non-regression; do not fail it for lack of demonstrated improvement.

**Policy.** Checklist §6:

> Apply OPT-135 when the task opts into it; do not invent a second policy in a
> stage prompt. A guard needs non-regression, not demonstrated improvement. If
> a target improves and a guard CI contains 1 but lies above its
> non-regression floor, apply the opted-in target/guard policy, not a
> guard-improvement test.

## Case 8 — Missing required capture or candidate NLL

**Scenario.** Required capture or candidate NLL was not run. A report still
emits `no_material_opportunity`, a quality pass, or a measured reject.

**Expected decision.** `incomplete`/`blocked`. Do not fabricate
`no_material_opportunity`, a quality pass, or a measured reject.

**Policy.** Checklist §7:

> Missing hardware is `incomplete`/`blocked`, not a measured rejection or no
> opportunity. Required capture or candidate NLL not run is
> `incomplete`/`blocked`; do not fabricate `no_material_opportunity`, a quality
> pass, or a measured reject.

# TASK-031 — EXP-H prefill recurrence algorithm

## Status
TODO
## Milestone
M10 — Architecture-V0 experiments
## Purpose
Test whether a 64-token chunkwise/WY evaluation can replace the selected serial register-resident prefill recurrence.
## Depends on
- TASK-030
## Normative references
- `docs/architecture/architecture-v0.md` — EXP-H; prefill GDN recurrence
- `docs/architecture/evaluation-policy-v0.md` — PERF-01; final parity-gap reassessment
- `docs/architecture/performance-validation.md`
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| G-02, S-02 | Selected prefill recurrence algorithm/ownership; experiment may recommend amendment | LOCKED UNDER TEST |
| S-01, P-01 | FP32 persistent state and recurrence arithmetic | LOCKED |
## Starting point
Serial 64-token recurrence has correctness/behavior/performance baseline; other experiments concluded.
## Scope
Specify and implement a mathematically equivalent 64-token chunkwise/WY candidate for the same GDN map, with explicit workspace and boundary state; independently verify equations; compare arbitrary tails/chunk partitions, finite-precision divergence, long-context behavior/continuation, workspace/memory, local recurrence and full prefill/TTFT/request timing. Keep weights, state ABI/precision, other kernels, and chunk size fixed.
## Out of scope
Changing recurrence semantics, decode recurrence, state layout/precision, chunk size sweep, combining candidate with rejected experiments.
## Required interfaces
Explicit experimental recurrence plan/algorithm ID and workspace description; production serial path retained.
## Required semantics
Candidate realizes the same ordered state map and produces one o/token; finite-precision equivalence is tested, never assumed. Tails <64 and cross-256/chunk continuation are exact semantic boundaries.
## Data representation
FP32 S remains `[head,value,key]`; all additional candidate workspace is typed, bounded, lifetime-planned, and byte-counted.
## Implementation constraints
Independent reference/derivation in tests or focused design note; no quality threshold relaxation; measure full prefill.
## Tuning defaults
64-token comparison interval and frozen measurement protocol.
## Expected files/modules
Experimental WY kernel/plan/workspace, reference/equivalence tests, final experiment report.
## Tests required
### Unit tests
Lengths/tails 1,2,63,64 and multiple intervals; workspace bounds; reset/snapshot.
### Reference/numerical tests
Candidate versus serial/reference for adversarial gates and long sequences; divergence tracked at S/o/logits.
### Integration tests
Arbitrary prompt partitions, prefill→decode continuation, frozen long-context behavior.
## Benchmark required
Yes: local 64-token recurrence, full prefill/TTFT/request, workspace/resources.
## Acceptance criteria
- [ ] Candidate equations and implementation have independent correctness evidence.
- [ ] All tail/chunk/handoff and long-context quality checks pass for a change recommendation.
- [ ] Workspace/resource/local/end-to-end deltas are identity-matched and uncertainty-aware.
- [ ] Recommendation follows EXP-H; no architecture amendment occurs.
- [ ] PERF-01 parity status is reassessed for the final accepted implementation using the matched llama.cpp baseline; any remaining speed gaps are explicit and do not invalidate a completed experiment.
## Architecture blocker rule
Candidate rejection is result; unrelated locked conflict requires full blocker report.
## Completion report
### Result
DONE | BLOCKED
### Changes made
### Tests run
Exact commands/results.
### Benchmark results
Quality/workspace/resource/local/end-to-end evidence.
### Architecture blocker
None/full report.
### Follow-up observations
Keep/change recommendation and any explicit proposed amendment only.

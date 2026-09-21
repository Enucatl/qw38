# TASK-027 — EXP-D GDN layout and ownership

## Status
TODO
## Milestone
M10 — Architecture-V0 experiments
## Purpose
Compare V0 recurrence ABI/ownership with the one named alternative using full costs.
## Depends on
- TASK-026
## Normative references
- `docs/architecture/architecture-v0.md` — EXP-D; chosen GDN ownership
- `docs/architecture/performance-validation.md`
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| S-02 | `[head,value,key]`, warp/value selected; experiment may recommend amendment | LOCKED UNDER TEST |
| S-01, P-01 | FP32 state/arithmetic stay fixed | LOCKED |
## Starting point
State-precision experiment concluded; reference recurrence and baseline profiles exist.
## Scope
Implement explicit experimental `[head,key,value]` state with one block/head ownership, including schema/layout version, reset/snapshot/restore, decode and 64-token prefill consumers. Compare reference correctness, q/k reuse, actual state traffic, spills, synchronization, occupancy, local recurrence, full decode/prefill/request latency, memory, and continuation with unrelated choices fixed.
## Out of scope
Other recurrence mappings, BF16 state, WY algorithm, equation/precision change, hidden state transpose outside explicitly measured candidate setup.
## Required interfaces
Distinct state ABI/plan IDs; candidate sessions/artifacts reject cross-layout snapshots.
## Required semantics
Same FP32 recurrence equation/order obligations and logical S; only physical axes/ownership change.
## Data representation
Baseline FP32 `[head,value,key]`; candidate FP32 `[head,key,value]`; all conversion/setup costs identified and excluded/included according to matching metric definition.
## Implementation constraints
No silent production ABI replacement; measure decode and 64-token prefill separately and end to end.
## Tuning defaults
Candidate block geometry may be minimally tuned and fully recorded; equations/other kernels fixed.
## Expected files/modules
Experimental state layout/kernel/schema, correctness/performance results.
## Tests required
### Unit tests
Indexing, schema/snapshot mismatch, reset, conversion diagnostic.
### Reference/numerical tests
One/many-step and partition continuation versus common reference.
### Integration tests
Full behavior continuation and matched engine measurements.
## Benchmark required
Yes: recurrence decode/64-token prefill plus complete decode/prefill/request and profiler metrics.
## Acceptance criteria
- [ ] Candidate ABI is explicit and independently correct.
- [ ] Traffic/spills/sync/occupancy and full-engine effects are measured.
- [ ] Quality/continuation remain within frozen gate.
- [ ] Recommendation follows EXP-D; architecture remains unchanged.
## Architecture blocker rule
Candidate failure is an outcome; unrelated locked conflict requires full blocker report.
## Completion report
### Result
DONE | BLOCKED
### Changes made
### Tests run
Exact commands/results.
### Benchmark results
All local/end-to-end/resource identities/deltas.
### Architecture blocker
None/full report.
### Follow-up observations
Keep/change recommendation only.


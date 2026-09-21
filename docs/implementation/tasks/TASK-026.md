# TASK-026 — EXP-C persistent-state precision

## Status
TODO
## Milestone
M10 — Architecture-V0 experiments
## Purpose
Test whether BF16 persistent GDN S can replace the conservative FP32 state without long-horizon loss.
## Depends on
- TASK-025
## Normative references
- `docs/architecture/architecture-v0.md` — EXP-C; GDN state; validation
- `docs/architecture/performance-validation.md`
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| S-01 | FP32 persistent S selected; experiment may recommend amendment | LOCKED UNDER TEST |
| S-02, P-01 | Layout/ownership and recurrence arithmetic stay fixed | LOCKED |
## Starting point
Prior experiment recommendations are recorded; production baseline remains identifiable.
## Scope
Implement explicit BF16-S candidate load/store while retaining FP32 recurrence registers, prediction/update/readout, physical `[head,value,key]`, ownership, weights, and schedules. Run one-step diagnosis but decide using long-horizon NLL, continuation/save-restore, capability/long-context, memory, decode, prefill recurrence, and request evidence.
## Out of scope
BF16 recurrence arithmetic, layout/ownership changes, KV/history precision, simultaneous activation changes.
## Required interfaces
Session/state schema and artifact policy explicitly version/identify FP32 versus BF16 candidate; snapshots cannot be misread across schemas.
## Required semantics
Only persistent S store/load rounds to BF16; every computation widens and remains FP32. Reset/continuation rules unchanged.
## Data representation
Candidate S BF16 same physical axes; report byte savings and conversion points.
## Implementation constraints
One-step agreement cannot justify narrowing; frozen inputs and conditions; production FP32 stays default.
## Tuning defaults
Frozen validation/performance protocol.
## Expected files/modules
Experimental state schema/runtime path, long-horizon tests/results.
## Tests required
### Unit tests
Schema mismatch, reset/snapshot, BF16 round-trip boundaries.
### Reference/numerical tests
One/many-step divergence tracking versus FP32 reference.
### Integration tests
Frozen behavior/long-context/continuation and matching performance cases.
## Benchmark required
Yes: memory, decode, 64-token prefill recurrence, full prefill/request.
## Acceptance criteria
- [ ] Candidate changes only persistent S precision.
- [ ] Long-horizon and continuation quality evidence is complete.
- [ ] Memory/performance deltas are matched and uncertainty-aware.
- [ ] Recommendation follows EXP-C; no architecture amendment occurs.
## Architecture blocker rule
Candidate failure is not a blocker; unrelated locked conflicts require complete blocker report.
## Completion report
### Result
DONE | BLOCKED
### Changes made
### Tests run
Exact commands/results.
### Benchmark results
Quality/memory/performance identities/deltas.
### Architecture blocker
None/full report.
### Follow-up observations
Keep/change recommendation only.


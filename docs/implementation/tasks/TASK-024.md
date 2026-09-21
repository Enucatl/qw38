# TASK-024 — EXP-A main weight policy

## Status
TODO
## Milestone
M10 — Architecture-V0 experiments
## Purpose
Test whether Q4G64 remains the main-weight policy under frozen behavioral and performance evidence.
## Depends on
- TASK-023
## Normative references
- `docs/architecture/architecture-v0.md` — EXP-A; quantizer definitions; validation policy
- `docs/architecture/quantization-validation.md`
- `docs/architecture/performance-validation.md`
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| Q-01 | Selected Q4G64 main weights; experiment may recommend amendment | LOCKED UNDER TEST |
| A-02, P-01–P-02 | Quantizer/layout separation and arithmetic stay fixed | LOCKED |
| L-01 | Physical layout gets explicit experimental versions where grouping changes | LOCKED |
## Starting point
Frozen behavior suite and identity-complete V0 baseline exist.
## Scope
Add explicitly versioned experimental Q4G32 and Q4G128 logical quantizers/packings/consumers using the same absmax/RNE/stored-scale recipe; compare to Q4G64 with unrelated choices fixed. Run full behavior gate, artifact/memory accounting, decode and request measurements. If G64 fails quality, isolate one tensor family at a time and test promotion to existing Q8G32. Produce keep/change recommendation and proposed affected decision/ABI IDs; do not amend architecture in this task.
## Out of scope
Mixed-width groups, calibration, outliers, activation quantization, simultaneous head/state/layout experiments, shipping a failed policy.
## Required interfaces
Experiment selection is explicit in compiler/runner/result identity and cannot be mistaken for production V0.
## Required semantics
Only group size/failing-family promotion varies; source, eval hashes, scale/code recipe, arithmetic, schedule, clocks, contexts, and timing procedure remain fixed.
## Data representation
Experimental logical/layout version IDs; complete payload/scale byte accounting and artifact hashes.
## Implementation constraints
No overwrite of V0 IDs; independent pack/unpack tests for each candidate; quality first, then end-to-end benefit.
## Tuning defaults
Use TASK-023 measurement protocol and ≥5% affected-mode improvement heuristic beyond noise.
## Expected files/modules
Experimental compiler/layout/consumer branches behind explicit IDs, tests, result records, recommendation report.
## Tests required
### Unit tests
Group/code/scale/golden-byte tests for G32/G128 and version rejection.
### Reference/numerical tests
Candidate contractions against independent unpack/reference.
### Integration tests
Frozen behavior suite and identity-matched baseline/candidate runs.
## Benchmark required
Yes: matching decode/request cases and local contraction evidence with memory accounting.
## Acceptance criteria
- [ ] All candidates have independent correctness evidence and unique identities.
- [ ] Frozen quality results are reported per domain/context before performance conclusions.
- [ ] End-to-end, local, and memory deltas use matched identities and uncertainty.
- [ ] Recommendation follows EXP-A falsification rule; architecture remains unamended.
## Architecture blocker rule
An expected hypothesis failure is an experiment result, not a blocker. For an unrelated locked conflict, stop and report all required blocker fields.
## Completion report
### Result
DONE | BLOCKED
### Changes made
### Tests run
Exact commands/results.
### Benchmark results
Candidate/baseline identities, raw paths, quality/memory/local/end-to-end deltas.
### Architecture blocker
None/full report.
### Follow-up observations
Keep/change recommendation and affected decisions only.


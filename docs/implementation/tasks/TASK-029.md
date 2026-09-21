# TASK-029 — EXP-F normalization/projection boundary

## Status
TODO
## Milestone
M10 — Architecture-V0 experiments
## Purpose
Test whether decode GDN should remove the separate normalized buffer through narrowly scoped RMS recomputation.
## Depends on
- TASK-028
## Normative references
- `docs/architecture/architecture-v0.md` — EXP-F; materialization/fusion
- `docs/architecture/performance-validation.md`
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| M-01 | Separate normalized scratch selected; experiment may recommend amendment | LOCKED UNDER TEST |
| P-01–P-02 | Norm math and BF16 projection operand semantics stay fixed | LOCKED |
## Starting point
Transport experiment concluded; eager region baseline/profiles exist.
## Scope
Implement only the named candidate: block-local hidden RMS recomputation fused into grouped GDN qkv/z decode contraction, while small a/b consumer recomputes its own norm. Leave prefill normalization buffer unchanged. Compare exact BF16 operands/results, saved launch/traffic, redundant reductions, registers/spills/occupancy, GDN-local and end-to-end decode/request timing, and behavior.
## Out of scope
Whole-layer fusion, MLP/attention fusion, prefill fusion, changing precision/layout/weights, eliminating other scratch.
## Required interfaces
Explicit experimental GDN plan/launch; production separate path retained and identity recorded.
## Required semantics
Each consumer computes the same zero-centered FP32 RMS and rounds identical normalized values to BF16 before contraction; a/b output remains FP32.
## Data representation
Candidate removes only decode normalized global store for this GDN region; report bytes/launches and extra compute/resources.
## Implementation constraints
No changed projection reduction/epilogue; compare matching full requests, not kernel-only wins.
## Tuning defaults
Baseline geometries unless minimal recorded adjustment is needed for fused resource validity.
## Expected files/modules
Experimental fused kernel/plan, operand-equivalence tests, profiler/results.
## Tests required
### Unit tests
Plan selection, scratch lifetime, separate a/b recomputation.
### Reference/numerical tests
Normalized BF16 operands and full GDN output/state versus baseline/reference.
### Integration tests
Frozen behavior and matched decode/request runs.
## Benchmark required
Yes: launch/traffic/resources, GDN local, full decode/request.
## Acceptance criteria
- [ ] Candidate scope is exactly EXP-F and prefill unchanged.
- [ ] Correctness/behavior gate passes.
- [ ] Saved traffic versus redundant work/resources and end-to-end delta are reported.
- [ ] Recommendation follows EXP-F; no architecture amendment.
## Architecture blocker rule
Candidate regression is outcome; unrelated locked conflict requires complete blocker report.
## Completion report
### Result
DONE | BLOCKED
### Changes made
### Tests run
Exact commands/results.
### Benchmark results
Matched local/resource/end-to-end evidence.
### Architecture blocker
None/full report.
### Follow-up observations
Keep/change recommendation only.


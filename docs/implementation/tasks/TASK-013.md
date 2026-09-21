# TASK-013 — Complete decode GDN mixer

## Status
TODO
## Milestone
M4 — GDN execution
## Purpose
Integrate all eight selected GDN regions into a continuation-correct mixer residual transition.
## Depends on
- TASK-010
- TASK-012
## Normative references
- `docs/architecture/architecture-v0.md` — GDN initial kernel sequence; precision/materialization
- `docs/architecture/model-semantics.md`
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| G-01, Q-01, Q-03, P-01–P-02, S-01–S-02, M-01 | Complete GDN semantics, precision, state, and cuts | LOCKED |
| T-01, T-03 | Initial kernel geometries | TUNING |
## Starting point
GDN steps 1–6, packed contractions, activation kernels, session state, and MLP exist.
## Scope
Implement gated output transform (per-value-head RMS with multiplicative gamma and FP32 SiLU(z), BF16 u), Q4 output projection with direct FP32 residual add, and a concrete eight-region GDN plan/executor. Add full CPU/BF16-control comparison and repeated-token continuation. Preserve each V0 diagnostic boundary.
## Out of scope
Prefill, region fusion, alternate layout/state precision, whole layer scheduler, attention.
## Required interfaces
GDN plan binds all named weights/parameters/state/scratch; decode mixer accepts FP32 residual and position/session and returns the other FP32 residual buffer without allocation.
## Required semantics
Steps exactly: input RMS; qkv/z; a/b; conv+SiLU; prep; recurrence; gated RMS/SiLU(z); out projection+residual. GDN gamma is multiplicative, not `1+gamma`. Input residual remains live until step 8.
## Data representation
Use normative shapes/dtypes and TASK-007 arena aliases; o FP32 `[48,128]`, u BF16 `[48,128]`.
## Implementation constraints
No fusion beyond grouped projections already selected; eager one-stream ordering; state commits once/token.
## Tuning defaults
Inherited TASK-009/011/012 geometries.
## Expected files/modules
Runtime GDN plan/executor, output transform kernel, complete GDN tests.
## Tests required
### Unit tests
Bindings/lifetimes, gamma role, z gate, residual preservation, state isolation.
### Reference/numerical tests
Every materialized boundary plus final residual and state against reference for one/many tokens.
### Integration tests
Complete GDN mixer followed by TASK-010 MLP for two sessions and restored continuation.
## Benchmark required
Report region timings diagnostically; no fusion authorization.
## Acceptance criteria
- [ ] Eight regions and declared stores remain separately attributable.
- [ ] Final residual/state pass reference and continuation tests.
- [ ] Gated norm/gate precision semantics are correct.
- [ ] Steady state performs no allocations.
## Architecture blocker rule
On locked conflict stop with full required blocker report; do not fuse or alter boundaries as a workaround.
## Completion report
### Result
DONE | BLOCKED
### Changes made
### Tests run
Exact commands/results.
### Benchmark results
Diagnostic region timings.
### Architecture blocker
None/full report.
### Follow-up observations
Concrete only.


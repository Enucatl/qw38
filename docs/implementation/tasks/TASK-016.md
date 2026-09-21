# TASK-016 — One-layer integration checkpoint

## Status
TODO
## Milestone
M6 — Integrated language decode
## Purpose
Expose binding, residual, scratch, and persistent-state errors before scaling to 64 layers.
## Depends on
- TASK-013
- TASK-015
## Normative references
- `docs/architecture/architecture-v0.md` — Semantic graph; decode schedule; materialization
- `docs/architecture/model-semantics.md`
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| G-01–G-02 | Layer semantics and decode physical schedule | LOCKED |
| P-01–P-02, S-01–S-02, M-01 | Precision/state/scratch contracts | LOCKED |
## Starting point
Complete GDN+MLP and attention+MLP paths work separately.
## Scope
Build immutable plans and deterministic fixtures for one GDN-style language layer and one full-attention-style language layer through the same runtime/session/tensor binding/scratch/residual machinery. Execute consecutive tokens and compare each materialized boundary, state update, and final residual to reference/BF16 control.
## Out of scope
64-layer model, embedding/head, scheduler optimization, prefill, fusion, general graph engine.
## Required interfaces
Concrete `LanguageLayerPlan`-equivalent tagged plan (no virtual operator framework); one decode-layer entry dispatches statically/tagged to mixer then shared MLP.
## Required semantics
Layer ordering is mixer residual transition then post-mixer MLP transition. Input residual remains valid for its owning add. Only the correct state family changes.
## Data representation
Bindings come from artifact tensor IDs/layouts; two FP32 residual buffers ping-pong; arena aliases follow proven lifetimes; session state remains layer-indexed.
## Implementation constraints
No per-call plan construction/allocation; no cross-layer scratch leakage; deterministic one-stream ordering.
## Tuning defaults
Inherited kernel defaults only.
## Expected files/modules
Runtime language-layer plan/executor and deterministic integration fixture/tests.
## Tests required
### Unit tests
Missing/wrong family binding, scratch overlap rejection, layer/state index isolation.
### Reference/numerical tests
Stage/final comparisons for each layer kind over 1 and multiple tokens.
### Integration tests
GDN layer then attention layer in one session; reset/restore continuation; second session isolation.
## Benchmark required
No.
## Acceptance criteria
- [ ] Both layer types use common runtime/session interfaces.
- [ ] Residual ping-pong, scratch reuse, and tensor bindings are correct.
- [ ] Only expected persistent state changes.
- [ ] Multi-token and restore continuation pass.
## Architecture blocker rule
On locked conflict stop with full blocker report; do not introduce a generic graph/operator architecture.
## Completion report
### Result
DONE | BLOCKED
### Changes made
### Tests run
Exact commands/results.
### Benchmark results
Not required.
### Architecture blocker
None/full report.
### Follow-up observations
Concrete only.


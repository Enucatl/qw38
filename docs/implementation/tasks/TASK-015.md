# TASK-015 — Segmented online decode attention

## Status
TODO
## Milestone
M5 — Attention execution
## Purpose
Implement bounded-memory causal GQA decode attention with deterministic segment merging and complete mixer output.
## Depends on
- TASK-014
## Normative references
- `docs/architecture/architecture-v0.md` — Full-attention sequence; initial attention ownership
- `docs/architecture/model-semantics.md`
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| P-01–P-02, G-01, M-01 | FP32 attention math, BF16 outputs/cache, online materialization | LOCKED |
| T-03 | 128 threads, 256-key segment, 32-key subtile | TUNING |
## Starting point
Prepared Q/g and correctly appended K/V exist; packed Q4 output projection exists.
## Scope
Implement an independent online-attention reference, per-query-head/per-256-key-segment FP32 online softmax scan, FP32 partial max/sum/256-value numerator, fixed-order merge, normalization, FP32 sigmoid(g), BF16 gated output, and Q4 output projection with direct FP32 residual add.
## Out of scope
Quadratic scores/probabilities, nondeterministic/atomic merge, prefill attention, cache paging, fused projection/preparation/scan, alternate GQA storage.
## Required interfaces
Attention-core plan binds Q/g/cache, populated length, partial workspace, output projection/residual; explicit scan and merge launch wrappers.
## Required semantics
Scores/softmax/value accumulation FP32 with causal range `[0,populated)`. Standard max-rescaled segment and merge equations; segments merge increasing index; empty/tail keys mask; six Q heads address each KV head; sigmoid gate precedes BF16 store.
## Data representation
Partial per segment/query head: FP32 max, sum, numerator `[256]`; shared staging reuses at most one K or V subtile, never both/full segment; no score vector in global memory.
## Implementation constraints
Initial scan and merge remain separate; one ordered stream; deterministic results for fixed inputs.
## Tuning defaults
128 threads; 256 keys/segment; 32-key subtiles.
## Expected files/modules
Reference online attention, CUDA scan/merge, full attention mixer executor/tests.
## Tests required
### Unit tests
Lengths 1,31,32,255,256,257; tail/empty segments; extreme logits; GQA mapping; causal exclusion; gate extremes.
### Reference/numerical tests
Segmented versus unsegmented stable FP32 reference and full mixer residual comparison.
### Integration tests
Repeated cache append+attention; snapshot/restore; deterministic repeated runs at several lengths.
## Benchmark required
Diagnostic timings at populated lengths 512,4096,32768 where memory permits.
## Acceptance criteria
- [ ] Online result meets reference tolerance across boundaries.
- [ ] Merge order is fixed and no quadratic global scores exist.
- [ ] Full attention mixer residual and cache continuation pass.
- [ ] Timings are diagnostic and do not alter architecture.
## Architecture blocker rule
On locked conflict stop with full blocker report; tuning limits do not authorize semantic/layout changes.
## Completion report
### Result
DONE | BLOCKED
### Changes made
### Tests run
Exact commands/results.
### Benchmark results
Diagnostic identities/timings.
### Architecture blocker
None/full report.
### Follow-up observations
Concrete only.


# TASK-014 — Attention preparation and KV cache

## Status
TODO
## Milestone
M5 — Attention execution
## Purpose
Implement full-attention projection semantics, per-head preparation, and persistent cache append independently of the attention scan.
## Depends on
- TASK-009
## Normative references
- `docs/architecture/architecture-v0.md` — Full-attention sequence; convolution/KV layout
- `docs/architecture/model-semantics.md`
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| Q-01, P-01–P-02, G-01, M-01 | Projection, precision, attention semantics and stores | LOCKED |
| T-03 | 128-thread preparation start | TUNING |
## Starting point
RMS, Q4 projections, QK RMS, partial RoPE, and session KV storage exist.
## Scope
Implement attention steps 1–3: input RMS; one Q4 launch for q/g, k, v ranges preserving per-query-head q/g association; preparation applying zero-centered QK norms in FP32 and partial RoPE; write BF16 Q and append one BF16 rotated K and BF16 V to cache; advance populated length transactionally after successful work. Add CPU reference and reset/restore tests.
## Out of scope
Attention score/softmax/value scan, output gating/projection, prefill cache batch append, repeated GQA cache copies.
## Required interfaces
Attention-prep plan binds norm/projections/QK norms/RoPE/cache/scratch; decode call takes absolute position and verifies it equals append/populated contract.
## Required semantics
24 query heads, four KV heads, width 256; q/g split within each head; QK norm before RoPE; only first 64 coordinates rotate; K stored after norm/RoPE, V stored once; six query heads map to each KV head.
## Data representation
Prepared Q BF16 `[24,256]`; g scratch aligned per query head; K/V BF16 `[4,capacity,256]`, coordinate-contiguous; no sixfold copies.
## Implementation constraints
Never expose advanced populated length on failure; capacity and position errors are typed; no full prepared K/V duplicate.
## Tuning defaults
One preparation block/head, 128 threads.
## Expected files/modules
Attention plan/preparation kernel/cache append tests.
## Tests required
### Unit tests
q/g ordering, head mapping, capacity/full cache, position mismatch, reset, unaffected RoPE suffix.
### Reference/numerical tests
Projected/prepared Q/K/V/g versus reference at positions 0,1,large; BF16 store tolerance.
### Integration tests
Multiple appends, snapshot/restore, two sessions, cache byte/index verification.
## Benchmark required
No.
## Acceptance criteria
- [ ] Per-head q/g, QK norm, and partial RoPE semantics pass.
- [ ] Cache contains one BF16 K/V copy per KV head/token.
- [ ] Append length changes only after success.
- [ ] Continuation/reset tests pass.
## Architecture blocker rule
On locked conflict stop with complete blocker fields; do not change cache layout or RoPE scope.
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


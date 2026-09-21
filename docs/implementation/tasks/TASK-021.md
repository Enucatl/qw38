# TASK-021 — Tiled causal prefill attention

## Status
TODO
## Milestone
M8 — Production V0 prefill
## Purpose
Implement the separate chunked causal attention core and cache append needed by production prefill.
## Depends on
- TASK-019
## Normative references
- `docs/architecture/architecture-v0.md` — Prefill full-attention strategy
- `docs/architecture/model-semantics.md`
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| G-02, P-01–P-02, M-01 | Separate tiled schedule and precision/materialization | LOCKED |
| T-03 | 128 threads, 32-query×64-key tile | TUNING |
## Starting point
Chunk projections and decode attention/cache provide semantics/reference.
## Scope
Implement batch Q/g/K/V preparation with completed BF16 cache append before reads, then one block/query-head/32-query tile scanning 64-key tiles with causal masking, FP32 online softmax/value accumulation, sigmoid gate, BF16 output, Q4 output projection/residual, and MLP. Support incoming populated cache and short/tail chunks.
## Out of scope
Quadratic score storage, decode segment kernel reuse as final path, simultaneous K/V shared buffers, projection fusion, paged KV.
## Required interfaces
Prefill attention plan accepts token-major chunk, absolute start, incoming populated length, capacity; commits cache/populated length transactionally.
## Required semantics
Each query attends all prior populated keys and valid keys up to itself; no future/padded keys. Q/K normalization and partial RoPE match decode. Cache contains prepared K and V once. Online merge is stable FP32; sigmoid gate applies before BF16 store.
## Data representation
32-query×64-key tile; shared Q/K/V staging with one 32 KiB buffer reused K then V and ≤8 KiB score/reduction scratch; output accumulators FP32 registers.
## Implementation constraints
Preparation/cache completion globally precedes attention; no full score/probability matrix; no K/V coexistence in shared memory.
## Tuning defaults
128 threads, 32 queries, 64 keys, single shared-buffer reuse.
## Expected files/modules
CUDA prefill preparation/attention, runtime layer integration, causal/cache tests.
## Tests required
### Unit tests
Lengths/tails around 32/64/256, incoming lengths 0/nonzero, capacity failure, causal mask, positions, transactional append.
### Reference/numerical tests
Outputs/cache versus stable reference and repeated decode for arbitrary chunk partitions.
### Integration tests
Multiple attention+MLP chunks followed by one decode token.
## Benchmark required
Diagnostic timings/resources for 256-token chunk with several incoming lengths.
## Acceptance criteria
- [ ] Causal outputs/cache match references across boundaries.
- [ ] Incoming and within-chunk keys are addressed correctly.
- [ ] No quadratic scores or duplicated GQA cache exist.
- [ ] Resource/timing evidence is recorded without architecture changes.
## Architecture blocker rule
On locked conflict stop with full blocker report; geometry pressure is tuning, not redesign permission.
## Completion report
### Result
DONE | BLOCKED
### Changes made
### Tests run
Exact commands/results.
### Benchmark results
Diagnostic only.
### Architecture blocker
None/full report.
### Follow-up observations
Concrete only.


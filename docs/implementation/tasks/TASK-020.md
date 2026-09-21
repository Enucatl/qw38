# TASK-020 — Chunked prefill GDN

## Status
TODO
## Milestone
M8 — Production V0 prefill
## Purpose
Implement the layer-wise chunk GDN schedule while preserving exact decode recurrence and history continuation.
## Depends on
- TASK-019
## Normative references
- `docs/architecture/architecture-v0.md` — Prefill GDN strategy; recurrence/state layout
- `docs/architecture/model-semantics.md`
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| S-01–S-02, G-02, P-01–P-02, M-01 | State ABI, separate prefill schedule, precision/cuts | LOCKED |
| T-02 | 256-token chunks and 64-token recurrence launches | TUNING |
## Starting point
Chunk projections/MLP work; decode GDN is reference authority.
## Scope
Implement parallel chunk FIR reading incoming three-vector history plus raw projected qkv; separate race-free history commit; chunk preparation/output transforms; and up to four ordered recurrence launches, each processing ≤64 valid tokens left-to-right while loading/storing FP32 S only at launch boundaries and emitting FP32 o per token. Integrate complete GDN+MLP prefill layer.
## Out of scope
Chunkwise/WY recurrence, state precision/layout changes, recurrence parallelization across time, history in-place races, attention/full-model scheduling.
## Required interfaces
Prefill GDN plan consumes token-major chunk, incoming state/history, absolute positions, valid count; commits exact outgoing state/history only for valid tokens.
## Required semantics
FIR first three positions read incoming history; commit final three vectors of incoming-history concatenated with valid new raw qkv. If fewer than three arrive, preserve required incoming tail. Recurrence equation/order equals decode; padded tokens never update state/cursor.
## Data representation
Token-major activations; history BF16 `[3,10240]`; S FP32 normative layout; o FP32 `[valid,48,128]`; output transforms/materializations match V0.
## Implementation constraints
Separate FIR and commit kernels; four ordered max-64 recurrence launches for 256; no finite-precision equivalence assumption beyond tested tolerance.
## Tuning defaults
64-token interval, channel-parallel FIR.
## Expected files/modules
CUDA prefill GDN FIR/commit/recurrence and runtime layer integration/tests.
## Tests required
### Unit tests
Lengths 0/1/2/3/63/64/65/255/256; history ordering; padded masks; cursor/position.
### Reference/numerical tests
Chunk results/state/history versus repeated decode/reference for arbitrary partitions and restored incoming state.
### Integration tests
Multiple chunks through one complete GDN+MLP layer; snapshot then continue.
## Benchmark required
Diagnostic region timings for 64/256 tokens.
## Acceptance criteria
- [ ] Every tested partition produces continuation-equivalent state/history/output within declared tolerance.
- [ ] S loads/stores occur only at recurrence-launch boundaries.
- [ ] History commit is race-free and correct for short chunks.
- [ ] No WY/alternate recurrence was introduced.
## Architecture blocker rule
On locked conflict stop with full blocker report; serialization performance is reserved for EXP-H.
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


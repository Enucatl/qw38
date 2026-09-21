# TASK-019 — Tensor-core prefill projections and chunk planning

## Status
TODO
## Milestone
M8 — Production V0 prefill
## Purpose
Establish the distinct bounded prefill activation schedule and common-view tensor-core dense consumers.
## Depends on
- TASK-018
## Normative references
- `docs/architecture/architecture-v0.md` — Prefill schedule; physical packing/consumption
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| A-01, G-02, L-01 | One packed view; separate prefill physical schedule | LOCKED |
| P-01–P-02, M-01 | BF16 operands/scratch, FP32 accumulation, bounded materialization | LOCKED |
| T-02 | 256-token chunks, 32×64×64 tiles, 128 threads | TUNING |
## Starting point
Decode/reference semantics and behavior baseline pass; runtime arena currently supports decode.
## Scope
Implement token-major chunk scratch/liveness planning for up to 256 valid tokens and tensor-core BF16 projection consumers that locally decode common Q4/Q8 tiles, stage single-buffered shared operands, reuse weights across 32 rows, accumulate FP32, and support ordinary BF16, paired SwiGLU, FP32 residual, and requested-logit epilogues. Handle short final tiles by masks. Add BF16 control/reference GEMM comparisons.
## Out of scope
Decode-kernel batching as final prefill, global decoded-weight cache, duplicate views, async double buffering/TMA/clusters, GDN recurrence/convolution, attention, end-to-end scheduler.
## Required interfaces
Typed prefill projection launch descriptor and immutable chunk plan with valid tokens/absolute positions; generation-head mode final row only and evaluation mode requested bounded rows.
## Required semantics
Decoded weights round to BF16 exactly as decode; activation/weight operands BF16; MMA accumulates FP32; padded rows never affect output/state. Paired gate/up applies FP32 SiLU×up and stores BF16 only.
## Data representation
Token-major `[valid_tokens,channels]`; output tile 32 tokens×64 outputs; K step 64; common eight-row packed source; 4 KiB activation and 8 KiB weight operands/tile before padding (two weight operands paired).
## Implementation constraints
One artifact view; local shared rearrangement only; stable preallocated arena; tensor-core use verified from generated code/profile evidence.
## Tuning defaults
256/32×64×64/128 threads, single buffering.
## Expected files/modules
Prefill chunk/arena plan, CUDA projection kernels/wrappers, GEMM reference/tests.
## Tests required
### Unit tests
Valid lengths 1,31,32,33,255,256; masks; arena overlap; epilogues; requested row sets.
### Reference/numerical tests
Q4/Q8/BF16 output against decoded-BF16 reference for representative model shapes.
### Integration tests
Prefill MLP and projection sets for one GDN/attention layer without recurrence/attention core.
## Benchmark required
Projection microbenchmarks for selected shapes/lengths; diagnostic only.
## Acceptance criteria
- [ ] Tensor-core path consumes the one V0 packed view.
- [ ] All length/tail/epilogue cases match reference tolerance.
- [ ] Chunk scratch is bounded, token-major, stable, and allocation-free in execution.
- [ ] Evidence confirms tensor-core instructions and records resources/timing.
## Architecture blocker rule
On locked conflict stop with full blocker report; common-view inefficiency belongs to EXP-G.
## Completion report
### Result
DONE | BLOCKED
### Changes made
### Tests run
Exact commands/results.
### Benchmark results
Diagnostic identities/results.
### Architecture blocker
None/full report.
### Follow-up observations
Concrete only.


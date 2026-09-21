# TASK-010 — Complete decode MLP

## Status
TODO
## Milestone
M3 — Core runtime and MLP
## Purpose
Deliver the first complete integrated quantized CUDA consumer and residual transition.
## Depends on
- TASK-009
## Normative references
- `docs/architecture/architecture-v0.md` — MLP kernel sequence; precision/materialization policy
- `docs/architecture/model-semantics.md`
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| Q-01, P-01–P-02, M-01 | Q4 MLP, precision, and scratch cuts | LOCKED |
| T-01 | Decode projection geometry | TUNING |
## Starting point
RMS and packed MMV with paired/residual epilogues work independently.
## Scope
Implement the complete decode MLP: FP32 `h_mid[5120]` → zero-centered RMS → BF16 normalized → paired Q4 gate/up `[17408]` with separate FP32 accumulators → FP32 SiLU(gate)×up → BF16 SwiGLU scratch → Q4 down `[5120,17408]` → FP32 accumulator added to original `h_mid` → FP32 next residual. Add a CPU/BF16-control reference and plan/runtime binding.
## Out of scope
Materialized gate/up arrays, down fusion across its reduction, prefill GEMM, other layers, architecture experiments.
## Required interfaces
Concrete MLP plan binds three tensor identities, norm, scratch, residual views, and ordered stream; decode call performs no allocation.
## Required semantics
`silu(x)=x*sigmoid(x)` evaluated FP32; only product stores BF16. Original residual stays live through down add. Gate/up rows are independent despite paired ownership.
## Data representation
FP32 residual 5120; BF16 normalized 5120; BF16 SwiGLU 17408; Q4 V0 matrices/scales.
## Implementation constraints
Exactly three initial regions (RMS, paired projection/epilogue, down/residual); no opportunistic fusion.
## Tuning defaults
TASK-009 projection geometry.
## Expected files/modules
Runtime MLP plan/execution, paired epilogue kernel support, reference/integration tests.
## Tests required
### Unit tests
Binding/shape/layout errors, zero/extreme activations, scratch lifetime, repeated call/no allocation.
### Reference/numerical tests
Packed and BF16-control MLP against CPU reference with stage-local and final tolerances.
### Integration tests
Two consecutive MLP invocations with deterministic residuals and scratch reuse.
## Benchmark required
Report one representative decode MLP timing as diagnostic only.
## Acceptance criteria
- [ ] Exact V0 sequence and precision boundaries execute.
- [ ] Only BF16 SwiGLU is globally materialized for gate/up.
- [ ] Final FP32 residual matches reference tolerance.
- [ ] Steady-state call allocates nothing.
- [ ] Tests pass.
## Architecture blocker rule
On locked conflict stop with full required blocker report; do not change fusion/materialization.
## Completion report
### Result
DONE | BLOCKED
### Changes made
### Tests run
Exact commands/results.
### Benchmark results
Diagnostic timing/identity.
### Architecture blocker
None/full report.
### Follow-up observations
Concrete only.


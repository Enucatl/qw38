# TASK-008 — Core reference math and activation kernels

## Status
TODO
## Milestone
M3 — Core runtime and MLP
## Purpose
Create understandable CPU references and focused CUDA primitives for shared semantic operations before layer integration.
## Depends on
- TASK-007
## Normative references
- `docs/architecture/architecture-v0.md` — Precision policy; semantic graph; CUDA ownership
- `docs/architecture/model-semantics.md`
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| P-01–P-02 | FP32 residual/reductions/nonlinear math; BF16 normalized transport | LOCKED |
| G-01 | Node semantic invariants | LOCKED |
| T-01, T-03 | 256-thread norm and 128-thread preparation starting points | TUNING |
## Starting point
Runtime buffers/views are available; packed contraction is not.
## Scope
Implement independent CPU references and CUDA kernels/wrappers for BF16 embedding gather→FP32, hidden zero-centered RMS (`1+gamma`)→BF16, GDN multiplicative gated RMS→BF16, QK RMS, FP32 SiLU/sigmoid, and 64-coordinate partial RoPE using 32 FP32 inverse frequencies; add simple BF16 dense reference contraction and deterministic argmax helper.
## Out of scope
Packed Q4/Q8 CUDA, attention softmax, GDN recurrence/convolution, fused projection, sampling, generic activation library.
## Required interfaces
Shape-specific typed launch wrappers returning expected errors; pure CPU reference functions over spans; explicit epsilon/position inputs from semantic metadata.
## Required semantics
Reductions/phases/nonlinear evaluation are FP32; rounding occurs only at declared BF16 stores. Attention q/g splitting remains per head. RoPE transforms first 64 of 256 coordinates; remaining 192 pass unchanged; integer positions convert only for FP32 phase evaluation.
## Data representation
FP32 residual `[5120]`; BF16 normalized `[5120]`; embedding row `[5120]`; head-contiguous vectors.
## Implementation constraints
No hidden allocations or global state; optimized kernels compare to CPU references; maintain separate norm roles.
## Tuning defaults
One 256-thread block/token for hidden RMS; one block/head for head norms.
## Expected files/modules
`reference/` numerical primitives, `cuda/` activation kernels/wrappers, focused tests.
## Tests required
### Unit tests
Zero/nonzero vectors, gamma roles, BF16 rounding, large positions, RoPE unaffected suffix, invalid indices/shapes.
### Reference/numerical tests
CUDA versus double/FP32 reference with declared per-operation tolerances and adversarial magnitudes.
### Integration tests
Embedding→hidden RMS and QK norm→RoPE pipelines on deterministic inputs.
## Benchmark required
No.
## Acceptance criteria
- [ ] Every operation has an independent reference.
- [ ] Norm roles, per-head association, and partial RoPE are explicitly tested.
- [ ] FP32/BF16 boundaries match V0.
- [ ] CUDA results meet recorded tolerances.
## Architecture blocker rule
On locked conflict stop and report all required blocker fields; do not change precision or semantic roles.
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


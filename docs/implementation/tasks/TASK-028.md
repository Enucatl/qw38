# TASK-028 — EXP-E activation transport

## Status
TODO
## Milestone
M10 — Architecture-V0 experiments
## Purpose
Diagnose whether BF16 normalized/projection transport causes quality loss by comparing an otherwise identical FP32 SIMT path.
## Depends on
- TASK-027
## Normative references
- `docs/architecture/architecture-v0.md` — EXP-E; precision policy
- `docs/architecture/quantization-validation.md`
- `docs/architecture/performance-validation.md`
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| P-02 | BF16 transport selected; experiment may recommend amendment | LOCKED UNDER TEST |
| P-01, Q-01–Q-03 | FP32 arithmetic and identical weight codes/scales | LOCKED |
## Starting point
Frozen behavior failures/successes and production baseline are available.
## Scope
Implement explicit FP32 normalized/projection scratch and diagnostic SIMT contractions using identical quantized codes, scales, equations, accumulation, state, and scheduling semantics. Compare stage attribution and full quality; account for scratch traffic/capacity and measure diagnostic decode. If widening repairs quality, separately document required prefill arithmetic/tensor-core consequences before any change recommendation.
## Out of scope
Weight/state precision changes, simultaneous fusion, claiming FP32 SIMT diagnostic performance as production prefill, threshold relaxation.
## Required interfaces
Transport policy is explicit in artifact/runtime/result identity; BF16 production remains default.
## Required semantics
Only declared activation stores/loads widen; nonlinear/reductions already FP32 remain unchanged; weight decoded values/codes identical.
## Data representation
Candidate FP32 normalized/projection scratch with exact byte/lifetime accounting.
## Implementation constraints
Use diagnostic SIMT to isolate transport; do not silently disable tensor-core production path.
## Tuning defaults
Frozen suite/baseline protocol.
## Expected files/modules
Experimental transport path, attribution tests, quality/memory/performance report.
## Tests required
### Unit tests
Policy/buffer dtype selection and no unintended tensor changes.
### Reference/numerical tests
Stage comparisons proving only transport rounding differs.
### Integration tests
Frozen behavior gate and diagnostic matched decode; prefill consequence analysis with executable evidence where feasible.
## Benchmark required
Yes: diagnostic decode, scratch/memory; prefill impact must be explicitly characterized, not conflated.
## Acceptance criteria
- [ ] Candidate differs only at activation transport boundaries.
- [ ] Evidence states whether widening repairs any quality failure.
- [ ] Traffic/capacity and prefill tensor-core implications are complete.
- [ ] Recommendation follows EXP-E without architecture amendment.
## Architecture blocker rule
Candidate result is not blocker; unrelated locked conflict requires full blocker fields.
## Completion report
### Result
DONE | BLOCKED
### Changes made
### Tests run
Exact commands/results.
### Benchmark results
Quality/diagnostic performance/memory/consequence evidence.
### Architecture blocker
None/full report.
### Follow-up observations
Keep/change recommendation only.


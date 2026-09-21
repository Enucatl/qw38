# TASK-011 — GDN convolution and preparation

## Status
TODO
## Milestone
M4 — GDN execution
## Purpose
Implement and isolate the stateful GDN front half before recurrence.
## Depends on
- TASK-009
## Normative references
- `docs/architecture/architecture-v0.md` — GDN sequence; convolution/state layout
- `docs/architecture/model-semantics.md`
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| Q-03 | Convolution, a/b, A_log, dt_bias stay BF16 | LOCKED |
| P-01–P-02 | FP32 convolution/gates/qk prep and BF16 stored qkv/v/z | LOCKED |
| M-01 | Selected GDN scratch boundaries | LOCKED |
| T-03 | Initial preparation block geometry | TUNING |
## Starting point
Large Q4 and grouped small BF16 projections plus RMS are available.
## Scope
Implement independent reference math and decode CUDA for four-tap causal convolution + SiLU, circular raw-qkv history update, q/k head RMS normalization, mapping `value_head/3` to 16 key heads, and FP32 alpha/beta from BF16 a/b, A_log, dt_bias according to normative equations. Integrate GDN steps 1–5 while preserving stage outputs for diagnosis.
## Out of scope
Recurrence, output transform/projection, prefill parallel FIR, fusion across listed regions, no-RoPE substitution questions.
## Required interfaces
Concrete GDN-front plan binds qkv/z Q4, a/b BF16, tap-major convolution, time parameters, history/cursor, and typed scratch.
## Required semantics
History contains last three raw pre-convolution qkv values oldest→newest; current raw qkv enters history, never SiLU output. Four taps accumulate FP32 then SiLU FP32/store BF16. GDN has no RoPE. q/k and alpha/beta staging are FP32; v aliases convolved BF16 slice.
## Data representation
qkv BF16 `[10240]`; z BF16 `[48,128]`; history BF16 `[3,10240]`; taps BF16 `[4,10240]`; normalized q/k FP32 `[16,128]`; gates FP32 `[48]`.
## Implementation constraints
Convolution/update and preparation remain separate launches; no physical triplication of q/k.
## Tuning defaults
Channel-parallel convolution; one block/head preparation.
## Expected files/modules
Reference GDN front math, CUDA convolution/preparation kernels, runtime plan/tests.
## Tests required
### Unit tests
History wrap/reset, tap order, zero/nonzero a/b/time values, `value_head/3` boundaries, q/k zero norms.
### Reference/numerical tests
One and many consecutive steps versus reference; stage outputs compared separately.
### Integration tests
RMS→large/small projections→conv/history→preparation on deterministic fixture; snapshot/restore history continuation.
## Benchmark required
No.
## Acceptance criteria
- [ ] Convolution tap/history semantics pass multi-step tests.
- [ ] Gate equations and q/k normalization match reference in FP32 tolerance.
- [ ] Required scratch/precision boundaries are visible.
- [ ] No q/k persistent duplication or fusion drift.
## Architecture blocker rule
On locked conflict stop with full blocker report; do not alter convolution or preparation semantics.
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


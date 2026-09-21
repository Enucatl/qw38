# TASK-012 — GDN recurrence and continuation

## Status
TODO
## Milestone
M4 — GDN execution
## Purpose
Validate the architecture-critical recurrent update independently of the complete layer.
## Depends on
- TASK-011
## Normative references
- `docs/architecture/architecture-v0.md` — Chosen GDN ownership and recurrence equation
- `docs/architecture/model-semantics.md`
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| S-01 | Persistent GDN S is FP32 | LOCKED |
| S-02 | Physical `[value_head,value,key]`, warp/value ownership | LOCKED |
| P-01 | Prediction/update/readout and reductions FP32 | LOCKED |
## Starting point
Prepared q/k/alpha/beta/v arrays and session S storage exist.
## Scope
Implement a simple CPU reference and decode CUDA recurrence: one warp owns one 128-key row for one value coordinate; four warps/block own adjacent values; load old S once, compute decay/prediction/error/update/readout, store new S once, emit FP32 o. Add reset, multi-token continuation, and exact snapshot/restore tests.
## Out of scope
Alternate `[head,key,value]`, BF16 state, prefill 64-token loop, gated RMS, projection, fusion, tuning experiments.
## Required interfaces
Typed recurrence launcher over prepared arrays, FP32 state and output, layer/state offsets, ordered stream; no allocation.
## Required semantics
For every row: `D_k=alpha*S_old_k`; `p=sum D_k*khat_k`; `e=beta*(v_j-p)`; `S_new_k=D_k+khat_k*e`; `o_j=sum S_new_k*qhat_k/sqrt(128)`. All operations/reductions FP32 and update precedes readout.
## Data representation
Per layer S FP32 `[48,128,128]` in `[value_head,value,key]`; lane `l` owns keys `l,l+32,l+64,l+96`; o FP32 `[48,128]`.
## Implementation constraints
No atomics/cross-block reductions; no state transpose; q/k shared within block and indexed rather than duplicated.
## Tuning defaults
128-thread blocks/four value rows; ownership axis is locked, block size may later tune consistently.
## Expected files/modules
Reference recurrence, CUDA kernel/wrapper, continuation fixtures.
## Tests required
### Unit tests
Physical indexing, zero state, reset, invalid views, independent layers/heads.
### Reference/numerical tests
One step and 1/2/17/128 consecutive steps against CPU reference; adversarial alpha/beta; stage-level S and o tolerances.
### Integration tests
Run N steps, snapshot, restore in new session, continue, and compare with uninterrupted reference/execute path.
## Benchmark required
Diagnostic one-step timing/resource report only.
## Acceptance criteria
- [ ] One-step and multi-step S/o match reference tolerance.
- [ ] Reset and restored continuation are correct.
- [ ] Compiled resource report and diagnostic timing are recorded.
- [ ] Physical layout and FP32 persistence exactly match S-01/S-02.
## Architecture blocker rule
If S-01/S-02 prevents correctness, stop with the complete required blocker report; never transpose/narrow silently.
## Completion report
### Result
DONE | BLOCKED
### Changes made
### Tests run
Exact commands/results.
### Benchmark results
Diagnostic timing/resources.
### Architecture blocker
None/full report.
### Follow-up observations
Concrete only.


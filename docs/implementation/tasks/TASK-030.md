# TASK-030 — EXP-G common weight view

## Status
TODO
## Milestone
M10 — Architecture-V0 experiments
## Purpose
Determine whether a second prefill-oriented MLP view justifies its large artifact/capacity cost.
## Depends on
- TASK-029
## Normative references
- `docs/architecture/architecture-v0.md` — EXP-G; one-view decision
- `docs/architecture/performance-validation.md`
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| A-01 | One packed view selected; experiment may recommend narrow amendment | LOCKED UNDER TEST |
| A-02, Q-01, L-01 | Same logical codes/scales; candidate needs distinct physical layout | LOCKED |
## Starting point
Production common-view prefill and complete baseline exist; prior experiments are recorded.
## Scope
Design and implement one explicit prefill-oriented physical view for MLP weights only, losslessly rearranging identical Q4 codes/scales; add manifest/layout version and tensor bindings; compare compiler/artifact size, load/upload/cold-request cost, capacity feasibility, local MLP prefill, TTFT/full prefill/request, and behavior. Evaluate workloads where roughly 8.47 GiB additional language-MLP Q4 payload fits.
## Out of scope
Second views for all weights, new quantizer, decode changes, runtime repack, ignoring cold/capacity costs, changing unrelated prefill kernels.
## Required interfaces
Candidate artifact explicitly carries both view IDs and shared logical identity; runtime selection is mode/plan-specific and rejects missing/mismatched view.
## Required semantics
Decoded BF16 operands are identical to common view; only byte arrangement/load path differs.
## Data representation
New experimental layout with fully specified order/alignment/scales; exact incremental artifact/device bytes reported.
## Implementation constraints
Independent pack/unpack golden tests; include compiler, disk, upload, VRAM, and cold request accounting.
## Tuning defaults
Frozen prefill cases and measurement protocol.
## Expected files/modules
Experimental layout/packer/consumer/bindings, artifact/capacity/performance report.
## Tests required
### Unit tests
Golden bytes/version/binding/corruption and missing-view behavior.
### Reference/numerical tests
Lossless logical codes/scales and matching projection/MLP outputs.
### Integration tests
Behavior gate and matched prefill/request runs including cold path.
## Benchmark required
Yes: local MLP, prefill/TTFT/request, upload/cold, memory/capacity.
## Acceptance criteria
- [ ] Candidate contains identical quantized values and only MLP duplicate view.
- [ ] ~8.47 GiB estimate is replaced/confirmed by exact measured accounting.
- [ ] Warm and cold end-to-end benefit, quality, and capacity are reported.
- [ ] Recommendation follows EXP-G; no architecture amendment.
## Architecture blocker rule
Candidate infeasibility/regression is result; unrelated locked conflict requires full blocker report.
## Completion report
### Result
DONE | BLOCKED
### Changes made
### Tests run
Exact commands/results.
### Benchmark results
Local/warm/cold/memory/capacity identities/deltas.
### Architecture blocker
None/full report.
### Follow-up observations
Keep/change recommendation only.


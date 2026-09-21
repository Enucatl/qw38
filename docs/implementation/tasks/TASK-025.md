# TASK-025 — EXP-B vocabulary precision

## Status
TODO
## Milestone
M10 — Architecture-V0 experiments
## Purpose
Determine whether Q4G64 can replace Q8G32 for the output head without unacceptable language behavior.
## Depends on
- TASK-024
## Normative references
- `docs/architecture/architecture-v0.md` — EXP-B; behavior/performance validation
- `docs/architecture/quantization-validation.md`
- `docs/architecture/performance-validation.md`
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| Q-02 | Q8G32 lm_head selected; experiment may recommend amendment | LOCKED UNDER TEST |
| Q-01, P-01–P-02 | Q4 candidate recipe and runtime arithmetic | LOCKED |
## Starting point
EXP-A outcome is recorded and baseline harness/suite remain frozen.
## Scope
Compile an explicit Q4G64-head candidate while keeping all other weights/semantics fixed; use BF16 identity for head-only attribution; verify pack/consumer; run domain-wise NLL, KL/distribution, capability/generation, decode/head-local/request timing, and artifact/memory accounting. Recommend change only if quality passes and decode improves end to end.
## Out of scope
Changing main-weight policy, embeddings/tie semantics, state/activation precision, alternate group sizes, threshold relaxation.
## Required interfaces
Candidate artifact/policy ID and results explicitly label head storage; production Q8 path remains available.
## Required semantics
All 248320 logits FP32; only lm_head storage changes Q8G32→Q4G64; embedding remains BF16 and untied.
## Data representation
Existing V0 Q4 physical layout with distinct policy/artifact identity; exact head bytes/scales reported.
## Implementation constraints
Parameter MSE cannot decide; source inputs, masks, schedules and measurement conditions fixed.
## Tuning defaults
Frozen TASK-018/023 policy.
## Expected files/modules
Experimental compiler policy selection, head comparison runner/results/tests.
## Tests required
### Unit tests
Policy/binding/version selection and no accidental embedding/tensor changes.
### Reference/numerical tests
Q4 head logits versus independent reference/BF16 attribution.
### Integration tests
Full frozen behavioral and matching performance cases.
## Benchmark required
Yes: lm_head local plus affected decode/request end-to-end.
## Acceptance criteria
- [ ] Candidate differs only in head representation.
- [ ] Domain NLL/distribution/capability evidence passes before performance recommendation.
- [ ] Matched end-to-end delta, uncertainty, and memory are reported.
- [ ] Recommendation follows EXP-B and does not amend architecture.
## Architecture blocker rule
Candidate rejection is not a blocker; unrelated locked conflicts require full blocker report.
## Completion report
### Result
DONE | BLOCKED
### Changes made
### Tests run
Exact commands/results.
### Benchmark results
Quality/local/end-to-end/memory with identities.
### Architecture blocker
None/full report.
### Follow-up observations
Keep/change recommendation only.


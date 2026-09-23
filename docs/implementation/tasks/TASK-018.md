# TASK-018 — Behavioral correctness baseline

## Status
TODO
## Milestone
M7 — Behavioral correctness baseline
## Purpose
Freeze and implement the language-only correctness evidence needed before performance experiments can change selected hypotheses.
## Depends on
- TASK-017
## Normative references
- `docs/architecture/architecture-v0.md` — Behavior and performance validation
- `docs/architecture/quantization-validation.md`
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| Q-01–Q-03, P-01–P-02, S-01–S-02 | V0 under test versus BF16 control | LOCKED |
| Initial validation policy | NLL/distribution/capability budgets | TUNING |
## Starting point
Complete V0 decode and BF16 diagnostic control exist; production prefill does not.
## Scope
Freeze hashed tokenized inputs/loss masks and scoring rules before comparison; implement teacher-forced target-token NLL with FP32 log-softmax, output KL/distribution summaries, deterministic greedy generation, reset/continuation/snapshot tests, declared language/code/arithmetic-reasoning/long-context capability checks, and feasible long-context retrieval. Compare V0 to BF16 identity, counting each target once and resetting at document boundaries. Clearly record untested coverage.
## Out of scope
Performance tuning, threshold relaxation, calibration on evaluation data, MTP metrics, intermediate equality as the quality verdict, llama/GGUF as numerical authority.
## Required interfaces
Evaluation CLI/config produces machine-readable results bound to artifact, binary, input, mask, tokenizer, and policy hashes; language-only labels are mandatory.
## Required semantics
Default hypotheses: mean NLL delta ≤+0.03 nats/token, each declared slice ≤+0.06, no capability regression >2 percentage points beyond paired uncertainty. Greedy strings are inspected, not required token-identical.
## Data representation
FP32 logits/log-softmax; immutable hashed token IDs/masks; results include counts by domain/context and uncertainty method.
## Implementation constraints
Source/BF16 control remains attribution authority. Failures block quality acceptance and are not silently repaired by wider precision.
## Tuning defaults
Validation thresholds above are policy defaults, not architecture.
## Expected files/modules
Evaluation tooling/harness, frozen manifests (not copyrighted corpus payload if licensing forbids), correctness integration tests.
## Tests required
### Unit tests
NLL/KL math, masks/counting, hash binding, reset boundaries, deterministic argmax.
### Reference/numerical tests
Hand-computed logits and paired BF16/V0 fixtures.
### Integration tests
Run full declared suite or explicitly documented feasible subset; repeated-decode continuation and save/restore; long-context boundary cases.
## Benchmark required
No throughput benchmark.
## Acceptance criteria
- [ ] Suite/scoring/input identities are frozen before V0 comparison.
- [ ] BF16 and V0 results report NLL, distribution, generation, capability, continuation, and coverage.
- [ ] Language-only quality gate passes or task is BLOCKED with failing slices preserved.
- [ ] MTP is excluded and no thresholds are silently changed.
## Architecture blocker rule
If locked V0 cannot produce correct behavior, report full `ARCHITECTURE_BLOCKER`; a quality failure alone does not authorize EXP-A/B/C/E early.

## Quality failure recovery
Preserve the failing baseline and frozen evaluation suite. First distinguish
source/semantic implementation bugs from a failure of the selected quantization
or activation-transport hypothesis. Fix implementation bugs under the existing
contract. For a hypothesis failure, propose a narrow architecture amendment that
names the decision, failing evidence, diagnostic comparison, affected weight
families, artifact identity, and memory impact. Record the decision authority
and explicit acceptance in the ledger before changing a locked policy or task
order. Re-run the unchanged quality gate before accepting a replacement baseline
and resuming the ledger. Do not relax thresholds or run an experiment early.
## Completion report
### Result
DONE | BLOCKED
### Changes made
### Tests run
Exact commands/results.
### Benchmark results
Not required; report behavioral metrics.
### Architecture blocker
None/full report.
### Follow-up observations
Concrete failing/passing slices only.

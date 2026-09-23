# TASK-017 — Complete primary-language decode

## Status
TODO
## Milestone
M6 — Integrated language decode
## Purpose
Reach the first complete batch-one autoregressive language-model execution through all 64 layers.
## Depends on
- TASK-016
## Normative references
- `docs/architecture/architecture-v0.md` — Decode schedule; end-to-end architecture
- `docs/architecture/model-semantics.md`
- `docs/architecture/model-inventory.md`
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| G-01–G-02 | 130-instance primary-language graph and decode schedule | LOCKED |
| Q-01–Q-03, P-01–P-02, S-01–S-02, L-01, M-01 | Full V0 storage/execution policy | LOCKED |
## Starting point
Both layer styles execute through shared runtime; artifact/compiler include complete language bindings.
## Scope
Implement model plan validation and decode CLI/API: BF16 embedding gather/widen; 64 layers in three-GDN/one-attention pattern repeated 16 times, each with MLP; all GDN/history/KV state; final zero-centered norm; Q8G32 lm_head producing all 248320 FP32 logits; deterministic argmax default while retaining logits. Add deliberately slow repeated-decode prompt-state setup for correctness only and BF16 one-layer-upload diagnostic if full BF16 control does not fit VRAM.
## Out of scope
Production prefill, sampling policy, tokenizer payload in artifact, MTP execution, vision, batching, paging, CPU offload production path.
## Required interfaces
Concrete compiled-model plan, session capacity, `decode_token(token_id, position)` returning FP32 logits/view and argmax; CLI clearly labels primary-language scope and slow prompt setup.
## Required semantics
Exactly 64 language layers; MTP disabled; embedding and lm_head untied; each token commits state in order; populated length/position remain coherent; logits are FP32. Per-layer progress remains pending until all layers and output work complete; one global token position commits at that boundary. A failure after any layer mutation poisons the session until reset or valid snapshot restore; it cannot be retried at the same position as if earlier layers had not advanced.
## Data representation
Use artifact bindings and session schema; no runtime repack/full dequantization; active state byte formula remains language-only.
## Implementation constraints
Stable allocations, eager one-stream schedule, no optimized prefill prerequisite, no model-wide BF16 residency requirement for diagnostic control.
## Tuning defaults
Inherited decode geometries.
## Expected files/modules
Runtime model plan/scheduler, decode CLI, full-model smoke/integration fixtures.
## Tests required
### Unit tests
Graph count/order/binding validation, token/position/capacity errors, MTP disabled, logits shape/dtype.
### Reference/numerical tests
Small deterministic full-stack fixture and selected authoritative-checkpoint checkpoints versus an independent source-model oracle. BF16 engine control must first agree with source logits and selected residual/state checkpoints within declared tolerances before V0-versus-BF16 differences are attributed to quantization.
### Integration tests
Reset, multi-token decode, slow prompt setup, save/restore continuation, deterministic repeated greedy output; complete authoritative model produces finite logits.
Inject failure after an earlier layer advanced; verify execution and snapshots reject while poisoned, then compare continuation after reset/restore with a clean control. Inspect persistent bytes and host metadata together.
## Benchmark required
No performance gate; record smoke latency only as diagnostic.
## Acceptance criteria
- [ ] Complete language graph executes one and multiple tokens.
- [ ] All state families update/continue correctly and reset works.
- [ ] Final output is 248320 finite FP32 logits plus deterministic argmax.
- [ ] MTP stays retained but disabled and output is labeled language-only.
- [ ] No production-prefill shortcut is claimed.
- [ ] Pin source weights/config/tokenizer, source software revision, token IDs, precision settings, and hashes for initial and continued-token logits plus selected residual/state checkpoints.
- [ ] Budget the one-layer-resident BF16 diagnostic explicitly; report untested coverage and preserve exact equality only for deterministic replay of the same implementation/schedule.
## Architecture blocker rule
On locked conflict stop with full blocker report; do not omit/reorder layers or enable unresolved MTP.
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

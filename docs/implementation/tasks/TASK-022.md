# TASK-022 — End-to-end prefill and decode handoff

## Status
TODO
## Milestone
M8 — Production V0 prefill
## Purpose
Assemble the 64-layer 256-token-chunk production prefill schedule and prove exact state handoff into decode.
## Depends on
- TASK-020
- TASK-021
## Normative references
- `docs/architecture/architecture-v0.md` — Prefill schedule; validation
- `docs/architecture/evaluation-policy-v0.md` — EVAL-01; core rerun, schedule comparisons and 32768 retrieval extension
- `docs/architecture/model-semantics.md`
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| G-01–G-02, A-01, P-01–P-02, S-01–S-02, M-01 | Full semantic graph, one view, state/precision, separate prefill | LOCKED |
| T-02–T-03 | Chunk/tile geometries | TUNING |
## Starting point
Complete GDN and attention prefill layers work separately.
## Scope
Implement full-model layer-wise prefill over ordered ≤256-token input chunks, embedding, all 64 layers, final norm/head policies, generation mode final-position logits and evaluation mode requested positions. Verify arbitrary chunk partitions, absolute positions, GDN S/history and KV handoff, then generate tokens with decode. Compare full/chunked prefill and repeated decode on identical IDs; rerun the EVAL-01 core through production prefill and complete the frozen 32768 retrieval extension with paired BF16 evidence. Follow EVAL-01's distinction between exact replay of one schedule and behavioral acceptance across schedules.
## Out of scope
Dynamic chunk autotuning, batching, paged KV, MTP, duplicate views, WY recurrence, CUDA graphs.
## Required interfaces
`prefill(token_ids, mode/requested_positions)` mutates one session and returns requested FP32 logits; subsequent `decode_token` continues without conversion/reset.
## Required semantics
Within each chunk execute all tokens for one layer before next layer. First chunk starts zero; later chunks use exact incoming state. Padding never advances state/position. First generated token is prefill/TTFT output, not steady decode.
## Data representation
Same artifact and session ABI as decode; bounded token-major arena; generation materializes only final prompt logits, evaluation requested rows in bounded tiles.
## Implementation constraints
No per-token full decode loop as production implementation; no handoff repack; stable allocations.
## Tuning defaults
256-token chunks and inherited tile sizes.
## Expected files/modules
Full prefill scheduler/API/CLI integration and handoff/equivalence tests.
## Tests required
### Unit tests
Empty/invalid prompt, 1/255/256/257 and arbitrary partitions, position/capacity, mode/output selection.
### Reference/numerical tests
Full/chunked/repeated-decode logits and all persistent states at boundaries with declared accumulation-order tolerances.
### Integration tests
Authoritative model prompt→prefill→128 decode continuation; snapshot at chunk boundary/restore; complete EVAL-01 core through production prefill and mandatory 32768 retrieval with paired BF16 evidence.
## Benchmark required
No acceptance speed threshold; capture smoke timings only.
## Acceptance criteria
- [ ] 64-layer production prefill is layer-wise and not batched decode.
- [ ] Arbitrary partitions and short finals preserve exact continuation semantics.
- [ ] Prefill→decode needs no state conversion and passes logits/state/behavior checks.
- [ ] Generation/evaluation output policies are distinct and correct.
- [ ] EVAL-01 core, schedule comparisons and frozen 32768 retrieval extension pass; missing mandatory evidence blocks completion.
## Architecture blocker rule
On locked conflict stop with full blocker report; do not substitute another prefill algorithm.
## Completion report
### Result
DONE | BLOCKED
### Changes made
### Tests run
Exact commands/results.
### Benchmark results
Smoke only.
### Architecture blocker
None/full report.
### Follow-up observations
Concrete only.

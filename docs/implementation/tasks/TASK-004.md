# TASK-004 — `.qw38` reader, metadata, and corruption validation

## Status
TODO
## Milestone
M1 — Runtime artifact and compiler foundation
## Purpose
Make `.qw38` independently readable and safe before any CUDA consumer trusts its bytes.
## Depends on
- TASK-003
## Normative references
- `docs/architecture/architecture-v0.md` — Artifact boundary; semantic graph; precision policy; state/cache layout
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| A-01, A-02, I-06 | Artifact/backend and logical/physical boundaries | LOCKED |
| Q-01–Q-03, P-01–P-02, S-01–S-02, G-01–G-02, L-01, M-01 | V0 tensor, graph, precision, state, and layout metadata | LOCKED |
## Starting point
TASK-003 emits deterministic integrity-protected fixtures.
## Scope
- Parse/map header and manifest safely; validate versions, sizes, arithmetic, alignment, ordering/overlap, hashes, enum values, layouts, shapes, storage/scale relationships, shared bindings, source/config/tokenizer identities, compiler revision, precision policy, semantic scope, graph bindings, and state/scratch schema before device allocation.
- Expose immutable typed views whose ownership/lifetime is explicit.
- Add corruption mutation tests covering every validation family and a writer→reader round trip.
## Out of scope
CUDA allocation/upload, checkpoint compilation, quantization, executing graph nodes, accepting unknown required layouts, or repairing malformed files.
## Required interfaces
`std::expected<Artifact, Error>` open/parse API; immutable tensor/span lookup by stable logical identity; typed architecture/state metadata access.
## Required semantics
No payload view is exposed before complete structural and integrity validation. Unsupported container, quantizer, or layout versions reject distinctly. External-data failures never rely on assertions.
## Data representation
Borrowed views use `std::span<const std::byte>` tied to RAII mapping/file ownership. Hashes cover exactly the schema-declared regions.
## Implementation constraints
Overflow-safe validation precedes pointer formation/allocation. Errors include version, tensor/field, and offset where applicable.
## Tuning defaults
None.
## Expected files/modules
Host `format/` reader/validator and corruption/round-trip fixtures.
## Tests required
### Unit tests
- Truncation at every record boundary; bad magic/version/enum/hash; misalignment; overflow; overlap; invalid layout/quantizer pair; bad shapes/scales/shared bindings/state schema.
### Reference/numerical tests
- Independent digest verification.
### Integration tests
- TASK-003 multi-tensor fixture round-trips with identical metadata and span bytes.
## Benchmark required
No.
## Acceptance criteria
- [ ] Valid artifacts expose all mandated metadata and exact bytes.
- [ ] Corruption and unsupported versions fail before allocation/use.
- [ ] Mapping/view ownership is unambiguous and RAII-managed.
- [ ] State schema represents language-only coefficients correctly and does not include live state.
- [ ] All tests pass.
## Architecture blocker rule
On a locked conflict, stop with all required `ARCHITECTURE_BLOCKER` fields; do not substitute.
## Completion report
### Result
DONE | BLOCKED
### Changes made
### Tests run
Exact commands/results.
### Benchmark results
Not required.
### Architecture blocker
None or full report.
### Follow-up observations
Concrete only.


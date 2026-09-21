# TASK-002 — `.qw38` format constants and schema

## Status
TODO
## Milestone
M1 — Runtime artifact and compiler foundation
## Purpose
Define the explicit, compiler-independent V0 file ABI and typed in-memory schema before any bytes are emitted.
## Depends on
- TASK-001
## Normative references
- `docs/architecture/architecture-v0.md` — Artifact and offline compiler; precision policy; state/cache layout
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| A-01 | One CUDA/Blackwell-oriented `.qw38` physical artifact | LOCKED |
| A-02 | Logical quantizer identity is separate from physical-layout identity | LOCKED |
| I-06 | Physical bytes may be CUDA-specific while semantic descriptors remain portable | LOCKED |
| Q-01–Q-03, P-01–P-02, S-01–S-02, G-01–G-02, L-01, M-01 | Stored policy, graph/state/layout identities | LOCKED |
## Starting point
TASK-001 provides native targets and tests; no artifact types exist.
## Scope
- Define magic, container/manifest versions, explicit-width little-endian encodings, 64-bit offsets/lengths, 256-byte span alignment, enum values, and canonical identifiers for storage, logical quantizer, physical layout, semantic node, precision policy, and supported scope.
- Define schema records for tensors/shapes/mappings/spans/scales/shared bindings, source/config/tokenizer hashes, compiler revision, graph bindings, and state/scratch allocation descriptions.
- Provide overflow-safe size/alignment/range helpers and deterministic schema-level validation independent of file I/O.
## Out of scope
Writer/reader I/O, hashing implementation, checkpoint parsing, quantization, CUDA, raw struct serialization, live state, tokenizer payloads, or CUDA binaries.
## Required interfaces
Typed value records and encode/decode primitives over `std::span<std::byte>` returning `std::expected`; stable enum-to-wire mappings; no generic tensor framework.
## Required semantics
All wire integers are little-endian. Unknown required versions/enums reject. Logical and physical layout versions are independent. The schema describes weights and state allocation, never live/zero state.
## Data representation
Header points to a manifest/directory; payload/scale spans use 64-bit fields and 256-byte-aligned bases. Never persist compiler padding, native pointers, `sizeof(struct)` layouts, or host endianness.
## Implementation constraints
Use explicit byte operations, bounds checks, `std::endian`/`std::byteswap` as needed, and typed errors with field/offset context.
## Tuning defaults
None.
## Expected files/modules
Host-only `format/` headers/sources and focused format-schema tests within TASK-001's tree.
## Tests required
### Unit tests
- Golden byte encodings, endian cases, enum/version rejection, overflow, alignment, invalid shape/span relationships, shared-binding rules.
### Reference/numerical tests
- None.
### Integration tests
- Schema records for representative embedding, Q4 matrix, Q8 head, GDN S, convolution history, and KV cache validate together.
## Benchmark required
No.
## Acceptance criteria
- [ ] Every required V0 metadata concept has an explicit wire representation.
- [ ] Logical quantizer and physical layout IDs cannot be conflated.
- [ ] ABI bytes do not depend on C++ object layout.
- [ ] Malformed schema cases return typed errors.
- [ ] Format tests pass.
## Architecture blocker rule
On a locked conflict, report `ARCHITECTURE_BLOCKER` with Decision ID, Attempted implementation, Observed problem, Evidence, architectural rationale, Smallest plausible alternative, and Affected downstream tasks; do not substitute.
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


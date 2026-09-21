# TASK-003 — `.qw38` writer and integrity records

## Status
TODO
## Milestone
M1 — Runtime artifact and compiler foundation
## Purpose
Emit deterministic, aligned `.qw38` artifacts with independently verifiable integrity metadata.
## Depends on
- TASK-002
## Normative references
- `docs/architecture/architecture-v0.md` — Selected artifact boundary; compiler pipeline
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| A-01, A-02, I-06 | Single CUDA artifact and separated logical/physical identities | LOCKED |
| L-01 | Named V0 dense layout versions | LOCKED |
## Starting point
TASK-002 defines validated wire/schema primitives.
## Scope
- Implement host-only streaming writer with deterministic ordering, 256-byte padding, separate payload/scale spans, 64-bit offset calculation, and atomic publish via temporary output plus successful close/rename.
- Compute SHA-256 for the manifest and every declared payload/scale span and encode the records without self-referential ambiguity.
- Reject incomplete, duplicate, overlapping, misaligned, overflowing, or inconsistent inputs before publication.
## Out of scope
Reader, checkpoint compiler, quantization math, CUDA upload, tokenizer embedding, zero state, generic archive support.
## Required interfaces
Fallible writer/builder APIs accept immutable schema plus streamed named byte spans and return typed errors; finalization returns artifact identity/path metadata.
## Required semantics
The same logical inputs and declared compiler revision produce byte-identical output. Failed finalization leaves no valid-looking destination artifact.
## Data representation
Little-endian explicit ABI; zero-filled alignment padding; separate aligned code/payload and scale bases; SHA-256 digests.
## Implementation constraints
RAII file ownership, bounded memory, no full-model buffering, no raw C++ struct dumps, actionable tensor/span/offset errors.
## Tuning defaults
None.
## Expected files/modules
Host `format/` writer and SHA-256 utility (small justified dependency permitted only with rationale), writer tests/fixtures.
## Tests required
### Unit tests
- Alignment/offset/golden bytes, deterministic repeat, empty/duplicate/missing span, overflow, write/finalize failure cleanup.
### Reference/numerical tests
- Independent SHA-256 comparison for known vectors and emitted spans.
### Integration tests
- Emit a multi-tensor fixture containing BF16, Q4 code+scale, Q8 code+scale, shared bindings, and state schema.
## Benchmark required
No.
## Acceptance criteria
- [ ] Writer emits deterministic valid bytes and all integrity records.
- [ ] It streams payloads with bounded memory.
- [ ] All spans are correctly aligned and nonoverlapping.
- [ ] Failure cannot publish a partial artifact as valid.
- [ ] Required tests pass.
## Architecture blocker rule
On a locked conflict, stop with the full required `ARCHITECTURE_BLOCKER` fields; do not substitute.
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


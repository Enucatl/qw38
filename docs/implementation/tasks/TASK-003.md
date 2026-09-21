# TASK-003 — `.qw38` writer and integrity records

## Status
DONE
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
- [x] Writer emits deterministic valid bytes and all integrity records.
- [x] It streams payloads with bounded memory.
- [x] All spans are correctly aligned and nonoverlapping.
- [x] Failure cannot publish a partial artifact as valid.
- [x] Required tests pass.
## Architecture blocker rule
On a locked conflict, stop with the full required `ARCHITECTURE_BLOCKER` fields; do not substitute.
## Completion report
### Result
DONE. Independent verification PASS (Debug/Release ctest 8/8).
### Changes made
- Added host-only `ArtifactWriter` / `ArtifactBuilder` under `src/format/`: RAII POSIX fd, temp file + `fsync` + rename publish, 256-byte zero padding, 64-bit sequential payload-then-scale placement, chunked named-span streaming (no full-model buffer).
- Local FIPS 180-4 SHA-256 (`sha256.hpp`/`sha256.cpp`) with no libcrypto; rationale is a small, vector-tested digest vs a new host dependency.
- Integrity records: SHA-256 of every declared payload/scale span (aliases reuse the owner digest/region) plus `Sha256Manifest` last. The manifest digest covers the encoded prefix that excludes that last 56-byte record, so the hash is not self-referential.
- Rejects empty/duplicate/missing/incomplete spans, misaligned or overlapping input spans, length/layout mismatch, overflow, and nonempty input integrity; failed finalize/destructor unlinks the temp file and never replaces an existing destination.
- Tests: `format_writer` (golden alignment/offsets, deterministic repeat, empty/duplicate/missing/incomplete, overflow, cleanup), `format_sha256` (NIST vectors + emitted span), `format_writer_integration` (BF16, Q4 code+scale, Q8 code+scale, MTP shared binding, GDN/conv/KV state schema).
### Tests run
Debug:

```text
docker run --gpus all --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 \
  bash -lc 'cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug && cmake --build build/debug && ctest --test-dir build/debug --output-on-failure'
```

Result: 8/8 tests passed (`host_expected_smoke`, `format_schema`, `format_schema_integration`, `format_writer` 0.03s, `format_sha256` 0.02s, `format_writer_integration` 0.01s, `cuda_runtime_smoke` 0.23s, `cuda_sm120_cubin`).

Release:

```text
docker run --gpus all --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 \
  bash -lc 'cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release && cmake --build build/release && ctest --test-dir build/release --output-on-failure'
```

Result: 8/8 tests passed (`host_expected_smoke`, `format_schema`, `format_schema_integration`, `format_writer` 0.03s, `format_sha256` 0.01s, `format_writer_integration` 0.01s, `cuda_runtime_smoke` 0.23s, `cuda_sm120_cubin`).
### Benchmark results
Not required.
### Architecture blocker
None.
### Follow-up observations
- TASK-004 should hash `Sha256Manifest.region` as the encoded-manifest prefix (`manifest_length - 56`), not the full directory including the trailing manifest-digest record.
- Input schemas must leave `integrity` empty; the writer owns those records. Unplaced payload/scale offsets (0,0) are assigned; nonzero caller placements are validated then replaced by the deterministic layout.


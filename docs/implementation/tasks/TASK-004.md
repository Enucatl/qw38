# TASK-004 — `.qw38` reader, metadata, and corruption validation

## Status
DONE
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
- [x] Valid artifacts expose all mandated metadata and exact bytes.
- [x] Corruption and unsupported versions fail before allocation/use.
- [x] Mapping/view ownership is unambiguous and RAII-managed.
- [x] State schema represents language-only coefficients correctly and does not include live state.
- [x] All tests pass.
## Architecture blocker rule
On a locked conflict, stop with all required `ARCHITECTURE_BLOCKER` fields; do not substitute.
## Completion report
### Result
DONE. Independent verification PASS (Debug/Release ctest 11/11).
### Changes made
- Host-only `Artifact` under `src/format/reader.hpp` / `reader.cpp`: `open(path)` memory-maps with RAII `mmap`/`munmap` + fd; `parse(vector|span)` owns a byte buffer. Payload/scale views are `std::span<std::byte const>` whose lifetime is the `Artifact`.
- `std::expected<Artifact, FormatError>` is returned only after header decode, overflow-safe range checks, exact file-size match, schema validation (versions, enums, layouts, shapes, storage/scale pairs, shared/graph bindings, compiler revision, precision policy, semantic scope, state/scratch schema including live-state rejection), unique-span ordering/overlap vs header+manifest, required integrity coverage, and SHA-256 of every schema-declared region.
- Manifest digest covers the encoded prefix `manifest_length - 56` (excludes the trailing digest record), matching the TASK-003 writer.
- Lookup by stable logical name or tensor id; typed accessors for compiler, precision, scope, graph bindings, and state/scratch schema. Unknown names return `TensorNotFound`; digest mismatch is `IntegrityDigestMismatch`; missing records are `MissingIntegrity`.
- Tests: truncation at every header/span/manifest record boundary; bad magic/version/enum/hash; misalignment, overflow, overlap; invalid layout/quantizer and storage/quantizer pairs; bad shapes/scales/shared bindings/state; independent digest verification; TASK-003 multi-tensor writer→reader round trip with identical metadata and span bytes; corruption mutations for every validation family.
### Tests run
Debug:

```text
docker run --gpus all --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 \
  bash -lc 'cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug && cmake --build build/debug && ctest --test-dir build/debug --output-on-failure'
```

Result: 11/11 tests passed (`host_expected_smoke`, `format_schema`, `format_schema_integration`, `format_writer` 0.03s, `format_sha256` 0.02s, `format_writer_integration` 0.01s, `format_reader` 0.03s, `format_reader_digest` 0.01s, `format_reader_integration` 0.01s, `cuda_runtime_smoke` 0.23s, `cuda_sm120_cubin`).

Release:

```text
docker run --gpus all --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 \
  bash -lc 'cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release && cmake --build build/release && ctest --test-dir build/release --output-on-failure'
```

Result: 11/11 tests passed (`host_expected_smoke`, `format_schema`, `format_schema_integration`, `format_writer` 0.03s, `format_sha256` 0.01s, `format_writer_integration` 0.01s, `format_reader` 0.03s, `format_reader_digest` 0.01s, `format_reader_integration` 0.01s, `cuda_runtime_smoke` 0.23s, `cuda_sm120_cubin`).
### Benchmark results
Not required.
### Architecture blocker
None.
### Follow-up observations
- POSIX fd RAII is duplicated in `writer.cpp` and `reader.cpp`; they are intentionally separate host I/O units. A shared helper would be a later cleanup, not a TASK-004 contract.


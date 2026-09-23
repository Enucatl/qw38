# TASK-002 — `.qw38` format constants and schema

## Status
DONE
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
- [x] Every required V0 metadata concept has an explicit wire representation.
- [x] Logical quantizer and physical layout IDs cannot be conflated.
- [x] ABI bytes do not depend on C++ object layout.
- [x] Malformed schema cases return typed errors.
- [x] Format tests pass.
## Architecture blocker rule
On a locked conflict, report `ARCHITECTURE_BLOCKER` with Decision ID, Attempted implementation, Observed problem, Evidence, architectural rationale, Smallest plausible alternative, and Affected downstream tasks; do not substitute.
## Completion report
### Result
DONE — independent verification PASSED (2026-09-21): all acceptance criteria met via Debug/Release `format_schema` and `format_schema_integration` tests (2/2 each); verified diff unchanged at delivery.
### Changes made
- Added host-only `qw38_format` static library under `src/format/`: little-endian `ByteWriter`/`ByteReader` over `std::span<std::byte>`, overflow-safe size/alignment/range helpers, and typed `FormatError` with field/offset context.
- Defined magic `QW38FMT\0`, container/manifest version 1, 64-byte header (manifest offset/length), 256-byte span alignment, and disjoint wire enums for storage class, logical quantizer (`q4g64_v0`/`q8g32_v0`/`none`), physical layout (`cuda_q4g64_v0`, `cuda_q8g32_v0`, BF16 layouts, GDN-S/conv/KV state layouts), semantic nodes, precision policy, and supported scope.
- Schema records cover tensors/shapes/mappings/payload+scale spans, legacy-compatible digest record kinds, compiler revision, precision-policy bindings, graph bindings, shared bindings (untied embed/`lm_head`; MTP aliases), integrity records, and state/scratch allocation descriptions (no live state). Current code standards permit only the small manifest digest; never compute payload/scale digests.
- Encode/decode is explicit byte ops (`std::endian`/`std::byteswap`); validation is deterministic and I/O-free. Logical quantizer and physical layout are separate fields with disjoint IDs; mismatched pairs are rejected.
- Wired `qw38_format` into `src/CMakeLists.txt` and added `format_schema` plus `format_schema_integration` CTest targets.
### Tests run
Debug configure/build/test:

```text
docker run --gpus all --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 \
  bash -lc 'cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug && cmake --build build/debug && ctest --test-dir build/debug --output-on-failure'
```

Result: 5/5 tests passed (`host_expected_smoke`, `format_schema`, `format_schema_integration`, `cuda_runtime_smoke` 0.23s, `cuda_sm120_cubin`).

Release configure/build/test:

```text
docker run --gpus all --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 \
  bash -lc 'cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release && cmake --build build/release && ctest --test-dir build/release --output-on-failure'
```

Result: 5/5 tests passed (`host_expected_smoke`, `format_schema`, `format_schema_integration`, `cuda_runtime_smoke` 0.24s, `cuda_sm120_cubin`).

Unit coverage: golden LE header/integers, swapped-endian version rejection, unknown enum, unsupported version, bad magic, Q4/Q8/BF16 pair rules, overflow, 256-byte alignment, invalid shape/span/scale relationships, shared-binding (untied embed/head, identical spans, self-bind, MTP alias rules), truncated input, layout-independent roundtrip.

Integration: representative embedding `[248320,5120]` BF16, Q4 MLP down `[5120,17408]`, Q8 head `[248320,5120]`, GDN S 150,994,944 B, conv history 2,949,120 B, and KV at T=4096 (268,435,456 B) validate and roundtrip together (`wire_bytes=1335`).
### Benchmark results
Not required.
### Architecture blocker
None.
### Follow-up observations
- Schema validation encodes sitting Qwen3.8 language state coefficients (48 GDN layers, 16 KV layers, `[48,128,128]` HVK S, `[3,10240]` conv, `65536*T` KV). A different SKU would need an artifact-versioned geometry change; none is specified for V0.
- Integrity records are slots only (kind, region, 32-byte digest). Digest computation remains TASK-003.

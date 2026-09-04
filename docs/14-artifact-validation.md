# 14. Artifact identity and tensor admission

[Previous](13-allocation-ledger.md) · [Index](README.md)

The model file is executable input, so Quartz admits it as a versioned binary
contract rather than accepting any file that happens to parse as GGUF. This
implements ledger tasks PIN-001, MDL-001, and MDL-002.

Feature contracts authenticate models, datasets, containers, generated
evidence, and other external artifacts at their real admission boundaries.
Source-file hashes are release provenance rather than feature contracts. Use
`make release-provenance` to emit Git, build, container, and selected evidence
identity at release time. Source edits are validated through behavior, builds,
and semantic fixtures.

## Admission sequence

`Engine::open` in [`src/engine.cpp`](../src/engine.cpp) checks the canonical byte
size, parses the GGUF v3 descriptors, validates the Qwen tensor contract, hashes
the complete file, and finally retains a private read-only mapping. A failed
check returns an explicit status before an Engine exists. The mapping owner is
move-only and releases `mmap` and its file descriptor together.

The parser in [`src/model.cpp`](../src/model.cpp) bounds counts and strings,
checks little-endian descriptor reads, rejects unknown tensor encodings, detects
offset overflow, and proves that each quantized payload fits before the next
tensor. Storage bytes are derived from the admitted GGML block formats:

| Type | Elements/block | Bytes/block |
|---|---:|---:|
| F32 | 1 | 4 |
| Q8_0 | 32 | 34 |
| Q4_K | 256 | 144 |
| Q6_K | 256 | 210 |

These are layout facts used by the loader, not accuracy or speed claims.

## Exact semantic inventory

The checked-in [`pins/tensor_inventory.json`](../pins/tensor_inventory.json) is
**Measured** from the pinned artifact. It contains every tensor name, semantic
role, shape, GGML type, relative and absolute byte range, storage/padded span,
and payload SHA-256. Its observed contract is:

- 851 unique tensors and no unassigned roles;
- 48 GDN layers with 14 tensors each;
- 16 full-attention layers with 11 tensors each;
- three global embedding/final-norm/output tensors;
- 353 F32, 288 Q8_0, 193 Q4_K, and 17 Q6_K tensors;
- 18,962,876,416 tensor payload bytes and no inter-tensor padding.

The 10,994,016 bytes before tensor data contain the GGUF header, metadata,
tokenizer arrays, tensor descriptors, and final header alignment. Whole-file
identity still uses the artifact SHA-256; per-tensor hashes do not replace it.

Inventory generation first runs full `Engine::open` validation, then maps the
file and hashes exact payloads. The measured run on 2026-08-29 took 1m56.997s.
This diagnostic cost is retained as evidence and is not on the token execution
path. See the corresponding UTC entries and remaining boundaries in
[`implementation_ledger.md`](../implementation_ledger.md).

## Failure boundaries

A valid GGUF container with an extra tensor, different layer shape, altered
quantization type, duplicate/missing name, misaligned range, or changed byte is
incompatible. There is intentionally no permissive fallback. CUDA repacking and
resident-weight preparation remain separate unfinished MDL-001 work; parsing and
mapping alone are not evidence that inference is implemented.

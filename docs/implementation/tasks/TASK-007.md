# TASK-007 — CUDA runtime ownership and session storage

## Status
DONE
## Milestone
M3 — Core runtime and MLP
## Purpose
Provide the narrow typed CUDA boundary, immutable model upload, session-owned persistent state, and reusable stable scratch needed by all kernels.
## Depends on
- TASK-004
## Normative references
- `docs/architecture/architecture-v0.md` — Materialization/memory hierarchy; state/cache layout
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| P-01–P-02 | Runtime storage/working precision | LOCKED |
| S-01–S-02 | FP32 GDN S and physical state ABI | LOCKED |
| M-01 | Persistent/materialized scratch obligations | LOCKED |
| T-02 | Initial 256-token chunk sizing informs arena capacity | TUNING |
## Starting point
Validated artifacts and state schemas exist; no CUDA ownership layer/runtime exists.
## Scope
Implement typed CUDA error translation; RAII device buffers, stream/events, and artifact upload; immutable model/tensor views; `Runtime`/`Session`-equivalent concrete types; zero/reset/save/restore for FP32 S, BF16 `[3,10240]` convolution history/cursor, and BF16 K/V `[kv_head,capacity,256]`; two FP32 residual buffers; fixed-address liveness-planned scratch; capacity/populated-length checks; one ordered eager stream.
## Out of scope
Kernels beyond test fills/copies, paging, eviction, CPU offload, multi-stream overlap, CUDA graphs, generic tensors/operators, MTP cache.
## Required interfaces
Fallible model load/upload and session creation/reset/snapshot/restore APIs; typed borrowed views carry pointer, dtype, logical extent, physical layout, and storage class.
## Required semantics
State is per-session, zero initially, survives calls, and is never shared. State size is `153,944,064 + 65,536*T` bytes for language-only persistent state. Allocation fails explicitly if requested capacity does not fit.
## Data representation
S FP32 `[layer,value_head,value,key]`; history BF16 `[layer,3,10240]`; K/V BF16 `[attention_layer,kv_head,capacity,256]`; stable device addresses for session lifetime.
## Implementation constraints
No per-token allocation/free; CUDA statuses translated at source; all owned resources RAII; host orchestration stays ordinary C++23.
## Tuning defaults
Single ordered stream; arena provision for 256 tokens.
## Expected files/modules
`runtime/` concrete model/session/arena and `cuda/` thin resources/upload wrappers.
## Tests required
### Unit tests
Size/overflow/liveness planning, move/destruction/error translation, capacity/populated validation.
### Reference/numerical tests
Pattern fill/copy verifies physical state indexing.
### Integration tests
Upload a fixture; create two isolated sessions; reset/snapshot/restore exact bytes; prove stable addresses and no hot-path allocations with instrumentation.
## Benchmark required
No.
## Acceptance criteria
- [x] All CUDA/model/session resources have explicit RAII ownership.
- [x] State sizes/layouts match V0 and sessions are isolated.
- [x] Scratch/residual addresses remain stable and reuse follows lifetimes.
- [x] Malformed artifacts reject before device allocation.
- [x] Tests pass on RTX 5090.
## Architecture blocker rule
On locked conflict stop with full required `ARCHITECTURE_BLOCKER`; do not alter state precision/layout or materialization.
## Completion report
### Result
DONE
### Changes made
- Thin CUDA RAII/error boundary in `cuda/`: typed status translation (`std::expected`), `DeviceBuffer`/`Stream`/`Event` ownership, H2D upload, D2H/D2D copy, pattern-fill kernel, and malloc/free instrumentation.
- Runtime types in `src/runtime/`: immutable uploaded `Model` tensor views; `Runtime` (one ordered eager stream); per-session `Session` owning FP32 GDN S `[layer,value_head,value,key]`, BF16 conv history `[layer,3,10240]` plus host cursor, BF16 K/V `[layer,component,kv_head,capacity,256]`, two FP32 residual buffers, and one liveness-planned scratch arena.
- Persistent language-only size `153,944,064 + 65,536*T` with overflow-checked capacity; populated length distinct from capacity; state zeroed at create, never shared.
- Arena T-02 default is 256 tokens; GDN/attention are exclusive-group packed; mixer/SwiGLU/logits reuse after last consumer; normalized scratch stays disjoint while live across mixer.
- Fallible load/upload/session APIs; malformed artifacts fail at `Artifact::open` with unchanged device malloc count; no per-token `cudaMalloc`/`cudaFree`.
- Tests: `runtime_plan` (size/overflow/liveness), `runtime_raii` (move/destruction/error translation), `runtime_state_index` (pattern fill/copy indexing), `runtime_session_integration` (fixture upload, two sessions, reset/snapshot/restore, stable addresses, hot-path allocation counter).
### Tests run
Debug:

```text
docker run --gpus all --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 \
  bash -lc 'cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug && cmake --build build/debug && ctest --test-dir build/debug --output-on-failure'
```

Result: 22/22 tests passed (`host_expected_smoke`, `format_schema`, `format_schema_integration`, `format_writer` 0.04s, `format_sha256` 0.02s, `format_writer_integration` 0.01s, `format_reader` 0.03s, `format_reader_digest` 0.01s, `format_reader_integration` 0.01s, `cuda_runtime_smoke` 0.23s, `cuda_sm120_cubin`, `compiler_identity` 0.02s, `compiler_transform`, `compiler_integration` 0.23s, `quantizer`, `pack_layout`, `quant_reference`, `quant_compiler_integration` 0.91s, `runtime_plan`, `runtime_raii` 0.20s, `runtime_state_index` 0.19s, `runtime_session_integration` 0.34s).

Release:

```text
docker run --gpus all --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 \
  bash -lc 'cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release && cmake --build build/release && ctest --test-dir build/release --output-on-failure'
```

Result: 22/22 tests passed (`host_expected_smoke`, `format_schema`, `format_schema_integration`, `format_writer` 0.02s, `format_sha256` 0.01s, `format_writer_integration` 0.01s, `format_reader` 0.03s, `format_reader_digest` 0.01s, `format_reader_integration` 0.01s, `cuda_runtime_smoke` 0.23s, `cuda_sm120_cubin`, `compiler_identity`, `compiler_transform`, `compiler_integration` 0.07s, `quantizer`, `pack_layout`, `quant_reference`, `quant_compiler_integration` 0.75s, `runtime_plan`, `runtime_raii` 0.17s, `runtime_state_index` 0.18s, `runtime_session_integration` 0.34s).
### Benchmark results
Not required.
### Architecture blocker
None.
### Follow-up observations
- Convolution cursors are session-owned host `uint32[48]` metadata included in snapshot/restore; they are not part of the 153,944,064-byte persistent VRAM formula. Later mixer kernels can take them as launch arguments.
- Artifact scratch records stay decode-sized (one token); the runtime scales the arena by the T-02 256-token default rather than treating those schema bytes as the VRAM footprint.
- Sessions share the `Runtime`'s single ordered stream; persistent state, residuals, and scratch buffers are not shared.



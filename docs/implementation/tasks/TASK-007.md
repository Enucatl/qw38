# TASK-007 — CUDA runtime ownership and session storage

## Status
TODO
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
- [ ] All CUDA/model/session resources have explicit RAII ownership.
- [ ] State sizes/layouts match V0 and sessions are isolated.
- [ ] Scratch/residual addresses remain stable and reuse follows lifetimes.
- [ ] Malformed artifacts reject before device allocation.
- [ ] Tests pass on RTX 5090.
## Architecture blocker rule
On locked conflict stop with full required `ARCHITECTURE_BLOCKER`; do not alter state precision/layout or materialization.
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


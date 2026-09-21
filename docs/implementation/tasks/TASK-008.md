# TASK-008 — Core reference math and activation kernels

## Status
DONE
## Milestone
M3 — Core runtime and MLP
## Purpose
Create understandable CPU references and focused CUDA primitives for shared semantic operations before layer integration.
## Depends on
- TASK-007
## Normative references
- `docs/architecture/architecture-v0.md` — Precision policy; semantic graph; CUDA ownership
- `docs/architecture/model-semantics.md`
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| P-01–P-02 | FP32 residual/reductions/nonlinear math; BF16 normalized transport | LOCKED |
| G-01 | Node semantic invariants | LOCKED |
| T-01, T-03 | 256-thread norm and 128-thread preparation starting points | TUNING |
## Starting point
Runtime buffers/views are available; packed contraction is not.
## Scope
Implement independent CPU references and CUDA kernels/wrappers for BF16 embedding gather→FP32, hidden zero-centered RMS (`1+gamma`)→BF16, GDN multiplicative gated RMS→BF16, QK RMS, FP32 SiLU/sigmoid, and 64-coordinate partial RoPE using 32 FP32 inverse frequencies; add simple BF16 dense reference contraction and deterministic argmax helper.
## Out of scope
Packed Q4/Q8 CUDA, attention softmax, GDN recurrence/convolution, fused projection, sampling, generic activation library.
## Required interfaces
Shape-specific typed launch wrappers returning expected errors; pure CPU reference functions over spans; explicit epsilon/position inputs from semantic metadata.
## Required semantics
Reductions/phases/nonlinear evaluation are FP32; rounding occurs only at declared BF16 stores. Attention q/g splitting remains per head. RoPE transforms first 64 of 256 coordinates; remaining 192 pass unchanged; integer positions convert only for FP32 phase evaluation.
## Data representation
FP32 residual `[5120]`; BF16 normalized `[5120]`; embedding row `[5120]`; head-contiguous vectors.
## Implementation constraints
No hidden allocations or global state; optimized kernels compare to CPU references; maintain separate norm roles.
## Tuning defaults
One 256-thread block/token for hidden RMS; one block/head for head norms.
## Expected files/modules
`reference/` numerical primitives, `cuda/` activation kernels/wrappers, focused tests.
## Tests required
### Unit tests
Zero/nonzero vectors, gamma roles, BF16 rounding, large positions, RoPE unaffected suffix, invalid indices/shapes.
### Reference/numerical tests
CUDA versus double/FP32 reference with declared per-operation tolerances and adversarial magnitudes.
### Integration tests
Embedding→hidden RMS and QK norm→RoPE pipelines on deterministic inputs.
## Benchmark required
No.
## Acceptance criteria
- [x] Every operation has an independent reference.
- [x] Norm roles, per-head association, and partial RoPE are explicitly tested.
- [x] FP32/BF16 boundaries match V0.
- [x] CUDA results meet recorded tolerances.
## Architecture blocker rule
On locked conflict stop and report all required blocker fields; do not change precision or semantic roles.
## Completion report
### Result
DONE
### Changes made
- CPU references in `src/reference/`: span APIs returning `std::expected`, explicit `eps`/`position`/`inv_freq`, FP32 math with declared BF16 stores, plus independent double golds. Operations: BF16 embed gather→FP32 `[5120]`; hidden zero-centered RMS `(1+γ)`→BF16; QK RMS `(1+γ)` on `[256]`; GDN multiplicative gated RMS `(γ ⊙ o/RMS(o)) ⊙ SiLU(z)`→BF16 `[128]`; FP32 SiLU/sigmoid; partial RoPE on the first 64 of 256 coords using 32 FP32 `ω_j`; row-major BF16 dense GEMV with FP32 accumulation; deterministic argmax (lowest index on ties).
- CUDA kernels/wrappers in `cuda/activation.{hpp,cu}`: shape-specific launches, no device allocations, T-01 256-thread hidden RMS (one block/token), T-03 128-thread head norms (one block/head). Integer positions convert only at FP32 phase evaluation. Host checks reject invalid token ids, empty streams, non-positive `eps`, and negative positions.
- Declared tolerances in `qw38::reference::tol` (embed exact; RMS 8e-3; gated RMS 1.6e-2; SiLU/sigmoid 2e-5; small-pos RoPE 8e-3; large-pos RoPE 5e-2).
- Tests: `reference_math_unit` (zero/nonzero, `1+γ` vs multiplicative γ, BF16 RNE store, large positions, RoPE suffix, invalid shapes/indices); `activation_reference` (CUDA vs FP32/double, adversarial magnitudes); `activation_integration` (embed→hidden RMS; 24-head QK RMS→RoPE).
- CMake: `qw38_reference`, `activation.cu` on `qw38_cuda`, three ctest targets.
### Tests run
Debug:

```text
docker run --gpus all --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 \
  bash -lc 'cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug && cmake --build build/debug && ctest --test-dir build/debug --output-on-failure'
```

Result: 25/25 tests passed (`host_expected_smoke`, `format_schema`, `format_schema_integration`, `format_writer` 0.03s, `format_sha256` 0.02s, `format_writer_integration` 0.01s, `format_reader` 0.03s, `format_reader_digest` 0.01s, `format_reader_integration` 0.01s, `cuda_runtime_smoke` 0.23s, `cuda_sm120_cubin`, `compiler_identity` 0.02s, `compiler_transform`, `compiler_integration` 0.23s, `quantizer`, `pack_layout`, `quant_reference`, `quant_compiler_integration` 0.91s, `runtime_plan`, `runtime_raii` 0.17s, `runtime_state_index` 0.19s, `runtime_session_integration` 0.34s, `reference_math_unit`, `activation_reference` 0.17s, `activation_integration` 0.17s).

Release:

```text
docker run --gpus all --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 \
  bash -lc 'cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release && cmake --build build/release && ctest --test-dir build/release --output-on-failure'
```

Result: 25/25 tests passed (`compiler_integration` 0.07s, `quant_compiler_integration` 0.75s, `activation_reference` 0.17s, `activation_integration` 0.17s; remaining tests as above).
### Benchmark results
Not required.
### Architecture blocker
None.
### Follow-up observations
Independent QK RMS then RoPE materializes a BF16 vector between the two launches. TASK-014's fused preparation kernel can keep the pre-RoPE head in FP32 and round once after rotation; that is a later fusion cut, not a V0 precision change. Text-only RoPE takes one integer position (equal T/H/W). Packed dense CUDA remains TASK-009.


# TASK-006 — Logical quantizers and CUDA V0 packers

## Status
DONE
## Milestone
M2 — Quantized contraction foundation
## Purpose
Establish independent mathematical and byte-level correctness definitions for Q4G64, Q8G32, and BF16 before optimized consumers exist.
## Depends on
- TASK-005
## Normative references
- `docs/architecture/architecture-v0.md` — Quantizer definitions; physical packing and consumption
- `docs/architecture/quantization-validation.md`
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| A-02 | Logical quantizers and physical packing are distinct | LOCKED |
| Q-01 | Main projections use Q4G64 | LOCKED |
| Q-02 | lm_head uses Q8G32 | LOCKED |
| Q-03 | BF16 passthrough families | LOCKED |
| L-01 | `cuda_q4g64_v0` and `cuda_q8g32_v0` layouts | LOCKED |
## Starting point
Identity compiler emits validated tensor families and BF16 control layout.
## Scope
- Implement deterministic FP32 logical quantizers: group absmax; zero group rule; smallest positive finite FP16 scale at least `a/qmax`; least-positive-normal floor; codes computed from stored scale; RNE; clamping; nonfinite/unrepresentable rejection.
- Implement physical packers for `[N/8,K/256,8,packed_256]` and separate scale order `[N/8,K/256,row,group]`, lower-nibble-first Q4, signed two's-complement codes, padding rules, and 256-byte bases.
- Implement independent simple unpackers/dequantizers and CPU FP32 reference contraction that first rounds decoded operands to BF16; implement BF16 passthrough/control decode.
- Integrate production/identity compiler format selection without changing family assignments.
## Out of scope
CUDA contraction, alternate group sizes, outlier handling, calibration, integer tensor cores, activation quantization, layout v1, quality claims.
## Required interfaces
Host quantize/pack/unpack/reference-contract APIs with logical quantizer and physical layout passed as distinct strong identifiers; compiler emits matching versions.
## Required semantics
Q4 range `[-7,7]` (`-8` invalid); 64 input weights/group, 32 code bytes + FP16 scale. Q8 range `[-127,127]` (`-128` invalid); 32/group, 32 bytes + scale. Matrix is `W[N,K]`; groups never cross output rows.
## Data representation
Eight rows × 256 K coordinates per physical tile. Q4 tile row 128 code bytes/four scales; Q8 tile row 256 bytes/eight scales. Scales are FP16 in a separate contiguous array.
## Implementation constraints
Optimized consumers must not become the correctness authority. Preserve one packed view and bounded streaming compilation.
## Tuning defaults
None; group sizes/layout versions are locked here.
## Expected files/modules
Host `compiler/quantization`, `format/layout`, reference numerical code, golden-byte fixtures.
## Tests required
### Unit tests
- Zero, ties/RNE, extrema, scale rounding-up/floor, nonfinite/reject, saturation, forbidden codes, nibble order, row/tile/group order, padding, deterministic bytes.
### Reference/numerical tests
- Quantize→pack→unpack agreement with logical codes/scales; reference contractions versus explicitly dequantized BF16 operands for varied aligned dimensions and real tensor samples.
### Integration tests
- Compile/read a mixed identity+Q4+Q8 fixture; verify declared byte counts, hashes, versions, and reconstructed values.
## Benchmark required
No.
## Acceptance criteria
- [x] Logical quantizer tests do not depend on physical packing.
- [x] Physical pack tests use independent expected/golden bytes.
- [x] Q4/Q8/BF16 reference contractions are understandable and deterministic.
- [x] Compiler produces only V0 family assignments and one physical view.
- [x] All tests pass.
## Architecture blocker rule
On a locked conflict, stop with every required `ARCHITECTURE_BLOCKER` field; do not change group size, scale rule, code range, or layout.
## Completion report
### Result
DONE. Independent verification PASS (Debug/Release ctest 18/18; delivery re-run 18/18).
### Changes made
- Host logical quantizers in `src/compiler/quantization/quantizer.cpp`: Q4G64 (`qmax=7`, G64) and Q8G32 (`qmax=127`, G32) with group absmax, zero-group scale 0, smallest finite FP16 `>= a/qmax`, least-positive-normal floor, codes from the stored FP16 scale, RNE, clamp, and nonfinite/unrepresentable rejection.
- Physical packers/unpackers in `src/format/pack.cpp` and independent `src/format/unpack.cpp` for `cuda_q4g64_v0` / `cuda_q8g32_v0`: `[N/8,K/256,8,packed_256]`, scales `[N/8,K/256,row,group]`, lower-nibble-first Q4, signed two's-complement, padded coordinates zero, 256-byte-aligned spans via the existing writer.
- CPU FP32 reference GEMV in `src/compiler/quantization/reference.cpp` that dequantizes, rounds operands to BF16, then accumulates; BF16 dense-tile passthrough decode.
- Compiler format selection: `WeightFormatPolicy::{IdentityBf16,ProductionV0}` without changing family assignments. Production uses Q4G64 for GDN qkv/z/out, full-attention q/g/k/v/o, MLP, matching MTP and `mtp.fc`; Q8G32 for `lm_head`; BF16 for embeddings, norms, conv, GDN a/b, and remaining families. CLI `--format identity|production`. Streaming emit processes eight output rows at a time.
- Tests: `quantizer` (logical only), `pack_layout` (golden bytes, no quantizer), `quant_reference` (round-trip + GEMV), `quant_compiler_integration` (mixed identity+Q4+Q8 artifact plus real 8×256 samples).
### Tests run
Debug:

```text
docker run --gpus all --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 \
  bash -lc 'cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug && cmake --build build/debug && ctest --test-dir build/debug --output-on-failure'
```

Result: 18/18 tests passed (`host_expected_smoke`, `format_schema`, `format_schema_integration`, `format_writer` 0.04s, `format_sha256` 0.02s, `format_writer_integration` 0.01s, `format_reader` 0.03s, `format_reader_digest` 0.01s, `format_reader_integration` 0.01s, `cuda_runtime_smoke` 0.23s, `cuda_sm120_cubin`, `compiler_identity` 0.02s, `compiler_transform`, `compiler_integration` 0.22s, `quantizer`, `pack_layout`, `quant_reference`, `quant_compiler_integration` 3.69s).

Release:

```text
docker run --gpus all --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 \
  bash -lc 'cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release && cmake --build build/release && ctest --test-dir build/release --output-on-failure'
```

Result: 18/18 tests passed (`host_expected_smoke`, `format_schema`, `format_schema_integration`, `format_writer` 0.03s, `format_sha256` 0.01s, `format_writer_integration` 0.01s, `format_reader` 0.03s, `format_reader_digest` 0.01s, `format_reader_integration` 0.01s, `cuda_runtime_smoke` 0.17s, `cuda_sm120_cubin`, `compiler_identity`, `compiler_transform`, `compiler_integration` 0.07s, `quantizer`, `pack_layout`, `quant_reference`, `quant_compiler_integration` 0.84s).
### Benchmark results
Not required.
### Architecture blocker
None.
### Follow-up observations
- Default ctest compiles mixed synthetic identity+Q4+Q8 fixtures and 8×256 slices of real `in_proj_qkv` / `lm_head`; it does not emit a full-model production `.qw38`. Use `qw38-compile --format production --checkpoint ... --output ...` for that path. Add `--verify-reconstruction` when the separate full second pass is required.
- CUDA decode contraction remains TASK-009; this task's unpacker/reference GEMV is the correctness authority for packed bytes.

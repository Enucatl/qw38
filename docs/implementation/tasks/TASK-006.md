# TASK-006 — Logical quantizers and CUDA V0 packers

## Status
TODO
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
- [ ] Logical quantizer tests do not depend on physical packing.
- [ ] Physical pack tests use independent expected/golden bytes.
- [ ] Q4/Q8/BF16 reference contractions are understandable and deterministic.
- [ ] Compiler produces only V0 family assignments and one physical view.
- [ ] All tests pass.
## Architecture blocker rule
On a locked conflict, stop with every required `ARCHITECTURE_BLOCKER` field; do not change group size, scale rule, code range, or layout.
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

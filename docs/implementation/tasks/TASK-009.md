# TASK-009 — Decode dense contraction consumers

## Status
TODO
## Milestone
M2 — Quantized contraction foundation
## Purpose
Consume the already-defined BF16/Q4/Q8 bytes in correct batch-one CUDA matrix-vector contractions without making kernels the format authority.
## Depends on
- TASK-006
- TASK-008
## Normative references
- `docs/architecture/architecture-v0.md` — Physical packing/consumption; precision policy; CUDA ownership
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| Q-01–Q-03, L-01 | Weight classes and physical layouts | LOCKED |
| P-01–P-02 | BF16 operands, FP32 accumulation/epilogues | LOCKED |
| T-01 | Eight warps/block, 256-input K tile | TUNING |
## Starting point
Independent pack/unpack/reference contractions and CUDA activation/runtime layers exist.
## Scope
Implement BF16-control, `cuda_q4g64_v0`, and `cuda_q8g32_v0` decode MMV consumers for aligned V0 matrices; one output row/warp, eight warps/block, shared BF16 input, subgroup scale broadcast, local decode rounded to BF16, FP32 FMA/reduction, no atomics; support BF16 output, FP32 output, and FP32 residual-add epilogues. Provide grouped small BF16 a/b contraction with FP32 output.
## Out of scope
Prefill GEMM, alternative packing, autotuning, integer MMA, fused norm, full MLP/GDN/attention, changing group sizes.
## Required interfaces
Typed launch descriptors include layout, `N,K`, code/scale/input/output views, epilogue, stream; reject mismatches before launch.
## Required semantics
Compute `y_n=sum_k BF16(dequant(W_nk))*BF16(x_k)` with FP32 accumulation. Q4 lane reads eight weights from a 32-bit word; Q8 from a 64-bit word. Paired mode owns matching independent rows/accumulators.
## Data representation
Exactly TASK-006 V0 layouts; no global decoded-weight cache; input up to 17408 BF16 staged once/block; output per declared epilogue.
## Implementation constraints
Reference comparison uses identical decoded BF16 operands. Keep eager measurable launches and narrow CUDA wrappers.
## Tuning defaults
Eight warps/256 threads, K tile 256; changes require correctness rerun but no ABI change.
## Expected files/modules
`cuda/` decode projection kernels/launchers and numerical tests.
## Tests required
### Unit tests
Launch validation, tails/padding metadata even though model dimensions align, zero/extreme codes/scales, all epilogues.
### Reference/numerical tests
Each format/dimension family against TASK-006 CPU reference; exact decoded operands and stated FP32 reduction tolerance.
### Integration tests
Run representative GDN, attention, MLP, and full Q8 head shapes on RTX 5090; compare BF16 control and packed paths.
## Benchmark required
Microbenchmark required only as smoke: report bytes, dimensions, launches, and timing; it cannot authorize redesign.
## Acceptance criteria
- [ ] Q4/Q8/BF16 consumers meet reference tolerances on all representative shapes.
- [ ] Layout/version mismatch rejects.
- [ ] Residual-add writes FP32 directly; ordinary staging writes BF16.
- [ ] No global dequantized weight buffer or atomics exist.
- [ ] Initial geometry and smoke timing are recorded.
## Architecture blocker rule
On locked conflict stop with full blocker fields; poor performance alone is tuning, not permission to repack.
## Completion report
### Result
DONE | BLOCKED
### Changes made
### Tests run
Exact commands/results.
### Benchmark results
Exact microbenchmark identity/results.
### Architecture blocker
None/full report.
### Follow-up observations
Concrete only.


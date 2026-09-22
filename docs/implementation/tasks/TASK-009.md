# TASK-009 — Decode dense contraction consumers

## Status
DONE
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
- [x] Q4/Q8/BF16 consumers meet reference tolerances on all representative shapes.
- [x] Layout/version mismatch rejects.
- [x] Residual-add writes FP32 directly; ordinary staging writes BF16.
- [x] No global dequantized weight buffer or atomics exist.
- [x] Initial geometry and smoke timing are recorded.
## Architecture blocker rule
On locked conflict stop with full blocker fields; poor performance alone is tuning, not permission to repack.
## Completion report
### Result
DONE
### Changes made
- CUDA decode MMV in `cuda/decode_mmv.{hpp,cu}`: T-01 geometry (8 warps/256 threads, K tile 256, shared BF16 input up to 17408), warp-row ownership, Q4 32-bit / Q8 64-bit lane loads, subgroup scale broadcast, local dequant rounded to BF16, FP32 FMA + warp reduction, no atomics and no global decoded-weight buffer.
- Typed `DecodeMmvDesc` / `DecodeMmvPairedDesc` launches reject layout/quantizer mismatches, padded N/K metadata errors, span-length mismatches, K>17408, and epilogue/pointer mismatches before launch.
- Epilogues: BF16 store, FP32 store, FP32 residual-add. Paired launch keeps two independent row accumulators. `launch_decode_ab_bf16` is the grouped GDN a/b BF16-tile / FP32-output path.
- `Event::create_timing` + `elapsed_ms` for measurable eager launches (existing `Event::create` stays timing-disabled).
- Tests: `decode_mmv_unit` (validation, 20×192 Q4 padding tails, zero/extreme, all epilogues), `decode_mmv_reference` (Q4/Q8/BF16 families vs TASK-006 CPU GEMV, paired, a/b), `decode_mmv_integration` (GDN qkv/z/out/a-b, attention qg/k/o, MLP paired gate/up + down, full Q8 `lm_head` 248320×5120).
- Smoke bench `qw38_bench_decode_mmv` (`EXCLUDE_FROM_ALL`, not ctest). Declared FP32 reduction tolerances in `qw38::cuda::decode_mmv_tol`.
### Tests run
Debug:

```text
docker run --gpus all --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 \
  bash -lc 'cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug && cmake --build build/debug && ctest --test-dir build/debug --output-on-failure'
```

Result: 28/28 tests passed (`decode_mmv_unit` 0.17s, `decode_mmv_reference` 0.18s, `decode_mmv_integration` 66.26s; remaining tests as TASK-008 plus the three new decode_mmv targets).

Release:

```text
docker run --gpus all --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 \
  bash -lc 'cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release && cmake --build build/release && ctest --test-dir build/release --output-on-failure'
```

Result: 28/28 tests passed (`decode_mmv_unit` 0.17s, `decode_mmv_reference` 0.17s, `decode_mmv_integration` 12.25s).
### Benchmark results
Historical diagnostic run: source revision `beacd98`; Release target
`qw38_bench_decode_mmv`, mutable container tag `qw38-dev:cuda13.4.1`, device
NVIDIA GeForce RTX 5090 `sm_120`. The executable SHA-256/build ID and immutable
container image digest were not preserved. Consequently, the timing rows below
are retained only as a historical log and are **withdrawn as reproducible
measurement evidence**. Future reported measurements must use the identity-bound
wrapper in `docs/implementation/dev-environment.md`. Geometry: 8 warps/block,
256 threads, K tile 256, max K 17408. Historical command:

```text
docker run --gpus all --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 \
  bash -lc 'cmake --build build/release --target qw38_bench_decode_mmv && ./build/release/benchmarks/qw38_bench_decode_mmv'
```

| Case | N×K | code_bytes | scale_bytes | iterations | kernel launches | ms |
|---|---| ---:| ---:| ---:| ---:| ---:|
| q4-8x256 | 8×256 | 1024 | 64 | 8 | 8 | 0.002368 |
| q4-mlp-down | 5120×17408 | 44564480 | 2785280 | 8 | 8 | 0.081916 |
| q4-mlp-gate | 17408×5120 | 44564480 | 2785280 | 8 | 8 | 0.07424 |
| q8-head | 248320×5120 | 1271398400 | 79462400 | 8 | 8 | 1.38918 |
| bf16-ab | 48×5120 | 491520 | 0 | 8 | 8 | 0.009056 |

Diagnostic only; does not authorize repack or geometry change.
### Architecture blocker
None.
### Follow-up observations
- Paired MMV writes two independent BF16 (or FP32) outputs. TASK-010 owns the SwiGLU fused epilogue that avoids materializing gate/up globally.
- Full Q8 head BF16-control comparison uses a 256-row slice; the packed Q8 path runs the full 248320×5120 matrix. Host unpack+dequant of the full head dominates debug integration time (~66s).

### Verification (delivery)
VERIFICATION: PASS

Commands run:
- Debug: `ctest --test-dir build/debug -R decode_mmv` — 3/3 passed
- Release: `ctest --test-dir build/release` — 28/28 passed

Unmet acceptance criteria: none.


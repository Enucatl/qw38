# TASK-010 — Complete decode MLP

## Status
DONE
## Milestone
M3 — Core runtime and MLP
## Purpose
Deliver the first complete integrated quantized CUDA consumer and residual transition.
## Depends on
- TASK-009
## Normative references
- `docs/architecture/architecture-v0.md` — MLP kernel sequence; precision/materialization policy
- `docs/architecture/model-semantics.md`
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| Q-01, P-01–P-02, M-01 | Q4 MLP, precision, and scratch cuts | LOCKED |
| T-01 | Decode projection geometry | TUNING |
## Starting point
RMS and packed MMV with paired/residual epilogues work independently.
## Scope
Implement the complete decode MLP: FP32 `h_mid[5120]` → zero-centered RMS → BF16 normalized → paired Q4 gate/up `[17408]` with separate FP32 accumulators → FP32 SiLU(gate)×up → BF16 SwiGLU scratch → Q4 down `[5120,17408]` → FP32 accumulator added to original `h_mid` → FP32 next residual. Add a CPU/BF16-control reference and plan/runtime binding.
## Out of scope
Materialized gate/up arrays, down fusion across its reduction, prefill GEMM, other layers, architecture experiments.
## Required interfaces
Concrete MLP plan binds three tensor identities, norm, scratch, residual views, and ordered stream; decode call performs no allocation.
## Required semantics
`silu(x)=x*sigmoid(x)` evaluated FP32; only product stores BF16. Original residual stays live through down add. Gate/up rows are independent despite paired ownership.
## Data representation
FP32 residual 5120; BF16 normalized 5120; BF16 SwiGLU 17408; Q4 V0 matrices/scales.
## Implementation constraints
Exactly three initial regions (RMS, paired projection/epilogue, down/residual); no opportunistic fusion.
## Tuning defaults
TASK-009 projection geometry.
## Expected files/modules
Runtime MLP plan/execution, paired epilogue kernel support, reference/integration tests.
## Tests required
### Unit tests
Binding/shape/layout errors, zero/extreme activations, scratch lifetime, repeated call/no allocation.
### Reference/numerical tests
Packed and BF16-control MLP against CPU reference with stage-local and final tolerances.
### Integration tests
Two consecutive MLP invocations with deterministic residuals and scratch reuse.
## Benchmark required
Report one representative decode MLP timing as diagnostic only.
## Acceptance criteria
- [x] Exact V0 sequence and precision boundaries execute.
- [x] Only BF16 SwiGLU is globally materialized for gate/up.
- [x] Final FP32 residual matches reference tolerance.
- [x] Steady-state call allocates nothing.
- [x] Tests pass.
## Architecture blocker rule
On locked conflict stop with full required blocker report; do not change fusion/materialization.
## Completion report
### Result
DONE
### Changes made
- Paired decode MMV epilogue `DecodeEpilogue::SwigluStoreBf16` in `cuda/decode_mmv.{hpp,cu}`: gate/up stay in FP32 accumulators, `silu(gate)*up` is evaluated FP32, and only the product is stored as BF16. Unpaired launches reject this epilogue. Gate/up rows remain independently reduced.
- CPU decode-MLP reference `qw38::reference::decode_mlp_reference` over decoded BF16 operands: zero-centered RMS → gate/up GEMV → FP32 SiLU×up BF16 store → down GEMV → FP32 residual add. Stage-local/final tolerances in `qw38::reference::tol`.
- Concrete `MlpPlan` in `src/runtime/mlp.{hpp,cpp}`: binds Q4 or BF16-control gate/up/down identities, post-attention RMS gamma, FP32 `h_mid`, BF16 normalized and SwiGLU scratch, and one ordered stream. `execute_decode_mlp` runs exactly three launches (RMS, paired SwiGLU, down residual-add) with no allocation. Model/session bind uses layer tensor names plus session `residual_h_mid` / `NormalizedHidden` / `MlpSwiglu` (decode token 0). Original residual is live through the in-place down add.
- Tests: `mlp_unit` (binding/shape/layout, unpaired SwiGLU, 8×256 fused epilogue, zero/extreme, scratch identity, repeated call malloc count, missing Model identities), `mlp_reference` (packed Q4 and BF16-control vs CPU stages), `mlp_integration` (`.qw38` upload, two consecutive MLP calls, deterministic residuals, scratch reuse).
- Smoke bench `qw38_bench_mlp` (`EXCLUDE_FROM_ALL`, not ctest). CMake/ctest Debug+Release GPU.
### Verification
VERIFICATION: PASS (composer-2.5, independent re-run of Debug and Release ctest suites; 31/31 each).

### Tests run
Debug:

```text
docker run --gpus all --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 \
  bash -lc 'cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug && cmake --build build/debug && ctest --test-dir build/debug --output-on-failure'
```

Result: 31/31 tests passed (`mlp_unit` 5.57s, `mlp_reference` 8.64s, `mlp_integration` 9.16s; remaining tests as TASK-009 plus the three new mlp targets).

Release:

```text
docker run --gpus all --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 \
  bash -lc 'cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release && cmake --build build/release && ctest --test-dir build/release --output-on-failure'
```

Result: 31/31 tests passed (`mlp_unit` 2.12s, `mlp_reference` 2.51s, `mlp_integration` 2.43s).
### Benchmark results
Identity: `qw38_bench_mlp` Release, container `qw38-dev:cuda13.4.1`, device NVIDIA GeForce RTX 5090 `sm_120`. Geometry: T-01 8 warps/block, 256 threads, K tile 256. Command:

```text
docker run --gpus all --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 \
  bash -lc 'cmake --build build/release --target qw38_bench_mlp && ./build/release/benchmarks/qw38_bench_mlp'
```

| Case | shape | regions | weight_bytes | launches | ms |
|---|---|---| ---:| ---:| ---:|
| decode-mlp q4 | hidden=5120 ffn=17408 | 3 (rms, paired-swiglu, down-residual) | 142049280 | 8 | 0.268244 |

Diagnostic only; does not authorize fusion or geometry change.
### Architecture blocker
None.
### Follow-up observations
- Down residual-add updates `h_mid` in place. The next-layer ping-pong into `residual_h` remains a layer-integration concern (TASK-016).
- Declared composed tolerances: RMS BF16 8e-3; SwiGLU BF16 2e-2 abs / 1e-3 rel; final residual 5e-2 abs / 1e-3 rel (GEMV reduction-order plus one BF16 store per stage).


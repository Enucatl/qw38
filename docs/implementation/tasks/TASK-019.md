# TASK-019 — SM120 quantization and kernel feasibility

## Status

IN_PROGRESS

## Milestone

M7 — Strategy and feasibility

## Purpose

Establish which compact representations and kernels are correct and useful on the actual SM120 platform before selecting a production ABI.

## Depends on

- [TASK-018](TASK-018.md)

## Normative references

- [Implementation ledger](../task_ledger.md) — OVERALL-01, revised task contract,
  overall decision rules, measurement envelope and migration of prior obligations.
- [Architecture V0](../../architecture/architecture-v0.md) — retained model semantics
  and controls; reopened decisions follow OVERALL-01.
- [EVAL-01 / PERF-01](../../architecture/evaluation-policy-v0.md) — unchanged
  quality criteria and measurement definitions, with task ownership remapped by the ledger.
- [Technology baseline](../technology-baseline.md).
- [Code standards](../code-standards.md).

## Architecture decisions consumed

| Decision | Contract for this task | Authority |
| -------- | ---------------------- | --------- |
| Q-01/Q-02, projection part of P-02 | Compare native FP4 weight/activation policies and existing Q4/BF16 controls | Reopened by OVERALL-01 |
| A-01/L-01, M-01, T-01–03 | Measure view/layout, workspace and shape choices | Reopened by OVERALL-01 |
| I-01–I-05, P-01, S-01/S-02 | RTX 5090 toolchain, FP32 arithmetic and initial state controls | Retained |

OVERALL-01 supersedes conflicting restrictions in the former task sequence.
Historical EXP-A–H ordering does not constrain this task. Decisions outside
the reopened scope remain binding.

## Starting point

TASK-018 has frozen the protocol and preserved the current development controls. Production quantizer/layout selection and full candidate quality acceptance remain open.

## Scope

Pin an appropriate CUTLASS revision and demonstrate dense NVFP4 and MXFP4
GEMMs on the actual RTX 5090/toolchain. Use minimal reference pack/quantize
paths before committing the production artifact ABI. Check scales, transpose
orientation, padding/tails, zero blocks, FP32 accumulation, and required output
precision against an independent contraction of reconstructed operands.
Separate arithmetic correctness from quantization error against BF16.

Run the measurement envelope above. Include a bounded Q4-to-BF16 GEMM control,
BF16 library GEMM where capacity allows, and estimates or probes for same-FP4
weight GEMV. Measure input scaling/packing and output work, not just GEMM on
prequantized inputs. Reject unsupported mixed formats explicitly. Compute full
resident/transient memory budgets for each viable policy, including selective
FP8/BF16 exceptions and optional views. **Exit:** reproducible support/cost table,
native MMA evidence, and a short list of feasible candidates with a fallback.
No native FP4 winner or large-GEMM-to-decode speedup is assumed.

## Out of scope

Freezing the production artifact ABI, model-wide integration, calibration on evaluation cases, structured sparsity, assuming SM100 instructions/resources apply to SM120, and claiming a native mixed-input kernel without proof.

## Required interfaces and data representation

A benchmark descriptor records family, M/N/K, logical and padded shapes, operand/scale types and layouts, output/epilogue, kernel/dispatch identity, workspace and conversion stages. Emit correctness results, raw timing samples, native-MMA evidence, toolchain/CUTLASS pin and full resident/transient memory estimates per candidate.

## Required semantics and constraints

Native NVFP4 uses quantized weights and activations. Distinguish its FP32 accumulation from output storage. Validate scale orientation, tensor-scale convention and zero/padding behavior against reconstructed operands, then assess quantization error separately against BF16. Keep both phases' comparison inputs, epilogues and output precision matched. Do not reinterpret Q4 codes as FP4.

Follow the code standards' identity and manifest-only digest policy. Keep
benchmarks separate from correctness checks and record source, binary,
artifact/policy, inputs, toolchain and hardware identities appropriate to each
result. Partial or invalid evidence cannot establish quality acceptance.

## Tuning defaults

Use real model families with M = 1, 2, 8, 32, 64, 128, 256, 512 and 1024 where workspace permits, plus unaligned tails. Record unsupported points and costs explicitly. Pin and verify the compiler target required by the chosen CUTLASS revision; a large-M winner does not select decode dispatch.

## Expected files/modules

Isolated projection/quantization benchmarks, minimal reference converters and packers, toolchain/dependency pin, instruction/profiler evidence and feasibility report. Production compiler/runtime integration follows later.

## CUTLASS candidate pin

Use the immutable NVIDIA CUTLASS `v4.8.0` revision `098de2a652cf8f00fd70b2df54051c7eccbb855a` for this feasibility investigation. The official release identifies it as the latest release as of 2026-09-24. Its SM120 example 79a explicitly implements dense NVFP4 GEMM and requires CUDA Toolkit 12.8 or newer; the release notes also call out code generation with CUDA 13.4. The official SM120 support table lists both `nv_float4_t` and `mx_float4_t` block-scaled paths. These sources make it a compatible candidate for the project toolchain, but do not substitute for building and running this exact revision on the project image and RTX 5090. Record the actual build and instruction evidence before treating it as demonstrated support. See [the v4.8.0 release](https://github.com/NVIDIA/cutlass/releases/tag/v4.8.0), [the pinned SM120 NVFP4 example](https://github.com/NVIDIA/cutlass/blob/v4.8.0/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu), and [the official SM120 GEMM support table](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#blackwell-sm120-gemms).

## Tests required

### Unit and contract checks

Scale/encoding reconstruction, matrix orientation, padding/tail masks, zero blocks, supported-shape checks and workspace bounds.

### Reference and numerical checks

Independent contractions of reconstructed NVFP4/MXFP4 operands and Q4-to-BF16 controls, with explicit tolerances and required output precision. Diagnose quantization error against BF16 separately.

### Integration checks

Build and run the pinned candidates on RTX 5090; prove the selected instruction path and identify fallbacks. Exercise real shapes/weights and representative activations. Account for weights, scales, embeddings/head, retained payloads, state/KV, scratch, optional views and transient peaks.

## Benchmark required

Required: native NVFP4/MXFP4, bounded Q4-to-BF16 GEMM, BF16 library controls where feasible, and same-FP4-weight GEMV probes or estimates. Report kernel-only and complete quantize/pack/dispatch/GEMM/epilogue costs, resource usage and memory, distinguishing measurements from estimates.

## Acceptance criteria

- [ ] Pinned SM120 build and native dense NVFP4/MXFP4 attempts have reproducible support and instruction evidence; any unsupported case has a concrete reason.
- [ ] Independent operand-reconstruction contractions validate implemented kernels and tails.
- [ ] The real-shape matrix reports conversion-inclusive latency, logical/padded work, output precision and resource/workspace costs.
- [ ] Full candidate memory budgets respect actual allocatable memory and the frozen reserve.
- [ ] A short list and eligible fallback are justified without claiming final quality or end-to-end speed.

## Architecture blocker rule

A rejected candidate is a recorded result; use the eligible fallback within OVERALL-01 without relaxing acceptance criteria. Missing required exit evidence prevents completion. A conflict outside the reopened decisions requires the full architecture-blocker report defined in the ledger; obsolete Q4-only or experiment-order restrictions are not blockers.

## Completion report

### Result

IN PROGRESS — selected the immutable CUTLASS v4.8.0 candidate pin and confirmed the available project image exposes CUDA 13.4.1, GCC 14.2.0, CMake 3.28.3, and the RTX 5090 (SM120). The official NVFP4 target and direct MXFP4 variant both built and ran on the GPU. An initial synthetic sweep completed 52 points but was stopped during the CUTLASS host reference for NVFP4 M=256, N=248320, K=5120. It is not a complete benchmark, independent kernel validation, or acceptance evidence for any criterion.

### Changes made

Pinned CUTLASS `v4.8.0` / `098de2a652cf8f00fd70b2df54051c7eccbb855a` in a separate detached worktree at `.cache/task019/cutlass-v4.8.0`. Preserved `.cache/task019/cutlass` at `v4.6.1` / `e05f953a5b3d38adc240df2ff928e0421c2abba3`. Added `scripts/task019_cutlass_sm120.sh`, which configures only the official SM120 NVFP4 example and prepares an MXFP4 × MXFP4 variant by changing the operand wrapper types in the same pinned example. After the interrupted sweep, the script was revised to run CUTLASS's host block-scaled reference only at bounded M=32, N=128, K=128 for each format, then use explicitly labeled benchmark-only copies with that host check disabled for the real-shape timing loop. It captures SASS before the timing loop and uses `grep` instead of an unverified `rg` dependency. The revised harness has not been built or run. Its timing path uses generated FP4 inputs and times initialized GEMM iterations, so it does not yet satisfy real-weight, conversion-inclusive, Q4/BF16-control, independent-reference, or full-envelope requirements. The earlier revision choice had no recorded comparison or compatibility rationale. No payload or scale-content digest was computed.

The pinned CUDA image lacks Python, which CUTLASS v4.8.0 requires during CMake configuration. The script read-only bind-mounts the host `/usr/bin/python3.12` interpreter and `/usr/lib/python3.12` standard library, and passes `-DPython3_EXECUTABLE=/usr/bin/python3.12`. The host interpreter is Python 3.12.3. This supplies Python only to the CMake configuration step; it does not change the Dockerfile/image or add a Python runtime dependency to the candidate binaries. The script fails with a clear message if either host path is unavailable.

The first v4.8.0 harness attempt configured successfully but failed compiling `79a_blackwell_geforce_nvfp4_bf16_gemm`: `cutlass/util/command_line.h` was not found. The example's CMake helper links `cutlass_tools_util_includes`, which CUTLASS declares from `tools/CMakeLists.txt` only when `CUTLASS_ENABLE_TOOLS=ON`; the harness had disabled tools. The harness now enables tools while retaining `CUTLASS_ENABLE_LIBRARY=OFF`, `CUTLASS_ENABLE_TESTS=OFF`, and `CUTLASS_ENABLE_PROFILER=OFF`. In the pinned CMake, tools with those options on adds `tools/util` and its INTERFACE include target only; it does not build the library, profiler, or tests as dependencies of the requested example target. With tools enabled, the official NVFP4 target built and linked successfully. Direct MXFP4 compilation initially failed at `sm120_blockscaled_mma_tma.hpp:643` because the command omitted `--expt-relaxed-constexpr`; adding the flag let it build successfully in the next run.

### Tests run

Environment probe, PASS: `docker run --rm --gpus all qw38-dev:cuda13.4.1-pinned bash -lc 'nvcc --version && cmake --version | head -1 && g++-14 --version | head -1 && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader'` reported CUDA 13.4.59, CMake 3.28.3, GCC 14.2.0 and `NVIDIA GeForce RTX 5090, 12.0`. `git fetch --filter=blob:none origin 098de2a652cf8f00fd70b2df54051c7eccbb855a` followed by `git worktree add --detach ../cutlass-v4.8.0 098de2a652cf8f00fd70b2df54051c7eccbb855a` created the requested source worktree and preserved the v4.6.1 checkout. Initial CMake configuration, PASS; its subsequent `cmake --build "${build}" --target 79a_blackwell_geforce_nvfp4_bf16_gemm -j2`, FAIL: compilation stopped because `cutlass/util/command_line.h` was missing after configuring with `-DCUTLASS_ENABLE_TOOLS=OFF`. With tools enabled, the official NVFP4 target built and linked successfully. Direct MXFP4 compilation first failed at `sm120_blockscaled_mma_tma.hpp:643` because the command omitted `--expt-relaxed-constexpr`, which CUTLASS CMake supplied to NVFP4; it built successfully after that flag was added. `bash -n scripts/task019_cutlass_sm120.sh` passes. Host probes found no `nvcc` or `cmake`; the container is the usable toolchain. No independent GPU operand-reconstruction comparison has run.

### Benchmark results

The exact interrupted command was `./scripts/task019_cutlass_sm120.sh`; its captured log is `.codex-wake-run/c965b940b139.log`. The log records 52 completed synthetic CUTLASS example points: all four listed N values at M=1, 2, 8, 32, 64 and 128 for both formats, then N=5120 and 17408 at M=256 for both. Each completed point says `Disposition: Passed` against CUTLASS's own host reference and reports one 20-iteration average initialized-GEMM time, not raw samples or conversion-inclusive cost. The next line is `NVFP4 m=256 n=248320 k=5120`, with no disposition or timing. The main thread stopped the harness container using `docker stop --time 1` after observing the example at roughly 100% host CPU, 0% GPU utilization, and about 1.5 GiB GPU allocation while it computed the full-vocabulary host reference. The remaining 19 points were not reached. Neither a complete benchmark nor native-MMA instruction, resource, memory-budget, or quality evidence is available. The revised bounded-reference/benchmark-only harness is pending a GPU run.

### Architecture blocker

Record none or the complete ledger-defined blocker report.

### Follow-up observations

Remaining for TASK-019: run the revised bounded-reference/benchmark-only harness; validate kernels against an independent contraction of reconstructed operands (including tails/scales/zero blocks and the BF16 output epilogue); use actual representative BF16 model weights and activations; measure quantization/packing plus dispatch/GEMM/output work; add feasible Q4-to-BF16 and BF16 library controls plus same-weight GEMV probes; establish profiler/instruction support; and complete the full memory budget. The script's command loop covers listed major projection dimensions and M values, but not unaligned tails or conversion costs. Its synthetic CUTLASS example operands are useful kernel bring-up only. This task remains IN_PROGRESS and no kernel, correctness, or performance acceptance criterion is yet claimed.

### CPU logical FP4 reference slice (2026-09-24)

Added `scripts/task019_fp4_reference.py` and its standard-library test
`tests/test_task019_fp4_reference.py`. The reference uses CUTLASS v4.8.0's
E2M1 values (including signed zero), UE4M3 scale encoding, UE8M0 scale
encoding, and 16-element NVFP4 / 32-element MXFP4 scale blocks. It stores
logical row-major E2M1 nibbles low-first and scale codes as row-major
`[row, ceil(K/block)]`; these are reference-only pack layouts, not a claim
about CUTLASS's interleaved device scale layout or a production payload ABI.

The explicitly parameterized NVFP4 convention is
`s = UE4M3_RNE(amax_block / (6 * tensor_scale))`, clamped to the smallest
positive UE4M3 scale when a nonzero block would round `s` to zero;
`q = E2M1_RNE(x / (tensor_scale * decode(s)))`, and
`xhat = tensor_scale * decode(s) * decode(q)`. MXFP4 uses
`s = UE8M0_ceil(amax_block / 6)`, `q = E2M1_RNE(x / decode(s))`, and
`xhat = decode(s) * decode(q)`. An all-zero block produces zero E2M1 codes;
its UE8M0 code is the defined minimum positive scale code, which does not
affect reconstructed zeros. K padding adds zero codes, and reconstruction
returns only logical columns. `contract_fp32` computes
`C[M,N] = A[M,K] * B[N,K]^T`, rounding products and each accumulated sum to
binary32. The tests use asymmetric M/N and check arithmetic on reconstructed
values separately from the BF16-input quantization-error comparison.

Exact focused command after scale repairs, PASS (6 tests, 0.004 s):
`PYTHONPATH=. python3 tests/test_task019_fp4_reference.py -v`. Syntax check, PASS:
`python3 -m py_compile scripts/task019_fp4_reference.py tests/test_task019_fp4_reference.py`.
Shell syntax check, PASS: `bash -n scripts/task019_cutlass_sm120.sh`.
Whitespace check, PASS: `git diff --check`. The new edge tests cover nonzero
NVFP4 underflow and MXFP4 rounding just above a power of two. The revised
harness has had shell syntax and source-transform checks only; it has not
been compiled or executed in the container.

Scope limit: the inspected CUTLASS v4.8.0 SM120 examples initialize packed
inputs and scales for host reference GEMM, but do not define a BF16-to-FP4
quantizer recipe or a project payload packing convention. This CPU reference
records a clear logical scaling convention and does not validate a GPU
candidate, CUTLASS physical scale-factor placement, full edge encodings, or
conversion costs. TASK-019 remains `IN_PROGRESS`; this slice satisfies no
kernel acceptance criterion.

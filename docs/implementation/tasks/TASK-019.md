# TASK-019 — SM120 quantization and kernel feasibility

## Status

DONE

The historical partial reports below retain their original status language;
the completion evidence at the end supersedes those interim findings.

## Historical blocked reason

Historical outcome of the 2026-09-24 review. The user explicitly reopened
TASK-019 to repair the remaining evidence in the main thread.

After one Sol repair pass and one supplemental evidence pass, independent GPU
operand-reconstruction validation, real-weight/activation measurements,
conversion-inclusive timing, Q4/BF16 and GEMV controls, complete memory budgets,
and a justified candidate shortlist remain absent. The final review also found
that the support replay prints no CUTLASS revision inside the container and
does not capture resource output because `cuobjdump` is invoked with an invalid
option combination. The permitted repair and supplemental-evidence rounds are
exhausted; do not promote or select an FP4 candidate from this evidence.

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

- [x] Pinned SM120 build and native dense NVFP4/MXFP4 attempts have reproducible support and instruction evidence; any unsupported case has a concrete reason.
- [x] Independent operand-reconstruction contractions validate implemented kernels and tails.
- [x] The real-shape matrix reports conversion-inclusive latency, logical/padded work, output precision and resource/workspace costs.
- [x] Full candidate memory budgets respect actual allocatable memory and the frozen reserve.
- [x] A short list and eligible fallback are justified without claiming final quality or end-to-end speed.

## Architecture blocker rule

A rejected candidate is a recorded result; use the eligible fallback within OVERALL-01 without relaxing acceptance criteria. Missing required exit evidence prevents completion. A conflict outside the reopened decisions requires the full architecture-blocker report defined in the ledger; obsolete Q4-only or experiment-order restrictions are not blockers.

## Completion report

### Historical initial result

IN PROGRESS — selected the immutable CUTLASS v4.8.0 candidate pin and confirmed the available project image exposes CUDA 13.4.1, GCC 14.2.0, CMake 3.28.3, and the RTX 5090 (SM120). The official NVFP4 target and direct MXFP4 variant both built and ran on the GPU. An initial synthetic sweep completed 52 points but was stopped during the CUTLASS host reference for NVFP4 M=256, N=248320, K=5120. It is not a complete benchmark, independent kernel validation, or acceptance evidence for any criterion.

### Changes made

Pinned CUTLASS `v4.8.0` / `098de2a652cf8f00fd70b2df54051c7eccbb855a` in a separate detached worktree at `.cache/task019/cutlass-v4.8.0`. Preserved `.cache/task019/cutlass` at `v4.6.1` / `e05f953a5b3d38adc240df2ff928e0421c2abba3`. Added `scripts/task019_cutlass_sm120.sh`, which configures only the official SM120 NVFP4 example and prepares an MXFP4 × MXFP4 variant by changing the operand wrapper types in the same pinned example. After the interrupted sweep, the script was revised to run CUTLASS's host block-scaled reference only at bounded M=32, N=128, K=128 for each format, then use explicitly labeled benchmark-only copies with that host check disabled for the real-shape timing loop. It captures SASS before the timing loop and uses `grep` instead of an unverified `rg` dependency. At that stage the revised harness had not been built or run; the later pinned GPU bring-up section records its subsequent run. Its timing path uses generated FP4 inputs and times initialized GEMM iterations, so it does not yet satisfy real-weight, conversion-inclusive, Q4/BF16-control, independent-reference, or full-envelope requirements. The earlier revision choice had no recorded comparison or compatibility rationale. No payload or scale-content digest was computed.

The pinned CUDA image lacks Python, which CUTLASS v4.8.0 requires during CMake configuration. The script read-only bind-mounts the host `/usr/bin/python3.12` interpreter and `/usr/lib/python3.12` standard library, and passes `-DPython3_EXECUTABLE=/usr/bin/python3.12`. The host interpreter is Python 3.12.3. This supplies Python only to the CMake configuration step; it does not change the Dockerfile/image or add a Python runtime dependency to the candidate binaries. The script fails with a clear message if either host path is unavailable.

The first v4.8.0 harness attempt configured successfully but failed compiling `79a_blackwell_geforce_nvfp4_bf16_gemm`: `cutlass/util/command_line.h` was not found. The example's CMake helper links `cutlass_tools_util_includes`, which CUTLASS declares from `tools/CMakeLists.txt` only when `CUTLASS_ENABLE_TOOLS=ON`; the harness had disabled tools. The harness now enables tools while retaining `CUTLASS_ENABLE_LIBRARY=OFF`, `CUTLASS_ENABLE_TESTS=OFF`, and `CUTLASS_ENABLE_PROFILER=OFF`. In the pinned CMake, tools with those options on adds `tools/util` and its INTERFACE include target only; it does not build the library, profiler, or tests as dependencies of the requested example target. With tools enabled, the official NVFP4 target built and linked successfully. Direct MXFP4 compilation initially failed at `sm120_blockscaled_mma_tma.hpp:643` because the command omitted `--expt-relaxed-constexpr`; adding the flag let it build successfully in the next run.

### Tests run

Environment probe, PASS: `docker run --rm --gpus all qw38-dev:cuda13.4.1-pinned bash -lc 'nvcc --version && cmake --version | head -1 && g++-14 --version | head -1 && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader'` reported CUDA 13.4.59, CMake 3.28.3, GCC 14.2.0 and `NVIDIA GeForce RTX 5090, 12.0`. `git fetch --filter=blob:none origin 098de2a652cf8f00fd70b2df54051c7eccbb855a` followed by `git worktree add --detach ../cutlass-v4.8.0 098de2a652cf8f00fd70b2df54051c7eccbb855a` created the requested source worktree and preserved the v4.6.1 checkout. Initial CMake configuration, PASS; its subsequent `cmake --build "${build}" --target 79a_blackwell_geforce_nvfp4_bf16_gemm -j2`, FAIL: compilation stopped because `cutlass/util/command_line.h` was missing after configuring with `-DCUTLASS_ENABLE_TOOLS=OFF`. With tools enabled, the official NVFP4 target built and linked successfully. Direct MXFP4 compilation first failed at `sm120_blockscaled_mma_tma.hpp:643` because the command omitted `--expt-relaxed-constexpr`, which CUTLASS CMake supplied to NVFP4; it built successfully after that flag was added. `bash -n scripts/task019_cutlass_sm120.sh` passes. Host probes found no `nvcc` or `cmake`; the container is the usable toolchain. No independent GPU operand-reconstruction comparison has run.

### Benchmark results

The exact interrupted command was `./scripts/task019_cutlass_sm120.sh`; its captured log is `.codex-wake-run/c965b940b139.log`. The log records 52 completed synthetic CUTLASS example points: all four listed N values at M=1, 2, 8, 32, 64 and 128 for both formats, then N=5120 and 17408 at M=256 for both. Each completed point says `Disposition: Passed` against CUTLASS's own host reference and reports one 20-iteration average initialized-GEMM time, not raw samples or conversion-inclusive cost. The next line is `NVFP4 m=256 n=248320 k=5120`, with no disposition or timing. The main thread stopped the harness container using `docker stop --time 1` after observing the example at roughly 100% host CPU, 0% GPU utilization, and about 1.5 GiB GPU allocation while it computed the full-vocabulary host reference. The remaining 19 points were not reached. Neither a complete benchmark nor native-MMA instruction, resource, memory-budget, or quality evidence was then available. The later pinned GPU bring-up section records the revised harness run.

### Architecture blocker

Record none or the complete ledger-defined blocker report.

### Follow-up observations

Remaining after this historical slice: validate kernels against an independent contraction of reconstructed operands (including tails/scales/zero blocks and the BF16 output epilogue); use actual representative BF16 model weights and activations; measure quantization/packing plus dispatch/GEMM/output work; add feasible Q4-to-BF16 and BF16 library controls plus same-weight GEMV probes; establish profiler/instruction support; and complete the full memory budget. The original script's command loop covered listed major projection dimensions and M values, but not unaligned tails or conversion costs. Its synthetic CUTLASS example operands are useful kernel bring-up only. This task remains IN_PROGRESS and no kernel, correctness, or performance acceptance criterion is yet claimed.

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
harness had shell syntax and source-transform checks only at this stage; the
later pinned GPU bring-up section records its first container execution.

Scope limit: the inspected CUTLASS v4.8.0 SM120 examples initialize packed
inputs and scales for host reference GEMM, but do not define a BF16-to-FP4
quantizer recipe or a project payload packing convention. This CPU reference
records a clear logical scaling convention and does not validate a GPU
candidate, CUTLASS physical scale-factor placement, full edge encodings, or
conversion costs. TASK-019 remains `IN_PROGRESS`; this slice satisfies no
kernel acceptance criterion.

### Pinned GPU bring-up evidence (2026-09-24)

The revised harness was run with `./scripts/task019_cutlass_sm120.sh` on the
actual GPU. The pinned checkout is clean at
`098de2a652cf8f00fd70b2df54051c7eccbb855a`; CMake configured
`CUTLASS_NVCC_ARCHS=120a`, enabled examples and tools, and disabled the CUTLASS
library, tests, and profiler. The project image is
`qw38-dev:cuda13.4.1-pinned`, image ID
`sha256:3844dc9c37087cecd40f96e62bd4f305ad405408f0d63312bda8aff8651f2b49`.
The toolchain reported CUDA 13.4.59, CMake 3.28.3, and GCC 14.2.0. The GPU was
an NVIDIA GeForce RTX 5090, compute capability 12.0, driver 590.48.01,
32,607 MiB total VRAM. The saved post-run idle reading was 34 MiB used; it is
not a candidate peak measurement. TASK-018 froze a 2 GiB free-VRAM reserve.

Both pinned candidates compiled for `sm_120a`. The official NVFP4×NVFP4 example
uses FP32 accumulation and writes BF16 output. The MXFP4×MXFP4 source was
produced by changing the example's operand wrapper types. `cuobjdump` found
`OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` in the NVFP4 binary and
`OMMA.SF.16864.F32.E2M1.E2M1.E8` in the MXFP4 binary. Static resource output
reports 168 registers, zero stack, and 1,024 bytes shared memory for the
reported kernel. This establishes generated native block-scaled MMA in these
two example binaries; it does not validate a project kernel or a mixed-format
path.

Exact bounded CUTLASS example checks were rerun with
`--m=32 --n=128 --k=128 --iterations=0`; both printed `Disposition: Passed`.
These are CUTLASS's own host-reference comparisons over its initialized
operands, not independent contractions of reconstructed project-packed
operands. The output is preserved at
`.cache/task019/bounded-host-reference.log`.

The benchmark-only binaries explicitly print `Verification: Skipped` and were
timed with `--iterations=20`. The selected re-run log is
`.cache/task019/representative-kernel-timings.log`; reported values are one
20-iteration GPU-event average per shape, not raw samples or conversion-
inclusive times:

| M | N | K | NVFP4 average (ms) | MXFP4 average (ms) |
| -: | -: | -: | -: | -: |
| 1 | 5,120 | 5,120 | 0.0345856 | 0.0324032 |
| 1 | 17,408 | 5,120 | 0.0645456 | 0.0668016 |
| 8 | 5,120 | 5,120 | 0.0398512 | 0.0321376 |
| 8 | 17,408 | 5,120 | 0.0670560 | 0.0610416 |

The broader harness sweep was stopped after 36 completed synthetic points:
both formats for all four scripted N values at M=1, 2, 8 and 32, and N=5,120
and 17,408 at M=64. The next point was NVFP4 M=64, N=248,320, K=5,120. At
that point the generated example was consuming one host CPU core to initialize
large synthetic host buffers while the GPU reported 0% utilization. The
container was stopped with `docker stop --time 1 ff583911fd87`; the harness
exited 137. The broad sweep's terminal output was not saved as a file, so only
the selected rerun above has durable timing logs. The revised script therefore
demonstrates bounded build, support, and instruction evidence, but the broad
matrix remains incomplete.

Candidate binary SHA-256 identities (binary files only):

| Candidate binary | SHA-256 |
| ---------------- | ------- |
| Official NVFP4 correctness | `6b0e3b1b3ddc457992dfdf090a451c302bc88c7535a8ef9336c6b15a4f8eeb9e` |
| Transformed MXFP4 correctness | `20437cbcf30985caf25051ccffec86484bb45c430a15e4539ee32a56affc56bc` |
| NVFP4 benchmark-only | `d5de5df4b1e29fff5a41ea501d3712ef108e126a803dde59771889e815fb55eb` |
| MXFP4 benchmark-only | `35a590f16f3b9c3c11851893834f6a3eefd82bc974228115d72cc96792837501` |

No BF16 checkpoint tensor, `.qw38` payload, or scale contents were hashed.
There is no TASK-019 artifact/policy identity yet: these runs use CUTLASS's
generated operands and seed, not model weights or activations. Existing
checkpoint metadata and the CPU logical reference are not substitutes for
real candidate inputs.

### Remaining acceptance gaps

The pinned binaries and native instruction evidence establish bounded CUTLASS
bring-up. The repaired support command and focused tests were run in the
supplemental evidence pass, but all five acceptance criteria remain open
pending complete evidence. Substantive gaps remain:

- independent device-kernel comparisons against FP32 contractions of
  reconstructed operands, including scale orientation, tails, zero blocks,
  padding masks, output conversion, and unsupported-shape behavior;
- the real projection-family matrix and representative BF16 model weights and
  activations; the repaired script now describes additional N/K pairs and an
  unaligned diagnostic, but its matrix mode has not run and all current timing
  points use generated operands;
- raw timing samples, quantize/scale/pack/dispatch and output costs, a bounded
  Q4-to-BF16 control, a feasible BF16 library control, same-weight FP4 GEMV
  probes, and measured workspace/peak memory;
- complete resident/transient candidate budgets including weights, scales,
  embeddings/head, state/KV, scratch, optional views, and the frozen 2 GiB
  reserve; and
- a justified candidate shortlist. Until comparative quality, conversion,
  memory, and workload costs exist, retain Q4G64 with BF16 activations only as
  the ledger's eligible fallback control; no FP4 winner is selected.

The original demonstration logs are
`.cache/task019/bounded-host-reference.log`,
`.cache/task019/instruction-evidence.log`, and
`.cache/task019/representative-kernel-timings.log`. These are diagnostic
bring-up results, not acceptance evidence for the missing criteria. No
architecture conflict was found.

### Review repair and supplemental evidence (2026-09-24)

The current `scripts/task019_cutlass_sm120.sh` has two modes. Its default
`support` mode builds the official NVFP4 example and direct MXFP4
operand-wrapper variant, runs only bounded CUTLASS host-reference checks at
M=32, N=128, K=128, prints the exact commands and correctness-binary hashes,
and checks for the format-specific `OMMA.SF` instruction in each correctness
binary. It checks the recorded local Docker image ID before launching. The
optional `matrix` mode additionally builds clearly labeled benchmark-only
copies and attempts the projection inventory at M=1, 2, 8, 32, 64, 128, 256,
512 and 1024. Its N/K pairs are 5120/5120, 17408/5120, 5120/17408,
10240/5120, 6144/5120, 5120/6144, 12288/5120, 1024/5120, 48/5120 and
248320/5120, plus unaligned diagnostic 129/513. Points above its conservative
256 MiB estimated host-buffer cap print `SKIPPED_BY_HARNESS`; example failures
print `UNSUPPORTED_BY_EXAMPLE`. Neither label proves a GPU hardware limit. The
full-vocabulary shape is included in the descriptor but is expected to hit
the harness cap; a bounded real-input consumer remains necessary. Matrix mode
still supplies synthetic kernel diagnostics only.

`scripts/task019_fp4_reference.py` now validates logical and padded geometry
before reconstruction and exposes BF16 output rounding after its FP32
contraction. Focused tests add known E2M1/scale-code bytes for both formats,
zero/tail and malformed-descriptor cases, and nearest-even BF16 store cases.
This remains a logical reference: no device payload/scale-layout adapter or
independent GPU comparison has been added.

The prior `.cache/task019/bounded-host-reference.log` and
`.cache/task019/instruction-evidence.log` came from the earlier full-sweep
harness, and `.cache/task019/representative-kernel-timings.log` came from a
selected manual rerun of its benchmark-only binaries. The instruction log was
dumped from the benchmark-only binaries, while the two bounded `Passed` lines
came from the correctness binaries. The four binary hashes above identify
those existing files; the three logs do not by themselves record the exact
binary path or hash for each command.

Supplemental checks passed:

- `PYTHONPATH=. python3 tests/test_task019_fp4_reference.py -v` — 9/9 passed.
- `python3 -m py_compile scripts/task019_fp4_reference.py tests/test_task019_fp4_reference.py` — passed.
- `bash -n scripts/task019_cutlass_sm120.sh` — passed.
- `./scripts/task019_cutlass_sm120.sh support` — exit 0; full output is
  `.cache/task019/support-evidence-20260924.log`.

The support log records bounded CUTLASS host-reference `Passed` results for
both formats at M=32, N=128, K=128, binary hashes matching the earlier report,
and the NVFP4 and MXFP4 `OMMA.SF` instructions. The container lacks `git`, so
its printed CUTLASS revision is empty; the harness checked the host checkout
against `098de2a652cf8f00fd70b2df54051c7eccbb855a` before launch. The replay's
`cuobjdump -res-usage` call with two binaries printed usage text instead of
resource output while returning success. Separate `cuobjdump
--dump-resource-usage` queries reported, for both kernels, 168 registers, zero
stack, 1,024 bytes shared, and zero local memory; NVFP4 common GLOBAL was 12
bytes and MXFP4 was 1,070 bytes. The corrected diagnostic was not captured in
the replay log, so replay resource reporting remains defective.

Final independent review: `REVIEW: CHANGES_REQUIRED`. It confirmed the
supplemental test and bounded support results, but required evidence remains
incomplete and identified the revision/resource logging defects above. Since
one code-repair round and one supplemental evidence round have been used,
TASK-019 is `BLOCKED`. No candidate is selected, no architecture conflict was
found, and TASK-020 remains ineligible.

### Completion after reopening (2026-09-24)

**Result: DONE for feasibility.** The earlier blocked result above is retained
as history. The current evidence establishes correct native dense NVFP4 and
MXFP4 execution, cost and memory bounds, and a shortlist for TASK-020. It does
not promote a precision policy, production artifact ABI, or model quality
result. The user required this repair to run in the main thread, so no new
subagent or independent reviewer was used; the checks below are a main-thread
audit of the raw logs and generated tables.

The reproducible sources are `scripts/task019_cutlass_sm120.sh`,
`scripts/task019_independent_probe.cu`, `scripts/task019_real_inputs.py`,
`scripts/task019_derive_activations.cu`, `scripts/task019_prepare_inputs.sh`,
`scripts/task019_controls.cu`, `scripts/task019_controls.sh`,
`scripts/task019_real_matrix.sh`, `scripts/task019_summarize.py`, and
`scripts/task019_memory_budget.py`. The complete 702-row support/cost table is
[`task019-real-matrix.csv`](../task019-real-matrix.csv); the detailed VRAM
budget and single-family exception arithmetic are
[`task019-memory-budget.json`](../task019-memory-budget.json). Raw 20-sample
records for all 117 cases are in `.cache/task019/real-matrix/core/` and
`.cache/task019/real-matrix/head/`, indexed by
`.cache/task019/real-matrix-core-final-index.log` and
`.cache/task019/real-matrix-head-index.log`. The CSV parser refuses missing
cases, missing samples, failed independent checks, or failed controls.

**Provenance and inputs.** The support replay checks the host CUTLASS checkout
at immutable revision `098de2a652cf8f00fd70b2df54051c7eccbb855a` before
launch and passes that verified revision into the container log. The pinned
image ID is `sha256:3844dc9c37087cecd40f96e62bd4f305ad405408f0d63312bda8aff8651f2b49`;
CUDA is 13.4.59, driver 590.48.01, GPU RTX 5090/SM120, target `sm_120a`.
The final matrix indices record source and binary identities: independent
probe source `e71a7f6698cf3519440500c17d8bd4e529afeb8777d2206f1c524385fecaae3b`,
NV/MX binaries `5eedb63aef09fca33e112062ef6d556aeae41000e446de176ae691043877f4f9` /
`05f474da3692378f2bc6e5fa4ca11e229e30b9ccc0e5a49af2b6cb7dd60cbe62`,
and control binary `1a8623feb61e04f5a1e050edffc2f0a710fa01a1735f46ddc19e82ee34a23415`.
These are code/binary hashes only. No BF16 checkpoint tensor, generated input,
packed payload, or scale content was hashed.

All 13 families use actual BF16 checkpoint weight tensors: layer-0 MLP/GDN,
layer-3 attention, and the full untied vocabulary head. The 5120-wide
activations are deterministic embedding rows normalized with the actual
layer-0 input norm; input metadata records tensor/shard names and token IDs.
The 6144-wide and 17408-wide inputs are computed from those actual weights:
SiLU of the layer-0 GDN z projection and a layer-0 SwiGLU gate/up projection.
They are **representative operator-derived proxies**, not complete GDN or
attention layer activations. TASK-020/022 must use full model traces for
quality and final dispatch decisions. Each M uses the first M rows of its
1024-row input file. The 13 families span M=1, 2, 8, 32, 64, 128, 256, 512,
1024, including N=48 small gates, K=6144/17408, and the full N=248320 head.

**Correctness and native resources.** The repaired support replay is
`.cache/task019/support-evidence-tiled-pack.log`: both bounded CUTLASS
host-reference examples passed, and the separate probe compared packed-byte
reconstruction to device BF16 output for both formats at M=3, logical
N=5/padded N=8, logical K=129/padded K=256, including zero blocks and tails,
and at M=32/N=128/K=128 (4,096 checked outputs per format). Both passed with
zero mismatches. Reconstruction decodes E2M1 nibbles and physical UE4M3/E8M0
scale bytes independently, contracts in FP32 with A[M,K] × B[N,K] transpose,
then rounds to BF16. It checks quantization error against the original BF16
inputs separately. The reference policy uses per-16-value UE4M3 NVFP4 scales
and per-32-value UE8M0 MXFP4 scales, E2M1 nearest-even codes and NVFP4
tensor scale 1; it is identified by the recorded source/binary revisions,
not a production artifact ABI. Every real-shape native result and
same-packed-weight BF16
activation GEMV probe passed; full-N BF16 cuBLAS and bounded Q4G64-to-BF16
unpack/GEMM controls passed independent sampled contractions in every case.
This is 117 cases × 6 paths × 20 raw timings = 14,040 samples, with five
warmups per path. Real-shape numerical checks sample representative output
rows/columns; the bounded edge check exhausts its output.

The replay disassembly contains
`OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` for NVFP4 and
`OMMA.SF.16864.F32.E2M1.E2M1.E8` for MXFP4. Corrected per-binary
`cuobjdump --dump-resource-usage` output is in the support log and
`.cache/task019/real-matrix-resources.log`:

| Kernel | Registers/thread | Stack | Shared/block | Local |
| --- | ---: | ---: | ---: | ---: |
| Native NVFP4 or MXFP4 GEMM | 168 | 0 | 1,024 B | 0 |
| Same-weight GEMV, either format | 20 | 0 | 2,048 B | 0 |
| Q4G64 unpack kernel | 20 | 0 | 0 | 0 |

The selected CUTLASS instantiations give both operands the same wrapper type:
NVFP4×NVFP4 or MXFP4×MXFP4, FP32 accumulation and BF16 output. A mixed
BF16×FP4 or NVFP4×MXFP4 native path is **unsupported by this candidate**:
there is no mixed operand instantiation, compiled kernel, or instruction proof.
The BF16-activation GEMV is a separate CUDA kernel using the identical packed
FP4 weight/scale bytes; it is not a mixed native MMA claim. Unsupported probe
dimensions or non-identity alpha/beta return a concrete error, while all
required real-shape points completed.

**Timing and conversion.** The CSV records logical M/N/K, actual storage
geometry, logical and storage-padded work counts, a threadblock tile-work
estimate from
the example's 128×128×128 tile, BF16 output, operand/scale/output/workspace
bytes, CUDA free memory, one-time BF16 weight packing/upload, and median/p95
of each raw kernel and host-complete time. The native paths have zero CUTLASS
workspace bytes. Control workspace is the explicit user allocation only;
cuBLAS internal allocation is included in its observed free-memory reading
and the model budget's library allowance. Kernel time uses CUDA events;
host-complete time includes
BF16 activation scaling/packing, A/scale upload, dispatch, GEMM, BF16 store,
synchronization and D2H output copy. Weight packing/upload is recorded
separately because it is a one-time preparation cost. The conversion path is a
minimal **CPU reference packer**; it is not a production GPU converter or a
steady-state model timing. The BF16 control transfers BF16 A and D without
packing; the Q4 control measures GPU unpack plus GEMM for at most 128 output
rows. Q4 timings with N>128 therefore cannot be compared as full-N timings.

Selected medians from the complete CSV, in milliseconds; each cell is
kernel/host-complete. Q4's measured width is shown explicitly:

| Family (M,N,K) | NVFP4 | MXFP4 | BF16 full N | Q4G64 measured N: time |
| --- | ---: | ---: | ---: | ---: |
| MLP gate (1,17408,5120) | 0.062/0.600 | 0.061/0.558 | 0.113/0.129 | 128: 0.010/0.022 |
| MLP gate (1024,17408,5120) | 0.166/507.611 | 0.164/488.548 | 0.919/3.459 | 128: 0.020/0.519 |
| GDN small gate (1,48,5120) | 0.026/0.560 | 0.026/0.520 | 0.007/0.017 | 48: 0.009/0.019 |
| Attention q/g (128,12288,5120) | 0.028/63.409 | 0.028/60.973 | 0.098/0.380 | 128: 0.016/0.099 |
| Head (1,248320,5120) | 0.718/1.269 | 0.716/1.245 | 1.641/1.691 | 128: 0.011/0.021 |
| Head (1024,248320,5120) | 2.644/536.557 | 2.600/517.147 | 11.051/40.598 | 128: 0.020/0.515 |

The large native kernel advantage in some aligned shapes is a feasibility
signal only. CPU activation packing reaches about 508 ms at M=1024 MLP gate
and about 1.7 s at M=1024 MLP down, overwhelming the GEMM saving. Small N=48
is faster with the BF16 control at M=1. Same-weight GEMV is measured in every
case but is not automatically better: M=1 MLP gate NVFP4 GEMV kernel median is
about 0.136 ms versus 0.062 ms for native GEMM; the M=1 head GEMV is about
2.2 ms versus 0.718 ms. The present GEMV is a correctness/control probe,
not a tuned decode dispatch. Observed sampled quantization differences from
BF16 reach 0.703125 (NVFP4), 0.958008 (MXFP4), and 0.570312 (bounded Q4G64)
in absolute BF16 output units. Their differing input formats and sampled
coverage prohibit a quality ranking; TASK-020 owns calibration and screening.

**VRAM budget.** The budget is inventory-derived for the active language model
with inactive MTP payload retained, full BF16 embedding, Q8G32 head, all
projection payloads/scales, 144 MiB-class FP32 GDN/history state, BF16 KV for
32,768 tokens (2 GiB), 1 GiB activation/output/scratch, 512 MiB library
workspace, 256 MiB graphs, 128 MiB metadata/alignment and a frozen 2 GiB free
reserve. The independently measured `cudaMemGetInfo` starting free amount is
33,099,350,016 bytes (31,566 MiB). Native logs record allocation before/after
and exact operands/scales/output/workspace; control logs record free memory
after allocation. Whole-model candidate
peaks are **estimates**, not measured integrated runtime peaks:

| One-view projection policy, Q8 head | Resident weights/scales/retained payload (MiB) | Estimated peak incl. state/transients (MiB) | Headroom after 2 GiB reserve (MiB) |
| --- | ---: | ---: | ---: |
| Q4G64 | 16,303.5 | 20,418.3 | 9,099.7 |
| NVFP4 | 17,028.5 | 21,143.3 | 8,374.7 |
| MXFP4 | 16,303.5 | 20,418.3 | 9,099.7 |

All one-view policies fit this measured free-capacity boundary under those
allowances. An additional complete Q4 projection view adds 12,923,699,200
bytes and does **not** fit any policy with the reserve. The JSON gives
per-family FP8/BF16 replacement deltas and single-family fit results for each
base policy, plus Q8/BF16/FP8/FP4 head option headroom. All individual
family exceptions fit; simultaneous exceptions must be summed against the
remaining headroom. FP8 and FP4 head entries are **storage estimates only**,
not validated kernels or quality candidates. This is a budget for choosing
which policies merit TASK-020 screening; actual integrated peak remains a
TASK-022/026 obligation.

**Feasible shortlist and fallback.** Carry NVFP4×NVFP4 and MXFP4×MXFP4
forward as native large-projection feasibility candidates, with a single
resident view and Q8 head baseline. Carry BF16 for narrow gates and selective
high-precision exceptions. Retain Q4G64 weights with BF16 activations and the
existing Q8 head as the eligible fallback. TASK-020 must assess activation
and weight error with representative full-model traces, calibrate selective
precision, and decide if GPU conversion and decode dispatch can recover the
kernel-only benefit. TASK-019 makes no final candidate, quality, or
end-to-end performance claim and found no architecture blocker.

Reproduction commands (after preparing the pinned checkpoint and container):
`./scripts/task019_cutlass_sm120.sh support`,
`./scripts/task019_prepare_inputs.sh`,
`./scripts/task019_controls.sh`,
`./scripts/task019_real_matrix.sh core`,
`./scripts/task019_real_matrix.sh head`,
`python3 scripts/task019_summarize.py`, and
`python3 scripts/task019_memory_budget.py --free-mib 31566 > docs/implementation/task019-memory-budget.json`.
Focused final checks passed: all nine
`tests/test_task019_fp4_reference.py` cases, `python3 -m py_compile` for the
TASK-019 Python sources, `bash -n` for the four TASK-019 shell scripts,
matrix/CSV/budget assertions, and staged whitespace validation.

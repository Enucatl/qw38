# TASK-001 — Reproducible CUDA build foundation

## Status
DONE

## Milestone
M0 — Reproducible implementation foundation

## Purpose
Create the architecture-neutral container, build, test, and GPU-smoke foundation on which every V0 task runs.

## Depends on
- None.

## Normative references
- `docs/architecture/architecture-v0.md` — Implementation baseline
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`

## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| I-01–I-05 | Linux/Docker/NVIDIA runtime, CUDA 13.4.x, RTX 5090 `sm_120`, C++23/CUDA C++23 | LOCKED |
| I-07 | No old-GPU or old-toolchain compatibility in V0 | LOCKED |

## Starting point
The repository has documentation and Python analysis scripts only: no Dockerfile, CMake project, native sources, native tests, benchmarks, or engine CLI.

## Scope
- Pin a CUDA 13.4.x development image/base distribution and compatible host compiler/CMake versions; use an immutable digest when obtainable.
- Add the minimal CMake project with separate ordinary C++ and narrow CUDA targets, Debug/Release support, native `sm_120`, C++23, CUDA C++23, CTest, and a distinct benchmark target area.
- Add host `std::expected` compile smoke, CUDA kernel/runtime smoke, and documented container build, test, and GPU-run commands.
- Validate and report driver/container compatibility from inside the container.

## Out of scope
- Engine functionality, model loading, `.qw38`, performance kernels, old targets, PTX fallback, packaging, CI, or a generic dependency framework.

## Required interfaces
- One documented container build command, one CMake configure/build command, and one normal CTest command.
- A trivial CUDA executable that launches a kernel, checks its result, and reports the active device and compiled target evidence.

## Required semantics
Smoke failures return nonzero. The CUDA test must execute on hardware, not merely compile.

## Data representation
No model data. Generated build output remains outside source-controlled source directories.

## Implementation constraints
- Keep host-only code outside CUDA compilation.
- Do not add compatibility shims for pre-C++23 or pre-Blackwell systems.
- Benchmarks must not run as correctness tests.

## Tuning defaults
- Native CUDA architecture: `120` / `sm_120`.

## Expected files/modules
Root build/container files plus minimal `src/`, `cuda/`, `tests/`, and `benchmarks/` skeletons justified by the existing empty native tree.

## Tests required
### Unit tests
- Compile and execute a host C++23 `std::expected` smoke.
### Reference/numerical tests
- None.
### Integration tests
- Build Debug and Release in the container; run CTest; run the CUDA smoke on RTX 5090; inspect the produced binary for native `sm_120` code.

## Benchmark required
No.

## Acceptance criteria
- [x] Image/tool versions are pinned and documented.
- [x] Host and CUDA C++23 targets compile.
- [x] CUDA smoke executes correctly on RTX 5090.
- [x] Binary inspection proves the intended Blackwell target.
- [x] The normal test command passes.
- [x] No engine behavior or compatibility work was added.

## Architecture blocker rule
If a locked decision prevents correct implementation, stop and report `ARCHITECTURE_BLOCKER` with: Decision ID; Attempted implementation; Observed problem; Evidence; Why this is architectural rather than tuning; Smallest plausible alternative; Affected downstream tasks. Do not silently substitute a design.

## Completion report
### Result
DONE — independent verification PASSED (2026-09-21): all acceptance criteria met via docker build, Debug/Release cmake+ctest (3/3 each), CUDA smoke on RTX 5090, and cuobjdump `sm_120` proof.
### Changes made
- Added `Dockerfile` pinning `nvidia/cuda:13.4.1-devel-ubuntu24.04@sha256:a3f014424ccc86cf66c84d700d2896491424205dd9457e3116ed4f0e96381311` (Ubuntu 24.04.5, CUDA 13.4.1 / nvcc V13.4.59, GCC 14.2.0, CMake 3.28.3, Ninja 1.11.1).
- Added root `CMakeLists.txt` with separate host CXX and CUDA targets, Debug/Release, native `120-real` (`sm_120` cubin only), C++23, CUDA C++23, CTest, and an `EXCLUDE_FROM_ALL` benchmark placeholder.
- Added host `std::expected` smoke in `src/`, CUDA kernel/runtime smoke in `cuda/`, CTest wiring plus `cuobjdump` `sm_120` check in `tests/`, and `benchmarks/` skeleton.
- Documented image pins and the three required commands in `docs/implementation/dev-environment.md`.
- GCC 14 is the image host compiler because nvcc 13.4 ignores `-std=c++23` with GCC 13. CMake 3.28's NVIDIA module does not record CUDA23 flags; `CMakeLists.txt` supplies `-std=c++23` for nvcc 13.3+ so the required dialect is actually enabled.
### Tests run
Exact commands and results.

Container build:

```text
docker build -t qw38-dev:cuda13.4.1 -f Dockerfile .
```

Result: success. Image `qw38-dev:cuda13.4.1` (`sha256:7161dc0d3410a1a8afad60e09977d5ab57fc74d7fec8cda845ac513f14a8fe13`).

Debug configure/build/test:

```text
docker run --gpus all --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 \
  bash -lc 'cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug && cmake --build build/debug && ctest --test-dir build/debug --output-on-failure'
```

Result: configure GNU 14.2.0 + NVIDIA 13.4.59, arch `120-real`; 3/3 tests passed (host_expected_smoke, cuda_runtime_smoke 0.23s, cuda_sm120_cubin).

Release configure/build/test (normal CTest command after the matching configure/build):

```text
docker run --gpus all --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 \
  bash -lc 'cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release && cmake --build build/release && ctest --test-dir build/release --output-on-failure'
```

Result: 3/3 tests passed (host_expected_smoke, cuda_runtime_smoke 0.17s, cuda_sm120_cubin).

CUDA smoke on RTX 5090 (`./build/release/cuda/qw38_cuda_runtime_smoke`):

```text
device_name=NVIDIA GeForce RTX 5090
device_compute_capability=12.0
cuda_driver_version=13.4
cuda_runtime_version=13.4
compiled_cuda_arch=1200
kernel_result=42
cuda runtime smoke ok
```

Binary inspection (`cuobjdump -lelf build/release/cuda/qw38_cuda_runtime_smoke`):

```text
ELF file    1: qw38_cuda_runtime_smoke.1.sm_120.cubin
```

`cuobjdump -ptx` reported fatbin ELF `arch = sm_120` and no PTX image.

Driver/container check (`docker run --gpus all --rm qw38-dev:cuda13.4.1 nvidia-smi`): driver 590.48.01, container CUDA 13.4, GPU `NVIDIA GeForce RTX 5090`, 32607 MiB. Kernel launch succeeded, so the NVIDIA runtime + this driver ran CUDA 13.4 `sm_120` code.

CTest lists only the three correctness tests. `qw38_bench_placeholder` builds only when requested and is not registered with CTest.
### Benchmark results
Not required.
### Architecture blocker
None.
### Follow-up observations
- Host `nvidia-smi` on this machine reports CUDA 13.1 while the CUDA 13.4.1 container reports CUDA 13.4; the 590.48.01 driver still executed the smoke kernel. Downstream GPU work should keep validating runtime/driver pairing from inside the container rather than trusting the host `nvidia-smi` CUDA column alone.
- CMake 3.28 does not yet map NVIDIA CUDA C++23; the project records `-std=c++23` itself. A later CMake that knows CUDA23 for nvcc 13.4 can drop that assignment without changing dialect policy.


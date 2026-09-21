# V0 development environment

Pinned CUDA 13.4.x container, CMake, and GPU-smoke commands for Architecture V0. This is the TASK-001 implementation of [technology-baseline.md](technology-baseline.md). It does not change Architecture V0.

## Pinned image and tools

| Item | Pin |
| --- | --- |
| Base image | `nvidia/cuda:13.4.1-devel-ubuntu24.04` |
| Manifest-list digest | `sha256:a3f014424ccc86cf66c84d700d2896491424205dd9457e3116ed4f0e96381311` |
| linux/amd64 image digest | `sha256:f0eec85d23f7fdff7b6d74feb1d63e1e5921b0a92c3adfd9847875f830eb0382` |
| Base distribution | Ubuntu 24.04.5 LTS (noble) |
| CUDA Toolkit | 13.4.1 (`nvcc` V13.4.59, built 2026-08-03) |
| Host compiler | GCC/G++ 14.2.0 (`g++-14`) |
| CMake | 3.28.3 (`3.28.3-1build7`) |
| Ninja | 1.11.1 |
| Native CUDA architecture | `120-real` (`sm_120` cubin only) |
| Language dialects | C++23 host, CUDA C++23 device |

The Dockerfile `FROM` line pins the manifest-list digest. `CC`/`CXX`/`CUDAHOSTCXX` are `gcc-14`/`g++-14` because `nvcc` 13.4 ignores `-std=c++23` with GCC 13.

CMake 3.28's NVIDIA module records CUDA standards only through C++20. `CMakeLists.txt` supplies `-std=c++23` for nvcc 13.3+ so `CMAKE_CUDA_STANDARD 23` with `CMAKE_CUDA_STANDARD_REQUIRED ON` is real, not decayed to C++20.

## Commands

Run these from the repository root. The image provides the toolchain; source is bind-mounted.

**Container build**

```bash
docker build -t qw38-dev:cuda13.4.1 -f Dockerfile .
```

**CMake configure/build** (Release shown; use `build/debug` and `-DCMAKE_BUILD_TYPE=Debug` for Debug)

```bash
docker run --gpus all --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 \
  bash -lc 'cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release && cmake --build build/release'
```

**Normal CTest command**

```bash
docker run --gpus all --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 \
  ctest --test-dir build/release --output-on-failure
```

`ctest` runs host `std::expected` smoke, CUDA kernel/runtime smoke on the GPU, and `cuobjdump` confirmation of native `sm_120` cubin. The `qw38_bench_placeholder` target is `EXCLUDE_FROM_ALL` and is not a CTest test.

## Driver / container compatibility

Validate from inside the container:

```bash
docker run --gpus all --rm qw38-dev:cuda13.4.1 nvidia-smi
```

The CUDA smoke also prints `device_name`, compute capability, `cuda_driver_version`, and `cuda_runtime_version`. The host driver must expose the GPU through the NVIDIA Container Toolkit. V0 does not hard-code a host driver version.

## Binary inspection

```bash
docker run --gpus all --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 \
  cuobjdump -lelf build/release/cuda/qw38_cuda_runtime_smoke
```

Expected: `ELF file    1: qw38_cuda_runtime_smoke.1.sm_120.cubin`.

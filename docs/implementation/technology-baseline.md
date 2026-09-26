# Technology baseline

This document is the normative V0 reference development, build, and runtime environment. [Architecture V0](../architecture/architecture-v0.md) defines retained semantics and controls; OVERALL-01, FP4-01, DELIVERY-01 and [the task ledger](task_ledger.md) govern candidate selection and implementation order for TASK-018–032. [Code standards](code-standards.md) define how its C++ and CUDA implementation is written.

# Reference platform

| Concern | V0 reference |
| --- | --- |
| Operating system | Linux |
| GPU | NVIDIA GeForce RTX 5090, Blackwell, compute capability 12.0 |
| Native CUDA target | `sm_120` |
| CUDA Toolkit | CUDA 13.4.x |
| Host language | C++23 |
| Device language | CUDA C++23, limited to the subset supported by the pinned CUDA toolchain |
| Build system | CMake |
| Build and deployment environment | Docker with NVIDIA Container Toolkit |

V0 is allowed to optimize specifically for RTX 5090 / Blackwell `sm_120`. Compatibility with pre-Blackwell NVIDIA GPUs is not a V0 constraint. A later compatibility phase may add older GPU architectures, PTX fallback, or other backends after the reference implementation is correct and measured.

# Container model

```text
HOST
Linux
NVIDIA driver
RTX 5090
Docker
NVIDIA Container Toolkit
        │
        ▼
CONTAINER
CUDA 13.4.x toolkit/runtime
host compiler toolchain
CMake
qw38 binaries
        │
        ▼
GPU
sm_120
```

The host owns the NVIDIA GPU, a compatible NVIDIA driver, Docker, and NVIDIA Container Toolkit/NVIDIA runtime integration. The container owns CUDA userspace and toolkit components, `nvcc`, the C++ compiler toolchain, CMake, build dependencies, runtime userspace libraries, and the qw38 source, build, and runtime environment.

The engine must not require the CUDA Toolkit to be installed directly on the host. The host driver must be compatible with the selected CUDA container/toolkit. The compatible driver range is an environment prerequisite to validate when the image is selected; V0 does not hard-code a host driver version without a demonstrated deployment requirement.

Containerization supplies a reproducible build environment, reproducible CUDA/userspace stack, and deployment isolation. It does not alter the retained model semantics or state contracts. OVERALL-01 reopens the V0 weight/activation formats, physical weight views, and schedules for measured candidate selection; platform/toolchain requirements in this document remain binding.

# Reproducibility policy

The implementation must eventually use a pinned development image. Do not depend on a floating `latest` CUDA image. The environment must pin at least:

- a CUDA 13.4.x image/tag;
- the base Linux distribution;
- the compiler version;
- the CMake version where reproducibility requires it.

Prefer an immutable image digest once the initial environment works. The later implementation ledger must include an early environment/bootstrap task that validates the GPU target, driver/container compatibility, compiler pair, and build tools. This policy does not define or modify a Dockerfile in the present documentation task.

# Toolchain policy

The reference toolchain prioritizes modern C++ support, CUDA 13.4, native Blackwell support, reproducibility, developer productivity, profiling, and kernel development. Architecture V0 need not support old host compilers or old CUDA releases. Compatibility substitutes for old C++17 toolchains are not required.

CMake is the normative V0 build system. Its language-standard requirements are conceptually equivalent to:

```cmake
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
```

The reference environment compiles natively for `sm_120`. PTX fallback and additional architecture targets are later compatibility decisions, not V0 requirements. The pinned host compiler, `nvcc`, standard library, and CUDA Toolkit combination must be validated together before implementation depends on particular language features.

# Host and device language boundary

Host code is ordinary C++23 and may use modern standard-library features supported by the pinned compiler. CUDA code is CUDA C++23 using the subset supported by the pinned CUDA toolchain; this policy does not claim that every C++23 feature is valid in device code.

```text
compiler / format / runtime orchestration
        → ordinary C++23

thin CUDA launch/resource boundary
        ↓

CUDA kernels and device utilities
        → CUDA-compatible C++23 subset
```

Keep CUDA compilation surfaces narrow. Host-only compiler, artifact, and orchestration logic should not acquire CUDA language constraints unnecessarily.

# Build-system scope

The later CMake implementation must support at least:

- an ordinary debug build;
- an optimized release build;
- CUDA architecture selection, with `sm_120` as the reference;
- correctness tests;
- benchmarks kept distinct from correctness tests.

Do not design a complicated packaging system for V0. Containerization is the primary reproducible development and deployment environment.

# Normative authority

Future implementation task specifications must treat:

- [Architecture V0](../architecture/architecture-v0.md) as authority for model/runtime architecture;
- this technology baseline as authority for platform, toolchain, build, and container assumptions;
- [Code standards](code-standards.md) as authority for C++/CUDA implementation conventions.

If a task conflicts with the platform or toolchain requirements here, report the conflict. For candidate architecture and task ordering in TASK-018–032, follow OVERALL-01, FP4-01, DELIVERY-01 and their revised ledger contracts; this baseline does not require the historical Q4/Q8 implementation to be completed as a quality/performance gate first.

# Code standards

This document is normative for V0 implementation work. [Architecture V0](../architecture/architecture-v0.md) defines the model/runtime architecture, while the [technology baseline](technology-baseline.md) defines the reference platform, toolchain, build, and container environment. The future `implementation_ledger.md` will define implementation order; it is intentionally not created here.

# Design philosophy

The engine is specialized for Qwen3.8. Prefer static, explicit, value-oriented, model-specific C++ abstractions over generic runtime-polymorphic operator frameworks. Suitable concepts may include `CompiledModel`, `Runtime`, `Session`, `GdnState`, `KvCache`, `ScratchArena`, `GdnPlan`, `AttentionPlan`, and `MlpPlan`; these names are illustrative and are not frozen APIs.

Do not center the implementation on interfaces such as:

```cpp
class Operator {
public:
    virtual Tensor forward(...) = 0;
};
```

V0 does not implement a universal `Tensor` class, generic dynamic shape inference, a generic operator registry, a runtime graph optimizer, arbitrary model loading, or a plugin operator system. The implementation is allowed to know the Qwen3.8 architecture.

# Errors and exceptions

Fallible engine APIs use explicit typed propagation:

```cpp
std::expected<T, Error>
std::expected<void, Error>
```

Use this model for file access, artifact parsing and validation, compiler stages, quantization, CUDA initialization, allocations, uploads, session creation, and runtime setup. CUDA failures must be translated close to their source into the engine's typed error model; raw CUDA status handling must not leak through unrelated layers.

Engine APIs do not use exceptions as their operational contract. Device code must not depend on exceptions, and implementation code must not deliberately throw for ordinary expected failures. If external or standard-library internals can throw unavoidably, contain and translate those exceptions at the relevant boundary. This policy does not mandate global `-fno-exceptions`.

# Ownership and borrowed memory

RAII is mandatory for owned resources, including files, mappings, host allocations, device allocations, CUDA streams, CUDA events, CUDA graphs, and future CUDA handles. Prefer value ownership and use `std::unique_ptr` only when dynamic ownership is genuinely needed. Do not spread manual allocate/free pairs across call sites, rely on implicit global lifetime, or leave pointer ownership ambiguous.

Use explicit non-owning views for borrowed contiguous memory. Prefer `std::span<T>` and `std::span<const T>` to raw pointer/length pairs where appropriate. Typed model or tensor views may carry a pointer, logical dimensions, storage class, and physical-layout identity. Such a view describes known memory; it neither owns storage nor dynamically dispatches computation, and it must not grow into a generic tensor framework.

# Modern C++ policy

Use supported C++23 facilities when they improve clarity or safety, including:

- `std::expected`, `std::span`, and `std::byte`;
- `std::bit_cast`, `std::endian`, and `std::byteswap`;
- `std::source_location` and `std::to_underlying`;
- concepts and `requires` when they materially improve correctness;
- ranges when they clarify host-side compiler or tooling code.

Do not add compatibility substitutes solely for old C++17 toolchains. Use advanced language machinery only when it improves clarity or safety; avoid template metaprogramming for its own sake.

# Polymorphism and runtime structure

Avoid virtual dispatch in hot inference paths. Prefer concrete types, enums or tags, direct calls, immutable execution plans, and compile-time specialization where useful. Core runtime behavior must not require RTTI. Do not create object hierarchies merely to appear extensible; V0 prioritizes clarity and model specialization.

# CUDA boundary and wrappers

Organize the code conceptually as:

```text
compiler/   ordinary C++23
format/     ordinary C++23
runtime/    ordinary C++23
cuda/       kernels, launch wrappers, CUDA resource helpers
```

Repository context may justify different directories. The normative rule is to concentrate `.cu` files, device implementation, and CUDA-specific headers behind a narrow boundary. Do not spread `__host__ __device__` annotations through the engine unnecessarily.

Ordinary engine code should not repeatedly perform a CUDA call, manually inspect its status, translate it, and arrange cleanup. Provide small typed RAII/error wrappers for device memory, streams, events, launches, and other CUDA resources. Keep them thin: the purpose is explicit ownership and consistent error handling, not hiding CUDA behind a large abstraction framework.

# Allocation policy

Steady-state inference reuses preallocated memory:

```text
model load
    ↓
session creation
    ↓
allocate persistent state + scratch
    ↓
decode / prefill
    ↓
reuse stable buffers
```

Avoid uncontrolled allocation and free operations in per-token hot paths. Stable addresses support predictable latency, profiling, future CUDA graph capture, and clear lifetime reasoning. Temporary host objects outside hot paths remain acceptable.

# Dependencies

Keep the dependency surface small. The default foundation is the C++ standard library, CUDA Toolkit, CMake, and ordinary testing/build utilities. PyTorch, TensorFlow, generic tensor libraries, and large ML runtimes must not become inference-engine dependencies.

A third-party dependency is acceptable when it supplies substantial capability that is risky or wasteful to reproduce. Any significant dependency added later requires explicit rationale.

# Binary artifact implementation

Implement `.qw38` parsing and writing explicitly. Prefer `std::byte`, `std::span`, `std::bit_cast`, `std::endian`, `std::byteswap`, and strong enum types. Avoid pointer punning, unaligned reinterpretation, and other undefined behavior.

Validate all offsets, lengths, layout versions, and relationships before use. Do not serialize compiler-dependent C++ struct layout directly as the persistent file ABI unless Architecture V0 explicitly specifies that representation.

# Validation, assertions, and diagnostics

External or untrusted data failures return typed errors. This includes malformed `.qw38` data, invalid offsets, unsupported versions, mismatched source hashes, and allocation failures. Assertions are appropriate for internal programmer invariants; they are never the only validation of artifact input.

Keep diagnostics simple and explicit; do not introduce a large logging framework by default. Important errors should carry enough structured context to identify the operation and, where relevant, tensor or layout, CUDA error, file offset, and format version. `std::source_location` may support internal diagnostics where useful.

# Testing policy

Implementation tasks should prefer deterministic unit tests, understandable reference paths for numerical kernels and formats, and integration tests across explicit interfaces. Keep benchmarks separate from correctness tests.

An optimized CUDA kernel must not become the sole definition of correctness. Maintain an independently understandable reference implementation for formats and numerical operations where practical.

## Verification scope

Verification is proportional to the change. A repair must run the smallest
test set that directly exercises its changed behavior, plus any focused
regression needed to cover a shared interface. Do not run the complete Debug
and Release suites merely because a repair changes one component.

Use one configured build by default. Prefer Debug for host-side parsing,
format, compiler, and validation changes. Prefer Release for CUDA numerical
or runtime changes. Run both configurations only when the change is known to
be configuration-sensitive, changes build/toolchain behavior, or a task file
explicitly requires both.

Every test command must identify the behavior it validates. Reuse an existing
configured build and build only affected targets when practical. A delivery or
review step must inspect the diff and reported focused results; it must not
repeat an unchanged test run without a concrete reason.

## Fast and extended tests

The normal repair command is the fast suite, selected with CTest labels that
exclude `extended` tests. Extended tests cover expensive evidence such as
authoritative checkpoint scans, full-vocabulary contractions, maximum-context
allocations, long-context runs, sanitizers, and benchmarks. They are not a
default requirement for unrelated repairs.

Run an extended test only when the changed behavior can affect the property it
covers, when a task file explicitly requires it, or at a deliberate release
or repair-batch checkpoint. State the reason before running it. Use the
smallest fixture that reproduces the defect: for example, test a segment
boundary at 255/256/257 rather than a long-context sweep, and test malformed
hash handling with a small fixture rather than repeatedly hashing an
unchanged checkpoint.

Cached or previously computed identities may be reused within one operation
when their source bytes are immutable for that operation. Content identity
remains mandatory at artifact/compiler boundaries; this policy only avoids
repeating the same expensive evidence in unrelated test executions.

# Performance policy

Do not sacrifice clarity outside demonstrated hot paths. Use profiling evidence before introducing difficult low-level host-side optimizations. CUDA kernels and model-format consumers are intentionally performance-oriented; host compiler and tooling code should emphasize correctness, explicit ownership, understandable transformations, and deterministic behavior.

# Normative authority

Future implementation task specifications must treat:

- [Architecture V0](../architecture/architecture-v0.md) as authority for model/runtime architecture;
- [Technology baseline](technology-baseline.md) as authority for platform, toolchain, build, and container assumptions;
- this code standard as authority for C++/CUDA implementation conventions.

If task instructions conflict with these documents, report the conflict rather than silently resolving it.

# 12. Language and platform strategy

[Previous](11-glossary-worksheets.md) · [Index](README.md) · [Sources](sources.md)

## Why this matters

Language choice does not make matrix multiplication intrinsically faster. The
kernel algorithm, generated instructions, memory traffic, and launch schedule do.
The language decides how directly this narrow engine can reach those mechanisms,
how safely it owns resources, and how much glue each platform requires.

This chapter makes two **Proposed** decisions: use restricted C++17 for the
portable production core, and support hosts in tiers rather than forcing CUDA,
Metal, and CPUs through a lowest-common-denominator GPU abstraction. The RTX
5090 remains the primary optimization target. DGX Spark and Apple Silicon must
be useful, while CPU execution supplies portable correctness and selected
practical optimizations.

## Host language is not kernel language

The host runtime loads artifacts, validates shapes, owns buffers, schedules
work, manages sessions, and serializes checkpoints. A GPU kernel runs thousands
of parallel invocations over tensors. They need not use the same language:

| Role | Choice | Boundary |
|---|---|---|
| Portable host runtime | restricted C++17 | model, session, plan, and backend contracts |
| NVIDIA kernels | CUDA C++ and CUTLASS where useful | CUDA translation units behind the backend |
| Apple runtime and kernels | Objective-C++ bridge and Metal Shading Language | Metal objects do not leak into the portable core |
| Offline and test tooling | Python | conversion, reference traces, calibration, and benchmarks |

### Why restricted C++17

C++ is the best fit here because CUDA and CUTLASS expose their native interfaces
in C++, Objective-C++ can call both C++ and Apple's Objective-C Metal APIs in one
translation unit, and compilers expose x86-64 and ARM64 SIMD from C++. RAII also
ties CUDA streams, Metal objects, mappings, files, and allocations to explicit
lifetimes. A narrow C ABI can sit at the public edge without constraining the
implementation. These are integration and ownership advantages, not a claim
that C++ syntax produces faster code.

The costs are real: C++ permits hidden allocation, complex lifetime behavior,
slow builds, template error cascades, and accidental framework growth. Keep the
core deliberately small:

- Use plain data structures for shapes, manifests, plans, and serialized state.
- Use move-only owners for resources; make borrowed views visibly non-owning.
- Limit templates to local numeric or dispatch code with demonstrated value.
- Do not build a generic tensor framework or operator registry.
- Do not build an inheritance-heavy backend hierarchy; prefer a small table of
  coarse operations and backend-owned state.
- Keep exceptions and RTTI out of ABI and device boundaries; report failures in
  an explicit, testable form.

C remains reasonable for a smaller single-platform runtime, a stable ABI layer,
or a project whose team values manual ownership over C++ interoperation. Rust is
reasonable when memory safety dominates and the required CUDA/Metal bindings
are already proven; here, native CUDA/CUTLASS and Objective-C++ integration
would otherwise add a second ownership and FFI system around the hot path.
Python is ideal for iteration and scientific comparison, but interpreter and
object overhead, dependency behavior, and indirect resource ownership make it
the wrong final scheduler for latency-sensitive decode. It remains essential
outside that path.

## Portable logical architecture

Portability begins with shared meaning, not identical buffers. Every backend
uses the same validated `ModelSpec`, tokenizer and chat template, canonical
model artifact, sampling rules, session semantics, and versioned checkpoint
format. Those components define what an execution means.

`BackendOps` is coarse. It provides capability discovery, weight preparation,
execution-plan creation, session allocation, prefill, decode, and checkpoint
import/export. The test surface also exposes semantic primitives—projection,
norm, RoPE, GDN, attention, FFN, and logits—so a backend can be compared with
the scalar oracle at named taps. Production does not have to dispatch those
primitives individually: CUDA or Metal may fuse them and schedule a whole layer
or request.

```text
canonical artifact + ModelSpec
              |
        prepare_weights
              |
   backend repack cache -------- architecture + quant policy + layout version
              |
     execution plan + session
              |
          prefill/decode
```

There is one canonical *logical* artifact. A backend may create a repacked cache
whose key includes the model identity, quantization policy, target architecture,
and layout version. A cache entry is disposable and reproducible; it is never a
second authority for tensor meaning.

A checkpoint similarly records logical state: committed token and position
frontiers, recurrent matrices and convolution history, logical KV rows, sampler
state, and format/version metadata. Export converts from a live tiled layout;
import validates and repacks for the destination. Thus a CPU checkpoint can
continue on CUDA or Metal without those backends using identical pitches,
tiles, dtypes, or allocation addresses. Cross-backend continuation still must
pass declared numeric tolerances and sequence tests; bit identity is not assumed.

## Target policies

### Tier 1: RTX 5090

The **Proposed** host target is Linux/x86-64. The **External** GPU facts are CUDA
compute capability 12.0 (`sm_120`) and 32 GiB of device memory; see the vendor
references in the [source ledger](sources.md). The **Proposed** policy keeps
quantized weights resident, makes host/device transfers explicit, proves fit
with an allocation ledger that includes state, workspaces, repacks, graphs, and
reserve, and uses CUDA Graphs only after address and shape stability are
established. Correctness, fit, TTFT, ITL, and throughput gates on this exact
target are the primary gates.

### Tier 2: DGX Spark

The **External** platform is Linux/ARM64 with CUDA, a compute capability 12.1
GB10 GPU (`sm_121`), and 128 GB of coherent unified system memory. Toolchain
support still must be pinned before treating that target spelling as a build fact.
The **Proposed** policy avoids duplicate host and device copies of the full
weight set, admits work by total capacity and working set, and tunes specifically
for Spark rather than reusing 5090 launch parameters. Unified memory does not
make migration or locality free. Spark therefore gets its own correctness and
performance gates, not scaled 5090 expectations.

The 5090 and Spark share model semantics and CUDA kernel sources where the
generated code is valid. They do not share a memory policy: discrete 5090 memory
requires an explicit 32 GiB residency proof, while Spark's larger coherent pool
favors avoiding duplication and controlling placement and working-set pressure.

### Tier 2: Apple Silicon

The **External** platform is macOS/ARM64 with Metal and a CPU/GPU unified-memory
architecture. This is one physical memory pool, not separate RAM plus VRAM.
The **Proposed** backend owns Metal buffers and pipelines behind an Objective-C++
bridge, applies working-set admission, and makes CPU/GPU command ordering and
synchronization explicit. It uses Metal-specific kernels and a Metal-specific
command-buffer or indirect-command replay strategy; CUDA Graphs are not treated
as a portable API or design requirement.

### Tier 3: CPU-only

The scalar CPU backend is the portable semantic oracle. It favors obvious loops,
intermediate taps, and deterministic small fixtures, not production throughput.
Separate optimized x86-64 and ARM64 paths use runtime ISA dispatch, quantized
SIMD kernels, bounded threading, and an explicit NUMA policy where applicable.
Any speed claim is **Proposed** until measured on a named machine. For this 27B
engine, expectations should be set by measured memory bandwidth and bytes read
per token rather than advertised vector-operation peaks.

## Staged implementation

Each stage keeps the preceding correctness gates live:

1. Implement portable scalar CPU semantics and the canonical artifact format.
2. Bring up an RTX 5090 high-precision CUDA path against scalar and external traces.
3. Add 5090 quantization, the 32 GiB fit proof, graphs, and profiler-led optimization.
4. Build on DGX Spark ARM64 and bring up its CUDA correctness path.
5. Add Spark's unified-memory policy and architecture-specific tuning.
6. Implement the Apple Metal runtime and high-precision correctness path.
7. Add Metal quantized performance kernels and replay tuning.
8. Add optimized x86-64 and ARM64 CPU backends behind runtime dispatch.
9. Verify checkpoint continuation in every source/destination backend pair.
10. Add MTP and vision only after every required text path passes.

These are **Proposed** acceptance stages, not evidence that any backend has
already reached its gate. A stage records pinned artifacts, compiler and driver,
hardware, tolerances, allocation logs, and labeled performance results.

## DwarfStar transfer boundary

DwarfStar supplies evidence for narrow APIs, explicit session ownership,
checkpoint payloads, CPU oracle code, Metal scheduling, CUDA kernels, and
allocation accounting. This chapter does not change DwarfStar's implementation
language or runtime API. Its model-specific graph and storage choices remain a
case study, not the portable interface for this Qwen engine.

## Common failures

- Calling C++ “faster” without identifying generated code or system access.
- Making primitive operator dispatch the production scheduling boundary.
- Treating a backend repack as the canonical model artifact.
- Requiring checkpoint files to preserve device addresses or tiled layouts.
- Applying the 5090 memory plan unchanged to Spark because both use CUDA.
- Describing Apple unified memory as a hidden copy between RAM and VRAM.
- Optimizing the scalar oracle until it is no longer easy to inspect.
- Reporting a cross-platform performance number without its evidence label.

## Exercise and expected result

Trace one checkpoint from scalar CPU export to Metal import and one weight tensor
from the canonical artifact to `sm_120` and ARM64 SIMD repacks. Expected: the
reader can name the shared logical fields, explain why physical layouts may
differ, identify every repack cache-key component, and state why the 5090 and
Spark require different memory policies despite sharing CUDA.

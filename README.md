# Quartz Watch 38

**Quartz Watch 38 — Swiss-made inference for Qwen3.8-27B. It ticks fast.**

Quartz Watch 38 (`qw38`) is a deliberately narrow, work-in-progress inference
engine for the pinned Qwen3.8-27B Q4_K_M artifact on one RTX 5090. The approved
scope is in [plan.md](plan.md), and implementation claims and evidence are in
[implementation_ledger.md](implementation_ledger.md).

The repository is not yet a usable inference server. Current binaries fail
closed when an operation has not passed its delivery gate.

The beginner-oriented [implementation handbook](docs/README.md) explains each
admitted concept and links it to code, fixtures, failures, and evidence. The
latest chapter covers [GGUF parameter conversion](docs/22-gguf-conversion.md),
including folded decay, RMSNorm scales, squeezed shapes, and GDN head order.
The following [typed-weight chapter](docs/23-typed-model-weights.md) explains how
all 851 admitted ranges become safe global and per-layer fields.
The [packed-projection chapter](docs/24-packed-projections.md) then fixes the
different GDN and attention slicing rules used by scalar execution.

## Build

```sh
make
make test
```

The host-only build establishes and tests the public boundary. CUDA compilation
uses the immutable CUDA 13.0 container:

```sh
make cuda-image
make cuda-build
```

Run `build/qw38-eval --build-info` to inspect the compiled target and artifact
pin. No model is downloaded by the build.
